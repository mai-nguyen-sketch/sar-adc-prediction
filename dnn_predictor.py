from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sar_adc import Predictor

torch.set_num_threads(1)

def _make_windows_advanced(x_norm: np.ndarray, L: int):
    """
    Erstellt Sliding-Windows mit Absolutwerten UND differentiellen Deltas Input-Feature-Dim verdoppelt sich: [x_t, delta_x_t]
    """
    X_vals = np.array([x_norm[i - L:i] for i in range(L, len(x_norm))])
    # Differentieller Feature-Kanal
    deltas = np.diff(x_norm, prepend=x_norm[0])
    X_deltas = np.array([deltas[i - L:i] for i in range(L, len(x_norm))])

    # Kanäle zusammenfügen (Stacking: L Absolutwerte + L Differenzen = 2*L Features)
    X = np.concatenate([X_vals, X_deltas], axis=1)
    Y = x_norm[L:]
    return X, Y


def _baseline_mix_alpha(recent_norm: np.ndarray) -> float:
    '''
    Berechnet dynamisch den Mischfaktor alpha für die Baseline-Extrapolation.
    Bestimmt das Gewicht `alpha` basierend auf den Varianzen der ersten (`diff1`) und zweiten Ableitung (`diff2`) des kausalen Historienfensters. Ein höheres
    `alpha` bevorzugt die 2-Punkt-Extrapolation bei stetigen Verläufen, während ein kleineres `alpha` bei verrauschten Signalen zur stabilen Konstanten-Projektion neigt.

    Args:
        recent_norm (np.ndarray): Ausschnitt des bisherigen normalisierten Signals (kausales Fenster).

    Returns:
        float: Der berechnete Mischfaktor `alpha` im geschlossenen Intervall [0.0, 1.0].
            Gibt 1.0 als Standardwert bei unzureichender Fenstergröße (< 4) oder verschwindender Gesamtvarianz zurück.
    '''
    if recent_norm.size < 4:
        return 1.0  # zu wenig Kontext -> Standardverhalten (reine 2-Punkt-Baseline)
    diff1 = np.diff(recent_norm)
    diff2 = np.diff(recent_norm, n=2)
    if diff2.size < 1:
        return 1.0
    var1 = float(np.var(diff1))
    var2 = float(np.var(diff2))
    denom = var1 + var2
    if denom < 1e-12:
        return 1.0  # entartetes/konstantes Fenster -> Baseline-Wahl irrelevant
    alpha = var1 / denom
    return float(np.clip(alpha, 0.0, 1.0))


def _baseline_series(x_norm: np.ndarray, window: int = 32) -> np.ndarray:
    """
    Berechnet eine kausale Baseline-Reihe für ein normalisiertes Signal.
    Die Baseline wird schrittweise für jeden Zeitpunkt $i$ basierend auf den vorherigen Werten bestimmt (ohne Zugriff auf $x[i]$). Sie kombiniert eine Konstanten-Projektion ($x[i-1]$) und eine lineare 2-Punkt-Extrapolation
    ($2 \cdot x[i-1] - x[i-2]$) über einen dynamischen Mischfaktor ($\alpha$),der aus dem kausalen Historienfenster ermittelt wird.

    Argunente:
        x_norm (np.ndarray): Normalisiertes 1D-Eingangssignal
        window (int, optional): Größe des Historienfensters zur Bestimmung des Mischfaktors `alpha`. Standardmäßig 32.

    Returns:
        np.ndarray: Array der gleichen Länge wie `x_norm` mit der berechneten Baseline-Reihe.
    """
    n = len(x_norm)
    baseline = np.empty(n, dtype=float)
    if n > 0:
        baseline[0] = x_norm[0]
    if n > 1:
        baseline[1] = x_norm[0]
    for i in range(2, n):
        recent = x_norm[max(0, i - window):i]  # kausal: nur Vergangenheit, ohne x[i]
        alpha = _baseline_mix_alpha(recent)
        x_prev, x_prev2 = x_norm[i - 1], x_norm[i - 2]
        extrap_2pt = 2.0 * x_prev - x_prev2
        baseline[i] = (1.0 - alpha) * x_prev + alpha * extrap_2pt
    return baseline


def _make_windows_residual(x_norm: np.ndarray, L: int, baseline_window: int = 32):
    """
    Wie _make_windows_advanced(), liefert aber zusätzlich das Residual-Ziel (Y_true - Y_baseline) statt des rohen Zielwerts, sowie die Baseline-Werte selbst (für Diagnosezwecke/Verifikation).
    baseline_window steuert die Fenstergröße der adaptiven Baseline-Mischung (Punkt 1), siehe _baseline_series().
    """
    X, Y_true = _make_windows_advanced(x_norm, L)
    baseline = _baseline_series(x_norm, window=baseline_window)
    Y_baseline = baseline[L:]
    Y_residual = Y_true - Y_baseline
    return X, Y_residual, Y_baseline


class _TorchDNNNet(nn.Module):
    """
    DNN mit GRU-Backbone und Multi-Feature-Eingabe (Absolutwerte + Deltas)
    (Layer='gru', Aktivierung='relu', Optimizer='Adam', Normalisierung='group'):
    - Rekurrenz: eine GRU-Schicht (statt LSTM) verarbeitet die Eingabe als Sequenz von L Zeitschritten mit je 2 Features (Absolutwert + Delta)
      GRU besitzt nur den versteckten Zustand h; der letzte Zeitschritt liefert den Zustandsvektor für den Kopf.
    - Aktivierung: ReLU nach jeder Normalisierung.
    - Normalisierung: GroupNorm (num_groups Gruppen) nach dem GRU-Zustand und nach der zweiten Dense-Schicht.
    """

    def __init__(self, L: int, H1: int, H2: int, num_groups: int = 4):
        super().__init__()
        self.L = L
        self.gru = nn.GRU(input_size=2, hidden_size=H1, batch_first=True)
        self.norm1 = nn.GroupNorm(num_groups, H1)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(H1, H2)
        self.norm2 = nn.GroupNorm(num_groups, H2)
        self.act2 = nn.ReLU()

        self.head = nn.Linear(H2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2*L) - erste Hälfte Absolutwerte, zweite Hälfte Deltas
        vals = x[:, :self.L]
        deltas = x[:, self.L:]
        seq = torch.stack([vals, deltas], dim=-1)  # (batch, L, 2)

        # GRU liefert nur h (kein Zellzustand c wie beim LSTM)
        out, _ = self.gru(seq)
        h_last = out[:, -1, :]  # letzter Zeitschritt -> (batch, H1)

        h = self.act1(self.norm1(h_last))
        h = self.act2(self.norm2(self.fc2(h)))

        y = self.head(h).squeeze(-1)
        return y


class DnnTorchPredictor(Predictor):
    name = "DNN (GRU, Tanh, Adam, GroupNorm)"

    def __init__(
            self,
            L: int = 16,
            H1: int = 32,
            H2: int = 16,
            num_groups: int = 4,
            lr: float = 0.002,
            weight_decay: float = 1e-4,
            n_epochs: int = 60,
            batch_size: int = 32,
            patience: int = 12,
            val_fraction: float = 0.2,
            clip_grad: float = 1.0,
            online_lr: float = 0.0001,
            norm_window: int = 200,
            norm_margin_frac: float = 0.05,
            baseline_window: int = 32,
            adaptive_search_bits: bool = True,
            # (random_walk) verschlechtert.
            search_percentile: float = 90.0,
            search_adapt_rate: float = 0.08,
            search_window_samples: int = 100,
            search_margin_bits: int = 1,
            min_search_bits: int = 2,
            max_search_bits_cap: Optional[int] = None,
            n_bits: int = 10,
            v_ref: float = 1.0,
            seed: Optional[int] = None,
    ):
        self.L = L
        self.H1 = H1
        self.H2 = H2
        self.num_groups = num_groups
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.clip_grad = clip_grad
        self.online_lr = online_lr
        self.norm_window = norm_window
        self.norm_margin_frac = norm_margin_frac
        self.baseline_window = baseline_window
        self._max_search_bits_init = max_search_bits
        self.adaptive_search_bits = adaptive_search_bits
        self.search_percentile = search_percentile
        self.search_adapt_rate = search_adapt_rate
        self.search_window_samples = search_window_samples
        self.search_margin_bits = search_margin_bits
        self.min_search_bits = min_search_bits
        self.n_bits = n_bits
        self.v_ref = v_ref
        self.max_search_bits_cap = max_search_bits_cap if max_search_bits_cap is not None else n_bits // 2
        self._lsb = self.v_ref / (2 ** self.n_bits)
        self._seed = seed
        self._x_min = 0.0
        self._x_max = 1.0
        self._buffer: list[float] = []
        self._is_trained = False
        self._init_weights()

    def _init_weights(self) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)

        self._net = _TorchDNNNet(self.L, self.H1, self.H2, self.num_groups)

        # Adam
        self._opt = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Separater Online-Optimizer für die inkrementelle Feinabstimmung pro neu beobachtetem Abtastwert (Batch-Größe 1)
        self._online_opt = optim.Adam(self._net.parameters(), lr=self.online_lr, weight_decay=self.weight_decay)

        # Kein Scheduler -> konstante Learning Rate über das gesamte Training

        self._loss_fn = nn.MSELoss()
        self._buffer = []
        self._is_trained = False

        # Reset des laufzeitadaptiven Suchfensters auf den konfigurierten Startwert sowie des Online-Quantil-Trackers.
        self.max_search_bits = self._max_search_bits_init
        self._q_log: Optional[float] = None  # laufender Quantil-Schätzer in log2(Codes)
        self._n_obs: int = 0

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._x_min) / max(self._x_max - self._x_min, 1e-12)

    def _denormalize(self, y: float) -> float:
        return float(y * (self._x_max - self._x_min) + self._x_min)

    def _baseline_adaptive(self, norm_hist_tail: np.ndarray) -> float:
        """
        Adaptive Baseline zur Inferenzzeit, analog zu _baseline_series() für die Trainingsphase.
        norm_hist_tail ist die kausale, normierte Historie bis einschließlich des zuletzt beobachteten Samples (norm_hist_tail[-1]); der vorherzusagende Wert ist nicht enthalten.
        """
        if norm_hist_tail.size == 0:
            return 0.0
        if norm_hist_tail.size == 1:
            return float(norm_hist_tail[-1])
        recent = norm_hist_tail[-self.baseline_window:]
        alpha = _baseline_mix_alpha(recent)
        x_prev, x_prev2 = norm_hist_tail[-1], norm_hist_tail[-2]
        extrap_2pt = 2.0 * x_prev - x_prev2
        return float((1.0 - alpha) * x_prev + alpha * extrap_2pt)

    def _update_norm_stats(self) -> None:
        """
        laufende Normierung, siehe SnnTorchPredictor._update_norm_stats() für Begründung. Wird kausal nach jedem neu beobachteten Abtastwert aufgerufen (in update()).
        """
        if len(self._buffer) < max(8, self.L // 2):
            return
        recent = np.asarray(self._buffer[-self.norm_window:], dtype=float)
        new_min, new_max = float(np.min(recent)), float(np.max(recent))
        span = new_max - new_min
        if span < 1e-9:
            return
        margin = self.norm_margin_frac * span
        self._x_min = new_min - margin
        self._x_max = new_max + margin

    def _update_search_window(self, x_true: float, x_hat: float) -> None:
        """
        Aktualisiert das adaptive Suchfenster (in Bits) basierend auf dem Quantisierungsfehler.
        Verwendet einen stochastischen Gradientenabstieg (Quantilsschätzer), um das erforderliche Bit-Budget für den Kodierungsfehler zu schätzen.
        Bei einem Kaltstart wird der Schätzer mit dem ersten beobachteten Wert initialisiert.
        Die Anpassung der aktiven Bit-Breite (`max_search_bits`) erfolgt erst nach Erreichen einer Mindestanzahl an Beobachtungen.

            Argumente:
                x_true (float): Der tatsächliche Zielwert.
                x_hat (float): Der vorhergesagte/rekonsruierte Wert.
            """
        if not self.adaptive_search_bits:
            return

        err_codes = abs(x_true - x_hat) / self._lsb
        y = float(np.log2(max(err_codes, 1e-6)))
        tau = self.search_percentile / 100.0

        if self._q_log is None:
            self._q_log = y  # Kaltstart: erste Beobachtung initialisiert den Schätzer
        else:
            indicator = 1.0 if y < self._q_log else 0.0
            self._q_log += self.search_adapt_rate * (tau - indicator)

        self._n_obs += 1

        # Erst ab einer Mindestmenge an Beobachtungen umschalten, sonst bleibt der konfigurierte Startwert (self._max_search_bits_init) aktiv
        if self._n_obs < max(8, self.search_window_samples // 4):
            return

        needed_bits = int(np.ceil(self._q_log)) + 1 + self.search_margin_bits
        needed_bits = int(np.clip(needed_bits, self.min_search_bits, self.max_search_bits_cap))
        self.max_search_bits = needed_bits

    def train_offline(self, x_raw: np.ndarray, verbose: bool = True) -> dict:
        self._x_min = float(np.min(x_raw))
        self._x_max = float(np.max(x_raw))
        x_norm = self._normalize(x_raw)

        X, Y, _Y_baseline = _make_windows_residual(x_norm, self.L, baseline_window=self.baseline_window)

        n_val = max(1, int(len(X) * self.val_fraction))
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        X_tr, Y_tr = X[:-n_val], Y[:-n_val]

        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

        best_val = float("inf")
        best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
        no_improve = 0
        rng = np.random.default_rng(self._seed)
        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

        for epoch in range(self.n_epochs):
            self._net.train()
            idx = rng.permutation(len(X_tr))
            ep_loss = 0.0
            n_batches = 0

            for start in range(0, len(idx), self.batch_size):
                batch = idx[start: start + self.batch_size]
                xb = torch.tensor(X_tr[batch], dtype=torch.float32)
                yb = torch.tensor(Y_tr[batch], dtype=torch.float32)

                self._opt.zero_grad()
                y_pred = self._net(xb)
                loss = self._loss_fn(y_pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), self.clip_grad)
                self._opt.step()

                ep_loss += loss.item()
                n_batches += 1

            # Validierung
            self._net.eval()
            with torch.no_grad():
                y_val = self._net(X_val_t)
                val_loss = self._loss_fn(y_val, Y_val_t).item()

            history["train_loss"].append(ep_loss / max(n_batches, 1))
            history["val_loss"].append(val_loss)

            if verbose and epoch % 5 == 0:
                print(f"  Epoche {epoch:3d}: val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                history["best_epoch"] = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(f"  Early Stopping (beste Epoche: {history['best_epoch']})")
                    break

        self._net.load_state_dict(best_state)
        self._is_trained = True
        if verbose:
            print(f"  Bester Val-Loss: {best_val:.6f}")
        return history

    def predict(self, history: np.ndarray) -> float:
        if history.size < 2:
            return float(history[-1]) if history.size > 0 else 0.0
        if history.size < self.L:
            # Nicht genug Kontext für ein volles Fenster: reine (adaptive) Baseline-Extrapolation ohne Netz
            norm_hist = self._normalize(history)
            baseline_norm = self._baseline_adaptive(norm_hist)
            return self._denormalize(baseline_norm)

        # Kontextfenster für die Baseline-Mischung kann breiter sein als L (baseline_window ist unabhängig vom Netz-Eingabefenster L)
        ctx_len = max(self.L, self.baseline_window)
        norm_ctx = self._normalize(history[-ctx_len:])
        norm_hist = norm_ctx[-self.L:]

        # Aufbereitung der historischen Daten mit Absolut- und Differenzwerten
        deltas = np.diff(norm_hist, prepend=norm_hist[0])
        combined_feat = np.concatenate([norm_hist, deltas])
        x_t = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0)
        baseline_norm = self._baseline_adaptive(norm_ctx)

        self._net.eval()
        with torch.no_grad():
            residual_norm = self._net(x_t)

        y_norm = baseline_norm + float(residual_norm.item())
        return self._denormalize(float(np.clip(y_norm, -0.5, 1.5)))

    def update(self, x_true: float, x_hat: float) -> None:
        """
        Online-Feinabstimmung:
        ein einzelner, gradientenbegrenzter AdamW-Schritt mit separater Online-Lernrate pro neu beobachtetem Abtastwert (Batch-Größe 1)
        """
        self._buffer.append(x_true)
        # laufende Normierung nach dem neuen Sample aktualisieren, damit predict() beim nächsten Aufruf die aktuellsten Grenzen nutzt
        self._update_norm_stats()
        # laufzeitadaptives Suchfenster nach jedem neu beobachteten Ist/Soll-Paar aktualisieren, damit die SARConverter-Suche beim nächsten Sample bereits das neue max_search_bits nutzt
        self._update_search_window(x_true, x_hat)

        if not self._is_trained or len(self._buffer) < self.L + 1:
            return None

        # gleiche adaptive Baseline wie in predict()/Training, aus einem ggf. breiteren Kontextfenster (baseline_window) als L
        ctx_len = max(self.L, self.baseline_window)
        ctx_vals = self._normalize(np.array(self._buffer[-(ctx_len + 1):-1]))
        window_vals = ctx_vals[-self.L:]
        deltas = np.diff(window_vals, prepend=window_vals[0])
        combined_feat = np.concatenate([window_vals, deltas])
        baseline_norm = self._baseline_adaptive(ctx_vals)
        y_true_norm = (x_true - self._x_min) / max(self._x_max - self._x_min, 1e-12)
        residual_target = y_true_norm - baseline_norm

        x_t = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor([residual_target], dtype=torch.float32)

        self._net.train()
        self._online_opt.zero_grad()
        y_pred = self._net(x_t)
        loss = self._loss_fn(y_pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), self.clip_grad)
        self._online_opt.step()
        self._net.eval()

    def reset(self) -> None:
        self._init_weights()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._net.parameters())