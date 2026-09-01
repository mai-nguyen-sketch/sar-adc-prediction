from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from sar_adc import Predictor, SARConfig, SARConverter
from simulation import SignalGenerator
from metrics import prediction_gain

# Aktivierungsfunktionen
activation_functions: dict[str, Callable[[], nn.Module]] = {
    "relu":   nn.ReLU,
    "linear": nn.Identity,
    "tanh":   nn.Tanh,
}

# Optimierer
optimizer_fact: dict[str, Callable] = {
    "SGD":          lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.0),
    "SGD_momentum": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "RMSprop":      lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.99),
    "Adam":         lambda params: torch.optim.Adam(params, lr=0.001),
    "AdamW":        lambda params: torch.optim.AdamW(params, lr=0.001),
}

LAYER_TYPES = ("feedforward", "cnn", "lstm", "gru", "transformer")
NORMALIZATIONS = ("batch", "layer", "group")
ALL_SIGNAL_TYPES = ("sine", "multitone", "ecg_like", "quiescent", "random_walk")


# Normalisierungs-Wrapper
class _Norm2D(nn.Module):
    def __init__(self, kind: str, features: int, num_groups: int = 4):
        super().__init__()
        if kind not in NORMALIZATIONS:
            raise ValueError(f"Unbekannte Normalisierung: {kind!r}")
        self.kind = kind
        if kind == "batch":
            self.norm = nn.BatchNorm1d(features)
        elif kind == "layer":
            self.norm = nn.LayerNorm(features)
        else:
            g = num_groups if features % num_groups == 0 else 1
            self.norm = nn.GroupNorm(g, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "group":
            return self.norm(x.unsqueeze(-1)).squeeze(-1)
        return self.norm(x)


class _Norm3D(nn.Module):
    def __init__(self, kind: str, channels: int, num_groups: int = 4):
        super().__init__()
        if kind not in NORMALIZATIONS:
            raise ValueError(f"Unbekannte Normalisierung: {kind!r}")
        self.kind = kind
        if kind == "batch":
            self.norm = nn.BatchNorm1d(channels)
        elif kind == "layer":
            self.norm = nn.LayerNorm(channels)
        else:
            g = num_groups if channels % num_groups == 0 else 1
            self.norm = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects x shape: (Batch, Sequence, Channels)
        if self.kind == "layer":
            return self.norm(x)
        # BatchNorm1d & GroupNorm expect (Batch, Channels, Sequence)
        x_t = x.transpose(1, 2)
        x_t = self.norm(x_t)
        return x_t.transpose(1, 2)


# DNN-Netzwerk
class _DNNNet(nn.Module):
    def __init__(self, L: int, H1: int, H2: int, layer_type: str, activation: str,
                 normalization: str, n_heads: int = 2, kernel_size: int = 3,
                 num_groups: int = 4):
        super().__init__()
        if layer_type not in LAYER_TYPES:
            raise ValueError(f"Unbekannter layer_type: {layer_type!r}")
        if activation not in activation_functions:
            raise ValueError(f"Unbekannte Aktivierung: {activation!r}")

        self.layer_type = layer_type
        act_cls = activation_functions[activation]

        if layer_type == "feedforward":
            self.fc1 = nn.Linear(L, H1)
            self.norm1 = _Norm2D(normalization, H1, num_groups)
            self.act1 = act_cls()
            self.fc2 = nn.Linear(H1, H2)
            self.norm2 = _Norm2D(normalization, H2, num_groups)
            self.act2 = act_cls()

        elif layer_type == "cnn":
            self.conv1 = nn.Conv1d(1, H1, kernel_size=kernel_size, padding=kernel_size // 2)
            self.norm1 = _Norm3D(normalization, H1, num_groups)
            self.act1 = act_cls()
            self.conv2 = nn.Conv1d(H1, H2, kernel_size=kernel_size, padding=kernel_size // 2)
            self.norm2 = _Norm3D(normalization, H2, num_groups)
            self.act2 = act_cls()

        elif layer_type in ("lstm", "gru"):
            rnn_cls = nn.LSTM if layer_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(input_size=1, hidden_size=H1, batch_first=True)
            self.norm1 = _Norm2D(normalization, H1, num_groups)
            self.act1 = act_cls()
            self.fc2 = nn.Linear(H1, H2)
            self.norm2 = _Norm2D(normalization, H2, num_groups)
            self.act2 = act_cls()

        else:  # "transformer"
            if H1 % n_heads != 0:
                n_heads = 1
            self.embed = nn.Linear(1, H1)
            self.pos_embed = nn.Parameter(torch.zeros(1, L, H1))
            nn.init.normal_(self.pos_embed, std=0.02)
            self.attn = nn.MultiheadAttention(H1, n_heads, batch_first=True)
            self.norm1 = _Norm3D(normalization, H1, num_groups)
            self.ff1 = nn.Linear(H1, H1 * 2)
            self.act1 = act_cls()
            self.ff2 = nn.Linear(H1 * 2, H1)
            # KORREKTUR: norm2 benötigt H1 als Channel-Dimension, da tok & ff die Form (B, L, H1) haben
            self.norm2 = _Norm3D(normalization, H1, num_groups)
            self.pool_fc = nn.Linear(H1, H2)
            self.act2 = act_cls()

        self.head = nn.Linear(H2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "feedforward":
            h = self.act1(self.norm1(self.fc1(x)))
            h = self.act2(self.norm2(self.fc2(h)))

        elif self.layer_type == "cnn":
            h = x.unsqueeze(1)
            h = self.conv1(h)
            h_cl = h.transpose(1, 2)
            h_cl = self.act1(self.norm1(h_cl))
            h = h_cl.transpose(1, 2)
            h = self.conv2(h)
            h_cl = h.transpose(1, 2)
            h_cl = self.act2(self.norm2(h_cl))
            h = h_cl.mean(dim=1)

        elif self.layer_type in ("lstm", "gru"):
            seq = x.unsqueeze(-1)
            out, _ = self.rnn(seq)
            h_last = out[:, -1, :]
            h = self.act1(self.norm1(h_last))
            h = self.act2(self.norm2(self.fc2(h)))

        else:  # "transformer"
            tok = self.embed(x.unsqueeze(-1)) + self.pos_embed
            attn_out, _ = self.attn(tok, tok, tok)
            tok = self.norm1(tok + attn_out)
            ff = self.ff2(self.act1(self.ff1(tok)))
            tok = self.norm2(tok + ff)
            pooled = tok.mean(dim=1)
            h = self.act2(self.pool_fc(pooled))

        return self.head(h).squeeze(-1)


# DNNPredictor
class DNNPredictor(Predictor):
    name = "DNN (konfigurierbar)"

    def __init__(
            self, L: int = 16, H1: int = 32, H2: int = 16,
            layer_type: str = "feedforward", activation: str = "relu",
            normalization: str = "batch", optimizer: str = "Adam",
            n_heads: int = 2, kernel_size: int = 3, num_groups: int = 4,
            n_epochs: int = 20, batch_size: int = 16,
            patience: int = 8, val_fraction: float = 0.2, clip_grad_norm: float = 1.0,
            online_lr: float = 0.0001, seed: Optional[int] = None, device: str = "cpu",
    ):
        self.L, self.H1, self.H2 = L, H1, H2
        self.layer_type = layer_type
        self.activation = activation
        self.normalization = normalization
        self.optimizer_name = optimizer
        self.n_heads, self.kernel_size, self.num_groups = n_heads, kernel_size, num_groups
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.patience, self.val_fraction = patience, val_fraction
        self.clip_grad_norm = clip_grad_norm
        self.online_lr = online_lr
        self._seed = seed
        self.device = torch.device(device)
        self._x_min, self._x_max = 0.0, 1.0
        self._is_trained = False
        self._buffer: list[float] = []
        self._loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)
        self.net = _DNNNet(self.L, self.H1, self.H2, self.layer_type, self.activation,
                           self.normalization, self.n_heads, self.kernel_size,
                           self.num_groups).to(self.device)
        self._opt = optimizer_fact[self.optimizer_name](self.net.parameters())
        self._online_opt = torch.optim.Adam(self.net.parameters(), lr=self.online_lr)
        self._is_trained = False
        self._buffer = []

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        span = max(self._x_max - self._x_min, 1e-12)
        return (x - self._x_min) / span

    def _denormalize(self, y: float) -> float:
        return float(y * (self._x_max - self._x_min) + self._x_min)

    def train_offline(self, x_raw: np.ndarray) -> dict:
        self._x_min, self._x_max = float(np.min(x_raw)), float(np.max(x_raw))
        x_norm = self._normalize(x_raw)

        X = np.stack([x_norm[i - self.L:i] for i in range(self.L, len(x_norm))])
        Y = x_norm[self.L:]
        n_val = max(1, int(len(X) * self.val_fraction))
        X_tr, Y_tr = X[:-n_val], Y[:-n_val]
        X_val = torch.as_tensor(X[-n_val:], dtype=torch.float32, device=self.device)
        Y_val = torch.as_tensor(Y[-n_val:], dtype=torch.float32, device=self.device)

        best_val = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        no_improve = 0
        rng = np.random.default_rng(self._seed)
        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

        for epoch in range(self.n_epochs):
            idx = rng.permutation(len(X_tr))
            self.net.train()
            ep_loss, n_batch = 0.0, 0

            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]
                if len(batch) < 2:
                    continue
                xb = torch.as_tensor(X_tr[batch], dtype=torch.float32, device=self.device)
                yb = torch.as_tensor(Y_tr[batch], dtype=torch.float32, device=self.device)

                self._opt.zero_grad()
                y_pred = self.net(xb)
                loss = self._loss_fn(y_pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
                self._opt.step()

                ep_loss += loss.item()
                n_batch += 1

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val)
                val_loss = self._loss_fn(val_pred, Y_val).item()

            history["train_loss"].append(ep_loss / max(n_batch, 1))
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                history["best_epoch"] = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.net.load_state_dict(best_state)
        self._is_trained = True
        return history

    def predict(self, history: np.ndarray) -> float:
        if history.size < self.L:
            return float(history[-1]) if history.size > 0 else 0.0
        window = self._normalize(history[-self.L:])
        x_t = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            y_norm = self.net(x_t)
        return self._denormalize(float(np.clip(y_norm.item(), -0.5, 1.5)))

    def update(self, x_true: float, x_hat: float) -> None:
        self._buffer.append(x_true)
        if not self._is_trained or len(self._buffer) < self.L + 1:
            return None

        window = self._normalize(np.array(self._buffer[-(self.L + 1):-1]))
        y_true_norm = (x_true - self._x_min) / max(self._x_max - self._x_min, 1e-12)

        x_t = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_t = torch.as_tensor([y_true_norm], dtype=torch.float32, device=self.device)

        self.net.eval()
        self._online_opt.zero_grad()
        y_pred = self.net(x_t)
        loss = self._loss_fn(y_pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
        self._online_opt.step()

    def reset(self) -> None:
        self._init_weights()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


# Benchmark-Runner
@dataclass
class BenchmarkResult:
    signal_type: str
    layer_type: str
    activation: str
    optimizer: str
    normalization: str
    seed: int
    best_epoch: int
    best_val_loss: float
    test_rmse: float
    test_pred_gain_db: float
    mean_cycles: float
    energy_saving: float
    n_parameters: int
    train_time_s: float


def run_dnn_benchmark(signal_types: tuple = ALL_SIGNAL_TYPES,
                      signal_kwargs_map: Optional[dict[str, dict]] = None,
                      n_samples: int = 2000, n_runs: int = 1, base_seed: int = 77,
                      train_frac: float = 0.7, L: int = 16, H1: int = 32, H2: int = 16,
                      n_epochs: int = 15, fs: float = 1000.0,
                      layer_types: tuple = LAYER_TYPES,
                      activations: tuple = tuple(activation_functions.keys()),
                      optimizers: tuple = tuple(optimizer_fact.keys()),
                      normalizations: tuple = NORMALIZATIONS,
                      ) -> list[BenchmarkResult]:
    """
    Rastersuche über Signaltyp × Layer-Typ × Aktivierung × Optimizer × Normalisierung.
    """
    signal_kwargs_map = signal_kwargs_map or {}
    sar_cfg = SARConfig(n_bits=10, v_ref=1.0, unipolar=True, comparator_noise_std=0.0005)
    results: list[BenchmarkResult] = []
    # Äußere Schleife geht alle gewünschten Signale nacheinander durch
    for sig_type in signal_types:
        sig_kwargs = signal_kwargs_map.get(sig_type, {})
        configs = product(layer_types, activations, optimizers, normalizations)

        for (layer_type, act, opt, norm), seed_offset in product(configs, range(n_runs)):
            seed = base_seed + seed_offset
            gen = SignalGenerator(fs=fs, seed=seed)
            x = gen.generate(sig_type, n_samples, **sig_kwargs)
            n_train = int(n_samples * train_frac)
            x_train, x_test = x[:n_train], x[n_train:]

            dnn = DNNPredictor(L=L, H1=H1, H2=H2, layer_type=layer_type, activation=act,
                               normalization=norm, optimizer=opt, n_epochs=n_epochs, seed=seed)

            t0 = time.perf_counter()
            hist = dnn.train_offline(x_train)
            train_time = time.perf_counter() - t0

            rng = np.random.default_rng(seed)
            conv = SARConverter(sar_cfg, predictor=dnn, rng=rng)
            conv_results = conv.convert_sequence_pretrained(x_test)

            recon = np.array([r.voltage for r in conv_results])
            x_hats = np.array([r.x_hat if r.x_hat is not None else recon[i]
                               for i, r in enumerate(conv_results)])
            cycles = np.array([r.n_cycles for r in conv_results])

            rmse = float(np.sqrt(np.mean((recon - x_test) ** 2)))
            gp = prediction_gain(x_test, x_hats)
            mean_c = float(np.mean(cycles))
            e_save = 1.0 - mean_c / sar_cfg.n_bits

            results.append(BenchmarkResult(
                signal_type=sig_type, layer_type=layer_type, activation=act,
                optimizer=opt, normalization=norm, seed=seed,
                best_epoch=hist["best_epoch"], best_val_loss=float(min(hist["val_loss"])),
                test_rmse=rmse, test_pred_gain_db=gp if np.isfinite(gp) else float("nan"),
                mean_cycles=mean_c, energy_saving=e_save,
                n_parameters=dnn.n_parameters, train_time_s=train_time,
            ))
    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    from collections import defaultdict
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r.signal_type, r.layer_type, r.activation, r.optimizer, r.normalization)].append(r)

    print(f"\n{'Signal':<13}{'Layer':<13}{'Aktivierung':<12}{'Optimizer':<15}{'Norm':<7}"
          f"{'Val-Loss':>10}{'best ep.':>9}{'RMSE':>10}{'Gp [dB]':>9}{'E_save[%]':>10}"
          f"{'#Params':>10}{'t_train':>10}")
    print("─" * 135)

    for (sig_type, layer_type, act, opt, norm), rlist in sorted(grouped.items()):
        val_losses = [r.best_val_loss for r in rlist]
        best_epochs = [r.best_epoch for r in rlist]
        rmses = [r.test_rmse for r in rlist]
        gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
        esaves = [r.energy_saving * 100 for r in rlist]
        n_params = [r.n_parameters for r in rlist]
        times = [r.train_time_s for r in rlist]

        gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
        print(f"{sig_type:<13}{layer_type:<13}{act:<12}{opt:<15}{norm:<7}"
              f"{np.mean(val_losses):>10.6f}{np.mean(best_epochs):>9.1f}"
              f"{np.mean(rmses):>10.5f}{gp_str:>9}{np.mean(esaves):>10.1f}"
              f"{int(np.mean(n_params)):>10}{np.mean(times):>9.2f}s")

    best = min(grouped.items(), key=lambda kv: np.mean([r.best_val_loss for r in kv[1]]))
    print(f"\n→ Beste Gesamtkonfiguration (geringster Val-Loss): "
          f"Signal='{best[0][0]}', Layer='{best[0][1]}', Aktivierung='{best[0][2]}', "
          f"Optimizer='{best[0][3]}', Normalisierung='{best[0][4]}'")


def print_axis_comparison(results: list[BenchmarkResult]) -> None:
    from collections import defaultdict
    axes = {
        "Signaltyp":       lambda r: r.signal_type,
        "Layer-Typ":       lambda r: r.layer_type,
        "Aktivierung":     lambda r: r.activation,
        "Optimizer":       lambda r: r.optimizer,
        "Normalisierung":  lambda r: r.normalization,
    }
    print("\n Einzelvergleich aller Achsen "
          "(gemittelt über alle übrigen Konfigurationen) ──")

    for axis_name, keyfn in axes.items():
        grouped: dict[str, list] = defaultdict(list)
        for r in results:
            grouped[keyfn(r)].append(r)

        print(f"\n  {axis_name}:")
        print(f"    {'Wert':<14}{'Val-Loss':>10}{'RMSE':>10}{'Gp [dB]':>9}"
              f"{'E_save[%]':>11}{'#Params':>10}{'t_train':>10}")
        for val, rlist in sorted(grouped.items()):
            val_losses = [r.best_val_loss for r in rlist]
            rmses = [r.test_rmse for r in rlist]
            gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
            esaves = [r.energy_saving * 100 for r in rlist]
            n_params = [r.n_parameters for r in rlist]
            times = [r.train_time_s for r in rlist]

            gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
            print(f"    {val:<14}{np.mean(val_losses):>10.6f}{np.mean(rmses):>10.5f}"
                  f"{gp_str:>9}{np.mean(esaves):>11.1f}{int(np.mean(n_params)):>10}"
                  f"{np.mean(times):>9.2f}s")


def print_best_overall_config(results: list[BenchmarkResult]) -> None:
    """
    Ermittelt die Konfiguration (Layer, Aktivierung, Optimizer, Normalisierung), die, gemittelt über alle Signaltypen, den geringsten Val-Loss erzielt.
    """
    from collections import defaultdict

    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r.layer_type, r.activation, r.optimizer, r.normalization)].append(r)

    all_signals = {r.signal_type for r in results}
    n_signals = len(all_signals)

    # Nur Konfigurationen berücksichtigen, die auf allen Signaltypen getestet wurden
    complete = {k: v for k, v in grouped.items()
                if len({r.signal_type for r in v}) == n_signals}
    if not complete:
        print("\n⚠ Keine Konfiguration deckt alle Signaltypen ab – "
              "'Beste Konfiguration über alle Signale' kann nicht berechnet werden.")
        return

    def per_signal_means(rlist, attr, scale=1.0, only_finite=False):
        by_signal: dict[str, list] = defaultdict(list)
        for r in rlist:
            v = getattr(r, attr) * scale
            if only_finite and not np.isfinite(v):
                continue
            by_signal[r.signal_type].append(v)
        means = [np.mean(v) for v in by_signal.values() if v]
        return means

    def mean_val_loss(rlist):
        means = per_signal_means(rlist, "best_val_loss")
        return np.mean(means)

    best_key, best_rlist = min(complete.items(), key=lambda kv: mean_val_loss(kv[1]))

    print(f"\n── Beste Konfiguration über alle Signaltypen gemittelt "
          f"({n_signals} Signale, je gleich gewichtet) ──")
    print(f"→ Layer='{best_key[0]}', Aktivierung='{best_key[1]}', "
          f"Optimizer='{best_key[2]}', Normalisierung='{best_key[3]}'")
    print(f"   mittlerer Val-Loss: {mean_val_loss(best_rlist):.6f}\n")

    by_signal: dict[str, list] = defaultdict(list)
    for r in best_rlist:
        by_signal[r.signal_type].append(r)

    print(f"    {'Signal':<13}{'Val-Loss':>10}{'RMSE':>10}{'Gp [dB]':>9}"
          f"{'E_save[%]':>11}{'t_train':>10}")
    for sig, rlist in sorted(by_signal.items()):
        val_losses = [r.best_val_loss for r in rlist]
        rmses = [r.test_rmse for r in rlist]
        gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
        esaves = [r.energy_saving * 100 for r in rlist]
        times = [r.train_time_s for r in rlist]
        gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
        print(f"    {sig:<13}{np.mean(val_losses):>10.6f}{np.mean(rmses):>10.5f}"
              f"{gp_str:>9}{np.mean(esaves):>11.1f}{np.mean(times):>9.2f}s")


def print_config_overview() -> None:
    print("\n── Verglichene Methodiken ──")
    print(f"  Signaltypen:               {', '.join(ALL_SIGNAL_TYPES)}")
    print(f"  Layer-Konzepte:            {', '.join(LAYER_TYPES)}")
    print(f"  Aktivierungsfunktionen:    {', '.join(activation_functions.keys())}")
    print(f"  Optimierer:                {', '.join(optimizer_fact.keys())}")
    print(f"  Normalisierung:            {', '.join(NORMALIZATIONS)}")


if __name__ == "__main__":
    print(" DNN-Benchmark: Signaltyp × Layer-Typ × Aktivierung × Optimizer × Normalisierung ")
    print_config_overview()

    # Parameterzuordnung für spezifische Signale
    signal_kwargs_map = {
        "quiescent": {"change_prob": 0.015, "step_std": 0.07},
        "sine": {"freq": 5.0, "amplitude": 0.45},
        "multitone": {"freqs": (3.0, 7.0, 11.0)},
        "ecg_like": {"beat_period": 150},
        "random_walk": {"step_std": 0.01},
    }

    results = run_dnn_benchmark(
        signal_types=ALL_SIGNAL_TYPES,
        signal_kwargs_map=signal_kwargs_map,
        n_samples=1500,
        n_runs=1,
        base_seed=77,
        train_frac=0.7,
        L=50,
        H1=32,
        H2=16,
        n_epochs=15,
    )

    print_benchmark_table(results)
    print_axis_comparison(results)
    print_best_overall_config(results)

    elapsed = time.perf_counter()
    print(f"\n{'═' * 70}")
    print(f"  Simulation abgeschlossen in {elapsed:.1f} s")
    print(f"{'═' * 70}")