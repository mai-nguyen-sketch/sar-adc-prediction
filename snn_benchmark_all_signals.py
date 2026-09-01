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

torch.set_num_threads(1)

ALL_SIGNAL_TYPES = ("sine", "multitone", "ecg_like", "quiescent", "random_walk")


# Spike-Funktion mit austauschbarem Surrogate-Gradient
class _FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float, sharpness: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold, ctx.sharpness = threshold, sharpness
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.sharpness * (u - ctx.threshold))
        grad = ctx.sharpness * sig * (1.0 - sig)
        return grad_output * grad, None, None


class _ArcTanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float, sharpness: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold, ctx.sharpness = threshold, sharpness
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        z = ctx.sharpness * (u - ctx.threshold)
        grad = (ctx.sharpness / np.pi) / (1.0 + z ** 2)
        return grad_output * grad, None, None


class _TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float, sharpness: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold, ctx.sharpness = threshold, sharpness
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        grad = torch.clamp(1.0 - ctx.sharpness * torch.abs(u - ctx.threshold), min=0.0)
        return grad_output * grad, None, None

surrogate_functions: dict[str, Callable] = {
    "fast_sigmoid": _FastSigmoidSpike.apply,
    "arctan":       _ArcTanSpike.apply,
    "triangular":   _TriangularSpike.apply,
}

optimizer_fact: dict[str, Callable] = {
    "SGD":     lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "RMSprop": lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.99),
    "Adam":    lambda params: torch.optim.Adam(params, lr=0.001),
    "AdamW":   lambda params: torch.optim.AdamW(params, lr=0.001),
}

# LIF-Layer und SNN-Netzwerk
class _LIFLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, beta: float, threshold: float, surrogate: str, sharpness: float):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        nn.init.normal_(self.fc.weight, mean=0.0, std=np.sqrt(2.0 / n_in))  # He-Init
        nn.init.zeros_(self.fc.bias)
        self.beta, self.threshold, self.sharpness = beta, threshold, sharpness
        self.spike_fn = surrogate_functions[surrogate]
        self.n_out = n_out

    def forward(self, x: torch.Tensor, mem: torch.Tensor, spk_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self.fc(x)
        mem_new = self.beta * mem + cur - self.threshold * spk_prev
        spk = self.spike_fn(mem_new, self.threshold, self.sharpness)
        return spk, mem_new

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mem0 = torch.zeros(batch_size, self.n_out, device=device)
        spk0 = torch.zeros(batch_size, self.n_out, device=device)
        return mem0, spk0


class _SNNNet(nn.Module):
    def __init__(self, L: int, H1: int, H2: int, beta: float, threshold: float, surrogate: str, sharpness: float = 5.0, n_steps: int = 10):
        super().__init__()
        self.layer1 = _LIFLayer(L, H1, beta, threshold, surrogate, sharpness)
        self.layer2 = _LIFLayer(H1, H2, beta, threshold, surrogate, sharpness)
        limit = np.sqrt(6.0 / (H2 + 1))
        self.head = nn.Linear(H2, 1)
        nn.init.uniform_(self.head.weight, -limit, limit)
        nn.init.zeros_(self.head.bias)
        self.readout_beta = beta
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        device = x.device

        mem1, spk1 = self.layer1.init_state(batch, device)
        mem2, spk2 = self.layer2.init_state(batch, device)
        mem_out = torch.zeros(batch, 1, device=device)

        spk1_sum = torch.zeros_like(mem1)
        spk2_sum = torch.zeros_like(mem2)

        for _ in range(self.n_steps):
            spk1, mem1 = self.layer1(x, mem1, spk1)
            spk2, mem2 = self.layer2(spk1, mem2, spk2)
            mem_out = self.readout_beta * mem_out + self.head(spk2)
            spk1_sum = spk1_sum + spk1
            spk2_sum = spk2_sum + spk2

        y = mem_out.squeeze(-1)
        spike_rate1 = spk1_sum / self.n_steps
        spike_rate2 = spk2_sum / self.n_steps
        return y, spike_rate1, spike_rate2


# SNNPredictor
class SNNPredictor(Predictor):
    name = "SNN (LIF, 2-schichtig)"

    def __init__(
            self, L: int = 16, H1: int = 32, H2: int = 16, beta: float = 0.9,
            threshold: float = 1.0, surrogate: str = "fast_sigmoid",
            surrogate_sharpness: float = 5.0, n_steps: int = 10,
            optimizer: str = "Adam", n_epochs: int = 20, batch_size: int = 16,
            patience: int = 8, val_fraction: float = 0.2, clip_grad_norm: float = 1.0,
            online_lr: float = 0.0001, seed: Optional[int] = None, device: str = "cpu",
    ):
        self.L, self.H1, self.H2 = L, H1, H2
        self.beta, self.threshold = beta, threshold
        self.surrogate, self.surrogate_sharpness = surrogate, surrogate_sharpness
        self.n_steps = n_steps
        self.optimizer_name = optimizer
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.patience, self.val_fraction = patience, val_fraction
        self.clip_grad_norm = clip_grad_norm
        self.online_lr = online_lr
        self._seed = seed
        self.device = torch.device(device)
        self._x_min, self._x_max = 0.0, 1.0
        self._is_trained = False
        self._buffer: list[float] = []
        self._last_spike_rate = float("nan")
        self._test_spike_rates: list[float] = []
        self._init_weights()

    def _init_weights(self) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)
        self.net = _SNNNet(self.L, self.H1, self.H2, self.beta, self.threshold,
                           self.surrogate, self.surrogate_sharpness, self.n_steps).to(self.device)
        self._opt = optimizer_fact[self.optimizer_name](self.net.parameters())
        self._online_opt = torch.optim.Adam(self.net.parameters(), lr=self.online_lr)
        self._is_trained = False
        self._buffer = []
        self._test_spike_rates = []

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
        history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "train_spike_rates": []}

        for epoch in range(self.n_epochs):
            idx = rng.permutation(len(X_tr))
            self.net.train()
            ep_loss, ep_spikes, n_batch = 0.0, 0.0, 0

            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = torch.as_tensor(X_tr[batch], dtype=torch.float32, device=self.device)
                yb = torch.as_tensor(Y_tr[batch], dtype=torch.float32, device=self.device)

                self._opt.zero_grad()
                y_pred, s1, s2 = self.net(xb)
                loss = nn.functional.mse_loss(y_pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
                self._opt.step()

                ep_loss += loss.item()
                ep_spikes += ((s1.mean() + s2.mean()) / 2).item()
                n_batch += 1

            self.net.eval()
            with torch.no_grad():
                val_pred, _, _ = self.net(X_val)
                val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

            history["train_loss"].append(ep_loss / max(n_batch, 1))
            history["val_loss"].append(val_loss)
            history["train_spike_rates"].append(ep_spikes / max(n_batch, 1))

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
            y_norm, s1, s2 = self.net(x_t)
            rate = ((s1.mean() + s2.mean()) / 2).item()
        self._last_spike_rate = rate
        self._test_spike_rates.append(rate)
        return self._denormalize(float(np.clip(y_norm.item(), -0.5, 1.5)))

    def update(self, x_true: float, x_hat: float) -> None:
        self._buffer.append(x_true)
        if not self._is_trained or len(self._buffer) < self.L + 1:
            return None

        window = self._normalize(np.array(self._buffer[-(self.L + 1):-1]))
        y_true_norm = (x_true - self._x_min) / max(self._x_max - self._x_min, 1e-12)

        x_t = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_t = torch.as_tensor([y_true_norm], dtype=torch.float32, device=self.device)

        self.net.train()
        self._online_opt.zero_grad()
        y_pred, _, _ = self.net(x_t)
        loss = nn.functional.mse_loss(y_pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
        self._online_opt.step()
        self.net.eval()

    def reset(self) -> None:
        self._init_weights()

    def start_test_phase(self) -> None:
        self._test_spike_rates = []

    @property
    def mean_spike_rate(self) -> float:
        return self._last_spike_rate

    @property
    def mean_test_spike_rate(self) -> float:
        return float(np.mean(self._test_spike_rates)) if self._test_spike_rates else float("nan")

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


# Benchmark-Runner
@dataclass
class BenchmarkResult:
    signal_type: str
    surrogate: str
    optimizer: str
    seed: int
    best_epoch: int
    best_val_loss: float
    test_rmse: float
    test_pred_gain_db: float
    mean_cycles: float
    energy_saving: float
    train_spike_rate: float
    test_spike_rate: float
    train_time_s: float


def run_snn_benchmark(signal_types: tuple = ALL_SIGNAL_TYPES,
                      signal_kwargs_map: Optional[dict[str, dict]] = None,
                      n_samples: int = 2000, n_runs: int = 3, base_seed: int = 77,
                      train_frac: float = 0.7, L: int = 16, H1: int = 32, H2: int = 16,
                      n_steps: int = 10, n_epochs: int = 20, fs: float = 1000.0,
                      surrogates: tuple = tuple(surrogate_functions.keys()),
                      optimizers: tuple = tuple(optimizer_fact.keys()),
                      ) -> list[BenchmarkResult]:
    """
    Rastersuche über Signaltyp × Surrogate-Gradient × Optimizer.
    """
    signal_kwargs_map = signal_kwargs_map or {}
    sar_cfg = SARConfig(n_bits=10, v_ref=1.0, unipolar=True, comparator_noise_std=0.0005)
    results: list[BenchmarkResult] = []

    # Äußere Schleife läuft alle Signaltypen ab
    for sig_type in signal_types:
        sig_kwargs = signal_kwargs_map.get(sig_type, {})
        configs = product(surrogates, optimizers)

        for (surr, opt), seed_offset in product(configs, range(n_runs)):
            seed = base_seed + seed_offset
            gen = SignalGenerator(fs=fs, seed=seed)
            x = gen.generate(sig_type, n_samples, **sig_kwargs)
            n_train = int(n_samples * train_frac)
            x_train, x_test = x[:n_train], x[n_train:]

            snn = SNNPredictor(L=L, H1=H1, H2=H2, surrogate=surr, optimizer=opt, n_steps=n_steps, n_epochs=n_epochs, seed=seed)

            t0 = time.perf_counter()
            hist = snn.train_offline(x_train)
            train_time = time.perf_counter() - t0

            snn.start_test_phase()

            rng = np.random.default_rng(seed)
            conv = SARConverter(sar_cfg, predictor=snn, rng=rng)
            conv_results = conv.convert_sequence_pretrained(x_test)

            recon = np.array([r.voltage for r in conv_results])
            x_hats = np.array([r.x_hat if r.x_hat is not None else recon[i]
                               for i, r in enumerate(conv_results)])
            cycles = np.array([r.n_cycles for r in conv_results])

            rmse = float(np.sqrt(np.mean((recon - x_test) ** 2)))
            gp = prediction_gain(x_test, x_hats)
            mean_c = float(np.mean(cycles))
            e_save = 1.0 - mean_c / sar_cfg.n_bits
            train_rates = hist.get("train_spike_rates", [])
            train_rate = float(np.mean(train_rates)) if train_rates else float("nan")

            results.append(BenchmarkResult(
                signal_type=sig_type, surrogate=surr, optimizer=opt, seed=seed,
                best_epoch=hist["best_epoch"], best_val_loss=float(min(hist["val_loss"])),
                test_rmse=rmse, test_pred_gain_db=gp if np.isfinite(gp) else float("nan"),
                mean_cycles=mean_c, energy_saving=e_save,
                train_spike_rate=train_rate, test_spike_rate=snn.mean_test_spike_rate,
                train_time_s=train_time,
            ))
    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    from collections import defaultdict
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r.signal_type, r.surrogate, r.optimizer)].append(r)

    print(f"\n{'Signal':<13}{'Surrogate':<16}{'Optimizer':<11}{'Val-Loss':>10}{'best ep.':>10}"
          f"{'RMSE':>10}{'Gp [dB]':>9}{'E_save [%]':>12}{'Spikerate(test)':>16}{'t_train [s]':>13}")
    print("─" * 120)

    for (sig_type, surr, opt), rlist in sorted(grouped.items()):
        val_losses = [r.best_val_loss for r in rlist]
        best_epochs = [r.best_epoch for r in rlist]
        rmses = [r.test_rmse for r in rlist]
        gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
        esaves = [r.energy_saving * 100 for r in rlist]
        spikes = [r.test_spike_rate for r in rlist if np.isfinite(r.test_spike_rate)]
        times = [r.train_time_s for r in rlist]

        gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
        sr_str = f"{np.mean(spikes):.3f}" if spikes else "n/a"
        print(f"{sig_type:<13}{surr:<16}{opt:<11}{np.mean(val_losses):>10.6f}{np.mean(best_epochs):>10.1f}"
              f"{np.mean(rmses):>10.5f}{gp_str:>9}{np.mean(esaves):>12.1f}%{sr_str:>16}"
              f"{np.mean(times):>13.2f}s")

    best = min(grouped.items(), key=lambda kv: np.mean([r.best_val_loss for r in kv[1]]))
    print(f"\n→ Beste Konfiguration (geringster Val-Loss): "
          f"Signal='{best[0][0]}', Surrogate='{best[0][1]}', Optimizer='{best[0][2]}'")


def print_axis_comparison(results: list[BenchmarkResult]) -> None:
    from collections import defaultdict
    axes = {
        "Signaltyp":  lambda r: r.signal_type,
        "Surrogate":  lambda r: r.surrogate,
        "Optimizer":  lambda r: r.optimizer,
    }
    print("\n Einzelvergleich aller Achsen "
          "(gemittelt über alle übrigen Konfigurationen) ──")

    for axis_name, keyfn in axes.items():
        grouped: dict[str, list] = defaultdict(list)
        for r in results:
            grouped[keyfn(r)].append(r)

        print(f"\n  {axis_name}:")
        print(f"    {'Wert':<14}{'Val-Loss':>10}{'RMSE':>10}{'Gp [dB]':>9}"
              f"{'E_save[%]':>11}{'Spikerate':>12}{'t_train':>10}")
        for val, rlist in sorted(grouped.items()):
            val_losses = [r.best_val_loss for r in rlist]
            rmses = [r.test_rmse for r in rlist]
            gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
            esaves = [r.energy_saving * 100 for r in rlist]
            spikes = [r.test_spike_rate for r in rlist if np.isfinite(r.test_spike_rate)]
            times = [r.train_time_s for r in rlist]

            gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
            sr_str = f"{np.mean(spikes):.3f}" if spikes else "n/a"
            print(f"    {val:<14}{np.mean(val_losses):>10.6f}{np.mean(rmses):>10.5f}"
                  f"{gp_str:>9}{np.mean(esaves):>11.1f}{sr_str:>12}"
                  f"{np.mean(times):>9.2f}s")


def print_best_overall_config(results: list[BenchmarkResult]) -> None:
    """
    Ermittelt die Konfiguration (Surrogate, Optimizer), die – gemittelt über ALLE Signaltypen – den geringsten Val-Loss erzielt (statt der global besten Einzelzeile, die von leicht lernbaren Signalen wie 'sine' dominiert wird).
    """
    from collections import defaultdict

    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r.surrogate, r.optimizer)].append(r)

    all_signals = {r.signal_type for r in results}
    n_signals = len(all_signals)

    complete = {k: v for k, v in grouped.items()
                if len({r.signal_type for r in v}) == n_signals}
    if not complete:
        print("\n⚠ Keine Konfiguration deckt alle Signaltypen ab – "
              "'Beste Konfiguration über alle Signale' kann nicht berechnet werden.")
        return

    def mean_val_loss(rlist):
        by_signal: dict[str, list] = defaultdict(list)
        for r in rlist:
            by_signal[r.signal_type].append(r.best_val_loss)
        return np.mean([np.mean(v) for v in by_signal.values()])

    best_key, best_rlist = min(complete.items(), key=lambda kv: mean_val_loss(kv[1]))

    print(f"\n── Beste Konfiguration über alle Signaltypen gemittelt "
          f"({n_signals} Signale, je gleich gewichtet) ──")
    print(f"→ Surrogate='{best_key[0]}', Optimizer='{best_key[1]}'")
    print(f"   mittlerer Val-Loss: {mean_val_loss(best_rlist):.6f}\n")

    by_signal: dict[str, list] = defaultdict(list)
    for r in best_rlist:
        by_signal[r.signal_type].append(r)

    print(f"    {'Signal':<13}{'Val-Loss':>10}{'RMSE':>10}{'Gp [dB]':>9}"
          f"{'E_save[%]':>11}{'Spikerate':>11}{'t_train':>10}")
    for sig, rlist in sorted(by_signal.items()):
        val_losses = [r.best_val_loss for r in rlist]
        rmses = [r.test_rmse for r in rlist]
        gps = [r.test_pred_gain_db for r in rlist if np.isfinite(r.test_pred_gain_db)]
        esaves = [r.energy_saving * 100 for r in rlist]
        spikes = [r.test_spike_rate for r in rlist if np.isfinite(r.test_spike_rate)]
        times = [r.train_time_s for r in rlist]
        gp_str = f"{np.mean(gps):.2f}" if gps else "n/a"
        sr_str = f"{np.mean(spikes):.3f}" if spikes else "n/a"
        print(f"    {sig:<13}{np.mean(val_losses):>10.6f}{np.mean(rmses):>10.5f}"
              f"{gp_str:>9}{np.mean(esaves):>11.1f}{sr_str:>11}"
              f"{np.mean(times):>9.2f}s")

# Visualisierung der Surrogate-Gradienten
def print_surrogate_shapes() -> None:
    u_vals = np.linspace(-1.0, 3.0, 200)
    threshold, sharpness = 1.0, 5.0

    numpy_equivalents = {
        "fast_sigmoid": lambda u: sharpness * (1 / (1 + np.exp(-sharpness * (u - threshold)))) * (
                    1 - 1 / (1 + np.exp(-sharpness * (u - threshold)))),
        "arctan": lambda u: (sharpness / np.pi) / (1.0 + (sharpness * (u - threshold)) ** 2),
        "triangular": lambda u: np.maximum(0.0, 1.0 - sharpness * np.abs(u - threshold)),
    }
    print("\n── Surrogate-Gradienten-Verläufe (U ∈ [-1, 3], Schwelle=1.0) ──")

    for name, fn in numpy_equivalents.items():
        vals = fn(u_vals)
        v_max = np.max(vals) if np.max(vals) > 0 else 1.0
        vals_norm = vals / v_max
        n_cols = 60
        bar = ""
        for j in range(n_cols):
            v = vals_norm[int(j / n_cols * len(u_vals))]
            bar += "█" if v > 0.75 else "▆" if v > 0.5 else "▃" if v > 0.25 else "▁" if v > 0.05 else " "
        print(f"  {name:<14}│{bar}│ max={v_max:.3f}")
    print("            └" + "─" * 60 + "┘")


# Einstiegspunkt
if __name__ == "__main__":
    print(" SNN-Benchmark: Signaltyp × 3 Surrogate-Gradienten × 4 Optimierer ")
    print_surrogate_shapes()

    signal_kwargs_map = {
        "sine": {"freq": 5.0, "amplitude": 0.45, "offset": 0.5},
        "multitone": {"freqs": (3.0, 7.0, 11.0)},
        "ecg_like": {"beat_period": 150},
        "quiescent": {"change_prob": 0.015, "step_std": 0.07},
        "random_walk": {"step_std": 0.01},
    }

    results = run_snn_benchmark(
        signal_types=ALL_SIGNAL_TYPES,
        signal_kwargs_map=signal_kwargs_map,
        n_samples=1500,
        n_runs=3,
        base_seed=77,
        train_frac=0.7,
        L=16,
        H1=32,
        H2=16,
        n_steps=10,
        n_epochs=20,
    )

    print_benchmark_table(results)
    print_axis_comparison(results)
    print_best_overall_config(results)