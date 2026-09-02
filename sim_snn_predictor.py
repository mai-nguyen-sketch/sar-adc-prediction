"""Simuliert und visualisiert das Blockschaltbild aus snn_predictor.py:
    Eingabemerkmale (2L) -> FC1 -> LIF1 -> FC2 -> LIF2 (über n_steps) -> Rate-Readout -> FC3 -> Prädiktion x_hat

Trainiert einen kleinen, echten SnnTorchPredictor (aus snn_predictor.py) auf einem synthetischen Sinussignal, lässt ihn Sample für Sample vorhersagen und zeigt ein Spike-Raster
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from snn_predictor import SnnTorchPredictor


STAGES = [
    ("in",   "Eingabemerkmale", "x_t, dx_t"),
    ("fc1",  "FC1 -> LIF1",     "Spikes S1"),
    ("fc2",  "FC2 -> LIF2",     "Spikes S2"),
    ("rate", "Rate-Readout",    "Mittelwert n_steps"),
    ("fc3",  "FC3",             "Prädiktion x_hat"),
]
COLORS = {"in": "#D3D1C7", "fc1": "#AFA9EC", "fc2": "#AFA9EC",
          "rate": "#5DCAA5", "fc3": "#F0997B"}


def draw_diagram(ax, dims):
    n = len(STAGES)
    box_w, box_h, gap = 2.9, 1.2, 0.55
    total_h = n * box_h + (n - 1) * gap
    y0 = total_h

    dims_text = {
        "in": f"2 x L = {dims['L'] * 2}",
        "fc1": f"H1 = {dims['H1']}",
        "fc2": f"H2 = {dims['H2']}",
        "rate": f"n_steps = {dims['n_steps']}",
        "fc3": "1",
    }
    patches = {}
    for i, (key, title, sub) in enumerate(STAGES):
        y = y0 - i * (box_h + gap)
        rect = mpatches.FancyBboxPatch(
            (-box_w / 2, y - box_h), box_w, box_h,
            boxstyle="round,pad=0.25", lw=1.2,
            facecolor=COLORS[key], edgecolor="black",
        )
        ax.add_patch(rect)
        ax.text(0, y - box_h * 0.30, title, ha="center", va="center",
                 fontsize=9, fontweight="bold")
        ax.text(0, y - box_h * 0.62, sub, ha="center", va="center", fontsize=7)
        ax.text(0, y - box_h * 0.88, dims_text[key], ha="center", va="center",
                 fontsize=7, style="italic", color="#444441")
        patches[key] = rect
        if i > 0:
            ax.annotate("", xy=(0, y), xytext=(0, y + gap),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2))
    # Rückkopplungsschleife FC2/LIF2 über n_steps
    y_fc2 = y0 - 2 * (box_h + gap)
    ax.annotate("", xy=(box_w / 2 + 0.05, y_fc2 - box_h * 0.4),
                xytext=(box_w / 2 + 0.9, y_fc2 - box_h * 0.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                 connectionstyle="arc3,rad=-0.6"))
    ax.text(box_w / 2 + 1.05, y_fc2 - box_h * 0.65, "x n_steps",
            fontsize=7, ha="left", va="center")

    ax.set_xlim(-2.2, 2.6)
    ax.set_ylim(-0.3, total_h + 0.3)
    ax.axis("off")
    ax.set_title("SNN-Prädiktor Architektur\n(snn_predictor.py)", fontsize=10)
    return patches


def make_toy_signal(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    x = 0.4 * np.sin(2 * np.pi * t / 90.0) + 0.05 * rng.normal(size=n_samples)
    return x.astype(float)


def record_spike_raster(predictor: SnnTorchPredictor, window: np.ndarray):
    """Führt denselben Forward-Pass wie _TorchSNNNet.forward() manuell aus"""
    net = predictor._net
    net.eval()
    x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    mem1 = net.lif1.init_leaky()
    mem2 = net.lif2.init_leaky()
    s1_hist, s2_hist = [], []
    with torch.no_grad():
        for _ in range(net.n_steps):
            cur1 = net.fc1(x_t)
            S1, mem1 = net.lif1(cur1, mem1)
            cur2 = net.fc2(S1)
            S2, mem2 = net.lif2(cur2, mem2)
            s1_hist.append(S1.squeeze(0).numpy())
            s2_hist.append(S2.squeeze(0).numpy())
    return np.array(s1_hist), np.array(s2_hist)  # (n_steps, H1) / (n_steps, H2)


def run_and_animate(L, H1, H2, n_epochs, patience, n_steps, n_bits, v_ref, seed):
    x = make_toy_signal(900, seed)
    n_train = int(len(x) * 0.6)

    # Nur Argumente übergeben, die der Konstruktor auch akzeptiert
    wanted_kwargs = dict(
        L=L, H1=H1, H2=H2, n_epochs=n_epochs, patience=patience,
        max_search_bits=None, n_bits=n_bits, v_ref=v_ref, seed=seed,
        n_steps=n_steps,
    )
    accepted = set(inspect.signature(SnnTorchPredictor.__init__).parameters)
    kwargs = {k: v for k, v in wanted_kwargs.items() if k in accepted}
    skipped = set(wanted_kwargs) - accepted
    if skipped:
        print(f"Hinweis: SnnTorchPredictor kennt folgende Argumente nicht, "
              f"sie werden übersprungen: {sorted(skipped)}")
    predictor = SnnTorchPredictor(**kwargs)
    n_steps = predictor._net.n_steps  # tatsächlich verwendeter Wert
    print(f"Trainiere SNN-Prädiktor auf {n_train} Samples ...")
    predictor.train_offline(x[:n_train], verbose=True)

    x_test = x[n_train:]
    preds, trues, history = [], [], []
    for xi in x_test:
        hist_arr = np.asarray(history, dtype=float)
        x_hat = predictor.predict(hist_arr)
        predictor.update(x_true=float(xi), x_hat=x_hat)
        preds.append(x_hat)
        trues.append(xi)
        history.append(xi)
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    print(f"Test-RMSE (Prädiktion x_hat vs. wahrer Wert): {rmse:.5f}")

    # Spike-Raster für ein representatives Testfenster aufzeichnen
    norm_hist = predictor._normalize(np.asarray(history[-max(L, 32):]))
    win = norm_hist[-L:]
    deltas = np.diff(win, prepend=win[0])
    feat = np.concatenate([win, deltas]).astype(np.float32)
    s1_hist, s2_hist = record_spike_raster(predictor, feat)

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.6])
    ax_diag = fig.add_subplot(gs[:, 0])
    ax_sig = fig.add_subplot(gs[0, 1])
    ax_raster = fig.add_subplot(gs[1, 1])

    patches = draw_diagram(ax_diag, dict(L=L, H1=H1, H2=H2, n_steps=n_steps))
    base_colors = {k: p.get_facecolor() for k, p in patches.items()}

    ax_sig.plot(trues, color="#378ADD", lw=1.3, label="wahres Signal x")
    pred_line, = ax_sig.plot([], [], color="#D85A30", lw=1.3, label="Prädiktion x_hat")
    ax_sig.set_xlim(0, len(x_test))
    ax_sig.set_ylim(min(trues.min(), preds.min()) - 0.1, max(trues.max(), preds.max()) + 0.1)
    ax_sig.set_xlabel("Testsample")
    ax_sig.legend(loc="upper right", fontsize=8)
    ax_sig.set_title(f"Test-RMSE = {rmse:.4f}", fontsize=9)

    raster = np.concatenate([s1_hist, s2_hist], axis=1).T  # (H1+H2, n_steps)
    ax_raster.imshow(raster, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                      interpolation="nearest")
    ax_raster.axhline(H1 - 0.5, color="#D85A30", lw=1.0)
    ax_raster.set_xlabel("Zeitschritt t (0..n_steps-1)")
    ax_raster.set_ylabel("Neuron (oben: LIF1, unten: LIF2)")
    ax_raster.set_title("Spike-Raster eines Testfensters", fontsize=9)

    stage_cycle = ["in", "fc1", "fc2", "rate", "fc3"]
    step = max(1, len(x_test) // 120)

    def update(frame):
        idx = min(frame * step, len(x_test) - 1)
        pred_line.set_data(np.arange(idx + 1), preds[: idx + 1])
        active = stage_cycle[frame % len(stage_cycle)]
        for k, p in patches.items():
            p.set_facecolor("#F2A623" if k == active else base_colors[k])
        return pred_line,

    n_frames = max(len(stage_cycle), (len(x_test) // step) + 1)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=120, repeat=False)
    fig.tight_layout()
    plt.show()
    return anim


def main():
    parser = argparse.ArgumentParser(description="SNN-Prädiktor Simulation/Visualisierung")
    parser.add_argument("--L", type=int, default=16, help="Fensterlänge")
    parser.add_argument("--h1", type=int, default=12, help="LIF1-Neuronen H1")
    parser.add_argument("--h2", type=int, default=8, help="LIF2-Neuronen H2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--steps", type=int, default=6, help="n_steps des SNN-Forward-Passes")
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--vref", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_and_animate(args.L, args.h1, args.h2, args.epochs, args.patience,
                     args.steps, args.bits, args.vref, args.seed)


if __name__ == "__main__":
    main()