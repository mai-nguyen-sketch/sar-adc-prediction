"""Simuliert und visualisiert das Blockschaltbild aus dnn_predictor.py:
    Eingabefenster (x_t, dx_t) -> GRU -> Dense-Kopf (GroupNorm+ReLU) -> Ausgabekopf (Residual) -> + Baseline-Extrapolation -> Prädiktion x_hat

Trainiert einen kleinen, echten DnnTorchPredictor (aus dnn_predictor.py) auf einem synthetischen Sinussignal und lässt ihn anschließend Sample für Sample vorhersagen (predict + update wie im SARConverter).
Die tatsächlichen Vorhersagen werden neben dem Blockschaltbild animiert dargestellt.
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from dnn_predictor import DnnTorchPredictor


STAGES = [
    ("in",   "Eingabefenster",  "x_t, dx_t"),
    ("gru",  "GRU-Schicht",     "H1 Neuronen"),
    ("dense", "Dense-Kopf",     "GroupNorm, ReLU"),
    ("head", "Ausgabekopf",     "Residualwert"),
    ("out",  "+ Baseline",      "Prädiktion x_hat"),
]
COLORS = {"in": "#D3D1C7", "gru": "#5DCAA5", "dense": "#5DCAA5",
          "head": "#AFA9EC", "out": "#F0997B"}


def draw_diagram(ax, cfg_dims):
    n = len(STAGES)
    box_w, box_h, gap = 2.8, 1.2, 0.55
    total_h = n * box_h + (n - 1) * gap
    y0 = total_h

    dims_text = {
        "in": f"2 x L = {cfg_dims['L'] * 2}",
        "gru": f"H1 = {cfg_dims['H1']}",
        "dense": f"H2 = {cfg_dims['H2']}",
        "head": "1",
        "out": "1",
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

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.3, total_h + 0.3)
    ax.axis("off")
    ax.set_title("DNN-Prädiktor Architektur\n(dnn_predictor.py)", fontsize=10)
    return patches


def make_toy_signal(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    x = 0.4 * np.sin(2 * np.pi * t / 90.0) + 0.05 * rng.normal(size=n_samples)
    return x.astype(float)


def run_and_animate(L, H1, H2, n_epochs, patience, n_bits, v_ref, seed):
    x = make_toy_signal(900, seed)
    n_train = int(len(x) * 0.6)

    # Nur Argumente übergeben, die der tatsächliche Konstruktor auch akzeptiert
    wanted_kwargs = dict(
        L=L, H1=H1, H2=H2, num_groups=min(4, H1),
        n_epochs=n_epochs, patience=patience,
        max_search_bits=None, n_bits=n_bits, v_ref=v_ref, seed=seed,
    )
    accepted = set(inspect.signature(DnnTorchPredictor.__init__).parameters)
    kwargs = {k: v for k, v in wanted_kwargs.items() if k in accepted}
    skipped = set(wanted_kwargs) - accepted
    if skipped:
        print(f"Hinweis: DnnTorchPredictor kennt folgende Argumente nicht, "
              f"sie werden übersprungen: {sorted(skipped)}")
    predictor = DnnTorchPredictor(**kwargs)
    print(f"Trainiere DNN-Prädiktor auf {n_train} Samples ...")
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

    fig, (ax_diag, ax_sig) = plt.subplots(
        1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.0, 1.6]}
    )
    patches = draw_diagram(ax_diag, dict(L=L, H1=H1, H2=H2))
    base_colors = {k: p.get_facecolor() for k, p in patches.items()}

    ax_sig.plot(trues, color="#378ADD", lw=1.3, label="wahres Signal x")
    pred_line, = ax_sig.plot([], [], color="#D85A30", lw=1.3, label="Prädiktion x_hat")
    ax_sig.set_xlim(0, len(x_test))
    ax_sig.set_ylim(min(trues.min(), preds.min()) - 0.1, max(trues.max(), preds.max()) + 0.1)
    ax_sig.set_xlabel("Testsample")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.legend(loc="upper right")
    ax_sig.set_title(f"Test-RMSE = {rmse:.4f}")

    stage_cycle = ["in", "gru", "dense", "head", "out"]
    step = max(1, len(x_test) // 120)  # Animation nicht Sample-für-Sample, sondern gerafft

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
    parser = argparse.ArgumentParser(description="DNN-Prädiktor Simulation/Visualisierung")
    parser.add_argument("--L", type=int, default=16, help="Fensterlänge")
    parser.add_argument("--h1", type=int, default=12, help="GRU-Neuronen H1")
    parser.add_argument("--h2", type=int, default=8, help="Dense-Neuronen H2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--vref", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_and_animate(args.L, args.h1, args.h2, args.epochs, args.patience,
                     args.bits, args.vref, args.seed)


if __name__ == "__main__":
    main()