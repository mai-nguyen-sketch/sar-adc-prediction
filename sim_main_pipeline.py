from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


STAGES = [
    ("1. Konfiguration", "SARConfig, Signaltypen,\nHyperparameter (build_global_config)"),
    ("2. Konventioneller\nVergleich", "Nullordnung, Arith. Tracking,\nAR(2), DPCM, LSB-first"),
    ("3. Train/Val/Test-\nProtokoll", "Training + Bewertung von\nSNN & DNN (PretrainedPredictorRunner)"),
    ("4. Metrikaus-\nwertung", "RMSE, Prediction Gain,\nEnergieeinsparung, Plots"),
    ("5. Zusammenfassung", "Bester Prädiktor über\nalle Signaltypen"),
]

COLORS = ["#D3D1C7", "#9FE1CB", "#AFA9EC", "#F0997B", "#D3D1C7"]
ACTIVE = "#F2A623"


def draw_static(ax):
    n = len(STAGES)
    box_w, box_h, gap = 2.6, 1.3, 0.6
    total_h = n * box_h + (n - 1) * gap
    y0 = total_h

    boxes = []
    for i, (title, sub) in enumerate(STAGES):
        y = y0 - i * (box_h + gap)
        rect = mpatches.FancyBboxPatch(
            (-box_w / 2, y - box_h), box_w, box_h,
            boxstyle="round,pad=0.25", lw=1.2,
            facecolor=COLORS[i], edgecolor="black",
        )
        ax.add_patch(rect)
        ax.text(0, y - box_h * 0.32, title, ha="center", va="center",
                 fontsize=9, fontweight="bold")
        ax.text(0, y - box_h * 0.72, sub, ha="center", va="center", fontsize=7)
        boxes.append(rect)
        if i > 0:
            ax.annotate("", xy=(0, y), xytext=(0, y + gap),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2))

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.3, total_h + 0.3)
    ax.axis("off")
    ax.set_title("Pipeline-Ablauf (main.py)", fontsize=11)
    return boxes


def animate_pipeline():
    fig, ax = plt.subplots(figsize=(5.5, 8))
    boxes = draw_static(ax)
    status = fig.text(0.5, 0.02, "", ha="center", fontsize=9)

    def update(frame):
        for i, b in enumerate(boxes):
            b.set_facecolor(ACTIVE if i == frame else COLORS[i])
        title = STAGES[frame][0].replace("\n", " ")
        status.set_text(f"Aktive Phase: {title}")
        return boxes

    anim = animation.FuncAnimation(
        fig, update, frames=len(STAGES), interval=1200, repeat=False
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()
    return anim


def run_real(quick: bool):
    """Führt main.py-Phasenfunktionen aus """
    import argparse as _argparse
    import main as pipeline  # das echte main.py aus dem Projekt

    args = _argparse.Namespace(quick=quick, signal="sine", no_neuromorphic=True)
    cfg = pipeline.build_global_config(args)

    for label, fn in [
        ("Konventioneller Vergleich", pipeline.run_2_conventional),
        ("Train/Val/Test-Protokoll", pipeline.run_3_protocol),
        ("Metrikauswertung", pipeline.run_4_metrics),
        ("Zusammenfassung", pipeline.run_6_summary),
    ]:
        t0 = time.perf_counter()
        print(f"\n>>> Starte Phase: {label}")
        fn(cfg)
        print(f">>> Phase '{label}' fertig in {time.perf_counter() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pipeline-Simulation/Visualisierung (main.py)")
    parser.add_argument("--real", action="store_true",
                         help="Echte Pipeline-Phasen aus main.py ausführen (braucht alle Projektmodule)")
    parser.add_argument("--quick", action="store_true", help="Schnellmodus für --real")
    args = parser.parse_args()

    if args.real:
        run_real(quick=args.quick)
    else:
        animate_pipeline()


if __name__ == "__main__":
    main()