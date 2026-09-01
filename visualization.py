from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGURES_DIR = Path("output/figures")

# Farbpalette
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })

def _color_for(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]

def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path_png = FIGURES_DIR / f"{filename}.png"
    path_pdf = FIGURES_DIR / f"{filename}.pdf"
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return path_png


# Gruppierte Balkendiagramme je Metrik (Signaltyp × Prädiktor)
def plot_metric_grouped_bars(
        rows: list[dict],
        metric: str,
        ylabel: str,
        title: str,
        filename: str,
        higher_is_better: bool = True,
) -> Path:
    """
    rows: Liste von Dicts mit mind. den Schlüsseln 'signal', 'predictor', metric.
    Ein Balken je (Signaltyp, Prädiktor)-Paar, gruppiert nach Signaltyp.
    """
    _setup_style()
    signals = sorted({r["signal"] for r in rows})
    predictors = sorted({r["predictor"] for r in rows})

    n_pred = len(predictors)
    x = np.arange(len(signals))
    width = 0.8 / max(n_pred, 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(signals)), 4.5))

    for i, pred in enumerate(predictors):
        vals = []
        for sig in signals:
            match = [r[metric] for r in rows
                     if r["signal"] == sig and r["predictor"] == pred
                     and np.isfinite(r[metric])]
            vals.append(np.mean(match) if match else np.nan)
        offset = (i - (n_pred - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=pred, color=_color_for(i))

    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    arrow = "↑ besser" if higher_is_better else "↓ besser"
    ax.set_title(f"{title}  ({arrow})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(n_pred, 4))
    fig.tight_layout()
    return _save(fig, filename)


# Heatmap (Prädiktor × Signaltyp) für eine Kennzahl
def plot_metric_heatmap(
        rows: list[dict],
        metric: str,
        title: str,
        filename: str,
        fmt: str = "{:.1f}",
        cmap: str = "viridis",
) -> Path:
    _setup_style()
    signals = sorted({r["signal"] for r in rows})
    predictors = sorted({r["predictor"] for r in rows})

    mat = np.full((len(predictors), len(signals)), np.nan)
    for r in rows:
        i = predictors.index(r["predictor"])
        j = signals.index(r["signal"])
        if np.isfinite(r[metric]):
            mat[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(1.4 * len(signals) + 2, 0.55 * len(predictors) + 2))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(signals)))
    ax.set_xticklabels(signals, rotation=20, ha="right")
    ax.set_yticks(range(len(predictors)))
    ax.set_yticklabels(predictors)

    # Textwerte für Lesbarkeit einblenden (Kontrastfarbe je nach Zellhelligkeit)
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            rel = (mat[i, j] - vmin) / (vmax - vmin + 1e-12)
            color = "white" if rel < 0.6 else "black"
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save(fig, filename)


# Trade-off-Scatter: Energieeinsparung vs. Genauigkeit
def plot_tradeoff_scatter(
        rows: list[dict],
        x_metric: str = "relative_energy_saving",
        y_metric: str = "rmse",
        x_label: str = "Energieeinsparung [%]",
        y_label: str = "RMSE",
        title: str = "Trade-off: Energieeinsparung vs. Genauigkeit",
        filename: str = "tradeoff_scatter",
        y_log: bool = True,
) -> Path:
    """
    Ein Punkt je (Prädiktor, Signaltyp) — Farbe = Prädiktor, Marker-Form
    optional je Signaltyp. Punkte oben-links (kleiner RMSE, hoher E_save)
    sind die wünschenswertesten Konfigurationen.
    """
    _setup_style()
    predictors = sorted({r["predictor"] for r in rows})
    signals = sorted({r["signal"] for r in rows})
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    for i, pred in enumerate(predictors):
        for j, sig in enumerate(signals):
            match = [r for r in rows if r["predictor"] == pred and r["signal"] == sig
                     and np.isfinite(r[x_metric]) and np.isfinite(r[y_metric])]
            if not match:
                continue
            xv = np.mean([r[x_metric] for r in match])
            yv = np.mean([r[y_metric] for r in match])
            ax.scatter(xv, yv, color=_color_for(i), marker=markers[j % len(markers)],
                       s=70, edgecolor="black", linewidth=0.4, alpha=0.9)

    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # zwei Legenden unterhalb des Plots: Farbe = Prädiktor, Form = Signaltyp
    pred_handles = [Patch(facecolor=_color_for(i), label=p) for i, p in enumerate(predictors)]
    sig_handles = [plt.Line2D([0], [0], marker=markers[j % len(markers)], color="gray",
                              linestyle="", markersize=8, label=s)
                   for j, s in enumerate(signals)]
    fig.subplots_adjust(bottom=0.30, top=0.93)
    leg1 = fig.legend(handles=pred_handles, title="Prädiktor",
                      loc="upper center", bbox_to_anchor=(0.5, 0.185),
                      ncol=min(len(predictors), 3), fontsize=8, title_fontsize=9)
    fig.add_artist(leg1)
    fig.legend(handles=sig_handles, title="Signaltyp",
               loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=min(len(signals), 5), fontsize=8, title_fontsize=9)
    return _save(fig, filename)


# Zusammenfassender Balkenplot über alle Signaltypen (Kap. 5.1 / 6.3)
def plot_summary_over_signals(
        labels: list[str],
        cycles_by_label: dict[str, list[float]],
        esave_by_label: Optional[dict[str, list[float]]] = None,
        filename: str = "summary_over_signals",
        n_bits: int = 16,
) -> Path:
    """
    Ein Balken je Prädiktor: mittlere Zyklenzahl über alle Signaltypen,
    absteigend sortiert (bester = wenigste Zyklen zuerst)
    """
    _setup_style()
    order = sorted(labels, key=lambda l: np.mean(cycles_by_label[l]))
    means = [np.mean(cycles_by_label[l]) for l in order]
    mins = [np.min(cycles_by_label[l]) for l in order]
    maxs = [np.max(cycles_by_label[l]) for l in order]
    err_low = [m - lo for m, lo in zip(means, mins)]
    err_high = [hi - m for m, hi in zip(means, maxs)]

    fig, ax1 = plt.subplots(figsize=(max(6, 1.1 * len(order)), 5))
    x = np.arange(len(order))
    colors = [_color_for(i) for i in range(len(order))]
    ax1.bar(x, means, yerr=[err_low, err_high], capsize=4, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=25, ha="right")
    ax1.set_ylabel("Ø Zyklen über alle Signaltypen  (↓ besser)")
    ax1.axhline(n_bits, color="gray", linestyle=":", linewidth=1)
    ax1.text(len(order) - 0.5, n_bits, f" Referenz: {n_bits} Bit",
             va="bottom", ha="right", fontsize=8, color="gray")

    if esave_by_label:
        ax2 = ax1.twinx()
        esave_means = [np.mean(esave_by_label[l]) for l in order]
        ax2.plot(x, esave_means, marker="o", color="black", linewidth=1.2,
                 markersize=6, label="E_save [%]")
        ax2.set_ylabel("Energieeinsparung [%]  (↑ besser)")
        ax2.legend(loc="upper right")

    ax1.set_title("Gesamtvergleich: mittlere Zyklenzahl & Energieeinsparung "
                  "über alle Signaltypen")
    fig.tight_layout()
    return _save(fig, filename)