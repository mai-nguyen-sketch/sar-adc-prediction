from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGURES_DIR = Path("output/figures/HP")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# Farbpalette
_PALETTE = [
    "#8FBBD9",
    "#E6B878",  
    "#91C7B1", 
    "#D99A7C",  
    "#C5A6C9",  
    "#8FC4DD", 
    "#E3D58A", 
    "#8A8A8A", 
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


# Balkendiagramme je Metrik (Signaltyp × Prädiktor)
def plot_metric_grouped_bars(rows: list[dict], metric: str, ylabel: str, title: str, filename: str, higher_is_better: bool = True) -> Path:
    """
    rows: Liste von Dicts mit mind. den Schlüsseln 'signal', 'predictor', metric
    Ein Balken je (Signaltyp, Prädiktor)-Paar, gruppiert nach Signaltyp
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
def plot_metric_heatmap(rows: list[dict], metric: str, title: str, filename: str, fmt: str = "{:.1f}", cmap: str = "viridis") -> Path:
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
    Ein Punkt je (Prädiktor, Signaltyp)
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
    Ein Balken je Prädiktor: mittlere Zyklenzahl über alle Signaltypen, absteigend sortiert (bester = wenigste Zyklen zuerst)
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


def _aggregate_history(runs: list[list[dict]], key: str):
    """
    Bringt mehrere Wiederholungsläufe unterschiedlicher Länge (Early Stopping) auf ein gemeinsames Epochenraster und berechnet je Epoche
    Median sowie 25./75.-Perzentil.
    Median/IQR statt Mittelwert/Std, da neuromorphe Prädiktoren teilweise exakt auf 0.0 konvergieren

    runs: Liste von Läufen, jeder Lauf eine Liste von Dicts mit mind. den Schlüsseln 'epoch' und `key` (z. B. 'val_loss' oder 'spike_rate').
    """
    all_epochs = sorted({pt["epoch"] for run in runs for pt in run if pt.get(key) is not None})
    if not all_epochs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    matrix = np.full((len(runs), len(all_epochs)), np.nan)
    for i, run in enumerate(runs):
        run_dict = {pt["epoch"]: pt[key] for pt in run if pt.get(key) is not None}
        last = None
        for j, ep in enumerate(all_epochs):
            if ep in run_dict:
                last = run_dict[ep]
            if last is not None:
                matrix[i, j] = last

    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    return np.array(all_epochs), median, q25, q75


def plot_learning_curves(histories: dict[str, dict[str, list[list[dict]]]], filename: str = "03_learning_curves", title: str = "Konvergenzverhalten der neuromorphen Prädiktoren") -> Path:
    """
    Validierungsfehlerverlauf je Signaltyp (ein Panel je Signal), je Prädiktor eine Linie (Median über die Wiederholungsläufe)
    histories: verschachteltes Dict signal -> label -> Liste von Läufen, wobei jeder Lauf eine Liste von Dicts {'epoch': int, 'val_loss': float, 'spike_rate': float | None} ist
    """
    _setup_style()

    desired_order = ["sine", "multitone", "ecg_like", "quiescent", "random_walk"]
    signals = sorted(
        histories.keys(),
        key=lambda s: desired_order.index(s) if s in desired_order else 999
    )

    if not signals:
        raise ValueError("Keine Trainingshistorien vorhanden – plot_learning_curves() übersprungen.")

    ncols = 2
    nrows = -(-len(signals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    all_labels = sorted({lbl for per_sig in histories.values() for lbl in per_sig})
    label_color = {lbl: _color_for(i) for i, lbl in enumerate(all_labels)}

    for ax, sig in zip(axes_flat, signals):
        for lbl in all_labels:
            runs = histories[sig].get(lbl)
            if not runs:
                continue
            epochs, median, q25, q75 = _aggregate_history(runs, key="val_loss")
            if len(epochs) == 0:
                continue
            ax.plot(epochs, median, label=lbl, color=label_color[lbl],
                    linewidth=2, marker="o", markersize=3)
            ax.fill_between(epochs, q25, q75, color=label_color[lbl], alpha=0.2)
        ax.set_yscale("log")
        ax.set_title(sig, fontsize=10)
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Val.-fehler (log)")
        ax.legend(fontsize=7)

    # Verbleibende ungenutzte Axes ausblenden
    for ax in axes_flat[len(signals):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, filename)


def plot_spike_rates(
        histories: dict[str, dict[str, list[list[dict]]]],
        filename: str = "03_spike_rate",
        title: str = "Spike-Rate-Verlauf (SNN)",
) -> Path:
    """
    Spike-Rate-Verlauf je Signaltyp/Prädiktor (Median über die Wiederholungsläufe, schattiertes 25.-75.-Perzentilband)
    Signale bzw. Prädiktoren ohne 'spike_rate'-Eintrag (z. B. das DNN) werden ignoriert
    Wirft ValueError, wenn überhaupt keine Spike-Rate-Daten vorliegen
    """
    _setup_style()
    signals = sorted(histories.keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plotted = False

    for i, sig in enumerate(signals):
        for lbl, runs in histories[sig].items():
            has_spike = any(pt.get("spike_rate") is not None for run in runs for pt in run)
            if not has_spike:
                continue
            epochs, median, q25, q75 = _aggregate_history(runs, key="spike_rate")
            if len(epochs) == 0:
                continue
            label = sig if len(histories[sig]) == 1 else f"{sig} ({lbl})"
            ax.plot(epochs, median, label=label, color=_color_for(i),
                     linewidth=2, marker="o", markersize=3)
            ax.fill_between(epochs, q25, q75, color=_color_for(i), alpha=0.15)
            plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError("Keine Spike-Rate-Daten in den Trainingshistorien gefunden.")

    ax.set_xlabel("Epoche")
    ax.set_ylabel("Spike-Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, filename)


def plot_snn_dnn_testcycles(rows: list[dict], filename: str = "03_snn_dnn_testcycles", title: str = "SNN vs. DNN – mittlere Testzyklenzahl (95\u2009%-KI)") -> Path:
    """
    Gruppierter Balkenplot: ein Balken je (Signaltyp, Prädiktor)-Paar, Fehlerbalken aus dem 95%-Konfidenzintervall über die Wiederholungsläufe
    rows: Liste von Dicts {'signal', 'predictor', 'mean_cycles', 'ci_low', 'ci_high'}, direkt aus aggregate_test_results() in run_3_protocol() befüllbar
    """
    _setup_style()
    signals = sorted({r["signal"] for r in rows})
    predictors = sorted({r["predictor"] for r in rows})
    x = np.arange(len(signals))
    width = 0.8 / max(len(predictors), 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(signals)), 5))
    for i, pred in enumerate(predictors):
        means, err_low, err_high = [], [], []
        for sig in signals:
            match = next((r for r in rows if r["signal"] == sig and r["predictor"] == pred), None)
            if match:
                means.append(match["mean_cycles"])
                err_low.append(max(0.0, match["mean_cycles"] - match["ci_low"]))
                err_high.append(max(0.0, match["ci_high"] - match["mean_cycles"]))
            else:
                means.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
        offset = (i - (len(predictors) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=[err_low, err_high], capsize=3,
               label=pred, color=_color_for(i))
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=20, ha="right")
    ax.set_ylabel("Mittlere Testzyklenzahl (95\u2009%-KI)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(len(predictors), 4))
    fig.tight_layout()
    return _save(fig, filename)

def plot_ttest_results(rows: list[dict], alpha: float = 0.05, filename: str = "03_ttest_forest", title: str = "Gepaarter t-Test SNN vs. DNN – t-Statistik je Signaltyp") -> Path:
    """
    Horizontaler Balkenplot der t-Statistik je Signaltyp, grün/rot codiert nach Signifikanz zum Niveau alpha, mit p-Wert als Beschriftung
    rows: Liste von Dicts {'signal', 't_stat', 'p_value'}, direkt aus paired_t_test_cycles() in run_3_protocol() befüllbar
    """
    _setup_style()
    signals = [r["signal"] for r in rows]
    t_vals = [r["t_stat"] for r in rows]
    p_vals = [r["p_value"] for r in rows]
    colors = ["#009E73" if p < alpha else "#D55E00" for p in p_vals]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(signals) + 2))
    y_pos = np.arange(len(signals))
    ax.barh(y_pos, t_vals, color=colors)
    span = max(1.0, max((abs(t) for t in t_vals), default=1.0))
    for yi, (t, p) in zip(y_pos, zip(t_vals, p_vals)):
        offset = 0.02 * span
        ax.text(t + (offset if t >= 0 else -offset), yi, f"p={p:.4f}",
                va="center", ha="left" if t >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signals)
    ax.set_xlabel("t-Statistik (SNN vs. DNN)")
    ax.set_title(title)
    handles = [Patch(facecolor="#009E73", label=f"signifikant (p<{alpha})"),
               Patch(facecolor="#D55E00", label=f"nicht signifikant (p\u2265{alpha})")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, filename)
