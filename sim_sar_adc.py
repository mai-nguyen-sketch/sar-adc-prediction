from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sar_adc import SARConfig, SampleAndHold, SARDac, Comparator


# Bit-für-Bit-Ablauf mit den ECHTEN Bauteilen aufzeichnen
def record_conversion(cfg: SARConfig, x: float, seed: int = 0):
    """Führt eine Wandlung durch und protokolliert jeden Zyklus für Animation"""
    rng = np.random.default_rng(seed)
    sh = SampleAndHold(rng=rng)
    dac = SARDac(cfg)
    comp = Comparator(noise_std=cfg.comparator_noise_std, rng=rng)

    v_in = sh.sample(x)
    code = 0
    steps = []
    for bit_pos in range(cfg.n_bits - 1, -1, -1):
        trial_code = code | (1 << bit_pos)
        v_dac = dac.code_to_voltage(trial_code)
        bit_set = comp.compare(v_in, v_dac)
        if bit_set:
            code = trial_code
        steps.append(dict(bit_pos=bit_pos, trial_code=trial_code,
                           v_dac=v_dac, bit_set=bit_set, code_after=code))
    v_out = dac.code_to_voltage(code)
    return v_in, code, v_out, steps


# Statisches Blockschaltbild
BOX_STYLE = dict(boxstyle="round,pad=0.3", lw=1.2)

def draw_diagram(ax):
    boxes = {
        "in":   (0.5, 8.6, "Analoger\nEingang"),
        "sh":   (0.5, 7.0, "Sample &\nHold"),
        "sar":  (0.5, 4.6, "SAR-Logik"),
        "dac":  (3.2, 5.6, "DAC"),
        "comp": (3.2, 3.6, "Komparator"),
        "pred": (-2.1, 4.6, "Prädiktor"),
        "out":  (0.5, 1.4, "Digitalcode"),
    }
    patches = {}
    for key, (x, y, label) in boxes.items():
        color = {
            "in": "#D3D1C7", "sh": "#D3D1C7", "out": "#D3D1C7",
            "sar": "#AFA9EC", "dac": "#85B7EB", "comp": "#F0997B",
            "pred": "#5DCAA5",
        }[key]
        p = mpatches.FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0,
                                     facecolor=color, edgecolor="black",
                                     **BOX_STYLE)
        ax.add_patch(p)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
        patches[key] = p

    def arrow(a, b, **kw):
        ax.annotate("", xy=b, xytext=a,
                     arrowprops=dict(arrowstyle="-|>", lw=1.2, **kw))

    arrow((0.5, 8.1), (0.5, 7.5))
    arrow((0.5, 6.5), (0.5, 5.1))
    arrow((1.4, 4.8), (2.3, 5.5))     # SAR -> DAC
    arrow((3.2, 5.1), (3.2, 4.1))     # DAC -> Komparator
    arrow((2.3, 3.7), (1.4, 4.4))     # Komparator -> SAR
    arrow((-1.2, 4.6), (-0.4, 4.6))   # Prädiktor -> SAR
    arrow((0.5, 4.1), (0.5, 1.9))

    ax.set_xlim(-3.2, 4.6)
    ax.set_ylim(0.5, 9.4)
    ax.axis("off")
    ax.set_title("SAR-ADU Blockschaltbild (sar_adc.py)", fontsize=11)
    return patches


ACTIVE_COLOR = "#F2A623"

def highlight(patches, active_keys, base_colors):
    for k, p in patches.items():
        p.set_facecolor(ACTIVE_COLOR if k in active_keys else base_colors[k])


# Animation über die Bit-Zyklen
def animate(cfg: SARConfig, x: float, seed: int):
    v_in, code, v_out, steps = record_conversion(cfg, x, seed=seed)

    fig, (ax_diag, ax_sig) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.1, 1.4]}
    )
    patches = draw_diagram(ax_diag)
    base_colors = {k: p.get_facecolor() for k, p in patches.items()}

    ax_sig.axhline(v_in, color="#3B8BD4", lw=1.5, label="v_in (abgetastet)")
    dac_line, = ax_sig.plot([], [], "o-", color="#D05538", label="v_dac (Suche)")
    ax_sig.set_xlim(-0.5, cfg.n_bits + 0.5)
    ax_sig.set_ylim(0, cfg.full_scale * 1.05)
    ax_sig.set_xlabel("Zyklus")
    ax_sig.set_ylabel("Spannung [V]")
    ax_sig.legend(loc="upper right")
    status = ax_sig.text(0.02, 0.02, "", transform=ax_sig.transAxes, fontsize=9)

    xs, ys = [], []

    def update(frame):
        if frame == 0:
            highlight(patches, {"in", "sh"}, base_colors)
            status.set_text("Sample & Hold tastet x(t) ab")
            return

        i = frame - 1
        if i < len(steps):
            s = steps[i]
            highlight(patches, {"sar", "dac", "comp"}, base_colors)
            xs.append(i + 1)
            ys.append(s["v_dac"])
            dac_line.set_data(xs, ys)
            entscheidung = "bit=1 (v_in >= v_dac)" if s["bit_set"] else "bit=0 (v_in < v_dac)"
            status.set_text(
                f"Zyklus {i+1}/{cfg.n_bits} | Bitposition {s['bit_pos']} | "
                f"v_dac={s['v_dac']:.4f} V | {entscheidung}"
            )
        else:
            highlight(patches, {"out"}, base_colors)
            status.set_text(
                f"Fertig: Code={code} ({cfg.n_bits} Bit) | v_out={v_out:.5f} V | "
                f"Fehler={abs(v_in - v_out):.5f} V"
            )
        return

    n_frames = cfg.n_bits + 2
    anim = __import__("matplotlib.animation", fromlist=["FuncAnimation"]).FuncAnimation(
        fig, update, frames=n_frames, interval=700, repeat=False
    )
    fig.tight_layout()
    plt.show()
    return anim  # Referenz halten, sonst wird die Animation ggf. vom GC entsorgt


def main():
    parser = argparse.ArgumentParser(description="SAR-ADU Simulation/Visualisierung")
    parser.add_argument("--bits", type=int, default=8, help="Auflösung n_bits")
    parser.add_argument("--vref", type=float, default=1.0, help="Referenzspannung")
    parser.add_argument("--value", type=float, default=None,
                         help="Analoger Eingangswert (Standard: zufällig)")
    parser.add_argument("--noise", type=float, default=0.0, help="Komparator-Rauschen (Std.)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SARConfig(n_bits=args.bits, v_ref=args.vref, unipolar=True,
                     comparator_noise_std=args.noise)
    rng = np.random.default_rng(args.seed)
    x = args.value if args.value is not None else float(rng.uniform(0, cfg.v_ref))

    print(f"SARConfig: n_bits={cfg.n_bits}, v_ref={cfg.v_ref}, LSB={cfg.lsb:.6f} V")
    print(f"Eingangswert x = {x:.5f} V")
    animate(cfg, x, seed=args.seed)


if __name__ == "__main__":
    main()