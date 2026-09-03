"""
main.py
========
Zentraler Einstiegspunkt der Simulationsarbeit:
"Vergleich von konventionellen und neuromorphen Prädikationsalgorithmen
 für energie-optimierte SAR-ADUs"
Dieses Skript orchestriert die gesamte Auswertungspipeline in sechs
aufeinanderfolgenden Phasen, die exakt der Kapitelstruktur der Arbeit
entsprechen:
    1  – Globale Konfiguration           (SARConfig, SplitConfig)
    2  – Konventioneller Vergleich        (Kapitel 4.2 / 5.1-5.3)
    3  – Trainings-/Testprotokoll für     (Kapitel 4.4 / 5.2)
               neuromorphe Prädiktoren (LSTM, SNN)
    4  – Vollständige Metrikauswertung    (Kapitel 4.5 / 5.3-5.5)
               (konventionell + neuromorph)
    5  – SNN-Architekturvergleich:        (Kapitel 4.6 / 5.3)
               Surrogate-Gradienten × Optimierer
    6  – Ergebniszusammenfassung          (Kapitel 5.1)

Modulstruktur:
    sar_adc.py          → SAR-ADU-Kernmodell                    (Kap. 4.1)
    simulation.py        → Signalgenerierung, Experiment-Runner   (Kap. 4.2)
    predictors.py         → Konventionelle Prädiktoren             (Kap. 4.3.1-4.3.5)
    snn_predictor.py      → SNN-Prädiktor (snnTorch, LIF)           (Kap. 4.3.7)
    snn_benchmark.py      → SNN-Architektur-/Surrogate-Benchmark    (Kap. 4.6)
    protocol.py            → Train/Val/Test-Protokoll                (Kap. 4.4)
    metrics.py             → Bewertungsmetriken                      (Kap. 4.5)

Verwendung:
    python main.py [--quick] [--signal SIGNALTYP] [--no-neuromorphic]

    --quick           Reduziert n_samples, n_runs und die Trainings-
                      dauer der neuronalen Prädiktoren stark, für eine
                      schnelle Funktionsprüfung (~1-2 Min.)
    --signal          Führt nur einen einzelnen Signaltyp aus
                      (sine | multitone | ecg_like | quiescent |
                       random_walk)
    --no-neuromorphic Überspringt SNN-Architekturbenchmark,
                      da dieser am rechenintensivsten ist. .
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sar_adc import SARConfig, SARConverter, ZeroOrderPredictor

from predictors import (
    ArithmeticTrackingPredictor,
    DPCMPredictor,
    LinearPredictor,
    LSBFirstPredictor,
)

from simulation import (
    ExperimentConfig,
    ExperimentRunner,
    SignalGenerator,
)

from protocol import (
    SplitConfig,
    PretrainedPredictorRunner,
    aggregate_test_results,
    paired_t_test_cycles,
)

from metrics import (
    ComplexityProfile,
    complexity_profile_arithmetic_tracking,
    complexity_profile_dpcm,
    complexity_profile_linear_ar,
    complexity_profile_lsb_first,
    complexity_profile_zero_order,
    evaluate,
    report_to_dict,
    hit_window_cost,
    expected_mean_cycles,
)

# Neuromorphe Prädiktoren (Kapitel 4.3.6 / 4.3.7)
from snn_predictor import SnnTorchPredictor

# DNN-Prädiktor (LSTM-Backbone, ReLU, AdamW, GroupNorm) mit Multi-Feature-
# Eingabe (Absolutwerte + Deltas) — zusätzliche neuromorphe Vergleichsbasis
from dnn_predictor import DnnTorchPredictor

# Graphische Auswertung
import visualization as viz

torch.set_num_threads(1)
# Hilfsfunktionen für konsolenbasierte Ausgabe
def _banner(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")

def _print_table(headers: list[str], rows: list[list], col_widths: list[int]) -> None:
    # Gibt eine einfache, ausgerichtete ASCII-Tabelle aus
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(col_widths))
    header_line = fmt.format(*headers)
    print(header_line)
    print("─" * len(header_line))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
'''
# Komplexitätsprofile für die neuromorphen Prädiktoren (Kapitel 2.5.3 / 4.6)
# metrics.py stellt für die konventionellen Prädiktoren fertige complexity_profile_*()-Funktionen bereit. Für LSTM und SNN existieren keine Referenzimplementierungen in der Literatur mit fixer Gatterzahl, daher werden die Operationszahlen hier aus der jeweiligen Architektur
def _complexity_profile_lstm(L: int, H1: int, H2: int) -> ComplexityProfile:
    """
    Zweischichtiger Stacked-LSTM (Kapitel 4.3.6): pro Zelle vier Gates (Input, Forget, Cell, Output), jedes Gate benötigt (n_in + H) MAC-
    Operationen plus Bias. Nichtlineare Operationen: 3 Sigmoid- + 1 Tanh-Aktivierung pro Gate-Satz, zzgl. Tanh der Zellzustands-Ausgabe
    """
    def cell_params(n_in: int, H: int) -> int:
        return 4 * H * (n_in + H) + 4 * H

    p1 = cell_params(L, H1)
    p2 = cell_params(H1, H2)
    p_head = H2 + 1  # linearer Ausgabekopf
    params = p1 + p2 + p_head

    return ComplexityProfile(
        name="LSTM (2-schichtig)",
        multiplications=params,          # MAC-Operationen ≈ Anzahl Gewichte
        additions=params,                # Akkumulation + Zustandskombination
        comparisons=0,
        nonlinear_ops=4 * (H1 + H2) + (H1 + H2),  # Gate-Aktivierungen + Zustands-Tanh
        parameters=params,
        state_size=2 * (H1 + H2),        # h & c je Schicht
    )
'''

def _complexity_profile_dnn(L: int, H1: int, H2: int, num_groups: int = 4) -> ComplexityProfile:
    """
    DNN mit LSTM-Backbone (Kap. 4.3.x, dnn_predictor.py): eine LSTM-Schicht verarbeitet pro Zeitschritt 2 Eingabe-Features (Absolutwert + Delta),
    gefolgt von zwei Dense-Schichten mit GroupNorm + ReLU. Die Gate-Zählung der LSTM-Zelle erfolgt analog zu _complexity_profile_lstm(), jedoch mit
    fixer Eingangsdimension 2 (statt L) pro Zeitschritt, da die Sequenz zeitlich statt feature-seitig aufgefächert wird. GroupNorm trägt pro
    Kanal zwei zusätzliche affine Parameter (Skalierung + Offset) bei.
    """
    def cell_params(n_in: int, H: int) -> int:
        return 3 * H * (n_in + H) + 3 * H

    p_lstm = cell_params(2, H1)     # LSTM: Input [x_t, delta_x_t] pro Zeitschritt
    p_fc2  = H1 * H2 + H2                # Dense-Schicht H1 -> H2
    p_head = H2 + 1                      # linearer Ausgabekopf
    p_norm = 2 * H1 + 2 * H2             # GroupNorm-Affinparameter (norm1 + norm2)
    params = p_lstm + p_fc2 + p_head + p_norm

    return ComplexityProfile(
        name="DNN (Eigen)",
        multiplications= p_lstm + p_fc2 + p_head,   # MAC-Operationen ≈ Gewichte (ohne Norm)
        additions= p_lstm + p_fc2 + p_head,
        comparisons=0,
        nonlinear_ops= 5 * H1 + H1 + H2,  # LSTM-Gate-Aktiv: (3 Sigmoid+Tanh+Zustands-Tanh) + 2x ReLU
        parameters=params,
        state_size= 2 * H1,               # h & c der LSTM-Schicht
    )


def _complexity_profile_snn(L: int, H1: int, H2: int, n_steps: int) -> ComplexityProfile:
    """
    Zweischichtiges SNN mit LIF-Neuronen (Kapitel 4.3.7): pro Zeitschritt
    ein vollständiger Forward-Pass durch beide FC-Schichten, die
    Spike-Erzeugung erfolgt über einen Schwellenvergleich (Comparator)
    statt einer stetigen Aktivierungsfunktion
    """
    p1 = L * H1 + H1
    p2 = H1 * H2 + H2
    p_head = H2 + 1
    params = p1 + p2 + p_head

    return ComplexityProfile(
        name="SNN (2-schichtig, LIF)",
        multiplications=params * n_steps,
        additions=params * n_steps,
        comparisons=(H1 + H2) * n_steps,  # Schwellenvergleiche der LIF-Neuronen
        nonlinear_ops=0,
        parameters=params,
        state_size=H1 + H2 + 1,           # Membranpotentiale + Ausgabeintegrator
    )


# Hilfsfunktion: Klassische vs. vortrainierte (neuromorphe) Prädiktoren
# Klassische Prädiktoren (Nullordnung, AR, DPCM, ...) benötigen keine Trainingsphase und werden auf der vollständigen Sequenz ausgewertet.
# LSTM/SNN müssen dagegen zunächst per train_offline() auf dem Trainings-/Validierungsanteil trainiert werden (Kapitel 4.4) und werden anschließend ausschließlich auf dem Testanteil bewertet, um Datenlecks (Train/Test-Leakage) zu vermeiden.

def _evaluate_predictor(factory, x, sar_cfg, split_config, seed):
    try:
        predictor = factory(seed)
    except TypeError:
        predictor = factory()
    rng = np.random.default_rng(seed)

    _, val_sl, test_sl = split_config.split_indices(len(x))

    if hasattr(predictor, "train_offline"):
        predictor.train_offline(x[:val_sl.stop])
        conv = SARConverter(sar_cfg, predictor=predictor, rng=rng)
        results = conv.convert_sequence_pretrained(x)
    else:
        conv = SARConverter(sar_cfg, predictor=predictor, rng=rng)
        results = conv.convert_sequence(x)

    return x[test_sl], results[test_sl]


def _print_search_window_tradeoff(cfg: dict) -> None:
    """
    Macht die Kompromiss-Rechnung aus Punkt 1/2 sichtbar: zeigt für jedes
    verwendete max_search_bits (globaler Default + prädiktorspezifische
    Overrides), wie sich die erwartete mittlere Zyklenzahl in Abhängigkeit
    von der Trefferquote verhält. Nützlich, um VOR einem vollen Simulations-
    lauf abzuschätzen, ob ein breiteres Fenster für die gemessene/erwartete
    Trefferquote eines Prädiktors überhaupt einen Nettovorteil bringt.
    """
    _section("Kompromiss-Rechnung: Suchfenster vs. Trefferquote")
    n_bits = cfg["sar_config"].n_bits
    neural = cfg["neural"]
    windows = {
        "Global (klassisch)": cfg["sar_config"].max_search_bits,
        # "LSTM": neural.get("max_search_bits_lstm", cfg["sar_config"].max_search_bits),
        "DNN": neural.get("max_search_bits_dnn", cfg["sar_config"].max_search_bits),
        "SNN": neural.get("max_search_bits_snn", cfg["sar_config"].max_search_bits),
    }
    hit_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    header = f"  {'Fenster':<20} {'bits':>5} {'hit-cost':>9}  " + \
             "  ".join(f"P_hit={p:.1f}" for p in hit_rates)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for label, bits in windows.items():
        cost = hit_window_cost(bits)
        vals = "  ".join(
            f"{expected_mean_cycles(p, bits, n_bits):9.2f}" for p in hit_rates
        )
        print(f"  {label:<20} {bits:>5} {cost:>9}  {vals}")
    print(f"  (Referenz konventionell: {n_bits} Zyklen bei P_hit=0)")


# Globale Konfiguration
def build_global_config(args: argparse.Namespace) -> dict:
    """
    Legt alle globalen Simulationsparameter fest, die in den nachfolgenden Phasen konsistent verwendet werden. Anpassungen an diesen Werten
    genügen, um die gesamte Simulation konsistent neu zu konfigurieren — kein Parameter ist hartcodiert in den einzelnen Phasen.
    """
    quick = args.quick

    # Hyperparameter der neuromorphen Prädiktoren (Kap. 4.3.6/4.3.7)
    neural = {
        "L": 24, # 24 bei ULP, 48 bei HP
        "H1": 12, # 12 bei ULP, 64 bei HP
        "H2": 8 , # 8 bei ULP, 32 bei HP
        "n_epochs": 15, # 15 bei ULP, 60 bei HP
        "patience": 10, # 10 bei ULP, 25 bei HP
        "n_steps":  6,      # Zeitschritte des SNN-Forward-Passes, 6 bei ULP, 16 bei HP
        "warmup_samples": 20,
        "seed": 0,
        # Prädiktorspezifische SAR-Suchfenster (Kompromiss-Punkte 1+2):
        # ein GLOBALES Anheben von SARConfig.max_search_bits würde auch
        # bereits sehr präzise klassische Prädiktoren (AR(2), Nullordnung)
        # unnötig verteuern, da jedes zusätzliche Fensterbit einen
        # zusätzlichen Zyklus PRO TREFFER kostet (siehe
        # metrics.hit_window_cost()) - unabhängig davon, ob der Prädiktor
        # das breitere Fenster überhaupt braucht. Stattdessen bekommen nur
        # die weniger präzisen neuronalen Prädiktoren ein breiteres
        # Fenster; die SNN etwas breiter als LSTM/DNN, da ihr
        # Rate-Readout (n_steps Zeitschritte) strukturell nur grob
        # quantisierte Ausgaben liefert. Der globale SARConfig-Default
        # bleibt unverändert bei 3, sodass klassische Prädiktoren
        # unbeeinflusst bleiben (siehe _print_search_window_tradeoff()).
        # "max_search_bits_lstm": 6,
        "max_search_bits_dnn": 4, # 6 bei HP, 4 bei ULP
        "max_search_bits_snn": 5, # 8 bei HP, 5 bei ULP
    }

    return {
        # SAR-ADU-Parameter (Kapitel 4.1)
        "sar_config": SARConfig(
            n_bits=10, # 16 bei HP, 10 bei ULP
            v_ref=1.0, # 5.0 bei HP, 1.0 bei ULP
            unipolar=True,
            comparator_noise_std=0.0005 ,#0.0005 bei ULP, 0.00002 bei HP
            # Globaler Default (gilt für alle Prädiktoren ohne eigenen
            # max_search_bits-Override, insb. die klassischen Prädiktoren).
            # Bewusst NICHT global erhöht (Kompromiss-Punkt 1) - siehe
            # Begründung bei neural["max_search_bits_*"] oben.
            max_search_bits=2,
        ),

        # Signalparameter (Kapitel 4.2)
        "signal_types": (
            [args.signal] if args.signal
            else ["sine", "multitone", "ecg_like", "quiescent", "random_walk"]
        ),
        "n_samples":    1500 if quick else 4000,
        "fs":           1000.0,
        "signal_kwargs": {
            "sine":        {"freq": 5.0, "amplitude": 0.4},
            "multitone":   {"freqs": (3, 7, 11), "amplitude": 0.3},
            "ecg_like":    {"beat_period": 192},
            "quiescent":   {"change_prob": 0.015, "step_std": 0.07},
            "random_walk": {"step_std": 0.012},
        },
        "jitter_frac": 0.02,

        # Protokollparameter (Kapitel 4.4)
        "split_config": SplitConfig(
            train_frac=0.50,
            val_frac=0.20,
            test_frac=0.30,
            test_warmup_fraction=0.10,
        ),
        "n_runs":       15, # 3 if quick else 8,
        "base_seed":    42,

        # Neuromorphe Hyperparameter (Kapitel 4.3.6/4.3.7)
        "neural": neural,

        # Komplexitätsprofile (Kapitel 4.5 / 2.5.3)
        "complexity_profiles": {
            "Nullordnung":            complexity_profile_zero_order(),
            "Arith. Tracking":        complexity_profile_arithmetic_tracking(window=8),
            "Lineare Präd. AR(2)":    complexity_profile_linear_ar(order=2),
            "DPCM (ord=3)":           complexity_profile_dpcm(order=3),
            "LSB-first":              complexity_profile_lsb_first(),
            # "LSTM (2-schichtig)":     _complexity_profile_lstm(neural["L"], neural["H1"], neural["H2"]),
            "SNN (2-schichtig, LIF)": _complexity_profile_snn(
                                          neural["L"], neural["H1"], neural["H2"],
                                          neural["n_steps"]),
            "DNN (Eigen)": _complexity_profile_dnn(
                                          neural["L"], neural["H1"], neural["H2"],
                                          neural.get("num_groups", 2)), # num_gruops=4 -> 2
        },

        # Laufzeit-Flags
        "quick": quick,
        "run_snn_architecture_benchmark": not args.no_neuromorphic,
    }


# Prädiktor-Factories (zentral definiert, um Duplikate zu vermeiden)
def _classic_predictor_registry() -> dict:
    # Konventionelle Prädiktoren (Kapitel 2.3), benötigen kein Training
    return {
        "Konventionell":        lambda: None,
        "Nullordnung":          lambda: ZeroOrderPredictor(),
        "Arith. Tracking":      lambda: ArithmeticTrackingPredictor(window=8),
        "Lineare Präd. AR(2)":  lambda: LinearPredictor(order=2),
        "DPCM (ord=3)":         lambda: DPCMPredictor(order=3),
        "LSB-first":            lambda: LSBFirstPredictor(),
    }


def _neural_predictor_registry(neural: dict, sar_cfg: SARConfig) -> dict:
    return {
        "SNN (2-schichtig, LIF)": lambda seed=None: SnnTorchPredictor(
            L=neural["L"], H1=neural["H1"], H2=neural["H2"],
            n_epochs=neural["n_epochs"], patience=neural["patience"],
            max_search_bits=neural.get("max_search_bits_snn"),
            n_bits=sar_cfg.n_bits,
            v_ref=sar_cfg.v_ref,
            seed=neural["seed"] if seed is None else seed,
        ),
        "DNN (Eigen)": lambda seed=None: DnnTorchPredictor(
            L=neural["L"], H1=neural["H1"], H2=neural["H2"],
            num_groups=neural.get("num_groups", 4),
            n_epochs=neural["n_epochs"], patience=neural["patience"],
            max_search_bits=neural.get("max_search_bits_dnn"),
            n_bits=sar_cfg.n_bits,
            v_ref=sar_cfg.v_ref,
            seed=neural["seed"] if seed is None else seed,
        ),
    }

# Konventioneller Vergleich (Kapitel 4.2 / 5.1–5.3)
def run_2_conventional(cfg: dict) -> None:
    """
    Führt den Basis-Vergleich aus Kapitel 4.2 (simulation.py / ExperimentRunner) für alle konventionellen Prädiktoren über alle
    Signaltypen aus. Liefert einen schnellen Überblick ohne vollständiges Train/Val/Test-Protokoll — alle Samples fließen in die Metriken ein.
    Die neuromorphen Prädiktoren benötigen zwingend eine Trainingsphase und werden daher erst später berücksichtigt.
    """
    _banner("Konventioneller Überblick (alle Samples, kein Split)")

    sar_cfg = cfg["sar_config"]
    runner = ExperimentRunner()
    predictor_registry = _classic_predictor_registry()

    for sig in cfg["signal_types"]:
        _section(f"Signal: {sig}")
        headers = ["Prädiktor", "RMSE", "MAE", "mean cyc.", "E_save [%]"]
        col_widths = [22, 9, 9, 10, 12]
        rows = []

        for label, factory in predictor_registry.items():
            exp = ExperimentConfig(
                label=label,
                predictor_factory=factory,
                sar_config=sar_cfg,
                signal_type=sig,
                signal_kwargs=cfg["signal_kwargs"].get(sig, {}),
                n_samples=cfg["n_samples"],
                n_runs=cfg["n_runs"],
                base_seed=cfg["base_seed"],
                fs=cfg["fs"],
                jitter_frac=cfg["jitter_frac"],
            )
            results = runner.run(exp)
            rmse = np.mean([r.rmse for r in results])
            mae  = np.mean([r.mae  for r in results])
            cyc  = np.mean([r.mean_cycles for r in results])
            esave = (1.0 - cyc / sar_cfg.n_bits) * 100
            rows.append([label, f"{rmse:.5f}", f"{mae:.5f}", f"{cyc:.2f}",  f"{esave:.1f}%"])
        _print_table(headers, rows, col_widths)

# Trainings-/Testprotokoll für neuromorphe Prädiktoren (Kapitel 4.4)
def run_3_protocol(cfg: dict) -> dict:
    """
    Führt für die neuromorphen Prädiktoren (SNN, DNN) das in Kapitel 4.4 definierte Train/Val/Test-Protokoll aus (PretrainedPredictorRunner):
    Training auf dem Train+Val-Anteil, Bewertung getrennt nach Train-, Val- und Test-Phase. Anschließend wird per gepaartem t-Test geprüft,
    ob sich die mittlere Testphasen-Zyklenzahl von SNN und DNN signifikant unterscheidet (Kapitel 4.4.4). Gibt die rohen ProtocolResult-Listen für  4 zurück.
    """
    _banner("Train/Val/Test-Protokoll für DNN & SNN (Kapitel 4.4)")

    sar_cfg = cfg["sar_config"]
    split   = cfg["split_config"]
    runner  = PretrainedPredictorRunner(split)
    predictor_registry = _neural_predictor_registry(cfg["neural"], cfg["sar_config"])
    all_raw: dict[str, dict[str, list]] = {}  # label → sig → results
    for sig in cfg["signal_types"]:
        _section(f"Signal: {sig}")
        headers = ["Prädiktor", "Train cyc.", "Val cyc.", "Test cyc.",
                   "Test RMSE", "95%-CI"]
        col_widths = [22, 11, 10, 10, 11, 18]
        rows = []

        for label, factory in predictor_registry.items():
            results = runner.run_repeated(
                label=label,
                sar_config=sar_cfg,
                predictor_factory=factory,
                signal_type=sig,
                signal_kwargs=cfg["signal_kwargs"].get(sig, {}),
                n_samples=cfg["n_samples"],
                n_runs=cfg["n_runs"],
                base_seed=cfg["base_seed"],
                fs=cfg["fs"],
                jitter_frac = cfg["jitter_frac"]
            )
            all_raw.setdefault(label, {})[sig] = results
            agg = aggregate_test_results(results)
            ci  = f"[{agg.ci95_cycles[0]:.2f}, {agg.ci95_cycles[1]:.2f}]"
            t_cyc = np.mean([r.train.mean_cycles for r in results])
            v_cyc = np.mean([r.val.mean_cycles   for r in results])
            rows.append([label,
                         f"{t_cyc:.2f}", f"{v_cyc:.2f}",
                         f"{agg.mean_cycles:.2f}",
                         f"{agg.mean_rmse:.5f}", ci])

        _print_table(headers, rows, col_widths)

    # Signifikanztest: SNN vs. DNN (gepaarter t-Test, Kap. 4.4.4)
    _section("Signifikanztest SNN vs. DNN (gepaarter t-Test, mittl. Testzyklen)")
    print(f"  {'Signal':<14} {'t-Stat':>8} {'p-Wert':>9}  Bewertung")
    print("  " + "─" * 46)
    for sig in cfg["signal_types"]:
        try:
            res_snn = all_raw["SNN (2-schichtig, LIF)"][sig]
            res_dnn = all_raw["DNN (Eigen)"][sig]
            t, p = paired_t_test_cycles(res_snn, res_dnn)
            sig_flag = "✓ sign. (p<0.05)" if p < 0.05 else "  nicht sign."
            print(f"  {sig:<14} {t:>+8.3f} {p:>9.4f}  {sig_flag}")
        except (KeyError, ValueError):
            print(f"  {sig:<14}  – nicht ausgeführt –")

    return all_raw

# Vollständige Metrikauswertung (Kapitel 4.5 / 5.3-5.5)
def run_4_metrics(cfg: dict) -> None:
    """
    Berechnet für jeden Prädiktor (konventionell + neuromorph) auf jedem Signaltyp den vollständigen FullEvaluationReport (alle drei
    Metrikgruppen aus Kapitel 2.5), inklusive Prediction Gain (Kapitel 2.5.1.3). Konventionelle Prädiktoren werden auf der
    gesamten Sequenz bewertet; LSTM/SNN werden zunächst auf dem Train+Val-Anteil trainiert und ausschließlich auf dem Testanteil
    bewertet (siehe _evaluate_predictor()), um Datenlecks zu vermeiden.
    """
    _banner("Vollständige Metrikauswertung (Kapitel 4.5)")

    sar_cfg = cfg["sar_config"]
    split_cfg = cfg["split_config"]
    profiles = cfg["complexity_profiles"]

    predictor_registry = {
        **_classic_predictor_registry(),
        **_neural_predictor_registry(cfg["neural"], cfg["sar_config"]),
    }

    # Sammelt alle (Signal × Prädiktor)-Kennzahlen als flache Dict-Liste,
    # damit sie im Anschluss an visualization.py übergeben werden können
    # (Plots ergänzen die ASCII-Tabellen, ersetzen sie aber nicht).
    plot_rows: list[dict] = []

    for sig in cfg["signal_types"]:
        _section(f"Signal: {sig}")
        headers = ["Prädiktor", "RMSE", "Gp [dB]", "mean cyc.", "max cyc.",
                   "E_save [%]", "ops/smpl", "mem [words]"]
        col_widths = [22, 9, 9, 10, 9, 12, 10, 9]
        rows = []

        gen = SignalGenerator(fs=cfg["fs"], seed=cfg["base_seed"])
        x = gen.generate(sig, cfg["n_samples"],
                         **cfg["signal_kwargs"].get(sig, {}))

        for label, factory in predictor_registry.items():
            x_eval, res = _evaluate_predictor(
                factory, x, sar_cfg, split_cfg, seed=cfg["base_seed"])
            cplx = profiles.get(label, complexity_profile_zero_order())  # Fallback

            report = evaluate(label, x_eval, res, n_bits=sar_cfg.n_bits, complexity=cplx)
            d = report_to_dict(report)

            if np.isnan(d["prediction_gain_db"]):
                gp_str = "n/a"
            elif np.isinf(d["prediction_gain_db"]):
                gp_str = "∞"
            else:
                gp_str = f"{d['prediction_gain_db']:.2f}"
            rows.append([
                label,
                f"{d['rmse']:.5f}",
                gp_str,
                f"{d['mean_cycles']:.2f}",
                str(d["max_cycles"]),
                f"{d['relative_energy_saving'] * 100:.1f}%",
                str(d["total_operations_per_sample"]),
                str(d["total_memory_words"]),
            ])

            plot_rows.append({
                "signal": sig,
                "predictor": label,
                "rmse": d["rmse"],
                "prediction_gain_db": d["prediction_gain_db"],
                "mean_cycles": d["mean_cycles"],
                "max_cycles": d["max_cycles"],
                "relative_energy_saving": d["relative_energy_saving"] * 100,
                "total_operations_per_sample": d["total_operations_per_sample"],
            })

        _print_table(headers, rows, col_widths)

    _section("Visualisierung")
    p1 = viz.plot_metric_grouped_bars(
        plot_rows, metric="relative_energy_saving",
        ylabel="E_save [%]", title="Energieeinsparung je Signaltyp",
        filename="04_energy_saving_by_signal")
    p2 = viz.plot_metric_grouped_bars(
        plot_rows, metric="rmse",
        ylabel="RMSE", title="Rekonstruktionsfehler je Signaltyp",
        filename="04_rmse_by_signal", higher_is_better=False)
    p3 = viz.plot_metric_heatmap(
        plot_rows, metric="relative_energy_saving",
        title="Energieeinsparung [%] – Prädiktor × Signaltyp",
        filename="04_energy_saving_heatmap")
    p4 = viz.plot_tradeoff_scatter(
        plot_rows, x_metric="relative_energy_saving", y_metric="rmse",
        title="Trade-off: Energieeinsparung vs. Genauigkeit",
        filename="04_tradeoff_energy_vs_rmse")
    for p in (p1, p2, p3, p4):
        print(f"  Plot gespeichert: {p}")

# Ergebniszusammenfassung (Kapitel 5.1)
def run_6_summary(cfg: dict) -> None:
    """
    Druckt eine kompakte Gesamtzusammenfassung der wichtigsten Kennzahlen
    über alle Signaltypen hinweg: mittlere Zyklenzahl und relative
    Energieeinsparung (Kapitel 2.5.2) pro Prädiktor (konventionell +
    neuromorph), gemittelt über alle ausgeführten Signaltypen, als
    Grundlage für Tabelle 5.1 / 6.3 der Arbeit. Für LSTM/SNN wird dabei
    wie in 4 nur der Testanteil je Signal gewertet, um
    Datenlecks zu vermeiden.
    """
    _banner("Gesamtzusammenfassung (Kapitel 5.1 / 6.3)")

    sar_cfg   = cfg["sar_config"]
    split_cfg = cfg["split_config"]
    predictor_registry = {
        **_classic_predictor_registry(),
        **_neural_predictor_registry(cfg["neural"], cfg["sar_config"]),
    }

    # Für jede Kombination (Prädiktor × Signaltyp) mittlere Zyklenzahl sammeln.
    all_cycles: dict[str, list[float]] = {lbl: [] for lbl in predictor_registry}
    all_esave: dict[str, list[float]] = {lbl: [] for lbl in predictor_registry}

    for sig in cfg["signal_types"]:
        gen = SignalGenerator(fs=cfg["fs"], seed=cfg["base_seed"])
        x   = gen.generate(sig, cfg["n_samples"],
                            **cfg["signal_kwargs"].get(sig, {}))
        for label, factory in predictor_registry.items():
            _, res = _evaluate_predictor(
                factory, x, sar_cfg, split_cfg, seed=cfg["base_seed"])
            cyc = float(np.mean([r.n_cycles for r in res]))
            all_cycles[label].append(cyc)
            all_esave[label].append((1.0 - cyc / sar_cfg.n_bits) * 100)

    _section("Mittlere Zyklenzahl und Energieeinsparung über alle Signaltypen")
    headers = ["Prädiktor", "Ø cyc. (alle Sig.)", "E_save [%]",
               "min cyc.", "max cyc."]
    col_widths = [22, 20, 12, 10, 10]
    rows = []
    for label, cycs in all_cycles.items():
        arr = np.array(cycs)
        mean_c = float(np.mean(arr))
        esave  = (1.0 - mean_c / sar_cfg.n_bits) * 100
        rows.append([label,
                     f"{mean_c:.3f}",
                     f"{esave:.1f}%",
                     f"{arr.min():.2f}",
                     f"{arr.max():.2f}"])
    _print_table(headers, rows, col_widths)

    # Beste Gesamtkombination hervorheben
    best = min(all_cycles, key=lambda lbl: np.mean(all_cycles[lbl]))
    best_cyc = np.mean(all_cycles[best])
    best_save = (1.0 - best_cyc / sar_cfg.n_bits) * 100
    print(f"\n  → Bester Prädiktor (geringstes Ø Zyklen): {best!r}")
    print(f"    {best_cyc:.3f} Zyklen  |  {best_save:.1f} % Energieeinsparung")
    print(f"    (Referenz konventionell: {sar_cfg.n_bits} Zyklen = 0.0 % Einsparung)")

    _section("Visualisierung")
    p = viz.plot_summary_over_signals(
        labels=list(predictor_registry.keys()),
        cycles_by_label=all_cycles,
        esave_by_label=all_esave,
        filename="06_summary_over_signals",
        n_bits=sar_cfg.n_bits,
    )
    print(f"  Plot gespeichert: {p}")


# Argument-Parser
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAR-ADU Prädiktionsalgorithmen – Hauptsimulation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Schnellmodus: reduzierte Sample-/Lauf-/Epochenzahl (~1-2 Min.)"
    )
    parser.add_argument(
        "--signal",
        choices=["sine", "multitone", "ecg_like", "quiescent", "random_walk"],
        default=None,
        help="Nur diesen Signaltyp ausführen (Standard: alle fünf)"
    )
    parser.add_argument(
        "--no-neuromorphic", action="store_true",
        help="SNN-Architekturbenchmark überspringen"
    )
    return parser.parse_args()


# main
def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  SAR-ADU Prädiktionsalgorithmen – Vgl. konv. & neurom. Verfahren ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    if args.quick:
        print("  [Schnellmodus aktiv: n_samples, n_runs und Epochenzahl stark reduziert]")

    # 1: Konfiguration
    _banner("Globale Konfiguration")
    cfg = build_global_config(args)
    sar = cfg["sar_config"]
    print(f"  SAR-ADU    : {sar.n_bits} Bit, V_ref={sar.v_ref} V, "
          f"unipolar={sar.unipolar}, σ_komp={sar.comparator_noise_std}")
    print(f"  Signaltypen: {cfg['signal_types']}")
    print(f"  n_samples  : {cfg['n_samples']}   n_runs: {cfg['n_runs']}")
    print(f"  Split      : train={cfg['split_config'].train_frac:.0%} / "
          f"val={cfg['split_config'].val_frac:.0%} / "
          f"test={cfg['split_config'].test_frac:.0%}")
    n = cfg["neural"]
    print(f"  Neuromorph : L={n['L']}  H1={n['H1']}  H2={n['H2']}  "
          f"Epochen≤{n['n_epochs']}  Patience={n['patience']}")
    _print_search_window_tradeoff(cfg)

    # 2: Konventioneller Überblick
    run_2_conventional(cfg)

    # 3: Train/Val/Test-Protokoll für LSTM & SNN
    run_3_protocol(cfg)

    # 4: Vollständige Metrikauswertung
    run_4_metrics(cfg)

    # 5: Zusammenfassung
    run_6_summary(cfg)

    elapsed = time.perf_counter() - t0
    print(f"\n{'═' * 70}")
    print(f"  Simulation abgeschlossen in {elapsed:.1f} s")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
