from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy import stats
from sar_adc import ConversionResult, Predictor, SARConfig, SARConverter
from simulation import SignalGenerator, _jitter_signal_kwargs

# Split-Konfiguration
@dataclass
class SplitConfig:
    # Relative Aufteilung einer Abtastwertfolge in Trainings-, Validierungs-und Testabschnitt
    train_frac: float = 0.5
    val_frac: float = 0.2
    test_frac: float = 0.3
    test_warmup_fraction: float = 0.1

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"train_frac + val_frac + test_frac muss 1.0 ergeben, ist aber {total:.3f}.")

    def split_indices(self, n_samples: int) -> tuple[slice, slice, slice]:
        n_train = int(round(n_samples * self.train_frac))
        n_val = int(round(n_samples * self.val_frac))
        n_test = n_samples - n_train - n_val
        return (
            slice(0, n_train),
            slice(n_train, n_train + n_val),
            slice(n_train + n_val, n_train + n_val + n_test),
        )


# Phasenweise Kennzahlen
@dataclass
class PhaseMetrics:
    phase: str
    n_samples: int
    rmse: float
    mae: float
    mean_cycles: float
    max_cycles: int
    worst_case_error: float


@dataclass
class ProtocolResult:
    label: str
    seed: int
    train: PhaseMetrics
    val: PhaseMetrics
    test: PhaseMetrics


def _compute_phase_metrics(phase_name: str, x_segment: np.ndarray, results_segment: list[ConversionResult], discard_initial: int = 0) -> PhaseMetrics:
    x_eval = x_segment[discard_initial:]
    results_eval = results_segment[discard_initial:]
    recon = np.array([r.voltage for r in results_eval])
    cycles = np.array([r.n_cycles for r in results_eval])
    errors = recon - x_eval
    return PhaseMetrics(phase=phase_name, n_samples=len(x_eval), rmse=float(np.sqrt(np.mean(errors ** 2))), mae=float(np.mean(np.abs(errors))), mean_cycles=float(np.mean(cycles)), max_cycles=int(np.max(cycles)), worst_case_error=float(np.max(np.abs(errors))))


# Runner für offline vortrainierte Prädiktoren
class PretrainedPredictorRunner:
    # Train/Val/Test-Protokoll
    def __init__(self, split_config: Optional[SplitConfig] = None):
        self.split_config = split_config if split_config is not None else SplitConfig()

    def run_single(self, label: str, sar_config: SARConfig, predictor_factory, signal_type: str,
                   signal_kwargs: dict, n_samples: int, seed: int, fs: float = 1000.0,
                   jitter_frac: float = 0.0) -> ProtocolResult:
        generator = SignalGenerator(fs=fs, seed=seed)
        jitter_rng = np.random.default_rng(seed + 1_000_000)
        signal_kwargs = _jitter_signal_kwargs(signal_kwargs, jitter_rng, jitter_frac)
        x = generator.generate(signal_type, n_samples, **signal_kwargs)
        train_sl, val_sl, test_sl = self.split_config.split_indices(n_samples)

        try:
            predictor = predictor_factory(seed)
        except TypeError:
            predictor = predictor_factory()

        if not hasattr(predictor, "train_offline"):
            raise TypeError(
                f"PretrainedPredictorRunner erwartet einen Prädiktor mit train_offline(); "
                f"'{type(predictor).__name__}' hat keine solche Methode."
            )

        # Trainingsphase = train_frac + val_frac
        predictor.train_offline(x[:val_sl.stop])

        # durchgängiger SAR-Konversionslauf über die gesamte Sequenz,
        rng = np.random.default_rng(seed)
        converter = SARConverter(sar_config, predictor=predictor, rng=rng)
        results = converter.convert_sequence_pretrained(x)

        train_metrics = _compute_phase_metrics("train", x[train_sl], results[train_sl])
        val_metrics = _compute_phase_metrics("val", x[val_sl], results[val_sl])

        n_test = test_sl.stop - test_sl.start
        discard = int(round(n_test * self.split_config.test_warmup_fraction))
        test_metrics = _compute_phase_metrics("test", x[test_sl], results[test_sl], discard_initial=discard)

        return ProtocolResult(label=label, seed=seed, train=train_metrics, val=val_metrics, test=test_metrics)

    def run_repeated(self, label: str, sar_config: SARConfig, predictor_factory: Callable[[], Predictor],
                     signal_type: str, signal_kwargs: dict, n_samples: int, n_runs: int,
                     base_seed: int = 42, fs: float = 1000.0, jitter_frac: float = 0.0) -> list[ProtocolResult]:
        # n_runs unabhängige Wiederholungen mit fortlaufend verschobenen Seeds
        return [
            self.run_single(label, sar_config, predictor_factory, signal_type, signal_kwargs,
                            n_samples, seed=base_seed + i, fs=fs, jitter_frac=jitter_frac)
            for i in range(n_runs)
        ]

# Statistische Auswertung über Wiederholungen
@dataclass
class AggregatedTestResult:
    label: str
    n_runs: int
    mean_rmse: float
    ci95_rmse: tuple[float, float]
    mean_mae: float
    mean_cycles: float
    ci95_cycles: tuple[float, float]
    mean_worst_case: float


def _confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else float("nan")
        return (m, m)
    mean = float(np.mean(values))
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return (mean - margin, mean + margin)


def aggregate_test_results(results: list[ProtocolResult]) -> AggregatedTestResult:
    label = results[0].label
    rmse_vals = np.array([r.test.rmse for r in results])
    mae_vals = np.array([r.test.mae for r in results])
    cycles_vals = np.array([r.test.mean_cycles for r in results])
    worst_vals = np.array([r.test.worst_case_error for r in results])
    return AggregatedTestResult(
        label=label, n_runs=len(results),
        mean_rmse=float(np.mean(rmse_vals)), ci95_rmse=_confidence_interval(rmse_vals),
        mean_mae=float(np.mean(mae_vals)),
        mean_cycles=float(np.mean(cycles_vals)), ci95_cycles=_confidence_interval(cycles_vals),
        mean_worst_case=float(np.mean(worst_vals)),
    )


def paired_t_test_cycles(results_a: list[ProtocolResult], results_b: list[ProtocolResult]) -> tuple[float, float]:
    #Gepaarter t-Test auf die mittlere Testphasen-Zyklenzahl
    if [r.seed for r in results_a] != [r.seed for r in results_b]:
        raise ValueError("Für einen gepaarten t-Test müssen beide Ergebnislisten identische, paarweise Seeds haben.")
    cycles_a = np.array([r.test.mean_cycles for r in results_a])
    cycles_b = np.array([r.test.mean_cycles for r in results_b])
    t_stat, p_value = stats.ttest_rel(cycles_a, cycles_b)
    return float(t_stat), float(p_value)

