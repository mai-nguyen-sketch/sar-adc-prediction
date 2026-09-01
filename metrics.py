"""
Modul implementiert eingeführte Bewertungsmetriken:

  1. Metriken der Vorhersagegüte      (Kapitel 2.5.1)
     - Root Mean Square Error (RMSE)          (2.5.1.1)
     - Mean Absolute Error (MAE)              (2.5.1.2)
     - Prediction Gain (Gp)                   (2.5.1.3)
     - Worst-Case-Fehler                       (2.5.1.4)

  2. Metriken der Konversions- und Energieeffizienz (Kapitel 2.5.2)
     - Durchschnittliche Anzahl benötigter SAR-Zyklen (2.5.2.1)
     - Relative Energieeinsparung                      (2.5.2.2)

  3. Metriken des Rechen- und Implementierungsaufwands (Kapitel 2.5.3)
     - Anzahl Operationen pro Sample                    (2.5.3.1)
     - Speicherbedarf                                    (2.5.3.2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from sar_adc import ConversionResult, Predictor


# 1. Metriken der Vorhersagegüte
def rmse(x_true: np.ndarray, x_reconstructed: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float)
    x_reconstructed = np.asarray(x_reconstructed, dtype=float)
    return float(np.sqrt(np.mean((x_true - x_reconstructed)**2)))


def mae(x_true: np.ndarray, x_reconstructed: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float)
    x_reconstructed = np.asarray(x_reconstructed, dtype=float)
    return float(np.mean(np.abs(x_true - x_reconstructed)))


def prediction_gain(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    error = x_true - x_hat

    var_signal = np.var(x_true)
    var_error = np.var(error)

    # Segment nahezu konstant (z.B. random_walk am Clipping-Rand): Gp ist auf einem quasi-varianzfreien Signal nicht aussagekräftig.
    if var_signal <= 1e-9:
        return float("nan")
    if var_error <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(var_signal / var_error))


def worst_case_error(x_true: np.ndarray, x_reconstructed: np.ndarray) -> float:
    x_true = np.asarray(x_true, dtype=float)
    x_reconstructed = np.asarray(x_reconstructed, dtype=float)
    return float(np.max(np.abs(x_true - x_reconstructed)))


@dataclass
class AccuracyMetrics:
    #Bündelung aller Metriken der Vorhersagegüte für einen Datenabschnitt
    rmse: float
    mae: float
    prediction_gain_db: float
    worst_case_error: float


def compute_accuracy_metrics(x_true: np.ndarray, x_reconstructed: np.ndarray, x_hat: Optional[np.ndarray] = None) -> AccuracyMetrics:
    # Berechnet alle vier Metriken der Vorhersagegüte (Kapitel 2.5.1) in einem Aufruf
    x_hat_for_gain = x_hat if x_hat is not None else x_reconstructed
    return AccuracyMetrics(rmse=rmse(x_true, x_reconstructed), mae=mae(x_true, x_reconstructed), prediction_gain_db=prediction_gain(x_true, x_hat_for_gain), worst_case_error=worst_case_error(x_true, x_reconstructed))


# Metriken der Konversions- und Energieeffizienz
def mean_sar_cycles(results: list[ConversionResult]) -> float:
    cycles = np.array([r.n_cycles for r in results], dtype=float)
    return float(np.mean(cycles))


def relative_energy_saving(results_predictive: list[ConversionResult], n_bits: int) -> float:
    c_mean = mean_sar_cycles(results_predictive)
    return float(1.0 - (c_mean / n_bits))


def hit_window_cost(max_search_bits: int) -> int:
    """
    Zyklenkosten eines "Treffers" (Prädiktion liegt im Suchfenster), vgl. SARConverter.convert(): (max_search_bits + 1) Binärsuchschritt für die verbleibenden Bit-Positionen + 2 Verifikationszyklen.
    """
    return int(max_search_bits) + 1 + 2


def expected_mean_cycles(hit_rate: float, max_search_bits: int, n_bits: int) -> float:
    """
    Kompromiss-Rechnung (Punkt 1/2 der Diskussion): erwartete mittlere Zyklenzahl in Abhängigkeit von Trefferquote und Fenstergröße.
        mean_cycles = n_bits - hit_rate * (n_bits - hit_window_cost(max_search_bits))
    Nützlich, um vorab abzuschätzen, ob ein breiteres Suchfenster für einen gegebenen (empirisch gemessenen oder angenommenen) hit_rate überhaupt einen Nettovorteil bringt:
    für hit_rate=1.0 ist ein schmaleres Fenster immer besser (geringere hit_window_cost), für kleine hit_rate lohnt sich ein breiteres Fenster trotz höherer Trefferkosten, weil deutlich mehr Prädiktionen überhaupt erst einen Treffer erzielen.
    """
    cost = hit_window_cost(max_search_bits)
    return float(n_bits - hit_rate * (n_bits - cost))


def hit_rate_from_mean_cycles(mean_cycles: float, max_search_bits: int, n_bits: int) -> float:
    """
    Kehrformel zu expected_mean_cycles(): rechnet aus einer gemessenen mittleren Zyklenzahl die implizite Trefferquote des Suchfensters zurück.
    Praktisch, um aus den Ergebnistabellen (Kap. 4.4/4.5) direkt abzulesen, wie oft ein Prädiktor tatsächlich im Fenster gelandet ist.
    """
    cost = hit_window_cost(max_search_bits)
    denom = n_bits - cost
    if denom <= 0:
        return float("nan")
    return float((n_bits - mean_cycles) / denom)


@dataclass
class EfficiencyMetrics:
    # Bündelung der Metriken der Konversions- und Energieeffizienz
    mean_cycles: float
    max_cycles: int
    relative_energy_saving: float

def compute_efficiency_metrics(results: list[ConversionResult], n_bits: int) -> EfficiencyMetrics:
    # Berechnet beide Effizienzmetriken in einem aufruf
    cycles = np.array([r.n_cycles for r in results], dtype=int)
    return EfficiencyMetrics(
        mean_cycles=float(np.mean(cycles)),
        max_cycles=int(np.max(cycles)),
        relative_energy_saving=relative_energy_saving(results, n_bits),
    )


# Metriken des Rechen- und Implementierungsaufwands
@dataclass
class ComplexityProfile:
    """
    Attribute
    multiplications, additions, comparisons, nonlinear_ops : int
        Anzahl der jeweiligen Grundoperationen
    parameters : int
        Anzahl der bei der Inferenz zu speichernden Parameter
    state_size : int
        Anzahl zusätzlich zu speichernder Zustandsgrößen
    """
    name: str
    multiplications: int
    additions: int
    comparisons: int
    nonlinear_ops: int
    parameters: int
    state_size: int

    @property
    def total_operations(self) -> int:
        # Gesamtzahl der Operationen pro Sample
        return self.multiplications + self.additions + self.comparisons + self.nonlinear_ops

    @property
    def total_memory_words(self) -> int:
        # Geschätzter Gesamtspeicherbedarf in Datenworten
        return self.parameters + self.state_size


def complexity_profile_zero_order() -> ComplexityProfile:
    # Nullordnungsprädiktion
    return ComplexityProfile(name="Nullordnungsprädiktor", multiplications=0, additions=0, comparisons=0, nonlinear_ops=0, parameters=0, state_size=1)


def complexity_profile_arithmetic_tracking(window: int) -> ComplexityProfile:
    # Arithmetisches Tracking
    return ComplexityProfile(name="Arithmetisches Tracking", multiplications=1, additions=window, comparisons=2, nonlinear_ops=0, parameters=0, state_size=window + 1)


def complexity_profile_linear_ar(order: int) -> ComplexityProfile:
    # Lineare Prädiktion / AR(p)
    return ComplexityProfile(name=f"Lineare Prädiktion (AR{order})", multiplications=order, additions=max(order - 1, 0), comparisons=0, nonlinear_ops=0, parameters=order, state_size=order)


def complexity_profile_dpcm(order: int) -> ComplexityProfile:
    # DPCM-Prädiktion
    return ComplexityProfile(name="DPCM-Prädiktion", multiplications=1, additions=max(order - 1, 0), comparisons=0, nonlinear_ops=0, parameters=0, state_size=order)


def complexity_profile_lsb_first() -> ComplexityProfile:
    # LSB-first/Bit-Repeating = Nullordnungsprädiktion
    return ComplexityProfile(name="LSB-first (Bit-Repeating)", multiplications=0, additions=0, comparisons=0, nonlinear_ops=0, parameters=0, state_size=1)


def complexity_profile_dense_feedforward(n_inputs: int, n_hidden: int) -> ComplexityProfile:
    # Dense-Feedforward-Netz
    mults = n_inputs * n_hidden + n_hidden
    adds = n_hidden * (n_inputs - 1) + n_hidden + (n_hidden - 1) + 1
    nonlin = n_hidden  # TanH/ReLU-Auswertungen der versteckten Schicht
    params = (n_hidden * n_inputs) + n_hidden + n_hidden + 1
    return ComplexityProfile(name="Dense-Feedforward-Netz", multiplications=mults, additions=adds, comparisons=0, nonlinear_ops=nonlin, parameters=params, state_size=n_inputs)


# Kombination aller drei Metrikgruppen
@dataclass
class FullEvaluationReport:
    label: str
    accuracy: AccuracyMetrics
    efficiency: EfficiencyMetrics
    complexity: ComplexityProfile


def evaluate(label: str, x_true: np.ndarray, results: list[ConversionResult], n_bits: int, complexity: ComplexityProfile, x_hat: Optional[np.ndarray] = None) -> FullEvaluationReport:
    x_true = np.asarray(x_true, dtype=float)
    x_reconstructed = np.array([r.voltage for r in results], dtype=float)

    if x_hat is None:
        raw_x_hat = [r.x_hat for r in results]
        if all(v is not None for v in raw_x_hat):
            x_hat = np.array(raw_x_hat, dtype=float)
            # Sicherheitsnetz: Extrapolationsausreißer (z.B. Train/Test-Distribution-Shiftm, bei random_walk) auf plausiblen Wertebereich begrenzen, statt den Gp-Wert an ihnen kollabieren zu lassen.
            lo, hi = float(np.min(x_true)), float(np.max(x_true))
            margin = 0.5 * (hi - lo)
            x_hat = np.clip(x_hat, lo - margin, hi + margin)

    accuracy = compute_accuracy_metrics(x_true, x_reconstructed, x_hat=x_hat)
    efficiency = compute_efficiency_metrics(results, n_bits=n_bits)

    return FullEvaluationReport(label=label, accuracy=accuracy, efficiency=efficiency, complexity=complexity)


def report_to_dict(report: FullEvaluationReport) -> dict:
    # FullEvaluationReport -> dict
    return {
        "label": report.label,
        "rmse": report.accuracy.rmse,
        "mae": report.accuracy.mae,
        "prediction_gain_db": report.accuracy.prediction_gain_db,
        "worst_case_error": report.accuracy.worst_case_error,
        "mean_cycles": report.efficiency.mean_cycles,
        "max_cycles": report.efficiency.max_cycles,
        "relative_energy_saving": report.efficiency.relative_energy_saving,
        "total_operations_per_sample": report.complexity.total_operations,
        "parameters": report.complexity.parameters,
        "state_size": report.complexity.state_size,
        "total_memory_words": report.complexity.total_memory_words,
    }