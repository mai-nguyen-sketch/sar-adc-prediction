"""
Simulationsumgebung und Versuchsdesign
  1. Eine Signalgenerator-Komponente zur reproduzierbaren Erzeugung der in Kapitel 2.5 zugrunde gelegten Testsignale (u.a. Sinus-, EKG-ähnliche und stückweise-konstante "Ruhephasen"-Signale, vgl. Kapitel 3.1.2).
  2. Ein Konfigurationsobjekt (ExperimentConfig), das ein vollständiges Experiment beschreibt
  3. Einen Experiment-Runner, der eine Liste von Konfigurationen gegeneinander ausführt und die Rohergebnisse für die in Kapitel 2.5 definierten Bewertungsmetriken aufbereitet.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sar_adc import ConversionResult, Predictor, SARConfig, SARConverter, ZeroOrderPredictor

# Schutz Underflow/Zero-Devision
EPSILON = 1e-12

# Hilfsfunktion
# Nur diese Parameter gelten als "Frequenz/Amplitude" und werden gejittert
_JITTER_KEYS = {"freq", "freqs", "amplitude"}
def _jitter_signal_kwargs(signal_kwargs: dict, rng: np.random.Generator, jitter_frac: float) -> dict:
    """Wendet pro Lauf einen relativen Gauß-Jitter auf Frequenz-/Amplitudenparameter an.
    jitter_frac=0.02 -> Standardabweichung = 2% des jeweiligen Basiswerts."""
    if jitter_frac <= 0.0:
        return signal_kwargs
    jittered = dict(signal_kwargs)
    for key, value in signal_kwargs.items():
        if key not in _JITTER_KEYS:
            continue
        if isinstance(value, (tuple, list)):
            jittered[key] = tuple(float(v + rng.normal(0.0, abs(v) * jitter_frac)) for v in value)
        elif isinstance(value, (int, float)):
            jittered[key] = float(value + rng.normal(0.0, abs(value) * jitter_frac))
    return jittered

# Signalgenerator
class SignalGenerator:
    """
    Erzeugt reproduzierbare synthetische Testsignale für die Simulation.
      - 'sine'       : reines Sinussignal als Grundlagentest
      - 'multitone'  : Summe mehrerer Sinusschwingungen zur Nachbildung komplexerer, aber weiterhin bandbegrenzter Signale
      - 'ecg_like'    : stückweise glattes Signal mit eingestreuten, impulsartigen "QRS-ähnlichen" Auslenkungen
      - 'quiescent'  : Signal mit langen Ruhephasen (konstante Abschnitte) und sporadischen Sprüngen
      - 'random_walk': Random-Walk-Prozess als Stresstest für Prädiktoren bei geringer struktureller Vorhersagbarkeit
    """

    def __init__(self, fs: float = 1000.0, seed: Optional[int] = None):
        self.fs = fs
        self.rng = np.random.default_rng(seed)

    def _time_vector(self, n_samples: int) -> np.ndarray:
        return np.arange(n_samples) / self.fs

    def sine(self, n_samples: int, freq: float = 5.0, amplitude: float = 0.45, offset: float = 0.5) -> np.ndarray:
        # Sinussignal: x[n] = offset + amplitude * sin(2*pi*f*t)
        t = self._time_vector(n_samples)
        return offset + amplitude * np.sin(2.0 * np.pi * freq * t)

    def multitone(self, n_samples: int, freqs=(3.0, 7.0, 11.0), amplitude: float = 0.3, offset: float = 0.5) -> np.ndarray:
        # Überlagerung mehrerer Sinustöne zur Nachbildung breitbandigerer Signale
        t = self._time_vector(n_samples)
        signal = np.zeros(n_samples)
        for k, f in enumerate(freqs, start=1):
            signal += (amplitude / k) * np.sin(2.0 * np.pi * f * t + k)
        return offset + signal / ((np.max(np.abs(signal))) + EPSILON) * amplitude

    def ecg_like(self, n_samples: int, beat_period: int = 150, offset: float = 0.5) -> np.ndarray:
        # EKG-ähnliches Signal
        t = np.arange(n_samples)
        baseline = 0.05 * np.sin(2.0 * np.pi * t / (beat_period * 4))
        signal = np.zeros(n_samples)
        beat_centers = np.arange(beat_period // 2, n_samples, beat_period)
        for c in beat_centers:
            window = np.arange(n_samples)
            signal += 0.4 * np.exp(-0.5 * ((window - c) / 4.0) ** 2)
        noise = self.rng.normal(0.0, 0.01, size=n_samples)
        return np.clip(offset + baseline + signal + noise, 0.0, 1.0)

    def quiescent(self, n_samples: int, change_prob: float = 0.02, step_std: float = 0.08, offset: float = 0.5) -> np.ndarray:
        #Stückweise-konstantes Signal mit seltenen Sprüngen
        signal = np.empty(n_samples)
        current = offset
        for i in range(n_samples):
            if self.rng.random() < change_prob:
                current += self.rng.normal(0.0, step_std)
                current = float(np.clip(current, 0.02, 0.98))
            signal[i] = current
        return signal

    def random_walk(self, n_samples: int, step_std: float = 0.01, offset: float = 0.5) -> np.ndarray:
        steps = self.rng.normal(0.0, step_std, size=n_samples)
        signal = offset + np.cumsum(steps)
        # Reflektierend statt hart geklippt, damit der Walk nicht am Rand "einfriert"
        period = 2.0
        folded = np.mod(signal, period)
        reflected = np.where(folded > 1.0, period - folded, folded)
        return reflected

    SIGNAL_TYPES: dict[str, str] = {
        "sine": "sine",
        "multitone": "multitone",
        "ecg_like": "ecg_like",
        "quiescent": "quiescent",
        "random_walk": "random_walk",
    }

    def generate(self, signal_type: str, n_samples: int, **kwargs) -> np.ndarray:
        if signal_type not in self.SIGNAL_TYPES:
            raise ValueError(
                f"Unbekannter Signaltyp '{signal_type}'. "
                f"Verfügbar: {list(self.SIGNAL_TYPES.keys())}"
            )
        method = getattr(self, self.SIGNAL_TYPES[signal_type])
        return method(n_samples, **kwargs)


# Experimentkonfiguration
@dataclass
class ExperimentConfig:
    """
    Simulationsexperiment mit:
      - eine SAR-ADU-Konfiguration (Auflösung, Referenzspannung, Rauschen),
      - einen Prädiktor (konventionell oder neuromorph, oder None für die konventionelle Vollwandlung als Referenz),
      - einen Signaltyp inkl. signalspezifischer Parameter,
      - die Anzahl der Wiederholungen (n_runs) mit unterschiedlichen Zufallszahlen-Seeds zur statistischen Absicherung der Ergebnisse
    """
    label: str
    predictor_factory: Callable[[int], Optional[Predictor]]
    sar_config: SARConfig = field(default_factory=SARConfig)
    signal_type: str = "sine"
    signal_kwargs: dict = field(default_factory=dict)
    n_samples: int = 2000
    n_runs: int = 5
    base_seed: int = 42
    fs: float = 1000.0
    jitter_frac: float = 0.0


# Ergebnisaggregation pro Lauf
@dataclass
class RunResult:
    # Aggregierte Kennzahlen eines Seeds
    label: str
    seed: int
    rmse: float
    mae: float
    mean_cycles: float
    max_cycles: int
    worst_case_error: float
    n_samples: int
    prediction_gain_db: float  # Vorhersagegewinn in dB
    cycles_saved_percent: float  # Prozentuale Zykleneinsparung


# Experiment-Runner
class ExperimentRunner:
    # Führt Liste von ExperimentConfig-Objekten aus und sammelt Rohergebnisse für die spätere Auswertung
    def run_single(self, config: ExperimentConfig, seed: int) -> RunResult:
        # reproduzierbares Signal erzeugen
        generator = SignalGenerator(fs=config.fs, seed=seed)
        jitter_rng = np.random.default_rng(seed + 1_000_000)
        signal_kwargs = _jitter_signal_kwargs(config.signal_kwargs, jitter_rng, config.jitter_frac)
        x = generator.generate(config.signal_type, config.n_samples, **signal_kwargs)

        # Prädiktor instanziieren
        rng = np.random.default_rng(seed)
        try:
            predictor = config.predictor_factory(seed)
        except TypeError:
            predictor = config.predictor_factory()  # Fallback für arglose Factories
        converter = SARConverter(config.sar_config, predictor=predictor, rng=rng)

        # Wandlung
        results: list[ConversionResult] = converter.convert_sequence(x)

        reconstructed = np.array([r.voltage for r in results])
        raw_cycles = np.array([r.n_cycles for r in results])
        cycles = raw_cycles

        # Fehler- & Leistungskennzahlen berechnen
        errors = reconstructed - x
        var_signal = float(np.var(x))
        var_error = float(np.var(errors))

        # Epsilon-Fix gegen ZeroDivision/Log-Underflow
        var_signal_safe = max(var_signal, EPSILON)
        var_error_safe = max(var_error, EPSILON)

        rmse = float(np.sqrt(np.mean(errors ** 2) + EPSILON))
        mae = float(np.mean(np.abs(errors)))
        worst_case = float(np.max(np.abs(errors)))

        # Vorhersagegewinn G_p in dB
        prediction_gain_db = 10.0 * np.log10(var_signal_safe / var_error_safe)

        # Prozentuale Zykleneinsparung gegenüber klassischem SAR-Standard
        baseline_cycles = float(config.sar_config.n_bits)
        mean_cycles = float(np.mean(cycles))
        cycles_saved_pct = ((baseline_cycles - mean_cycles) / baseline_cycles) * 100.0

        return RunResult(
            label=config.label,
            seed=seed,
            rmse=rmse,
            mae=mae,
            mean_cycles=mean_cycles,
            max_cycles=int(np.max(cycles)),
            worst_case_error=worst_case,
            prediction_gain_db=prediction_gain_db,
            cycles_saved_percent=cycles_saved_pct,
            n_samples=config.n_samples,
        )

    def run(self, config: ExperimentConfig) -> list[RunResult]:
        # Führt config.n_runs unabhängige Wiederholungen mit fortlaufend verschobenen Seeds aus
        return [
            self.run_single(config, seed=config.base_seed + i)
            for i in range(config.n_runs)
        ]

    def run_all(self, configs: list[ExperimentConfig]) -> dict[str, list[RunResult]]:
        # Führt mehrere Experimentkonfigurationen aus und gruppiert nach Label
        return {config.label: self.run(config) for config in configs}


# Beispielhaftes Versuchsdesign
def build_baseline_experiment_suite() -> list[ExperimentConfig]:
    # Definiert das in Kapitel 4.2 beschriebene Grund-Versuchsdesign
    sar_cfg = SARConfig(n_bits=16, v_ref=5.0, unipolar=True, comparator_noise_std=0.0005) #n_bits= 10 und v_ref = 1.0für ULP
    signal_types = ["sine", "multitone", "ecg_like", "quiescent", "random_walk"]

    configs: list[ExperimentConfig] = []
    for sig in signal_types:
        configs.append(
            ExperimentConfig(
                label=f"konventionell_{sig}",
                predictor_factory=lambda seed=None: None,
                sar_config=sar_cfg,
                signal_type=sig,
                n_samples=2000,
                n_runs=5,
            )
        )
        configs.append(
            ExperimentConfig(
                label=f"nullordnung_{sig}",
                predictor_factory=lambda seed=None: ZeroOrderPredictor(),
                sar_config=sar_cfg,
                signal_type=sig,
                n_samples=2000,
                n_runs=5,
            )
        )
    return configs
