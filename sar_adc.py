"""
Modellierung der SAR-ADU-Signalverarbeitung in Python
    Sample & Hold  -->  DAC  -->  Komparator  -->  SAR-Logik
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class SARConfig:
    """
    Parameter:
    n_bits : int
        Auflösung des Wandlers in Bit (vgl. Kapitel 2.1.2, Quantisierung).
    v_ref : float
        Referenzspannung des DAC; legt den Eingangsspannungsbereich [-v_ref, +v_ref] bzw. [0, v_ref] fest (je nach unipolar-Flag).
    unipolar : bool
        Falls True, wird der Bereich [0, v_ref] verwendet; andernfalls [-v_ref, +v_ref].
    comparator_noise_std : float
        Standardabweichung eines additiven, normalverteilten Rauschens am Komparatoreingang (vgl. Kapitel 2.2.3, Fehlerquellen).
    max_search_bits : int
        Maximale Anzahl zusätzlicher Suchschritte, die ein Prädiktor ausserhalb des klassischen Binärsuche-Fensters anfordern darf, um eine fehlerhafte Prädiktion zu korrigieren (Sicherheitsnetz, vgl. Kapitel 2.3.5, Bit-Repeating-Konzept).
        Dient als globaler Fallback-Wert; einzelne Prädiktoren können dies
        über ihr eigenes Attribut `max_search_bits` überschreiben
        (siehe Predictor-Basisklasse, Kompromiss-Punkt "prädiktorspezifische
        Fenstergröße").
    """
    n_bits: int = 10
    v_ref: float = 1.0
    unipolar: bool = True
    comparator_noise_std: float = 0.0
    max_search_bits: int = 2

    @property
    def full_scale(self) -> float:
        return self.v_ref if self.unipolar else 2.0 * self.v_ref

    @property
    def lsb(self) -> float:
        return self.full_scale / (2**self.n_bits)

    @property
    def code_max(self) -> int:
        return (2**self.n_bits) - 1

# Sample-&-Hold-Schaltung
class SampleAndHold:
    """
    Verhaltensmodell:
    "Einfrieren" eines analogen Momentanwerts für die Dauer eines Konversionszyklus
    """

    def __init__(self, aperture_noise_std: float = 0.0, rng: Optional[np.random.Generator] = None):
        self.aperture_noise_std = aperture_noise_std
        self._rng = rng if rng is not None else np.random.default_rng()

    def sample(self, x: float) -> float:
        #Tastet den analogen Momentanwert x ab und hält ihn
        if self.aperture_noise_std > 0.0:
            x = x+self._rng.normal(0.0, self.aperture_noise_std)
        return float(x)


# Digital-Analog-Umsetzer des SAR-Kerns
class SARDac:
    def __init__(self, config: SARConfig, gain_error: float = 0.0, offset_error: float = 0.0):
        self.config = config
        self.gain_error = gain_error      # relativer Gain-Fehler
        self.offset_error = offset_error  # absoluter Offset-Fehler

    def code_to_voltage(self, code: int) -> float:
        # setzt einen n_bits-Digitalcode in eine DAC-Ausgangsspannung um
        cfg = self.config
        code = int(np.clip(code, 0, cfg.code_max))
        v = code * cfg.lsb
        if not cfg.unipolar:
            v = v - cfg.v_ref
        v = v * (1.0 + self.gain_error) + self.offset_error
        return float(v)


# Komparator
class Comparator:
    # Vergleicht die abgetastete Eingangsspannung mit der vom DAC erzeugten Referenzspannung und liefert das binäre Vergleichsergebnis. Rauschen und Offset über ein additives Gauss'sches Rauschen modelliert
    def __init__(self, noise_std: float = 0.0, rng: Optional[np.random.Generator] = None):
        self.noise_std = noise_std
        self._rng = rng if rng is not None else np.random.default_rng()

    def compare(self, v_in: float, v_dac: float) -> bool:
        # True, wenn v_in >= v_dac, sonst false.
        noise = self._rng.normal(0.0, self.noise_std) if self.noise_std > 0.0 else 0.0
        return (v_in + noise) >= v_dac


# Prädiktor-Schnittstelle
class Predictor(ABC):
    """
    Schnitttstelle für alle Prädikationsalgorithmen.
    Die SAR-Konversionslogik (SARConverter) interagiert nur über predict() und update(), wodurch Prädiktor und Wandlerkern vollständig entkoppelt sind.

    predict(history) -> float
        Liefert Schätzung x_hat des nächsten Abtastwerts auf Basis der bisherigen Werte.
    update(x_true, x_hat) -> None
        Aktualisiert internen Zustand des Prädiktors nach Konversion, z.B. für adaptive/online-lernende Verfahren.
    name : str
        Bezeichner für Protokollierung und Diagrammlegenden.
    max_search_bits : Optional[int]
        Prädiktorspezifischer Override für SARConfig.max_search_bits (vgl.
        Kap. 2.3.5 / Diskussion "Fenstergröße prädiktorspezifisch statt
        global"). None (Default) bedeutet: der globale Wert aus SARConfig
        wird verwendet. Rauschempfindlichere Prädiktoren (z.B. neuromorphe
        Verfahren mit größerer Prädiktionsvarianz) können hierüber ein
        breiteres Suchfenster anfordern, ohne dass bereits präzise
        Prädiktoren (z.B. AR(2)) durch eine globale Fensteraufweitung
        unnötig verteuert werden (jedes zusätzliche Fensterbit kostet einen
        zusätzlichen Binärsuchzyklus bei jedem Treffer, vgl.
        metrics.hit_window_cost()).
    """
    name: str = "AbstractPredictor"
    max_search_bits: Optional[int] = None

    @abstractmethod
    def predict(self, history: np.ndarray) -> float:
        ...

    def update(self, x_true: float, x_hat: float) -> None:
        return None

    def reset(self) -> None:
        #Setzt internen Zustand zwischen unabhängigen Testläufen zurück
        return None


class ZeroOrderPredictor(Predictor):
    # Nullordnungsprädiktor: x_hat[n] = x[n-1]: dient als Referenz für den Vergleich aller weiteren Prädiktoren
    name = "Nullordnungsprädiktor"

    def predict(self, history: np.ndarray) -> float:
        if history.size == 0:
            return 0.0
        return float(history[-1])


# SAR-Konversionslogik (Successive-Approximation-Verfahren)
@dataclass
class ConversionResult:
    code: int                 # finaler Digitalcode
    voltage: float            # DAC-Ausgabe
    n_cycles: int             # Bitcycles
    bits_resolved: list = field(default_factory=list)  # Reihenfolge Bit-Entscheidungen
    x_hat: Optional[float] = None  # rohe, unquantisierte Prädiktion


class SARConverter:
    """
    Kernmodell des SAR-ADU:
    Konventioneller Modus (predictor=None): klassische Binärsuche über alle n_bits Bit-Positionen (MSB-first)
    Benötigt exakt n_bits Komparatorzyklen pro Abtastwert (Referenz)

    Prädiktionsgestützter Modus (predictor != None): Prädiktor liefert eine Schätzung x_hat des Abtastwerts
    Binärsuche beginnt in Umgebung des erwarteten Werts
    """

    def __init__(self, config: SARConfig, predictor: Optional[Predictor] = None, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.predictor = predictor
        self._rng = rng if rng is not None else np.random.default_rng()

        self.sh = SampleAndHold(rng=self._rng)
        self.dac = SARDac(config)
        self.comparator = Comparator(noise_std=config.comparator_noise_std, rng=self._rng)

        # Verlauf bereits gewandelter Werte
        self._history: list[float] = []

    def reset(self) -> None:
        # zurücksetzen Verlauf und Prädiktorzustand zwischen Testläufen
        self._history.clear()
        if self.predictor is not None:
            self.predictor.reset()

    def _voltage_to_code(self, v: float) -> int:
        # Rundet eine Spannung auf nächstgelegenen Wert
        cfg = self.config
        v_offset = v + cfg.v_ref if not cfg.unipolar else v
        code = int(round(v_offset / cfg.lsb))
        return int(np.clip(code, 0, cfg.code_max))

    def _binary_search(self, v_in: float, start_code: int, start_bit: int) -> ConversionResult:
        cfg = self.config
        code = start_code
        bits_resolved = []
        n_cycles = 0

        for bit_pos in range(start_bit, -1, -1):
            trial_code = code | (1 << bit_pos)
            v_dac = self.dac.code_to_voltage(trial_code)
            bit_set = self.comparator.compare(v_in, v_dac)
            n_cycles += 1
            if bit_set:
                code = trial_code
            bits_resolved.append(bit_set)

        voltage = self.dac.code_to_voltage(code)
        return ConversionResult(code=code, voltage=voltage, n_cycles=n_cycles, bits_resolved=bits_resolved)

    def convert(self, x: float) -> ConversionResult:
        """
        Wandelt einzelnen analogen Abtastwert x in Digitalcode -> vollständige Konversionsergebnis zurück
          1. Sample & Hold tastet x ab
          2. Falls Prädiktor konfiguriert ist, wird Schätzung x_hat berechnet
          3. Die SAR-Logik führt die Binärsuche durch
          4. Prädiktor erhält über update() eine Rückmeldung über tatsächlichen Wert
        """
        cfg = self.config
        v_sampled = self.sh.sample(x)

        if self.predictor is None:
            # Konventionelle Wandlung
            result = self._binary_search(v_sampled, start_code=0, start_bit=cfg.n_bits - 1)
        else:
            history_arr = np.asarray(self._history, dtype=float)
            x_hat = self.predictor.predict(history_arr)

            # Schätzung in vorläufigen Startcode umrechnen
            predicted_code = self._voltage_to_code(x_hat)

            # Prädiktorspezifisches Suchfenster (falls gesetzt), sonst
            # globaler SARConfig-Default (Kompromiss-Punkt 2).
            effective_search_bits = getattr(self.predictor, "max_search_bits", None)
            if effective_search_bits is None:
                effective_search_bits = cfg.max_search_bits

            # Reduziertes Suchfenster
            start_bit = min(effective_search_bits, cfg.n_bits - 1)
            window = 1 << (start_bit + 1)  # Breite des per Binärsuche abgedeckten Restfensters

            # Obere, als sicher angenommene Bits werden direkt aus dem vorhergesagten Code übernommen
            upper_mask = ~(window - 1) & cfg.code_max
            base_code = predicted_code & upper_mask

            # Verifikationszyklus
            v_window_high = self.dac.code_to_voltage(min(base_code + window, cfg.code_max))
            # compare() liefert true, falls v_in >= v_dac; für obere Fenstergrenze wird daher das Komplement benötigt
            window_ok_high = not self.comparator.compare(v_in=v_sampled, v_dac=v_window_high)
            v_window_low = self.dac.code_to_voltage(base_code)
            window_ok_low = self.comparator.compare(v_in=v_sampled, v_dac=v_window_low)  # v_in >= v_low?
            verification_cycles = 2

            if window_ok_high and window_ok_low:
                # Prädiktion liegt im erwarteten Fenster -> verkürzte Suche.
                result = self._binary_search(v_sampled, start_code=base_code, start_bit=start_bit)
                result.n_cycles = min(result.n_cycles + verification_cycles, cfg.n_bits)  # ← Kappung ergänzt
            else:
                # Prädiktion grob falsch -> vollständige Binärsuche als Rückfallebene
                result = self._binary_search(v_sampled, start_code=0, start_bit=cfg.n_bits - 1)
                result.n_cycles = min(result.n_cycles + verification_cycles, cfg.n_bits)

            self.predictor.update(x_true=v_sampled, x_hat=x_hat)
            result.x_hat = x_hat

        self._history.append(result.voltage)
        return result

    def convert_sequence(self, x: np.ndarray) -> list[ConversionResult]:
        #wandelt vollständige Abtastwertfolge sequenziell um
        return [self.convert(float(xi)) for xi in x]

    def convert_sequence_pretrained(self, x: np.ndarray) -> list[ConversionResult]:
        # wie convert_sequence(), aber setzt nur den internen Verlaufsspeicher (self._history) des Wandlers zurück
        self._history.clear()
        if self.predictor is not None and hasattr(self.predictor, "_buffer"):
            self.predictor._buffer.clear()  # type: ignore[attr-defined]
        return [self.convert(float(xi)) for xi in x]