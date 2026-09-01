"""
Implementierung der Prädikationsalgorithmen.
Enthaltene konventionelle Prädiktoren (Kapitel 2.3):
    - ZeroOrderPredictor            (Kapitel 2.3.1, bereits in sar_adc.py
                                      definiert, hier nur referenziert)
    - ArithmeticTrackingPredictor   (Kapitel 2.3.2)
    - LinearPredictor                (Kapitel 2.3.3, AR(p)-Modell)
    - DPCMPredictor                  (Kapitel 2.3.4)
    - LSBFirstPredictor              (Kapitel 2.3.5, mit Bit-Repeating)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from sar_adc import Predictor

# Arithmetisches Tracking
class ArithmeticTrackingPredictor(Predictor):
    name = "Arithmetisches Tracking"

    def __init__(self, window: int = 8, base_bits: int = 1, max_bits: int = 6):
        self.window = window
        self.base_bits = base_bits
        self.max_bits = max_bits
        self._recent_diffs: deque[float] = deque(maxlen=window)

    def predict(self, history: np.ndarray) -> float:
        if history.size == 0:
            return 0.0
        return float(history[-1])

    def update(self, x_true: float, x_hat: float) -> None:
        # Aktivität wird über die absolute Differenz zum Vorwert erfasst.
        self._recent_diffs.append(abs(x_true - x_hat))

    def suggested_search_bits(self, lsb: float) -> int:
        # Schätzt die Anzahl benötigter Suchbits anhand der jüngsten Signalaktivität
        if not self._recent_diffs:
            return self.base_bits
        mean_diff = float(np.mean(self._recent_diffs))
        # Anzahl Bits, um mean_diff/lsb Quantisierungsschritte abzudecken
        n_steps = max(1, int(np.ceil(mean_diff / max(lsb, 1e-12))))
        bits_needed = max(self.base_bits, int(np.ceil(np.log2(n_steps + 1))))
        return int(np.clip(bits_needed, self.base_bits, self.max_bits))

    def reset(self) -> None:
        self._recent_diffs.clear()



# Lineare Prädiktion / autoregressive Modelle
class LinearPredictor(Predictor):
    name = "Lineare Prädiktion (AR)"

    def __init__(self, order: int = 2, coeffs: Optional[list[float]] = None, adaptive: bool = False, learning_rate: float = 0.01):
        self.order = order
        if coeffs is not None:
            assert len(coeffs) == order, "Anzahl Koeffizienten muss order entsprechen."
            self.coeffs = np.array(coeffs, dtype=float)
        elif order == 1:
            self.coeffs = np.array([1.0])           # entspricht Nullordnungsprädiktion
        elif order == 2:
            self.coeffs = np.array([2.0, -1.0])      # lineare Extrapolation
        else:
            # Default
            self.coeffs = np.array([1.0] + [0.0] * (order - 1))

        self.adaptive = adaptive
        self.lr = learning_rate

    def predict(self, history: np.ndarray) -> float:
        if history.size == 0:
            return 0.0
        if history.size < self.order:
            return float(history[-1])  # Fallback bei zu kurzer Historie
        recent = history[-self.order:][::-1]  # [x[n-1], x[n-2], ..., x[n-p]]
        return float(np.dot(self.coeffs, recent))

    def update(self, x_true: float, x_hat: float) -> None:
        if not self.adaptive:
            return None
        error = x_true - x_hat
        # LMS-Update
        if len(self.coeffs) > 0:
            self.coeffs[0] += self.lr * error

    def reset(self) -> None:
        pass  # Koeffizienten bleiben über Testläufe hinweg erhalten



#  Differentielle Pulse-Code-Modulation
class DPCMPredictor(Predictor):
    name = "DPCM-Prädiktion"

    def __init__(self, order: int = 1):
        self.order = order

    def predict(self, history: np.ndarray) -> float:
        if history.size == 0:
            return 0.0
        recent = history[-self.order:]
        return float(np.mean(recent))


# LSB-first-Quantisierung mit Bit-Repeating
class LSBFirstPredictor(Predictor):
    name = "LSB-first (Bit-Repeating)"

    def predict(self, history: np.ndarray) -> float:
        if history.size == 0:
            return 0.0
        return float(history[-1])



