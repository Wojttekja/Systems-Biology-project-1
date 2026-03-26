# environment.py

import numpy as np
from .strategies import EnvironmentDynamics

# ---------------------------------------------------------------------------
# Linear Shift Environment Strategy
# ---------------------------------------------------------------------------


class LinearShiftEnvironment(EnvironmentDynamics):
    """
    Scenariusz globalnego ocieplenia: optymalny fenotyp przesuwa się liniowo
    z opcjonalnymi losowymi fluktuacjami w każdym pokoleniu.

        alpha(t) = alpha(t-1) + N(c, delta^2 * I)

    Jeśli delta=0, przesunięcie jest czysto deterministyczne:
        alpha(t) = alpha(t-1) + c
    """

    def __init__(self, alpha_init: np.ndarray, c: np.ndarray, delta: float = 0.0):
        """
        :param alpha_init: początkowy optymalny fenotyp
        :param c: wektor kierunkowej zmiany (średnie przesunięcie na pokolenie)
        :param delta: odch. std. losowych fluktuacji wokół c (0 = brak szumu)
        """
        self.alpha: np.ndarray = np.array(alpha_init, dtype=float)
        self.c: np.ndarray = np.array(c, dtype=float)
        self.delta: float = float(delta)

    def update(self) -> None:
        """alpha(t) = alpha(t-1) + N(c, delta^2 * I)"""
        if self.delta > 0:
            shift = np.random.normal(loc=self.c, scale=self.delta, size=len(self.alpha))
        else:
            shift = self.c.copy()
        self.alpha = self.alpha + shift

    def get_optimal_phenotype(self) -> np.ndarray:
        return self.alpha.copy()


# Alias dla kompatybilności wstecznej
Environment = LinearShiftEnvironment


class EnvironmentWithRandomChanges(EnvironmentDynamics):
    """
    Scenario with random rapid changes in drift direction.
    Shifts come from symetric Pareto distribution which is long-tailed.
    It supports drifts across only some dimensions. It is controlled via parameter `c`.
    """

    def __init__(self, alpha_init: np.ndarray, c: np.ndarray, a: float) -> None:
        """
        :param alpha_init: starting optimal phenotype
        :param c: weights for each direction. `0` at position `i` means no shift across this dimension.
        :param a: parameter for Pareto distribution.
        """
        self.alpha: np.ndarray = alpha_init
        self.c: np.ndarray = c
        self.a = a

    def update(self) -> None:
        r = np.random.pareto(self.a, self.alpha.size) * np.random.choice([-1, 1], size=self.alpha.size)
        self.alpha = self.alpha + self.c * r

    def get_optimal_phenotype(self) -> np.ndarray:
        return self.alpha.copy()
