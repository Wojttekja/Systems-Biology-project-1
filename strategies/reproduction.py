# reproduction.py

import copy
import numpy as np
from .strategies import ReproductionStrategy


class AsexualReproduction(ReproductionStrategy):
    """
    Reprodukcja bezpłciowa (klonowanie):
    Losowo wybiera rodziców spośród ocalałych (z powtórzeniami) i klonuje ich,
    aby uzyskać dokładnie target_size osobników nowego pokolenia.

    Każdy ocalały ma równe szanse na bycie rodzicem – selekcja już uwzględniła fitness.
    Po każdym wywołaniu reproduce() dostępne są statystyki przez get_reproduction_stats().
    """

    def __init__(self):
        self._last_counts: np.ndarray = np.array([])

    def reproduce(self, survivors: list, target_size: int) -> list:
        if not survivors:
            self._last_counts = np.array([])
            return []
        indices = np.random.randint(0, len(survivors), size=target_size)
        # bincount[i] = liczba potomków osobnika i
        self._last_counts = np.bincount(indices, minlength=len(survivors))
        return [copy.deepcopy(survivors[i]) for i in indices]

    def get_reproduction_stats(self) -> dict:
        """
        Zwraca statystyki z ostatniego reproduce():
          n_parents       – ilu osobników miało ≥1 potomka ("ewolucyjny sukces")
          median_offspring – mediana potomków wśród reprodukujących się osobników
          max_offspring   – maksymalna płodność (najlepiej przystosowany osobnik)
        """
        if len(self._last_counts) == 0:
            return {'n_parents': 0, 'median_offspring': 0.0, 'max_offspring': 0}
        reproducing = self._last_counts[self._last_counts > 0]
        return {
            'n_parents':        int(len(reproducing)),
            'median_offspring': float(np.median(reproducing)) if len(reproducing) else 0.0,
            'max_offspring':    int(self._last_counts.max()),
        }


# Funkcja pomocnicza zachowana dla kompatybilności wstecznej
def asexual_reproduction(survivors: list, N: int) -> list:
    return AsexualReproduction().reproduce(survivors, N)
