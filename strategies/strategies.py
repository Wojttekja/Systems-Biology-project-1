# strategies.py
"""
Abstrakcyjne klasy bazowe (interfejsy) dla wszystkich rozszerzeń modelu.

Aby zaimplementować własne rozszerzenie:
  1. Dziedzicz po odpowiedniej klasie abstrakcyjnej
  2. Nadpisz metody oznaczone @abstractmethod
  3. Przekaż instancję swojej klasy do run_simulation() w main.py

Przykład (reprodukcja płciowa):
    class SexualReproduction(ReproductionStrategy):
        def reproduce(self, survivors, target_size):
            # ... twoja implementacja ...
            return new_individuals
"""

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Typing Import
# ---------------------------------------------------------------------------

from run_elements.population import Population
from run_elements.individual import Individual

# ---------------------------------------------------------------------------
# Mutacja
# ---------------------------------------------------------------------------

class MutationStrategy(ABC):
    """
    Strategia mutacji: definiuje jak fenotypy osobników zmieniają się
    w każdym pokoleniu przed selekcją.

    Zaimplementowane strategie: IsotropicMutation (mutation.py)
    Możliwe rozszerzenia:
        DirectionalMutation  – mutacje biased w kierunku przesunięcia optimum
                               ("adaptive memory" / uczenie kierunku zmiany)
        AdaptiveMutation     – xi zmienia się w zależności od odległości od optimum
        PleiotropicMutation  – mutacje jednej cechy wpływają na inne (korelacje)
    """

    @abstractmethod
    def mutate(self, population: Population) -> None:
        """
        Mutuje in-place wszystkich osobników w populacji.
        :param population: obiekt Population
        """
        ...


# ---------------------------------------------------------------------------
# Reprodukcja
# ---------------------------------------------------------------------------

class ReproductionStrategy(ABC):
    """
    Strategia reprodukcji: z listy ocalałych osobników tworzy nową populację
    o rozmiarze target_size.

    Zaimplementowane strategie: AsexualReproduction (reproduction.py)
    Możliwe rozszerzenia: SexualReproduction, PoissonReproduction, ...
    """

    @abstractmethod
    def reproduce(self, survivors: list[Individual], target_size: int) -> list[Individual]:
        """
        :param survivors: lista osobników, które przeżyły selekcję
        :param target_size: docelowy rozmiar nowej populacji (N)
        :return: lista target_size osobników następnego pokolenia
        """
        ...

    def get_reproduction_stats(self) -> None|dict[str, float]:
        """
        Opcjonalna metoda zwracająca statystyki z ostatniego wywołania reproduce().
        Domyślnie None – podklasy mogą nadpisać, aby udostępnić np.:
          'n_parents'       – ilu osobników miało ≥1 potomka
          'median_offspring' – mediana liczby potomków wśród reprodukujących się
          'max_offspring'   – maksymalna liczba potomków jednego osobnika
        """
        return None


# ---------------------------------------------------------------------------
# Selekcja
# ---------------------------------------------------------------------------

class SelectionStrategy(ABC):
    """
    Strategia selekcji: z pełnej listy osobników wybiera tych, którzy przeżywają.

    Zaimplementowane strategie:
        ThresholdSelection, ProportionalSelection, TwoStageSelection (selection.py)
    Możliwe rozszerzenia: TournamentSelection, RankSelection, ...
    """

    @abstractmethod
    def select(self, individuals: list[Individual], alpha: np.ndarray) -> list[Individual]:
        """
        :param individuals: lista wszystkich osobników w populacji
        :param alpha: aktualny optymalny fenotyp środowiska
        :return: lista osobników, które przeżyły (pusta lista = wymarcie)
        """
        ...


# ---------------------------------------------------------------------------
# Dynamika środowiska
# ---------------------------------------------------------------------------

class EnvironmentDynamics(ABC):
    """
    Dynamika środowiska: definiuje jak zmienia się optymalny fenotyp α w czasie.

    Zaimplementowane strategie: LinearShiftEnvironment (environment.py)
    Możliwe rozszerzenia:
        MeteoriteEnvironment   – co T pokoleń nagła radykalna zmiana α
        OscillatingEnvironment – α oscyluje sinusoidalnie
        SpatialEnvironment     – optimum zależy od położenia (x, y) osobnika
        MultiOptimumEnvironment – zbiór kilku optimów prowadzący do specjacji
    """

    @abstractmethod
    def update(self) -> None:
        """Aktualizuje stan środowiska o jedno pokolenie."""
        ...

    @abstractmethod
    def get_optimal_phenotype(self) -> np.ndarray:
        """Zwraca aktualny optymalny fenotyp α(t)."""
        ...
