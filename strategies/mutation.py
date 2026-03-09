# mutation.py

import numpy as np
from .strategies import MutationStrategy


class IsotropicMutation(MutationStrategy):
    """
    Izotropowa mutacja punktowa (domyślna, zgodna z treścią zadania):
      - Z prawdopodobieństwem mu osobnik ulega mutacji
      - Każda cecha p_i mutuje niezależnie z prawdopodobieństwem mu_c
      - Zmiana mutacyjna ∆p_i ~ N(0, ξ²) – izotropowa, bez preferencji kierunku

    Parametry przechowywane jako atrybuty obiektu, dzięki czemu:
      - można tworzyć wiele instancji z różnymi parametrami (do parameter sweep)
      - nie trzeba modyfikować config.py między eksperymentami
    """

    def __init__(self, mu: float, mu_c: float, xi: float):
        """
        :param mu:   prawdopodobieństwo mutacji osobnika
        :param mu_c: prawdopodobieństwo mutacji pojedynczej cechy
        :param xi:   odchylenie std zmiany mutacyjnej ∆p_i
        """
        self.mu  = mu
        self.mu_c = mu_c
        self.xi  = xi

    def mutate(self, population) -> None:
        """Mutuje in-place wszystkich osobników w populacji."""
        for ind in population.get_individuals():
            self._mutate_individual(ind)

    def _mutate_individual(self, individual) -> None:
        if np.random.rand() < self.mu:
            phenotype = individual.get_phenotype().copy()
            for i in range(len(phenotype)):
                if np.random.rand() < self.mu_c:
                    phenotype[i] += np.random.normal(0.0, self.xi)
            individual.set_phenotype(phenotype)


# ---------------------------------------------------------------------------
# Funkcje pomocnicze – zachowane dla kompatybilności wstecznej
# ---------------------------------------------------------------------------

def mutate_individual(individual, mu: float, mu_c: float, xi: float) -> None:
    IsotropicMutation(mu, mu_c, xi)._mutate_individual(individual)


def mutate_population(population, mu: float, mu_c: float, xi: float) -> None:
    IsotropicMutation(mu, mu_c, xi).mutate(population)
