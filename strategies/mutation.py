# mutation.py

import numpy as np
from collections import deque
from .unified_mutations import UnifiedMutations


# ---------------------------------------------------------------------------
# Typing Import
# ---------------------------------------------------------------------------

from run_elements.population import Population
from run_elements.individual import Individual

# ---------------------------------------------------------------------------
# Isotropic Mutation Strategy
# ---------------------------------------------------------------------------


class IsotropicMutation(UnifiedMutations):
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
        self.mu: float = mu
        self.mu_c: float = mu_c
        self.xi: float = xi

    def mutate(self, population: Population) -> None:
        """Mutuje in-place wszystkich osobników w populacji."""
        for ind in population.get_individuals():
            self._mutate_individual(ind)

    def update_alpha(self, new_alpha: np.ndarray) -> None:
        """Empty Method"""
        pass

    def _mutate_individual(self, individual: Individual) -> None:
        if np.random.rand() < self.mu:
            phenotype = individual.get_phenotype().copy()
            for i in range(len(phenotype)):
                if np.random.rand() < self.mu_c:
                    phenotype[i] += np.random.normal(0.0, self.xi)
            individual.set_phenotype(phenotype)


# ---------------------------------------------------------------------------
# Directional Mutation Strategy
# ---------------------------------------------------------------------------


class DirectionalMutation(UnifiedMutations):
    """Directional mutation strategy with environment-informed drift.

    This strategy extends isotropic mutation by adding a directional term that
    follows recent changes in the environmental optimum (`alpha`).

    For each individual, with probability `mu`, phenotype components are
    mutated in two parts:
      1. Isotropic part: each trait mutates independently with probability
         `mu_c` by adding Gaussian noise ``N(0, xi^2)``.
      2. Directional part: a shared shift `b * d` is added, where `d` is
         the mean of the last `k` recorded environmental shifts.
    """

    def __init__(self, mu: float, mu_c: float, xi: float, k: int, b: float):
        """Initialize directional mutation parameters.

        :param mu: Probability that an individual mutates.
        :param mu_c: Probability that a single trait mutates isotropically.
        :param xi: Standard deviation of isotropic mutational change per trait.
        :param k: Number of most recent environmental shifts used to estimate
            the directional component.
        :param b: Scaling factor applied to the directional component.
        """
        self.mu: float = mu
        self.mu_c: float = mu_c
        self.xi: float = xi
        self.k: int = k
        self.b: float = b

        self.env_shifts: deque[np.ndarray] = deque(maxlen=k)
        self.previous_alpha: np.ndarray | None = None

    def mutate(self, population: Population) -> None:
        """Mutates in place all individuals in the Population.

        :param population: A population to which the mutations will be applied.
        """
        directional_component = self.calculate_directional_component()
        for ind in population.get_individuals():
            self._mutate_individual(ind, directional_component)

    def update_alpha(self, new_alpha: np.ndarray) -> None:
        """
        Calculates environmental shift and appends it to `self.env_shifts`

        :param new_alpha: new optimal phenotype
        """
        if self.previous_alpha is not None:
            self.env_shifts.append(new_alpha - self.previous_alpha)
        self.previous_alpha = new_alpha

    def calculate_directional_component(self) -> np.ndarray:
        """Returns average of `k` last shifts."""
        no_shifts = min(self.k, len(self.env_shifts))

        # for the first mutation, we don't have any previous shifts recorded
        if no_shifts == 0:
            if self.previous_alpha is None:
                return np.array(0.0)
            return np.zeros_like(self.previous_alpha)

        recent_shifts = list(self.env_shifts)[-no_shifts:]
        return np.mean(np.stack(recent_shifts, axis=0), axis=0)

    def _mutate_individual(self, individual: Individual, directional_component: np.ndarray) -> None:
        """Applies mutation to an individual with probability of `self.mu`. The mutation is a combination of isotropic and directional mutation.

        :param individual: Individual to mutate.
        :param directional_component: directional component of mutation calculated from recent environmental shifts.
        """
        if np.random.rand() < self.mu:
            old_phenotype = individual.get_phenotype().copy()
            if directional_component.shape == ():
                directional_component = np.zeros_like(old_phenotype)
            mask = np.random.random(old_phenotype.size) < self.mu_c
            isotropic_component = np.where(
                mask, np.random.normal(loc=0.0, scale=self.xi, size=old_phenotype.size), np.zeros_like(old_phenotype)
            )

            new_phenotype = old_phenotype + (1 - self.b) * isotropic_component + self.b * directional_component

            individual.set_phenotype(new_phenotype)


class WeightedShiftsMutation(UnifiedMutations):
    def __init__(self, mu: float, mu_c: float, xi: float, k: int, b: float):
        """Initialize WeightedShift mutation parameters.

        :param mu: Probability that an individual mutates.
        :param mu_c: Probability that a single trait mutates isotropically.
        :param xi: Standard deviation of isotropic mutational change per trait.
        :param k: Number of most recent environmental shifts used to estimate
            the directional component.
        :param b: Scaling factor applied to the directional component.
        """
        self.mu: float = mu
        self.mu_c: float = mu_c
        self.xi: float = xi
        self.k: int = k
        self.b: float = b

        self.env_shifts: deque[np.ndarray] = deque(maxlen=k)
        self.previous_alpha: np.ndarray | None = None

    def mutate(self, population: Population) -> None:
        """Mutates in place all individuals in the Population.

        :param population: A population to which the mutations will be applied.
        """
        for ind in population.get_individuals():
            self._mutate_individual(ind)

    def update_alpha(self, new_alpha: np.ndarray) -> None:
        """
        Calculates environmental shift and appends it to `self.env_shifts`

        :param new_alpha: new optimal phenotype
        """
        if self.previous_alpha is not None:
            self.env_shifts.append(new_alpha - self.previous_alpha)
        self.previous_alpha = new_alpha

    def calculate_directional_component(self, individual: Individual) -> np.ndarray:
        num_observations = len(self.env_shifts)
        if num_observations < 1:
            return np.zeros_like(individual.phenotype)
        if num_observations < 2:
            return self.env_shifts[-1]


        t = np.arange(num_observations)
        num_dimensions = self.previous_alpha.shape[0]

        shifts = np.array(self.env_shifts)
        predictions = []

        softmaxed_weights = np.exp(individual.weights[-num_observations:]) / np.sum(np.exp(individual.weights[-num_observations]))

        for d in range(num_dimensions):
            y = shifts[:, d]
            coeffs = np.polyfit(t, y, 1, w=softmaxed_weights)
            p = np.poly1d(coeffs)
            predictions.append(p(num_observations))

        return np.array(predictions)

    def _mutate_individual(self, individual: Individual) -> None:
        """Applies mutation to an individual with probability of `self.mu`. The mutation is a combination of isotropic and directional mutation.

        :param individual: Individual to mutate.
        :param directional_component: directional component of mutation calculated from recent environmental shifts.
        """

        if np.random.rand() < self.mu:
            # mutate phenotype
            directional_component = self.calculate_directional_component(individual)
            old_phenotype = individual.get_phenotype().copy()

            mask = np.random.random(old_phenotype.size) < self.mu_c
            isotropic_component = np.where(
                mask, np.random.normal(loc=0.0, scale=self.xi, size=old_phenotype.size), np.zeros_like(old_phenotype)
            )

            new_phenotype = old_phenotype + (1 - self.b) * isotropic_component + self.b * directional_component

            individual.set_phenotype(new_phenotype)

            # mutate weights
            old_weights = individual.weights.copy()
            new_weights = 0.5 * old_weights + 0.5 * np.random.uniform(0, 1, size=old_weights.size)
            individual.weights = new_weights


class AdaptiveDirectionalMutation(UnifiedMutations):
    def __init__(self, mu: float, mu_c: float, xi: float):
        """Initialize WeightedShift mutation parameters.

        :param mu: Probability that an individual mutates.
        :param mu_c: Probability that a single trait mutates isotropically.
        :param xi: Standard deviation of isotropic mutational change per trait.
        :param k: Number of most recent environmental shifts used to estimate
            the directional component.
        :param b: Scaling factor applied to the directional component.
        """
        self.mu: float = mu
        self.mu_c: float = mu_c
        self.xi: float = xi

        self.previous_shift: np.ndarray | None = None
        self.previous_alpha: np.ndarray | None = None

    def update_alpha(self, new_alpha: np.ndarray) -> None:
        """
        Calculates environmental shift.

        :param new_alpha: new optimal phenotype
        """
        if self.previous_alpha is not None:
            self.env_shift = self.previous_alpha - new_alpha
            # print(self.env_shift, self.previous_alpha)
        self.previous_alpha = new_alpha

    def calculate_directional_component(self) -> np.ndarray:
        """Returns average of `k` last shifts."""

        if self.previous_shift is None:
            return np.zeros_like(self.previous_alpha)

        return self.previous_shift.copy()
    
    def mutate(self, population: Population) -> None:
        """Mutates in place all individuals in the Population.

        :param population: A population to which the mutations will be applied.
        """
        directional_component = self.calculate_directional_component()
        for ind in population.get_individuals():
            self._mutate_individual(ind, directional_component)

    def _mutate_individual(self, individual: Individual, directional_component: np.ndarray) -> None:
        """Applies mutation to an individual with probability of `self.mu`. The mutation is a combination of isotropic and directional mutation.

        :param individual: Individual to mutate.
        :param directional_component: directional component of mutation calculated from recent environmental shifts.
        """
        if np.random.rand() < self.mu:
            old_phenotype = individual.get_phenotype().copy()
            if directional_component.shape == ():
                directional_component = np.zeros_like(old_phenotype)
            mask = np.random.random(old_phenotype.size) < self.mu_c
            isotropic_component = np.where(
                mask, np.random.normal(loc=0.0, scale=self.xi, size=old_phenotype.size), np.zeros_like(old_phenotype)
            )

            new_phenotype = old_phenotype + (1 - individual.lambdas) * isotropic_component + individual.lambdas * directional_component

            individual.set_phenotype(new_phenotype)
        
            # mutate lambdas
            old_lambdas = individual.lambdas.copy()
            
            new_lambdas = np.clip(old_lambdas + np.random.normal(0, 0.1, size=old_lambdas.size), -np.zeros_like(old_lambdas), np.ones_like(old_lambdas))
            # new_lambdas = 0.5 * old_lambdas + 0.5 * np.clip(np.random.normal(0, 1, size=old_lambdas.size), -1*np.ones_like(old_lambdas), np.ones_like(old_lambdas))
            individual.lambdas = new_lambdas

# ---------------------------------------------------------------------------
# Funkcje pomocnicze – zachowane dla kompatybilności wstecznej
# ---------------------------------------------------------------------------


def mutate_individual(individual, mu: float, mu_c: float, xi: float) -> None:
    IsotropicMutation(mu, mu_c, xi)._mutate_individual(individual)


def mutate_population(population, mu: float, mu_c: float, xi: float) -> None:
    IsotropicMutation(mu, mu_c, xi).mutate(population)
