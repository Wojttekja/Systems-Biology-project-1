# population.py

import numpy as np
from .individual import Individual

class Population:
    """
    Klasa przechowuje listę osobników (Individual)
    oraz pomaga w obsłudze różnych operacji na populacji.
    """
    def __init__(
        self, 
        size: int, 
        n_dim: int, 
        init_scale: float = 0.1,
        alpha_init: None|np.ndarray = None,
        weights_init: None|np.ndarray = None,
        ):
        """
        Inicjalizuje populację losowymi fenotypami w n-wymiarach.
        :param size:       liczba osobników (N)
        :param n_dim:      wymiar fenotypu (n)
        :param init_scale: odchylenie std rozkładu startowego wokół optimum.
                           Zalecana reguła: sigma / sqrt(n).
                           Przy zbyt dużej wartości cała populacja ma fitness ≈ 0
                           i wymiera w pierwszym pokoleniu.
        :param alpha_init: centrum inicjalizacji – powinno być równe alpha0
                           ze środowiska. None → inicjalizacja wokół [0,...,0],
                           co powoduje wymarcie gdy alpha0 ≠ 0.
        """
        center = (np.array(alpha_init, dtype=float)
                  if alpha_init is not None else np.zeros(n_dim))
        
        self.individuals: list[Individual] = []

        if weights_init is None:
            weights_init = np.ones(n_dim, dtype=float)
        
        for _ in range(size):
            phenotype = np.random.normal(loc=center, scale=init_scale, size=n_dim)
            self.individuals.append(Individual(phenotype, weights_init))

    def get_individuals(self) -> list[Individual]:
        return self.individuals

    def set_individuals(self, new_individuals: list[Individual]):
        self.individuals = new_individuals

    def __len__(self) -> int:
        return len(self.individuals)
