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
        weights_init: np.ndarray,
        init_scale: float = 0.1,
        alpha_init: None|np.ndarray = None,
        lambdas_init: None|np.ndarray = None,
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
        :param lambdas_init: zakres inicjalizacji lambd.
                   None -> U(0, 1) dla każdej cechy i osobnika.
                   skalar lub wektor (n_dim,) -> U(0, lambdas_init).
        """
        center = (np.array(alpha_init, dtype=float)
                  if alpha_init is not None else np.zeros(n_dim))
        
        self.individuals: list[Individual] = []

        if weights_init is None:
            weights_init = np.ones(n_dim, dtype=float)

        if lambdas_init is None:
            lambda_upper = np.ones(n_dim, dtype=float)
        else:
            lambda_upper = (
                np.full(n_dim, float(lambdas_init), dtype=float)
                if np.isscalar(lambdas_init)
                else np.array(lambdas_init, dtype=float)
            )
            if lambda_upper.shape != (n_dim,):
                raise ValueError(
                    f"lambdas_init must be scalar or shape ({n_dim},), got {lambda_upper.shape}"
                )

        if np.any(lambda_upper < 0.0):
            raise ValueError("lambdas_init upper bounds must be non-negative")
        
        for _ in range(size):
            phenotype = np.random.normal(loc=center, scale=init_scale, size=n_dim)
            lambdas = np.random.uniform(low=0.0, high=lambda_upper, size=n_dim)
            self.individuals.append(
                Individual(
                    phenotype,
                    np.array(weights_init, dtype=float, copy=True),
                    lambdas,
                )
            )

    def get_individuals(self) -> list[Individual]:
        return self.individuals

    def set_individuals(self, new_individuals: list[Individual]):
        self.individuals = new_individuals

    def __len__(self) -> int:
        return len(self.individuals)
