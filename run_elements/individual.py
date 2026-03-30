# individual.py

import numpy as np

class Individual:
    """
    Klasa opisująca pojedynczego osobnika.
    Przechowuje wektor fenotypu w n-wymiarowej przestrzeni.
    """
    def __init__(self, phenotype: np.ndarray, weights: np.ndarray, lambdas: np.ndarray):
        self.phenotype: np.ndarray = np.array(phenotype, dtype=float, copy=True)

        # weights for directional mutations based on multiple previous environmental shifts
        self.weights: np.ndarray = np.array(weights, dtype=float, copy=True)

        # adaptive blend coefficients between isotropic and directional components
        self.lambdas: np.ndarray = np.array(lambdas, dtype=float, copy=True)

    def get_phenotype(self) -> np.ndarray:
        return self.phenotype

    def set_phenotype(self, new_phenotype: np.ndarray):
        self.phenotype = new_phenotype
