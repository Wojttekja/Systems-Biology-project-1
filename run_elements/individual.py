# individual.py

import numpy as np

class Individual:
    """
    Klasa opisująca pojedynczego osobnika.
    Przechowuje wektor fenotypu w n-wymiarowej przestrzeni.
    """
    def __init__(self, phenotype: np.ndarray):
        self.phenotype: np.ndarray = phenotype

    def get_phenotype(self) -> np.ndarray:
        return self.phenotype

    def set_phenotype(self, new_phenotype: np.ndarray):
        self.phenotype = new_phenotype
