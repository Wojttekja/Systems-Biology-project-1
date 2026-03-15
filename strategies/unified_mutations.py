# unified_mutations.py

from abc import abstractmethod
from .strategies import MutationStrategy
import numpy as np

# ---------------------------------------------------------------------------
# Typing Import
# ---------------------------------------------------------------------------

from run_elements.population import Population

# ---------------------------------------------------------------------------
# Unified Mutation
# ---------------------------------------------------------------------------

class UnifiedMutations(MutationStrategy):
    """
    Unified Mutations Strategy class.

    It Inherits from MutationStrategy class.

    Forces usage of `update_alpha` method.
    """

    @abstractmethod
    def mutate(self, population: Population) -> None:
        """
        Mutates in-place all Individuals in Population.

        :param population: Population Object
        """
        ...


    @abstractmethod
    def update_alpha(self, new_alpha: np.ndarray) -> None:
        """
        Updates list of stored alpha parameters.

        :param new_alpha: new alpha passed to object.
        :type new_alpha: np.ndarray
        """
        ...


