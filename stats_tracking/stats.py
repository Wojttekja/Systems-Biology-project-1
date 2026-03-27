# stats.py
"""
Moduł zbierania i analizy statystyk symulacji.

SimulationStats zbiera metryki w każdym pokoleniu i udostępnia je
do analizy i wizualizacji wyników (np. heatmapy, wykresy czasowe).

Przykład użycia:
    stats = run_simulation(pop, env, selection, reproduction)
    print(stats.summary())
    plot_stats(stats)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class GenerationRecord:
    """Snapshot statystyk populacji z jednego pokolenia."""
    generation: int
    mean_fitness: float
    mean_phenotype: np.ndarray
    mean_weights: np.ndarray
    phenotype_variance: float     # uśredniona wariancja po wymiarach (miara różnorodności)
    weights_variance: float
    distance_from_optimum: float  # ||mean_phenotype - alpha||
    population_size: int
    # --- Statystyki reprodukcji ---
    n_parents: int        = 0    # ilu osobników miało ≥1 potomka
    median_offspring: float = 0.0  # mediana potomków wśród reprodukujących się
    max_offspring: int    = 0    # maksymalna płodność ("zwycięzca ewolucyjny")
    # --- Pole rozszerzenia – studenci mogą tu dodać własne metryki ---
    # Przykład użycia w własnej symulacji:
    #   stats.records[-1].extra['moja_metryka'] = wartość
    extra: dict           = field(default_factory=dict)


class SimulationStats:
    """
    Zbiera i przechowuje statystyki populacji z każdego pokolenia symulacji.

    Właściwości jako tablice numpy (do łatwego tworzenia wykresów):
        .generations              – numery pokoleń
        .mean_fitnesses           – średnie fitness populacji
        .distances_from_optimum   – odległość centroidu od optimum
        .phenotype_variances      – wariancja fenotypowa (różnorodność)
        .n_parents_series         – liczba osobników z ≥1 potomkiem
        .median_offspring_series  – mediana potomków wśród reprodukujących się
        .max_offspring_series     – maksymalna płodność

    Jak dodać własną statystykę (dla studentów):
    -----------------------------------------------
    Opcja A – pole extra (najprostsza, bez zmiany kodu bazowego):
        W swojej podklasie lub po wywołaniu record() dostęp przez:
            stats.records[-1].extra['moja_metryka'] = wartość
        Odczyt jako tablica:
            np.array([r.extra.get('moja_metryka', np.nan) for r in stats.records])

    Opcja B – podklasa SimulationStats:
        class MyStats(SimulationStats):
            def record(self, generation, population, alpha, sigma,
                       reproduction_strategy=None):
                super().record(generation, population, alpha, sigma,
                               reproduction_strategy)
                # dodaj swoje obliczenia do self.records[-1].extra
                self.records[-1].extra['moja_metryka'] = ...
    """

    def __init__(self) -> None:
        self.records: List[GenerationRecord] = []
        self.extinct_at: Optional[int] = None
        self.alpha_history: List[np.ndarray] = []  # pozycja optimum w każdym pokoleniu

    def record(self, generation: int, population, alpha: np.ndarray,
               sigma: float, reproduction_strategy=None) -> None:
        """Rejestruje stan populacji po kroku reprodukcji, przed zmianą środowiska."""
        from strategies.selection import compute_fitnesses

        individuals = population.get_individuals()
        if not individuals:
            return

        self.alpha_history.append(alpha.copy())

        phenotypes = np.array([ind.get_phenotype() for ind in individuals])
        weights = np.array([ind.weights for ind in individuals])
        fitnesses = compute_fitnesses(individuals, alpha, sigma)

        mean_phenotype = phenotypes.mean(axis=0)
        phenotype_variance = phenotypes.var(axis=0).mean()
        mean_weights = weights.mean(axis=0)
        weights_variance = weights.var(axis=0).mean()
        distance = float(np.linalg.norm(mean_phenotype - alpha))
        mean_fitness = float(fitnesses.mean())

        # Statystyki reprodukcji (jeśli strategia je udostępnia)
        repro = (reproduction_strategy.get_reproduction_stats()
                 if reproduction_strategy is not None else None) or {}

        self.records.append(GenerationRecord(
            generation=generation,
            mean_fitness=mean_fitness,
            mean_phenotype=mean_phenotype,
            mean_weights=mean_weights,
            phenotype_variance=float(phenotype_variance),
            weights_variance=float(weights_variance),
            distance_from_optimum=distance,
            population_size=len(individuals),
            n_parents=repro.get('n_parents', 0),
            median_offspring=repro.get('median_offspring', 0.0),
            max_offspring=repro.get('max_offspring', 0),
        ))

    def mark_extinct(self, generation: int) -> None:
        """Rejestruje wymarcie populacji."""
        self.extinct_at = generation

    # --- Wygodne właściwości jako tablice numpy ---

    @property
    def generations(self) -> np.ndarray:
        return np.array([r.generation for r in self.records])

    @property
    def mean_fitnesses(self) -> np.ndarray:
        return np.array([r.mean_fitness for r in self.records])

    @property
    def distances_from_optimum(self) -> np.ndarray:
        return np.array([r.distance_from_optimum for r in self.records])

    @property
    def phenotype_variances(self) -> np.ndarray:
        return np.array([r.phenotype_variance for r in self.records])

    @property
    def weights_variances(self) -> np.ndarray:
        return np.array([r.weights_variance for r in self.records])

    @property
    def population_sizes(self) -> np.ndarray:
        return np.array([r.population_size for r in self.records])

    @property
    def n_parents_series(self) -> np.ndarray:
        return np.array([r.n_parents for r in self.records])

    @property
    def median_offspring_series(self) -> np.ndarray:
        return np.array([r.median_offspring for r in self.records])

    @property
    def max_offspring_series(self) -> np.ndarray:
        return np.array([r.max_offspring for r in self.records])

    # --- Użytkowe metody ---

    def survived(self) -> bool:
        """Zwraca True jeśli populacja nie wymarła."""
        return self.extinct_at is None

    def final_mean_fitness(self) -> float:
        """Zwraca średnie fitness z ostatniego zarejestrowanego pokolenia."""
        if not self.records:
            return 0.0
        return self.records[-1].mean_fitness

    def summary(self) -> str:
        """Zwraca jednoliniowe podsumowanie symulacji."""
        if not self.records:
            return "Brak danych."
        last = self.records[-1]
        status = (f"Wymarła w pokoleniu {self.extinct_at}"
                  if self.extinct_at is not None else "Przeżyła")
        return (
            f"Pokoleń: {last.generation + 1} | Status: {status}\n"
            f"Ostatnie śr. fitness: {last.mean_fitness:.4f} | "
            f"Odległość od optimum: {last.distance_from_optimum:.4f} | "
            f"Wariancja fenotypowa: {last.phenotype_variance:.4f} | "
            f"Wariancja wag: {last.weights_variance:.4f}"
        )
