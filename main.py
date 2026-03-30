# main.py
"""
Główny skrypt symulacji Geometrycznego Modelu Fishera.

Uruchomienie:
    python main.py

Aby zmienić parametry symulacji, edytuj plik config.py.

Aby użyć innej strategii selekcji / reprodukcji / środowiska, zmień
obiekty przekazywane do run_simulation() w funkcji main() poniżej.
Dostępne klasy bazowe do rozszerzeń: strategies.py
"""

import os
import json
import argparse
from typing import Any
import numpy as np

import config
from strategies.environment import LinearShiftEnvironment
from run_elements.population import Population
from strategies.mutation import (
    IsotropicMutation,
    DirectionalMutation,
    WeightedShiftsMutation,
)
from strategies.selection import TwoStageSelection
from strategies.reproduction import AsexualReproduction
from run_visualization.visualization import plot_frame, plot_stats
from stats_tracking.stats import SimulationStats


# ---------------------------------------------------------------------------
# Główna pętla symulacji
# ---------------------------------------------------------------------------


def _value_from_mapping(
    cfg: dict,
    nested_cfg: dict,
    key: str,
    default=None,
    required: bool = False,
) -> Any:
    """Pobiera parametr: najpierw z nested_cfg, potem z cfg, potem default."""
    if key in nested_cfg:
        return nested_cfg[key]
    if key in cfg:
        return cfg[key]
    if required:
        raise ValueError(f"Brak wymaganego parametru '{key}'.")
    return default


def _coerce_drift_vector(c_raw, n: int) -> np.ndarray:
    """Konwertuje c (skalar lub listę) do wektora długości n."""
    c = np.full(n, c_raw) if np.isscalar(c_raw) else np.array(c_raw, dtype=float)
    if c.shape != (n,):
        raise ValueError(
            f"Parametr 'c' musi mieć długość n={n} (otrzymano kształt {c.shape})."
        )
    return c


def _coerce_vector_param(raw_value, n: int, name: str) -> np.ndarray:
    """Konwertuje skalar lub listę na wektor długości n."""
    vec = (
        np.full(n, raw_value, dtype=float)
        if np.isscalar(raw_value)
        else np.array(raw_value, dtype=float)
    )
    if vec.shape != (n,):
        raise ValueError(
            f"Parametr '{name}' musi mieć długość n={n} "
            f"(otrzymano kształt {vec.shape})."
        )
    return vec


def build_environment_from_config(cfg: dict, alpha_init: np.ndarray):
    """
    Buduje strategię środowiska na podstawie configu.

    Obsługiwany format (nowy):
      "environment": {"type": "linear_shift", "params": {...}}

    Kompatybilność wsteczna:
      c, delta na poziomie głównym configu.
    """
    env_cfg = cfg.get("environment", {})
    env_type = env_cfg.get("type", "linear_shift")
    env_params = env_cfg.get("params", {})

    if env_type == "linear_shift":
        n = int(cfg["n"])
        c_raw = _value_from_mapping(cfg, env_params, "c", default=0.01)
        delta = float(_value_from_mapping(cfg, env_params, "delta", default=0.01))
        c = _coerce_drift_vector(c_raw, n)
        return LinearShiftEnvironment(alpha_init.copy(), c, delta)

    raise ValueError(
        f"Nieznana strategia środowiska: '{env_type}'. "
        "Obsługiwane: linear_shift."
    )


def build_mutation_from_config(cfg: dict):
    """
    Buduje strategię mutacji na podstawie configu.

    Obsługiwany format (nowy):
      "mutation_strategy": {
        "type": "isotropic|directional|weighted_shifts|adaptive_directional",
        "params": {...}
      }

    Kompatybilność wsteczna:
      mu, mu_c, xi (+ k, b) na poziomie głównym configu.
    """
    mut_cfg = cfg.get("mutation_strategy", {})
    mut_type = mut_cfg.get("type", "isotropic")
    mut_params = mut_cfg.get("params", {})

    mu = float(_value_from_mapping(cfg, mut_params, "mu", required=True))
    mu_c = float(_value_from_mapping(cfg, mut_params, "mu_c", required=True))
    xi = float(_value_from_mapping(cfg, mut_params, "xi", required=True))

    if mut_type == "isotropic":
        return IsotropicMutation(mu, mu_c, xi)

    if mut_type == "directional":
        k = int(_value_from_mapping(cfg, mut_params, "k", required=True))
        b = float(_value_from_mapping(cfg, mut_params, "b", required=True))
        return DirectionalMutation(mu, mu_c, xi, k=k, b=b)

    if mut_type == "weighted_shifts":
        k = int(_value_from_mapping(cfg, mut_params, "k", required=True))
        b = float(_value_from_mapping(cfg, mut_params, "b", required=True))
        return WeightedShiftsMutation(mu, mu_c, xi, k=k, b=b)

    if mut_type == "adaptive_directional":
        from strategies.mutation import AdaptiveDirectionalMutation
        return AdaptiveDirectionalMutation(mu, mu_c, xi)

    raise ValueError(
        f"Nieznana strategia mutacji: '{mut_type}'. "
        "Obsługiwane: isotropic, directional, weighted_shifts, adaptive_directional."
    )


def _run_from_json_config(config_path: str) -> SimulationStats:
    """Uruchamia pojedynczą symulację na podstawie pliku JSON."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if "n" not in cfg or "N" not in cfg:
        raise ValueError("Config JSON musi zawierać co najmniej pola 'n' i 'N'.")

    n = int(cfg["n"])
    alpha0 = np.zeros(n)

    mut_cfg = cfg.get("mutation_strategy", {})
    mut_params = mut_cfg.get("params", {}) if isinstance(mut_cfg, dict) else {}

    weights_raw = _value_from_mapping(cfg, mut_params, "init_weights", default=None)
    if weights_raw is None:
        weights_raw = _value_from_mapping(cfg, mut_params, "weights", default=1.0)
    lambdas_raw = _value_from_mapping(cfg, mut_params, "lambdas", default=0.5)
    init_weights = _coerce_vector_param(weights_raw, n, "weights")
    lambdas = _coerce_vector_param(lambdas_raw, n, "lambdas")

    env = build_environment_from_config(cfg, alpha_init=alpha0)
    pop = Population(
        size=cfg["N"],
        n_dim=n,
        weights_init=init_weights,
        init_scale=cfg["init_scale"],
        alpha_init=alpha0,
        lambdas_init=lambdas,
    )
    selection = TwoStageSelection(cfg["sigma"], cfg["threshold"], cfg["N"])
    reproduction = AsexualReproduction()
    mutation = build_mutation_from_config(cfg)

    return run_simulation(
        population=pop,
        environment=env,
        selection_strategy=selection,
        reproduction_strategy=reproduction,
        mutation_strategy=mutation,
        max_generations=cfg["max_generations"],
        frames_dir=None,
        verbose=True,
        target_size=cfg["N"],
        sigma=cfg["sigma"],
    )


def run_simulation(
    population: Population,
    environment,
    selection_strategy,
    reproduction_strategy,
    mutation_strategy,
    max_generations: int = config.max_generations,
    frames_dir: str | None = None,
    verbose: bool = True,
    target_size: int | None = None,
    sigma: float | None = None,
) -> SimulationStats:
    """
    Uruchamia pętlę ewolucyjną i zwraca zebrane statystyki.

    Pętla ewolucyjna (4 kroki zgodnie z treścią zadania):
        1. Mutacja
        2. Selekcja
        3. Reprodukcja
        4. Zmiana środowiska

    :param population:            obiekt Population
    :param environment:           obiekt implementujący EnvironmentDynamics
    :param selection_strategy:    obiekt implementujący SelectionStrategy
    :param reproduction_strategy: obiekt implementujący ReproductionStrategy
    :param mutation_strategy:     obiekt implementujący MutationStrategy
    :param max_generations:       liczba pokoleń do zasymulowania
    :param frames_dir:            katalog do zapisu klatek PNG (None = brak)
    :param verbose:               czy drukować postęp co 10 pokoleń
    :param target_size:           docelowy rozmiar populacji (nadpisuje config.N)
    :param sigma:                 parametr selekcji (nadpisuje config.sigma)
    :return:                      obiekt SimulationStats z wynikami
    """
    if target_size is None:
        target_size = config.N
    if sigma is None:
        sigma = config.sigma

    stats = SimulationStats()

    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)

    for generation in range(max_generations):
        alpha = environment.get_optimal_phenotype()

        # Krok 1: Mutacja
        mutation_strategy.mutate(population)

        mutation_strategy.update_alpha(alpha)

        # Krok 2: Selekcja
        survivors = selection_strategy.select(population.get_individuals(), alpha)
        if not survivors:
            if verbose:
                print(f"Populacja wymarła w pokoleniu {generation}.")
            stats.mark_extinct(generation)
            break

        # Krok 3: Reprodukcja
        new_individuals = reproduction_strategy.reproduce(survivors, target_size)
        population.set_individuals(new_individuals)

        # Zbieranie statystyk i zapis klatki (nowa populacja vs aktualne optimum)
        stats.record(
            generation,
            population,
            alpha,
            sigma,
            reproduction_strategy=reproduction_strategy,
        )

        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f"frame_{generation:03d}.png")
            plot_frame(
                population,
                alpha,
                generation,
                stats,
                save_path=frame_path,
                show_plot=False,
                max_generations=max_generations,
                sigma=sigma,
            )

        # Krok 4: Zmiana środowiska
        environment.update()

        if verbose and generation % 10 == 0:
            r = stats.records[-1]
            print(
                f"  Pokolenie {generation:4d} | "
                f"śr. fitness: {r.mean_fitness:.3f} | "
                f"dist. od optimum: {r.distance_from_optimum:.3f} | "
                f"var. fenotyp.: {r.phenotype_variance:.3f}"
            )

    return stats


# ---------------------------------------------------------------------------
# Narzędzie do tworzenia GIF
# ---------------------------------------------------------------------------


def create_gif_from_frames(
    frames_dir: str, gif_filename: str, duration: float = 0.2
) -> None:
    """
    Łączy wszystkie pliki PNG z katalogu frames_dir w animację GIF.
    Wymaga: pip install imageio
    """
    import imageio

    filenames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not filenames:
        print("Brak klatek do złożenia w GIF.")
        return
    with imageio.get_writer(gif_filename, mode="I", duration=duration) as writer:
        for fname in filenames:
            writer.append_data(imageio.imread(os.path.join(frames_dir, fname)))


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia symulację z config.py (domyślnie) albo z pliku JSON."
        )
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=(
            "Ścieżka do pliku eksperymentu .json (zgodnego z run_experiment.py)."
        ),
    )
    args = parser.parse_args()

    if args.config_json:
        print(f"Rozpoczynam symulację z pliku: {args.config_json}\n")
        stats = _run_from_json_config(args.config_json)
        print(f"\n{stats.summary()}")
        return

    # --- Ziarno losowości (config.seed = None → inna symulacja za każdym razem) ---
    if config.seed is not None:
        np.random.seed(config.seed)

    # --- Inicjalizacja komponentów ---
    env = LinearShiftEnvironment(
        alpha_init=config.alpha0,
        c=config.c,
        delta=config.delta,
    )
    pop = Population(
        size=config.N,
        n_dim=config.n,
        init_scale=config.init_scale,
        alpha_init=config.alpha0,  # populacja startuje blisko alpha0, nie wokół zera
        weights_init=config.init_weights,
        lambdas_init=config.lambdas,
    )
    selection = TwoStageSelection(
        sigma=config.sigma,
        threshold=config.threshold,
        N=config.N,
    )
    reproduction = AsexualReproduction()
    mutation = IsotropicMutation(
        mu=config.mu,
        mu_c=config.mu_c,
        xi=config.xi,
    )

    # --- Uruchomienie symulacji ---
    print("Rozpoczynam symulację...\n")
    frames_dir = "frames"
    stats = run_simulation(
        population=pop,
        environment=env,
        selection_strategy=selection,
        reproduction_strategy=reproduction,
        mutation_strategy=mutation,
        frames_dir=frames_dir,
        verbose=True,
    )

    print(f"\n{stats.summary()}")

    # --- GIF ---
    print("\nTworzenie GIF-a...")
    create_gif_from_frames(frames_dir, "pictures/simulation.gif")
    print("GIF zapisany jako simulation.gif")

    # --- Wykres statystyk ---
    plot_stats(stats, save_path="simulation_stats.png", show_plot=False)
    print("Wykres statystyk zapisany jako simulation_stats.png")


if __name__ == "__main__":
    main()
