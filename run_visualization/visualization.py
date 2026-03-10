# visualization.py

import matplotlib.pyplot as plt
import numpy as np


def _draw_phenotype_panel(ax, individuals: list, alpha: np.ndarray,
                          sigma: float, window_size: float,
                          alpha_history=None, trail_len: int = 15):
    """
    Pomocnicza funkcja rysująca panel fenotypowy (wymiary 1–2) na podanym Axes:

      • Aura fitness – wypełnione pola funkcji Gaussa φ(p₁,p₂)
        (ceteris paribus: pozostałe cechy = optimum)
      • Przerywane izolinii oznaczone wartością φ – widać,
        jak szybko maleje szansa przeżycia wraz z oddalaniem od optimum
      • Ślad optimum – ostatnie trail_len pozycji α rysowane jako
        zanikająca linia + kropki (starsze = bardziej przezroczyste)
      • Strzałka prognozowana – kierunek i tempo ruchu optimum
        oszacowane z ostatnich kroków (widać, w którą stronę ucieka)
      • Chmura osobników pokolorowana ich rzeczywistym fitness (n-wymiarowym)
        – czerwony = zagrożony eliminacją, zielony = bezpieczny
      • Złota gwiazda = optimum α (kolor kojarzący się pozytywnie, nie z zagrożeniem)

    Uwaga: izolinii dotyczą wycinku 2D. Przy n>2 rzeczywiste fitness osobnika
    (kolor punktu) jest niższe niż sugeruje izolinia, bo uwzględnia odchylenia
    we wszystkich n wymiarach.

    :param ax:            obiekt Axes (matplotlib)
    :param individuals:   lista osobników
    :param alpha:         aktualny optymalny fenotyp
    :param sigma:         parametr siły selekcji
    :param window_size:   połowa rozmiaru okna wokół optimum
    :param alpha_history: lista np.ndarray – historia pozycji α (z SimulationStats)
    :param trail_len:     ile ostatnich pozycji α rysować jako ślad
    :return:              ScalarMappable (do stworzenia colorbar na zewnątrz)
    """
    # --- 1. Siatka fitness (2D wycinek) ---
    res = 160
    xs = np.linspace(alpha[0] - window_size, alpha[0] + window_size, res)
    ys = np.linspace(alpha[1] - window_size, alpha[1] + window_size, res)
    Xg, Yg = np.meshgrid(xs, ys)
    Z = np.exp(-((Xg - alpha[0])**2 + (Yg - alpha[1])**2) / (2 * sigma**2))

    # --- 2. Aura: wypełnione pola (białe → jasnozielone → ciemnozielone) ---
    fill_levels = [0.0, 0.01, 0.05, 0.20, 0.50, 0.80, 1.001]
    ax.contourf(Xg, Yg, Z, levels=fill_levels, cmap='YlGn',
                alpha=0.38, zorder=1)

    # --- 3. Izolinii z etykietami φ ---
    line_levels = [0.05, 0.25, 0.50, 0.75]
    cs = ax.contour(Xg, Yg, Z, levels=line_levels,
                    colors='darkgreen', linewidths=0.8,
                    linestyles='--', alpha=0.7, zorder=2)
    ax.clabel(cs, fmt={v: f'φ={v:.2f}' for v in line_levels},
              fontsize=7, inline=True)

    # --- 4. Ślad optimum (zanikające poprzednie pozycje α) + strzałka kierunku ---
    if alpha_history is not None and len(alpha_history) > 1:
        hist = np.array(alpha_history)                    # (T, n)
        # ostatnie trail_len pozycji BEZ bieżącej (bieżąca = złota gwiazda)
        past = hist[-min(trail_len + 1, len(hist)):-1, :2]  # (≤trail_len, 2)
        T = len(past)
        if T > 0:
            fracs = np.linspace(0.08, 0.65, T)           # stara=prawie niewidoczna
            # łącząca linia (segmenty, żeby można było różnicować przezroczystość)
            for i in range(T - 1):
                ax.plot(past[i:i+2, 0], past[i:i+2, 1],
                        color='white', lw=1.4, alpha=fracs[i + 1], zorder=6)
            # kropki na każdej poprzedniej pozycji
            for i in range(T):
                ax.scatter(past[i, 0], past[i, 1], color='white', s=14,
                           alpha=fracs[i], edgecolors='#bbbbbb',
                           linewidths=0.4, zorder=7)

        # strzałka prognozy: prędkość z ostatnich min(5, T+1) kroków
        n_v = min(5, len(hist))
        if n_v >= 2:
            velocity_2d = (hist[-1, :2] - hist[-n_v, :2]) / (n_v - 1)  # krok/gen
            speed = np.linalg.norm(velocity_2d)
            if speed > 1e-10:
                # długość strzałki = 5 kroków, max 35% window_size
                n_ahead = min(5.0, 0.35 * window_size / speed)
                tip = alpha[:2] + velocity_2d * n_ahead
                ax.annotate('',
                            xy=(tip[0], tip[1]),
                            xytext=(alpha[0], alpha[1]),
                            arrowprops=dict(arrowstyle='->', color='#ffe066',
                                           lw=2.0, mutation_scale=16),
                            zorder=11)

    # --- 5. Chmura osobników – kolor = n-wymiarowe fitness ---
    phenotypes = np.array([ind.get_phenotype() for ind in individuals])  # (N, n)
    x_pts = phenotypes[:, 0]
    y_pts = phenotypes[:, 1]
    diff = phenotypes - alpha                                             # broadcasting
    full_fitness = np.exp(-np.einsum('ij,ij->i', diff, diff) / (2 * sigma**2))

    sc = ax.scatter(x_pts, y_pts, c=full_fitness, cmap='RdYlGn',
                    vmin=0, vmax=1, s=28, alpha=0.88,
                    edgecolors='none', zorder=5)

    # --- 6. Optimum: złota gwiazda ---
    ax.scatter([alpha[0]], [alpha[1]], color='gold', marker='*',
               s=380, zorder=10, edgecolors='#b8860b', linewidths=1.2,
               label='Optimum α')

    ax.set_xlim(alpha[0] - window_size, alpha[0] + window_size)
    ax.set_ylim(alpha[1] - window_size, alpha[1] + window_size)
    ax.set_xlabel("Cecha 1", fontsize=9)
    ax.set_ylabel("Cecha 2", fontsize=9)
    ax.set_facecolor('#f7f7f7')

    return sc  # do colorbar


def plot_population(population, alpha: np.ndarray, generation: int,
                    save_path: str = None, show_plot: bool = False,
                    window_size: float = 0.75, sigma: float = 0.2,
                    alpha_history=None) -> None:
    """
    Rysuje populację w 2D wraz z optymalnym fenotypem alpha.
    Osie są wyśrodkowane na aktualnym optimum – śledzą przesuwające się środowisko.

    :param population:    obiekt Population
    :param alpha:         aktualny optymalny fenotyp
    :param generation:    numer pokolenia (do tytułu)
    :param save_path:     ścieżka do zapisu PNG (None = brak zapisu)
    :param show_plot:     True = wyświetl interaktywnie
    :param window_size:   połowa rozmiaru okna wokół optimum
    :param sigma:         parametr siły selekcji – wyznacza promień izolini φ
    :param alpha_history: historia pozycji α (z SimulationStats) – do śladu optimum
    """
    individuals = population.get_individuals()
    if not individuals:
        return

    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc = _draw_phenotype_panel(ax, individuals, alpha, sigma, window_size,
                               alpha_history=alpha_history)
    plt.colorbar(sc, ax=ax, label='fitness φ', shrink=0.88)
    ax.set_title(f"Pokolenie: {generation}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=80)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_stats(stats, save_path: str = None, show_plot: bool = True) -> None:
    """
    Rysuje wykresy podsumowujące przebieg symulacji.

    Rząd 1 (zawsze):
      • Średnie fitness populacji w czasie
      • Odległość centroidu populacji od optimum
      • Wariancja fenotypowa (miara różnorodności genetycznej)

    Rząd 2 (jeśli zebrane dane reprodukcji):
      • Liczba "ewolucyjnych zwycięzców" (osobników z ≥1 potomkiem) w czasie
      • Mediana potomków wśród reprodukujących się osobników
      • Maksymalna płodność (najbardziej dopasowany osobnik)

    :param stats:     obiekt SimulationStats
    :param save_path: ścieżka do zapisu PNG (None = brak zapisu)
    :param show_plot: True = wyświetl interaktywnie
    """
    gens = stats.generations
    has_repro = stats.n_parents_series.sum() > 0
    n_rows = 2 if has_repro else 1

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ujednolicź indeksowanie

    # --- Rząd 1: statystyki populacji ---
    axes[0, 0].plot(gens, stats.mean_fitnesses, color='steelblue')
    axes[0, 0].set_title("Średnie fitness w czasie")
    axes[0, 0].set_xlabel("Pokolenie")
    axes[0, 0].set_ylabel("Średnie fitness")
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].plot(gens, stats.distances_from_optimum, color='darkorange')
    axes[0, 1].set_title("Odległość centroidu od optimum")
    axes[0, 1].set_xlabel("Pokolenie")
    axes[0, 1].set_ylabel("||μ_p − α||")

    axes[0, 2].plot(gens, stats.phenotype_variances, color='seagreen')
    axes[0, 2].set_title("Wariancja fenotypowa (różnorodność)")
    axes[0, 2].set_xlabel("Pokolenie")
    axes[0, 2].set_ylabel("Var(p)")

    # --- Rząd 2: statystyki reprodukcji ---
    if has_repro:
        n_surv = stats.population_sizes  # N osobników po selekcji
        n_par  = stats.n_parents_series

        axes[1, 0].plot(gens, n_par, color='mediumpurple', lw=1.5)
        axes[1, 0].fill_between(gens, n_par, alpha=0.15, color='mediumpurple')
        axes[1, 0].plot(gens, n_surv, color='gray', lw=1, linestyle='--',
                        label='wszyscy ocalałi')
        axes[1, 0].set_title('Ewolucyjni „zwycięzcy" (osobnicy z ≥1 potomkiem)')
        axes[1, 0].set_xlabel('Pokolenie')
        axes[1, 0].set_ylabel('Liczba osobników')
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].plot(gens, stats.median_offspring_series,
                        color='mediumvioletred', lw=1.5)
        axes[1, 1].fill_between(gens, stats.median_offspring_series,
                                alpha=0.15, color='mediumvioletred')
        axes[1, 1].set_title('Mediana potomków (wśród reprodukujących się)')
        axes[1, 1].set_xlabel('Pokolenie')
        axes[1, 1].set_ylabel('Potomkowie (mediana)')

        axes[1, 2].plot(gens, stats.max_offspring_series,
                        color='tomato', lw=1.5)
        axes[1, 2].fill_between(gens, stats.max_offspring_series,
                                alpha=0.15, color='tomato')
        axes[1, 2].set_title('Maks. płodność (najlepszy osobnik)')
        axes[1, 2].set_xlabel('Pokolenie')
        axes[1, 2].set_ylabel('Potomkowie (max)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_frame(population, alpha: np.ndarray, generation: int, stats,
              save_path: str = None, show_plot: bool = False,
              window_size: float = 0.75, max_generations: int = 100,
              sigma: float = 0.2) -> None:
    """
    Renderuje jedną klatkę GIF-a jako figurę z trzema subplotami:
      [1] Aura fitness + chmura osobników (wym. 1–2)
      [2] Średnie fitness w czasie (do bieżącego pokolenia)
      [3] Odległość centroidu populacji od optimum w czasie

    Ponieważ stats.record() jest wywoływane przed zapisem klatki,
    każda klatka zawiera historię włącznie z bieżącym pokoleniem.

    :param population:      obiekt Population
    :param alpha:           aktualny optymalny fenotyp
    :param generation:      numer bieżącego pokolenia
    :param stats:           obiekt SimulationStats (zebrany do tej pory)
    :param save_path:       ścieżka do zapisu PNG (None = brak)
    :param show_plot:       True = wyświetl interaktywnie
    :param window_size:     połowa rozmiaru okna wokół optimum w scatter-plotcie
    :param max_generations: całkowita liczba pokoleń – ustala stałą skalę osi X
                            we wszystkich klatkach (bez tego oś skacze w GIF-ie)
    :param sigma:           parametr siły selekcji – wyznacza promień izolini φ
    """
    individuals = population.get_individuals()
    if not individuals:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: aura fitness + chmura osobników (wymiary 1–2) ---
    ax = axes[0]
    sc = _draw_phenotype_panel(ax, individuals, alpha, sigma, window_size,
                               alpha_history=stats.alpha_history)
    plt.colorbar(sc, ax=ax, label='fitness φ', shrink=0.82, pad=0.02)
    # Adnotacja reprodukcji: ilu osobników miało potomstwo w tym pokoleniu
    last = stats.records[-1] if stats.records else None
    repro_note = (
        f"Zwycięzcy: {last.n_parents}/{last.population_size}  "
        f"med. potomków: {last.median_offspring:.1f}"
        if last and last.n_parents > 0 else ""
    )
    ax.set_title(
        f"Populacja (wym. 1–2)  |  N = {len(individuals)}\n"
        f"{repro_note}",
        fontsize=9, linespacing=1.5,
    )
    ax.legend(loc="upper right", fontsize=7)

    gens = stats.generations
    has_data = len(gens) > 0
    x_max = max_generations  # stała skala we wszystkich klatkach GIF-a

    # --- Panel 2: średnie fitness w czasie ---
    ax = axes[1]
    if has_data:
        ax.plot(gens, stats.mean_fitnesses, color='steelblue', lw=1.5)
        ax.fill_between(gens, stats.mean_fitnesses, alpha=0.15, color='steelblue')
        ax.axvline(generation, color='red', linestyle='--', lw=1, alpha=0.6)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_title("Średnie fitness populacji")
    ax.set_xlabel("Pokolenie")
    ax.set_ylabel("Fitness")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: odległość centroidu od optimum ---
    ax = axes[2]
    if has_data:
        ax.plot(gens, stats.distances_from_optimum, color='darkorange', lw=1.5)
        ax.fill_between(gens, stats.distances_from_optimum, alpha=0.15, color='darkorange')
        ax.axvline(generation, color='red', linestyle='--', lw=1, alpha=0.6,
                   label=f"gen {generation}")
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_title("Odległość centroidu od optimum")
    ax.set_xlabel("Pokolenie")
    ax.set_ylabel("||\u03bc_p \u2212 \u03b1||")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Pokolenie {generation:03d}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
