# config.py

import numpy as np

# -------------------
# PARAMETRY POPULACJI
# -------------------
N = 100           # liczba osobników w populacji
n = 4             # wymiar przestrzeni fenotypowej
init_weights = np.ones(n) # Wagi dla mutacji kierunkowych
lambdas = np.full(n, 0.5) # Udział komponentu kierunkowego mutacji na wymiar
lambda_init_scale = 0.1   # Rozrzut początkowych lambd wokół lambdas (None -> U(0, lambdas))

# Rozrzut początkowych fenotypów wokół optimum.
# Zbyt duży → większość osobników ma fitness ≈ 0 i wymiera w pierwszym pokoleniu.
#
# Wyprowadzenie: każda cecha startuje jako p_i ~ N(alpha0_i, init_scale²),
# więc ||p - alpha0||² ~ init_scale² * chi²(n),  E[||p-alpha0||²] = n * init_scale²
#
# Oczekiwane fitness w pokoleniu 0 (całka po rozkładzie chi²):
#
#   E[phi] = (1 + init_scale² / sigma²)^(-n/2)
#
# Stąd wzór na dobór init_scale dla zadanego minimalnego E[phi] = f_min:
#
#   init_scale <= sigma * sqrt( f_min^(-2/n) - 1 )
#
# Reguła praktyczna: init_scale = sigma / sqrt(n)
#   → E[phi] = (1 + 1/n)^(-n/2)  ≈  e^(-1/2) ≈ 0.61  (dla każdego dużego n)
#   → dla n=4: E[phi] = 1.25^(-2) ≈ 0.64  ✓  (populacja startuje zdrowo)
#
# Przykłady przy sigma=0.2, n=4:
#   init_scale = 0.10  →  E[phi] ≈ 0.64   ← obecna wartość, dobry start
#   init_scale = 0.30  →  E[phi] ≈ 0.24   ← populacja ledwo przeżywa selekcję
#   init_scale = 1.00  →  E[phi] ≈ 0.001  ← natychmiastowe wymarcie
init_scale = 0.1   # = sigma / sqrt(n) = 0.2 / 2 = 0.1

# --------------------
# PARAMETRY MUTACJI
# --------------------
mu = 0.1          # prawdopodobieństwo mutacji dla osobnika
mu_c = 0.5        # prawdopodobieństwo mutacji konkretnej cechy, jeśli osobnik mutuje
xi = 0.05         # odchylenie standardowe mutacji
                  # (mniejsze niż w 2D: w wyższych wymiarach duże kroki
                  #  są proporcjonalnie bardziej szkodliwe – tw. Fishera)

# --------------------
# PARAMETRY SELEKCJI
# --------------------
sigma = 0.2       # parametr w funkcji fitness (kontroluje siłę selekcji)
threshold = 0.01  # próg selekcji progowej
                  # (obniżony z 0.1 do 0.01: w 4D maksymalna tolerowana
                  #  odległość od optimum rośnie z 0.43 do 0.61)

# --------------------
# PARAMETRY ŚRODOWISKA
# --------------------
# UWAGA: alpha0 i c są wyprowadzane z n.
# Wystarczy zmienić n powyżej – wektory środowiska dopasują się automatycznie.
alpha0 = np.zeros(n)       # początkowy optymalny fenotyp
c      = np.full(n, 0.01)  # kierunkowa zmiana α na pokolenie ("globalne ocieplenie")
delta  = 0.01              # odchylenie std losowych fluktuacji wokół c
max_generations = 200      # liczba pokoleń do zasymulowania

# ----------------------
# PARAMETRY REPRODUKCJI
# ----------------------
# W wersji bezpłciowej zakładamy klonowanie z uwzględnieniem mutacji.
# Jeśli chcemy modelować płciowo, trzeba dodać odpowiednie parametry.

# --------------------
# REPRODUKOWALNOŚĆ
# --------------------
# seed = None  →  inne wyniki przy każdym uruchomieniu
# seed = int   →  deterministyczne wyniki (do debugowania i raportów)
seed = 42
