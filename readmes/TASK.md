# Assignment: Fisher's Geometric Model (deadline: 30.03.2026, 19:00 CEST)

## Background

The baseline simulation implements the simplest possible version of FGM:
asexual cloning, isotropic mutation, and a linearly drifting optimum. This
is a useful theoretical benchmark, but real populations differ from this
baseline in many biologically important ways.

Your task is to pick **one** such difference, implement it as a new strategy
class, run a controlled experiment comparing it to the baseline, and report
what you find.

---

## Step 1 — Formulate a research question

Choose one of the suggested extensions below (or propose your own, subject
to prior approval). Write a precise, **falsifiable** research question and
state your prediction before running any simulations:

> *"Populations with [new condition] will show [higher / lower / faster /
> slower] [fitness / extinction risk / phenotypic variance / reproductive
> inequality] compared to the baseline because…"*

Explaining *why* you expect this outcome is as important as the prediction
itself.

---

### Option A — Sexual reproduction

**Research question:**
Does recombination accelerate or impede adaptation to a gradually shifting
optimum compared to asexual cloning?

**What to implement — `SexualReproduction(ReproductionStrategy)`:**
When producing each new offspring, randomly draw **two** parents from the
survivors. The offspring inherits each trait independently from one of the
two parents with equal probability (uniform crossover). The rest of the loop
(mutation, selection, environment) stays identical to the baseline.

**Competing biological mechanisms to consider:**
- *For* sexual reproduction: recombination can assemble beneficial mutations
  faster than cloning (Fisher–Muller hypothesis) and restores phenotypic
  variance lost to selection.
- *Against* sexual reproduction: the cost of mixing — a well-adapted
  phenotype near the optimum may be broken up by recombination with a
  mediocre partner; in a small population this can lower mean fitness.

---

### Option B — Catastrophic environment

**Research question:**
Is a population that has adapted to gradual linear drift more or less
resilient to a sudden large displacement of the optimum than a naive
(un-adapted) population?

**What to implement — `ShockEnvironment(EnvironmentDynamics)`:**
The optimum drifts linearly as in the baseline (`c` per generation), but
every `T_shock` generations it additionally jumps by a random displacement
drawn from N(0, σ_shock² I). Add `T_shock` and `sigma_shock` as parameters
in `config.py`.

**Key biological question:** Sustained directional selection reduces
phenotypic variance — the population "specialises" in one direction. Does
this specialisation make it *fragile* when the environment suddenly changes
direction? Compare populations that experienced many generations of drift
(low remaining variance) versus freshly initialised populations (high
variance) when a shock hits.

---

### Option C — Directional mutation bias

**Research question:**
Can a population that tracks the recent direction of environmental change
through a heritable mutation bias adapt faster than one with purely
isotropic mutations?

**What to implement — `DirectionalMutation(MutationStrategy)`:**
In addition to the usual isotropic noise, add a small deterministic bias
**b** · **v̂** to every mutation, where **v̂** is the unit vector of the
estimated optimum drift direction. Estimate **v̂** from the last *k* entries
of `alpha_history` (available via the `SimulationStats` object you pass in,
or by estimating from the config parameter `c` directly). The magnitude *b*
is a new parameter you tune.

**Competing biological mechanisms to consider:**
- *For* directional bias: fewer wasted mutations in unhelpful directions;
  the population tracks the optimum more efficiently.
- *Against* directional bias: if the environment reverses or the drift
  direction changes, the bias becomes a handicap that slows re-adaptation.
  Test this by flipping the sign of `c` mid-run.

---

## Step 2 — Implementation

- Create your new class in a **new file** (e.g. `sexual_reproduction.py`)
- Keep your implementation self-contained; do not modify any existing
  strategy class or the simulation loop
- Any new parameters should live in `config.py` with a brief comment
  explaining their meaning and sensible default value
- Verify that `python main.py` still produces correct output with the
  baseline strategies after your additions

---

## Step 3 — Data collection

Run both conditions (**baseline** and **your extension**) under identical
settings. For each condition:

- At least **20 independent replicates** (`seed=None`, loop over integer
  seeds 0 to 19 so results are reproducible)
- At least **two parameter variants** — for example, slow drift (`c = 0.005`)
  versus fast drift (`c = 0.02`), or with and without environmental noise
  (`delta = 0` vs `delta = 0.02`)
- Save the full `SimulationStats` objects to disk (e.g. using `pickle`) so
  you can re-run the analysis without re-running the simulation

A minimal data collection loop looks like this:

```python
import pickle, config
from main import run_simulation
# ... import your strategies ...

results = {}
for condition, rep_strategy in [('baseline', AsexualReproduction()),
                                 ('sexual',   SexualReproduction())]:
    results[condition] = []
    for seed in range(20):
        np.random.seed(seed)
        pop = Population(config.N, config.n, config.init_scale,
                         alpha_init=config.alpha0)
        env = LinearShiftEnvironment(config.alpha0.copy(),
                                     config.c.copy(), config.delta)
        sel = TwoStageSelection(config.sigma, config.threshold, config.N)
        mut = IsotropicMutation(config.mu, config.mu_c, config.xi)
        stats = run_simulation(pop, env, sel, rep_strategy, mut,
                               max_generations=config.max_generations,
                               frames_dir=None, verbose=False)
        results[condition].append(stats)

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

---

## Step 4 — Analysis and report

Your analysis should address the following questions. Not all of them will
be equally informative for every extension — use your judgement about which
are most relevant to your research question, and explain why.

### Required analyses

1. **Mean fitness trajectory** — plot mean ± standard deviation across
   replicates for both conditions on the same axes. Does one condition
   maintain higher fitness throughout, or only at certain phases?

2. **Lag from the optimum** — the `distance_from_optimum` series measures
   how far behind the population falls. A steadily growing lag predicts
   eventual extinction; a stable lag means the population is tracking.
   Compare across conditions.

3. **Phenotypic variance** — the `phenotype_variances` series is a proxy
   for genetic diversity. Conditions that deplete diversity early become
   more vulnerable later. Is this reflected in the fitness trajectories?

4. **Reproductive inequality** — compare `n_parents`, `median_offspring`,
   and `max_offspring` between conditions. A highly concentrated offspring
   distribution (few parents, high max) signals a fitness bottleneck. Does
   your extension increase or decrease reproductive inequality?

5. **Extinction rate** — what fraction of the 20 replicates went extinct
   (check `stats.extinct_at is not None`)? Is the difference between
   conditions statistically significant?  
   Use Fisher's exact test or a χ² test; a p-value threshold of 0.05 is
   conventional.

6. **Parameter sensitivity** — does the advantage or disadvantage of your
   extension depend on drift speed `c` or selection strength `sigma`?
   Plot the key outcome (e.g. mean fitness at generation 100, extinction
   rate) as a function of the parameter you varied. This point is important, 
   we want to explore grid of parameters, to see how model responds to 
   parameters perturbation.

### Suggested plots

- Time-series plot: mean ± std of fitness for both conditions, two
  parameter variants, in a 2 × 1 or 2 × 2 panel layout
- Bar chart or box plot: extinction rates per condition per parameter value
- Scatter or line plot: mean lag from optimum at the final generation,
  across replicates, for both conditions
- Optional: violin plot of individual fitness at generation 50 and 150
  to show the shape of the fitness distribution, not just its mean

### Report structure

Write a short report (recommended: **4 pages including figures**):

1. **Introduction** (~0.5 p.) — biological motivation; what do you expect
   and why?
2. **Methods** (~1 p.) — the model, your extension (describe it precisely),
   parameter values used, number of replicates, statistical tests
3. **Results** (~2 p.) — figures and their interpretation; state what you
   observe before interpreting it
4. **Discussion** (~0.5 p.) — does the result match your prediction?
   What biological mechanism explains the outcome? What would you test next?

---

## Practical hints

- Run `python main.py` first and watch the GIF before writing any code — it
  will give you intuition for the dynamics
- The `plot_stats(stats)` call at the end of `main.py` shows all six panels
  for a single run; compare these visually before committing to a parameter
  range for the full experiment
- If a condition frequently goes extinct before generation 200, shorten
  `max_generations` or widen `threshold` / reduce `c` — a run where
  everything dies immediately is not informative
- For the parameter sweep, 20 replicates × 2 conditions × 2 parameter
  values = 80 runs of 200 generations each; this takes a few minutes on a
  laptop. Add `verbose=False` to suppress per-generation output.

