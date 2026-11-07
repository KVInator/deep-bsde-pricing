# Deep BSDE pricing

European call pricing and spot risk under correlated geometric Brownian motion. The study compares a neural value function differentiated in spot with a terminal-loss Deep BSDE solver. It also measures whether either solver's frozen Brownian control reduces the cost of Monte Carlo pricing.

The test cases include a vanilla call, a five-asset geometric call and arithmetic calls with up to 100 assets. Black–Scholes and the geometric-basket formula provide exact benchmarks. Arithmetic prices and deltas use independent numerical references. All model inputs are synthetic.

## Run

```sh
conda env create -f environment.yml
conda activate bsde-pricing
python -W error::UserWarning -m unittest discover -s tests -v
python -m bsde_pricing --help
```

The notebooks use the `bsde-pricing` kernel and work from the repository root or the `notebooks/` directory.

| Notebook | What it measures |
| --- | --- |
| [Classical references](notebooks/01_classical_references.ipynb) | Vanilla price, delta and gamma against Black–Scholes |
| [Basket references](notebooks/02_basket_references.ipynb) | Geometric prices and coordinate deltas, arithmetic reference precision |
| [Neural solvers](notebooks/03_neural_solvers.ipynb) | Price and Greek errors by case and seed, and arithmetic time-grid sensitivity |
| [Control variates](notebooks/04_control_variates.ipynb) | Known-control checks and matched-precision costs for frozen controls |

## Results

All ten reference cases qualified and all 78 fits finished. One vanilla fit met both locked accuracy targets. Six of the 24 ten-asset grid fits met the price target, and all 24 missed the joint price-and-delta target. At 100 assets, absolute price errors ranged from 0.037 to 0.071.

All 48 estimator repetitions completed. At the 0.025 half-width target, both conventional geometric-payoff controls had lower total and per-query costs than the learned controls in all three seed cohorts. At 0.01, both conventional controls qualified in every cohort and every learned variant missed the target. [Locked results](experiments/results.json)

The locked run took 33 minutes on the Apple silicon Mac and about 97 minutes of worker time in the Linux workspace. Both runs selected the same 78 checkpoints, matched the pricing and risk decisions, and selected the same estimator allocations. The checked-in tables use the Mac report.

## Locked experiment

[Design](experiments/design.json) specifies 78 fits across three seeds, ten financial cases and three extra grids for the ten-asset arithmetic call. It fixes the reference budgets, training budget, price and delta tolerances, and independent evaluation streams.

```sh
python -m bsde_pricing experiment-design
python -m bsde_pricing final-suite --run-dir scratch/runs/final_suite
python -m experiments.summarize scratch/runs/final_suite/report.json scratch/runs/final_suite/summary.json
```

The suite runs one worker at a time and saves each task before moving on. Repeat the same command to resume with the original code, environment and specification. The run directory contains the ledger, worker logs, checkpoints and measured resource use. A full run can take several hours. The notebooks use the checked-in results by default and read the local summary when available.

[Methodology](docs/methodology.md) covers the pricing equations, reference construction, error classification and cost model.
