# Deep BSDE pricing

European call prices and spot sensitivities under correlated geometric Brownian motion. The study compares a neural value function differentiated in spot with a terminal-loss Deep BSDE solver. It also measures the cost of reusing each solver's frozen Brownian control in Monte Carlo.

The benchmarks include a vanilla call, a five-asset geometric call and arithmetic calls with up to 100 assets. Black–Scholes and the geometric-basket formula provide exact benchmarks. Arithmetic prices and deltas use numerical references with an independent sampling check. All inputs are synthetic.

## Observed results

All ten references qualified and all 78 fits completed. One vanilla fit met both the 0.01 price target and the 0.01 weight-scaled delta target. Six of the 24 ten-asset grid fits passed the price target, and all 24 missed the joint target. The 100-asset price errors ranged from 0.037 to 0.071 in absolute value.

In the fully reconciled estimator cohort, both conventional geometric payoff controls cost less than all eligible learned controls at the 0.025 half-width target. The learned controls missed the 0.01 target at every measured allocation. This cost comparison covers one training seed. The run was interrupted with 31 of 48 estimator repetitions verified. [Recorded observations](experiments/observations.json)

## Run

~~~sh
conda env create -f environment.yml
conda activate bsde-pricing
python -W error::UserWarning -m unittest discover -s tests -v
~~~

| Notebook | Question |
| --- | --- |
| [Classical references](notebooks/01_classical_references.ipynb) | How do Monte Carlo and a finite-difference solver compare with Black–Scholes? |
| [Basket references](notebooks/02_basket_references.ipynb) | How precise are geometric and arithmetic prices and coordinate deltas? |
| [Neural solvers](notebooks/03_neural_solvers.ipynb) | How do price and Greek errors vary by formulation, seed and grid? |
| [Control variates](notebooks/04_control_variates.ipynb) | How much does an adapted control reduce variance, and what does it cost? |

Use the Python kernel from the bsde-pricing environment. Each notebook runs from the repository root or the notebooks directory.

## Reproduce the locked study

[Experiment design](experiments/design.json) fixes the cases, seeds, accuracy targets and resource limits. The suite runs one worker at a time and saves the ledger, worker logs and checkpoints under the run directory. A full run can take several hours. Repeat the same suite command to resume an interrupted run with its original source, environment and specification.

~~~sh
python -m bsde_pricing experiment-design
python -m bsde_pricing final-suite --run-dir scratch/runs/final_suite
python -m experiments.summarize scratch/runs/final_suite/report.json scratch/runs/final_suite/summary.json
~~~

The neural and control notebooks can read this local summary when available. [Methodology](docs/methodology.md)
