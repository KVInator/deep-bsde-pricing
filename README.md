# Deep BSDE Pricing

A numerical study of neural BSDE methods for multi-asset derivative pricing and risk. The planned comparison covers value-gradient and terminal-loss formulations, independent price and delta references, and learned martingale control variates.

The current implementation provides validated financial problems, a single-asset Black-Scholes reference, exact-terminal Monte Carlo prices and deltas with sampling uncertainty, a one-dimensional finite-difference PDE reference, and correlated multi-asset terminal sampling with moment diagnostics. Environment diagnostics record the numerical runtime. Neural solvers are not yet implemented.

## Environment

```bash
conda env create -f environment.yml
conda activate bsde-pricing
```

The specification records the direct package versions verified on an Apple silicon Mac with Python 3.12.16. Conda supplies Python and pip; numerical packages use pip wheels. Transitive dependencies are resolved during installation.

## Runtime diagnostics

Run from the repository root:

```bash
python -m bsde_pricing environment --output scratch/environment.json
python -m pip check
```

The command imports the research packages, checks NumPy and PyTorch CPU matrix calculations in double precision, and records package versions, interpreter, architecture, physical memory and MPS availability. MPS availability identifies a usable backend; derivative accuracy and training performance require separate checks.

Generated reports, experiments and checkpoints belong under ignored `scratch/`. Python bytecode caches and local editor settings are also ignored.

## Financial specification

```bash
python -m bsde_pricing describe --case vanilla
python -m unittest discover -s tests -v
```

The base case is a one-year European call with spot and strike 100, annual volatility 20%, a continuously compounded rate of 3% and zero dividends. The model, payoff and initial state have separate validated contracts. Basket weights and correlation follow the same contracts when dimension is increased.

The [methodology](docs/methodology.md) defines the pricing measure, units, payoff formulas and sensitivity notation. The describe command reports inputs and conventions.

## Analytical reference

```bash
python -m bsde_pricing reference --case vanilla --output scratch/vanilla_reference.json
```

For the base case, the call price is **9.41340338**, spot delta is **0.59870633**, and gamma is **0.01933341**. Prices use the spot/strike currency unit; delta is price change per unit of spot, and gamma is delta change per unit of spot. Call and put prices are evaluated separately and checked against parity and numerical integration of the discounted terminal payoff.

The reference handles expiry and zero volatility explicitly. Classical Greeks at deterministic payoff kinks are reported as `null`. The methodology includes the derivation, numerical conventions and validation tolerances.

## Monte Carlo pricing

```bash
python -m bsde_pricing monte-carlo --case vanilla --paths 100000 --seed 2026 --output scratch/vanilla_mc.json
```

The command samples the terminal GBM state directly, applies the existing payoff and reports its discounted sample mean, standard error and approximate 95% confidence interval. The sampler supports correlated assets; this CLI price comparison remains the vanilla case. Its output retains the seed, generator type, number of positive payoffs and comparison with the separate analytical reference.

For 100,000 samples with seed 2026, the base-case estimate is **9.38965721**, with standard error **0.04441069** and approximate interval **[9.30261386, 9.47670057]**. The analytical value is **9.41340338**. These are seeded simulation results; a new seed changes the estimate and interval.

## Monte Carlo delta

```bash
python -m bsde_pricing monte-carlo --quantity delta --case vanilla --paths 100000 --seed 2026 --output scratch/vanilla_delta.json
```

The delta command compares a pathwise estimator with a central spot bump using common random numbers. Both reuse the exact terminal sampler. It reports uncertainty for each estimate and for their paired difference. The default bump is 0.1% of initial spot.

For the same sample count and seed, pathwise delta is **0.59985776** with standard error **0.00184923**. The central-bump estimate is **0.59987416**; analytical delta is **0.59870633**. The methodology separates sampling error, finite-bump bias and rounding effects. Deterministic exercise kinks are rejected as undefined derivatives.

## Finite-difference reference

```bash
python -m bsde_pricing finite-difference --case vanilla --output scratch/vanilla_pde.json
python -m bsde_pricing finite-difference --case vanilla --space-steps 800 --time-steps 800
```

The scalar call solver uses a uniform spot grid, Crank–Nicolson and four backward-Euler half steps across the first two time intervals. Delta and gamma come from the numerical price grid. The analytical reference enters only the comparison report. The default upper spot boundary is 400; both grids have 400 intervals.

For the base case, the PDE price is **9.41095007**, delta **0.59865223**, and gamma **0.01934023**. The price error is **−0.00245331**; doubling both grids to 800 reduces it to **−0.00061312**. The notebook separates spot, time and upper-boundary refinement and shows the gamma oscillations produced by plain Crank–Nicolson on a coarse time grid. Full equations, scope and error checks are in the [methodology](docs/methodology.md#finite-difference-reference).

## Correlated terminal simulation

```bash
python -m bsde_pricing describe --case correlated
python -m bsde_pricing sample --case correlated --paths 100000 --seed 2026 --output scratch/correlated_terminal.json
```

The three-asset example uses unequal spots, volatilities and dividends, with both positive and negative pair correlations. A Cholesky transform produces joint normal shocks for exact GBM terminal transitions. The report compares observed log-return means/covariances and terminal spot means with theoretical values and their sampling standard errors. The scalar transition retains its original seeded stream and calculation order.

[Basket references](notebooks/02_basket_references.ipynb) begins with these distribution checks, covariance heatmaps, pairwise scatter plots and a correlation sweep. The [methodology](docs/methodology.md#correlated-terminal-gbm) explains the matrix orientation, uncertainty formulas and deterministic limits. The current basket experiments validate simulated states; basket-option reference prices and coordinate deltas are subsequent calculations.

## Notebook exploration

Open [Classical references](notebooks/01_classical_references.ipynb) or [Basket references](notebooks/02_basket_references.ipynb) in VS Code with the Microsoft Jupyter extension. Select the `bsde-pricing` Python environment and run all cells. The first setup cell supports a kernel working directory at either the repository root or `notebooks/`.

For an existing environment, install and register the notebook kernel:

```bash
python -m pip install "ipykernel==7.4.0"
python -m ipykernel install --user --name bsde-pricing --display-name "Python (bsde-pricing)"
```

Editable inputs control spot, strike, volatility, maturity, rates, dividends and simulation settings. The notebook provides price/Greek curves, Monte Carlo error bars and tables, sample-count/seed studies, paired bump diagnostics, and PDE price/Greek comparisons with grid and boundary studies. Its numerical calculations use the package classes. Retained outputs show the default configuration; rerun all cells after editing inputs.
