# Deep BSDE Pricing

A numerical study of neural BSDE methods for multi-asset derivative pricing and risk. The planned comparison covers value-gradient and terminal-loss formulations, independent price and delta references, and learned martingale control variates.

The current implementation provides validated financial problems, a single-asset Black-Scholes reference, and exact-terminal Monte Carlo prices and deltas with sampling uncertainty. Environment diagnostics record the numerical runtime. Neural solvers are not yet implemented.

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

The command samples the terminal GBM state directly, applies the existing payoff and reports its discounted sample mean, standard error and approximate 95% confidence interval. The current sampler supports one asset. Its output retains the seed, generator type, number of positive payoffs and comparison with the separate analytical reference.

For 100,000 samples with seed 2026, the base-case estimate is **9.38965721**, with standard error **0.04441069** and approximate interval **[9.30261386, 9.47670057]**. The analytical value is **9.41340338**. These are seeded simulation results; a new seed changes the estimate and interval.

## Monte Carlo delta

```bash
python -m bsde_pricing monte-carlo --quantity delta --case vanilla --paths 100000 --seed 2026 --output scratch/vanilla_delta.json
```

The delta command compares a pathwise estimator with a central spot bump using common random numbers. Both reuse the exact terminal sampler. It reports uncertainty for each estimate and for their paired difference. The default bump is 0.1% of initial spot.

For the same sample count and seed, pathwise delta is **0.59985776** with standard error **0.00184923**. The central-bump estimate is **0.59987416**; analytical delta is **0.59870633**. The methodology separates sampling error, finite-bump bias and rounding effects. Deterministic exercise kinks are rejected as undefined derivatives.
