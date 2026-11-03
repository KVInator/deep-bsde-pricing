# Deep BSDE Pricing

A numerical study of neural BSDE methods for multi-asset derivative pricing and risk. The planned comparison covers value-gradient and terminal-loss formulations, independent price and delta references, and learned martingale control variates.

The current implementation provides environment diagnostics, validated financial problem definitions and a single-asset Black-Scholes reference for European prices, delta and gamma. Neural solvers are not yet implemented.

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
