# Deep BSDE Pricing

A numerical study of neural BSDE methods for multi-asset derivative pricing and risk. The planned comparison covers value-gradient and terminal-loss formulations, independent price and delta references, and learned martingale control variates.

The implementation includes classical vanilla and basket references, exact GBM simulation, a finite-difference PDE reference, and BSDE derivative/loss diagnostics. A first one-asset value-gradient network is trained on simulated paths, with its initial price, delta and independent path errors reported below.

## First value-gradient experiment

```bash
python -m bsde_pricing train --run-dir scratch/runs/vanilla_fbsnn
```

The default experiment uses CPU float64, two 16-unit tanh layers, 16 time intervals and 3,000 Adam updates with 128 fresh paths per update. A separate fixed stream monitors the loss; fresh post-training paths measure errors against Black-Scholes. The final update is retained without tuning the budget to meet the inspection targets.

| Initial quantity | Learned | Analytical | Signed error | Absolute-error target |
| --- | ---: | ---: | ---: | ---: |
| Price | 9.56229763 | 9.41340338 | +0.14889424 | 0.25, met |
| Delta | 0.51607919 | 0.59870633 | -0.08262714 | 0.02, missed |

These are single-seed development-workspace results. The delta target is missed despite a lower training loss. On 2,048 fresh paths, pre-expiry price RMSE is 0.92230, delta RMSE is 0.11624 and terminal payoff RMSE is 1.75548. The unconstrained network can also violate option bounds in sampled tails. This is an initial experiment, with its errors retained for analysis.

The command saves `model.pt` and `report.json` under the chosen new run directory. Existing runs are protected. Open [Neural solvers](notebooks/03_neural_solvers.ipynb) for saved-run learning curves, price/delta errors, fresh trajectories, conditional state slices and terminal-fit diagnostics. The [training specification](docs/methodology.md#first-value-gradient-training-experiment) records the objective, data streams, checkpoint rule, timing scope and limits of the fixed-start experiment.

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

The command samples the terminal GBM state directly, applies the existing payoff and reports its discounted sample mean, standard error and approximate 95% confidence interval. The sampler supports correlated assets; the price command compares vanilla and geometric cases with their separate analytical references. Its output retains the seed, generator type, number of positive payoffs and comparison with the separate analytical reference.

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

[Basket references](notebooks/02_basket_references.ipynb) begins with these distribution checks, covariance heatmaps, pairwise scatter plots and a correlation sweep. The [methodology](docs/methodology.md#correlated-terminal-gbm) explains the matrix orientation, uncertainty formulas and deterministic limits. The current basket experiments validate simulated states; the geometric-basket price and delta checks follow in the same notebook.

## Geometric-basket reference

```bash
python -m bsde_pricing reference --case geometric --output scratch/geometric_reference.json
python -m bsde_pricing monte-carlo --case geometric --paths 100000 --seed 2026 --output scratch/geometric_mc.json
python -m bsde_pricing monte-carlo --case geometric --quantity delta --coordinate 0 --paths 100000 --seed 2026 --output scratch/geometric_delta_0.json
```

The weighted geometric basket is lognormal under the correlated GBM model. Its effective drift includes the variance correction and generally differs from the risk-free rate. The analytical reference derives the effective parameters and reuses the scalar Black–Scholes implementation. Each asset delta includes the geometric-basket chain rule.

For the asymmetric three-asset case with weights 0.2, 0.3 and 0.5, the call price is **10.80584305** and coordinate deltas are **(0.15794784, 0.18953741, 0.26324641)**. With 100,000 paths and seed 2026, the Monte Carlo price is **10.85409388** with estimated standard error **0.05001722**. The zero-based `--coordinate` selects the starting spot for a pathwise/common-random-number bump comparison. Full formulas, numerical limits and independent checks are in the [methodology](docs/methodology.md#geometric-basket-reference).

The basket notebook now includes these price/delta checks, selected-spot and correlation sensitivities, and paired bump diagnostics. Its geometric contract maturity follows the editable simulation horizon.

## Arithmetic-basket numerical reference

```bash
python -m bsde_pricing basket-reference --case correlated --method antithetic --pairs 1048576 --seed 2027 --output scratch/arithmetic_antithetic.json
python -m bsde_pricing basket-reference --case correlated --method sobol --power 14 --scrambles 16 --seed 2027 --output scratch/arithmetic_sobol.json
```

The reference estimates the asymmetric arithmetic call price and all coordinate deltas. A separate pilot fits geometric-payoff and geometric-delta control coefficients. Antithetic uncertainty uses pair averages; Sobol uncertainty uses independent scramble means and a Student-t interval. Plain and controlled results remain visible, including cases where the control increases measured uncertainty.

For seed 2027, the controlled price estimates are **13.35403144** (antithetic SE **0.00229101**) and **13.35667441** (Sobol SE **0.00028312**). Both confirmation methods meet the notebook's approximate 95% half-width targets of 0.005 price units and 0.0005 per delta. These are numerical estimates with sampling uncertainty. The [methodology](docs/methodology.md#arithmetic-basket-numerical-reference) records the initial under-resolved run, independent confirmation, estimator formulas, bounds and computing limits.

Basket-notebook sections 11-14 retain matched-work comparisons, independent confirmation, sample-allocation/seed variation and complete-call timings. Runtime measurements apply to the execution environment recorded in the notebook.

## PDE and BSDE conventions

```bash
python -m bsde_pricing bsde-diagnostic --case vanilla --paths 8192 --seed 2026 --output scratch/vanilla_bsde.json
```

The pricing BSDE uses independent Brownian coordinates, driver `f(Y) = -rY`, forward drift `+rY`, and control `Z = A.T @ delta`, where `A = diag(spots * volatilities) @ L` and `L @ L.T` is correlation. The [derivation](docs/methodology.md#pricing-pde-and-bsde) connects self-financing replication, the pricing PDE and discounted martingales.

The diagnostic substitutes analytical vanilla deltas into the discounted BSDE on shared paths across nested time grids. GBM states are exact at grid points. With 8,192 paths and seed 2026, residual RMS falls from **2.791483** at eight intervals to **0.512118** at 256. The final residual mean is **-0.00276424**, with standard error **0.00565842**. Classical-notebook sections 11-12 show the control and refinement plots. These quantify stochastic-integral discretisation before neural training.

## Automatic derivatives and backend checks

```bash
python -m bsde_pricing autograd --output scratch/autograd.json
```

The command checks time derivatives, spot deltas and Hessians against scaled polynomials and the analytical vanilla reference. A small two-hidden-layer tanh network then exercises value/gradient/Hessian evaluation and backpropagation to its weights. CPU float64 is the reference; CPU float32 and available MPS float32 use the same weights and inputs. Reports retain errors, tolerances, backend failures, warmup settings and repeated synchronized workload times. No pricing network is trained by this command.

The [derivative specification](docs/methodology.md#automatic-derivatives-and-backend-checks) explains physical-unit scaling, graph ownership, the independent checks and timing scope. MPS availability alone does not establish derivative support or a speed advantage. Use `--backend cpu` for a CPU-only report or `--backend mps` to require an MPS trial. The neural-results notebook begins with the first trained solver.

## Known-solution BSDE loss

```bash
python -m bsde_pricing bsde-loss --case vanilla --output scratch/vanilla_loss.json
python -m bsde_pricing bsde-loss --case geometric --output scratch/geometric_loss.json
```

`LocalBSDELoss` combines discounted one-step consistency with terminal payoff matching. A differentiable analytical oracle checks the same loss that will be used by the value network. The reports separate local and terminal terms, raw residual RMS, path-level sampling uncertainty and deliberately incorrect driver/control conventions.

On 2,048 coupled paths and grids from eight to 64 intervals, the default vanilla local loss falls from **8.01864e-4** to **1.05459e-4**; the geometric loss falls from **9.54428e-4** to **1.23041e-4**. Analytical terminal errors are zero in these runs. The wrong driver is detected by its signed residual bias, even though its squared loss also decreases. A wrong correlation transpose leaves the finest geometric loss approximately **30.9 times** the correct loss. The [loss specification](docs/methodology.md#local-discounted-bsde-loss) defines the scaling, checks and finite-grid interpretation. These are CPU float64 consistency checks before pricing-network training.

## Notebook exploration

Open [Classical references](notebooks/01_classical_references.ipynb), [Basket references](notebooks/02_basket_references.ipynb) or [Neural solvers](notebooks/03_neural_solvers.ipynb) in VS Code with the Microsoft Jupyter extension. Select the `bsde-pricing` Python environment and run all cells. The first setup cell supports a kernel working directory at either the repository root or `notebooks/`. The neural notebook loads a saved run produced by the training command above.

For an existing environment, install and register the notebook kernel:

```bash
python -m pip install "ipykernel==7.4.0"
python -m ipykernel install --user --name bsde-pricing --display-name "Python (bsde-pricing)"
```

Editable inputs control spot, strike, volatility, maturity, rates, dividends and simulation settings. The notebook provides price/Greek curves, Monte Carlo error bars and tables, sample-count/seed studies, paired bump diagnostics, and PDE price/Greek comparisons with grid and boundary studies. Its numerical calculations use the package classes. Retained outputs show the default configuration; rerun all cells after editing inputs.
