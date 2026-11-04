# Deep BSDE Pricing

A numerical study of neural BSDE methods for multi-asset derivative pricing and risk. The planned comparison covers value-gradient and terminal-loss formulations, independent price and delta references, and learned martingale control variates.

The implementation includes classical vanilla and basket references, exact GBM simulation, a finite-difference PDE reference, and BSDE derivative/loss diagnostics. One-asset value-gradient and terminal-loss networks are trained on simulated paths, with independent price and path-error checks. The value-gradient seed study retains its mixed sensitivity results.

## First value-gradient experiment

```bash
python -m bsde_pricing train --run-dir scratch/runs/vanilla_fbsnn
```

The default experiment uses CPU float64, two 16-unit tanh layers, 16 time intervals and 3,000 Adam updates with 128 fresh paths per update. A separate fixed stream monitors the loss; fresh post-training paths measure errors against Black-Scholes. The first recorded run retained the final update without tuning the budget to meet the inspection targets. Current runs also retain validation-selected weights and the final-update evaluation, as described below.

| Initial quantity | Learned | Analytical | Signed error | Absolute-error target |
| --- | ---: | ---: | ---: | ---: |
| Price | 9.56229763 | 9.41340338 | +0.14889424 | 0.25, met |
| Delta | 0.51607919 | 0.59870633 | -0.08262714 | 0.02, missed |

These are the retained first-run results, subsequently reproduced on the ARM64 MacBook. The delta target is missed despite a lower training loss. On 2,048 fresh paths, pre-expiry price RMSE is 0.92230, delta RMSE is 0.11624 and terminal payoff RMSE is 1.75548. The unconstrained network can also violate option bounds in sampled tails. This is an initial experiment, with its errors retained for analysis.

The command now saves selected inference weights in `model.pt`, continuation state in `resume.pt` and both evaluations in `report.json` under the chosen new run directory. The original version 1 checkpoint remains readable. Existing runs are protected. Open [Neural solvers](notebooks/03_neural_solvers.ipynb) for saved-run learning curves, price/delta errors, fresh trajectories, conditional state slices and terminal-fit diagnostics. The [training specification](docs/methodology.md#first-value-gradient-training-experiment) records the objective, data streams, checkpoint rule, timing scope and limits of the fixed-start experiment.

## Reproducibility and loss weighting

```bash
python -m bsde_pricing training-study --run-dir scratch/runs/reproducibility
```

The study fixes seeds 2026, 2027 and 2028 and compares terminal-loss weights 1 and 5 with paired initial weights and path streams. Architecture, time grid and the 3,000-update budget stay fixed. Checkpoint selection minimises the fixed validation objective; fresh analytical errors assess the selected and final models separately. A short continuation check must reproduce the uninterrupted model, Adam state, streams and evaluation exactly before the study runs.

All six development-workspace runs completed. Selected-model delta errors for weight 1 were -0.08272, -0.02765 and +0.00447; weight 5 gave -0.10935, -0.05881 and +0.04488. Stronger terminal weighting improved final-update fresh-path RMSEs but did not consistently improve the initial delta. The [comparison specification and results](docs/methodology.md#training-reproducibility-and-loss-weighting) retain all seeds, checkpoint choices, runtime and limitations. The neural notebook adds tables and three figures without retraining.

Each run directory is new and protected. The study writes `design.json`, incremental `study.json`, a continuation check, and six run directories with inference and continuation checkpoints. Optional `train --patience N` stops after N scheduled validation checks without improvement; the paired study disables early stopping to keep its update budget fixed. Pause/resume commands are in the methodology.

## Terminal-loss Deep BSDE

```bash
python -m bsde_pricing train-terminal --run-dir scratch/runs/vanilla_terminal_bsde
```

The second formulation learns a free initial price and discounted Brownian controls. It propagates values along exact GBM paths and minimises discounted terminal-payoff mismatch. The initial control is free; each later time point has its own two-layer tanh network. The 16-interval, width-16 model has 4,817 parameters and uses the same 3,000-update pilot budget. Training begins with zero initial price/control and uses no analytical labels.

In the first development-workspace run, validation selects update 2,800. Its initial price is **9.40694237** against **9.41340338**, an error of **-0.00646101**. On 2,048 fresh paths, discounted terminal RMSE is **2.49503**, compared with **2.03220** when using analytical controls on the same grid. This is one seed with an encouraging initial-price result and remaining terminal error. It does not establish control accuracy, Greek accuracy or a cost advantage.

The command protects existing directories and saves selected `model.pt`, final `last_model.pt` and `report.json`. These terminal-solver checkpoints support inference; they do not contain optimiser continuation state. The neural notebook adds terminal learning curves, residual distributions and propagated paths. The [terminal-loss specification](docs/methodology.md#terminal-loss-deep-bsde-in-one-asset) derives the discounted recurrence, scaling, stream/selection rules and scope. Control-to-delta recovery and a matched comparison of the two solvers follow separately.

## Saved-model price and risk comparison

```bash
python -m bsde_pricing compare-risk --run-dir scratch/runs/vanilla_risk_comparison
```

The comparison loads the selected seed-2026, weight-one value-gradient checkpoint and the selected terminal-loss checkpoint without retraining. The terminal solver's discounted control is first undiscounted, then delta is recovered by solving the transposed diffusion system. Both models are evaluated on the same 2,048 fresh paths and 16-interval grid.

| Selected model | Initial price error | Initial delta error | Matched-state delta RMSE |
| --- | ---: | ---: | ---: |
| Value-gradient | +0.15375889 | -0.08271617 | 0.11278521 |
| Terminal loss | -0.00646101 | -0.00746586 | 0.08397111 |

The terminal model meets both initial inspection targets in this one-seed comparison. Tail risk remains inaccurate: it violates call delta bounds at 3,136 of 32,768 sampled path/time points, versus 1,541 for the value-gradient model. The value-gradient initial gamma is 0.00578372 against 0.01933341 analytically; finite delta bumps agree with its automatic derivative. Reconstruction residuals are at floating-point scale and do not certify the learned controls. The [common risk evaluation](docs/methodology.md#common-price-and-risk-evaluation) records selection provenance, diffusion conditioning, numerical checks and timing scope. The notebook retains raw state slices and tail errors. These are frozen single-seed models with different parameter counts, not a robust solver or speed ranking.

## Five-asset geometric basket

```bash
python -m bsde_pricing train --case geometric-five
python -m bsde_pricing train-terminal --case geometric-five
python -m bsde_pricing compare-risk \
  --value-run scratch/runs/geometric_five_fbsnn \
  --terminal-run scratch/runs/geometric_five_terminal_bsde \
  --run-dir scratch/runs/geometric_five_risk
```

The basket uses unequal spots, weights, volatilities and dividend yields with mixed-sign correlations. Both fits complete 3,000 updates at seed 2026. The terminal model meets the price and all five initial delta inspection bands. The value-gradient model misses the price band and the third delta band. On 2,048 common paths, terminal delta RMSE is 0.025250 versus 0.027914, while its worst delta error is 0.452072 versus 0.219419. The five coordinate results, bound counts and diffusion diagnostics are recorded in the [basket experiment](docs/methodology.md#five-asset-geometric-basket).

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
