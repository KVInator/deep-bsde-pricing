# Deep BSDE Pricing and Basket Greeks

A numerical study of European basket calls under correlated geometric Brownian
motion. The value-gradient solver learns a differentiable price surface. The
terminal-loss solver learns the initial price and Brownian controls directly.
The study evaluates both against independent prices and spot Greeks, then
measures the cost of reusing their frozen controls at matched precision.

[Technical report](docs/report.md) · [Executed notebook](notebooks/deep_bsde_study.ipynb) ·
[Locked design](bsde_pricing/design.json) · [Complete result data](docs/results.json)

![Price and scaled delta errors across all fits and cost at matched precision](docs/figures/accuracy_and_cost.png)

The left panel shows all 78 fits. The dotted lines mark the point-error
tolerances. Formal acceptance also accounts for uncertainty in the numerical
references. The right panel shows the median total cost of 100 estimates of
the same ten-asset contract at a 0.025 interval half-width on the recorded Mac.

## Results at a glance

| Formulation | Fits completed | Price passes | All-coordinate delta passes | Joint passes | Fits outside a delta bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Value-gradient | 39 | 3 | 0 | 0 | 4 |
| Terminal loss | 39 | 14 | 3 | 1 | 6 |

The only joint pass was terminal loss on the vanilla call with seed 41002.
At 100 assets, all six fits missed the price target. On the ten-asset grid
study, six of 24 fits passed on price and all 24 missed the joint target.
The cheapest learned control cost **68.36 s** for 100 estimates at half-width
0.025. The antithetic geometric-payoff control cost **0.14 s**, including
its coefficient pilot. At 0.01, every learned variant missed the target.

## Contracts and synthetic data

All prices, references and paths are generated under specified model
parameters. The study uses no observed quotes, calibrations or historical
returns. The vanilla and geometric calls give exact prices and Greeks.
The arithmetic ladder tests nonlinear basket payoffs without a scalar
Black–Scholes formula. Keeping the other inputs fixed isolates each
strike, maturity or correlation change within the ten-asset stress set.

| Case | Assets | Initial spot | Strike | Maturity (years) | Rate | Volatility | Dividend yield | Correlation |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| Vanilla | 1 | 100 | 100 | 1 | 3% | 20% | 0% | — |
| Geometric | 5 | See below | 100 | 1 | 3% | See below | See below | Factor correlation |
| Arithmetic baseline | 10, 25, 50, 100 | 100 in every coordinate | 100 | 1 | 3% | 20% in every coordinate | 0% | 0.30 |
| Lower strike | 10 | As baseline | 90 | 1 | 3% | As baseline | As baseline | 0.30 |
| Higher strike | 10 | As baseline | 110 | 1 | 3% | As baseline | As baseline | 0.30 |
| Short maturity | 10 | As baseline | 100 | 0.25 | 3% | As baseline | As baseline | 0.30 |
| High correlation | 10 | As baseline | 100 | 1 | 3% | As baseline | As baseline | 0.80 |

The arithmetic baskets are equally weighted. The geometric case uses
$`s=(90,100,110,95,105)`$, $`w=(0.10,0.15,0.20,0.25,0.30)`$,
$`\sigma=(0.16,0.19,0.23,0.26,0.31)`$ and
$`q=(0.005,0.010,0.015,0.020,0.025)`$. With loadings
$`\ell=(0.65,0.50,-0.35,0.45,0.55)`$, its correlation matrix has
$`\rho_{ii}=1`$ and $`\rho_{ij}=\ell_i\ell_j`$ for $`i\ne j`$.
The unequal inputs and mixed signs make the coordinate Greeks and the
Brownian-to-spot transformation observable in the analytical check.

## Model and reference prices

### Risk-neutral dynamics

Let $`W`$ have independent Brownian components, $`LL^\top=\rho`$ and
$`\Sigma(s)=\mathrm{diag}(s_i\sigma_i)L`$. The drift is
$`b_i(s)=(r-q_i)s_i`$. Rates and yields compound continuously, time is in
years, and payoff and spot use the same price unit.

```math
\begin{aligned}
dS_t^i &= (r-q_i)S_t^i\,dt+(\Sigma(S_t)dW_t)_i, \\
S_{t+h}^i &= S_t^i\exp\!\left[(r-q_i-\tfrac{1}{2}\sigma_i^2)h+\sigma_i(L\Delta W)_i\right], \\
\Delta W &\sim \mathcal{N}(0,hI).
\end{aligned}
```

The basket is $`A(s)=\sum_i w_i s_i`$ or $`G(s)=\prod_i s_i^{w_i}`$.
For $`g(s)=(B(s)-K)^+`$ and remaining time $`\tau=T-t`$,

```math
\begin{aligned}
u(t,s) &= e^{-r\tau}\mathbb{E}^{\mathbb{Q}}[g(S_T)\mid S_t=s], \\
0 &= u_t+b^\top\nabla_s u+\tfrac{1}{2}\mathrm{tr}(\Sigma\Sigma^\top D_s^2u)-ru, \\
u(T,s) &= g(s).
\end{aligned}
```

The forward lognormal step is exact. Time discretization enters the
stochastic integral and the fitted loss. A terminal-payoff reference can
sample $`S_T`$ in one step, while a martingale control evaluates intermediate
states at the left endpoints of the integration grid.

### Vanilla and geometric benchmarks

Black–Scholes supplies the vanilla price, delta and gamma. The
finite-difference comparison solves the same scalar PDE with Crank–Nicolson,
backward-Euler half steps at startup, and the discounted intrinsic
asymptote at the upper boundary. At 800 intervals in each dimension,
the signed errors were $`-6.13\times10^{-4}`$ in price,
$`-1.35\times10^{-5}`$ in delta and $`+1.71\times10^{-6}`$ in gamma
at the initial spot. The executed notebook retains the refinement table.

For the geometric basket, set $`a_i=w_i\sigma_i`$,
$`v=a^\top\rho a`$ and $`m=\sum_i w_i(r-q_i-\sigma_i^2/2)`$.
The conditional log basket return has mean $`m\tau`$ and variance $`v\tau`$.
The scalar Black–Scholes formula at $`G(s)`$ uses volatility $`\sqrt v`$
and effective yield $`r-m-v/2`$. Its spot derivatives give the coordinate
deltas $`\Delta_i=C_G w_iG(s)/s_i`$.

### Arithmetic-basket price and delta references

Exact GBM transitions give $`\partial S_T^i/\partial s_i=S_T^i/s_i`$.
For the nondegenerate calls, the strike event has zero probability.
The integrable pathwise derivative yields

```math
\Delta_i=e^{-r\tau}\mathbb{E}^{\mathbb{Q}}\!\left[
\mathbf{1}_{\{A(S_T)>K\}}\,w_i\frac{S_T^i}{s_i}
\;\middle|\; S_t=s
\right].
```

A geometric payoff and its pathwise delta contributions provide controls
with known expectations. Separate pilot samples fit the coefficients.
The primary reference uses 16 randomized Sobol scrambles with $`2^{16}`$
terminal states per scramble at the first allocation. A separate estimate
uses 131,072 antithetic pairs. The standard-error units are independent
scramble means and independent pair means, respectively.

The approximate pointwise 95% half-width must be at most 0.002 for the
price and every delta divided by its weight. The two estimators must also
agree within four combined standard errors for each quantity. All eight
arithmetic cases qualified at the first allocation. The largest price
half-width was 0.0005613, the largest scaled-delta half-width was 0.0005672,
and the largest discrepancy was 1.95 combined standard errors.

## Neural BSDE formulations

Applying Itô's formula to $`Y_t=u(t,S_t)`$ gives
$`dY_t=rY_t\,dt+Z_t^\top dW_t`$ with $`Z_t=\Sigma(S_t)^\top\nabla_su(t,S_t)`$.
For $`\widetilde{Y}_t=e^{-rt}Y_t`$ and $`\widetilde{Z}_t=e^{-rt}Z_t`$,

```math
\begin{aligned}
d\widetilde{Y}_t &= \widetilde{Z}_t^\top dW_t, \\
\widetilde{Y}_T &= X=e^{-rT}g(S_T).
\end{aligned}
```

The value-gradient solver represents $`u_\theta(t,s)`$ with one smooth
network. Differentiation in physical spot coordinates supplies the
discounted control. The locked objective combines the local martingale
residuals with the terminal residual,

```math
\frac{1}{K^2}\mathbb{E}\!\left[
\begin{aligned}
&\sum_{j=0}^{N-1}
(\widetilde{Y}_{j+1}-\widetilde{Y}_j-\widetilde{Z}_j^\top\Delta W_j)^2 \\
&\quad +(\widetilde{Y}_N-X)^2
\end{aligned}
\right].
```

The terminal-loss solver learns a free initial value $`y`$ and initial
control, with a separate state-only network at each later grid point.
It minimizes $`\mathbb{E}[(y+\sum_j\widehat{Z}_j^\top\Delta W_j-X)^2]/K^2`$.
For square-integrable predictable controls on exact forward paths,
martingale representation and Itô isometry give the population identity

```math
\begin{aligned}
&\mathbb{E}[(y+\sum_j\widehat{Z}_j^\top\Delta W_j-X)^2] \\
&\quad=(y-u(0,S_0))^2+\sum_j\int_{t_j}^{t_{j+1}}\mathbb{E}\|\widehat{Z}_j-\widetilde{Z}_t\|^2\,dt.
\end{aligned}
```

The first term is initial price error and the second is integrated
control error. Spot-risk validation is pointwise. For the terminal
formulation, delta comes from solving
$`\Sigma(s)^\top\Delta=e^{rt}\widehat{Z}_t`$. The inverse diffusion norm
governs amplification of control error. The value-gradient formulation
also permits spot curvature by differentiating the learned value twice.
The arithmetic call satisfies $`0\le\Delta_i\le w_i e^{-q_i\tau}`$.
The geometric upper bound is $`e^{-q_{\mathrm{eff}}\tau}w_iG/s_i`$.
The [technical report](docs/report.md#3-bsde-formulations-and-sources-of-error)
derives the fixed-grid projection and discusses the scope of the
published convergence bounds.

## Locked evaluation

Both architectures use two hidden tanh layers of width 16, CPU float64,
3,000 Adam updates, and batches of 128 paths. The learning rate starts
at 0.001 and drops to 0.0003 after 70% of the updates. Checkpoints use
the lowest scheduled validation objective, evaluated every 100 updates
on 256 reserved paths. Conditional diagnostics use 2,048 separate paths.
The final training seeds are 41001, 41002 and 41003. References, validation,
path diagnostics, coefficient pilots and estimator evaluation use
separate random streams.

Each of ten contracts is fitted on a 16-interval grid with two formulations
and three seeds. The ten-asset base case is also fitted at 10, 20 and 40
intervals, giving 78 fits in total. The precision gates were locked before
the final fits. For signed error $`e`$ and reference half-width $`h`$ in the
same units, $`|e|+h\le0.01`$ is a pass, and $`|e|-h>0.01`$ is a miss.
The remaining overlap is unresolved. A joint pass requires a price
pass and a pass in every $`\Delta_i/w_i`$ coordinate.

### Prices and Greeks across contracts

The table covers every 16-interval case. Price columns show the minimum
and maximum absolute error over three seeds. Delta columns show the
largest absolute scaled coordinate error over those seeds.

| Contract | Reference price | Value-gradient price error | Terminal-loss price error | Value-gradient max delta error / weight | Terminal-loss max delta error / weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla | 9.413403 | 0.0759–0.2292 | 0.002763–0.03763 | 0.0597 | 0.0071 |
| geometric-five | 5.494719 | 0.01939–0.1622 | 0.004056–0.04504 | 0.1868 | 0.0637 |
| arithmetic-10 | 6.406185 | 0.01049–0.122 | 3.923e-05–0.0435 | 0.2860 | 0.1171 |
| arithmetic-25 | 6.132674 | 0.0192–0.07655 | 0.003624–0.03721 | 0.2963 | 0.2292 |
| arithmetic-50 | 6.038341 | 0.0009879–0.1001 | 0.004428–0.04565 | 0.2675 | 0.6969 |
| arithmetic-100 | 5.990219 | 0.04096–0.0705 | 0.0373–0.07105 | 0.5181 | 2.1342 |
| arithmetic-10-strike-90 | 13.422761 | 0.03267–0.1524 | 0.007783–0.03001 | 0.2261 | 0.0915 |
| arithmetic-10-strike-110 | 2.349156 | 0.02248–0.1216 | 0.01078–0.08257 | 0.1885 | 0.1009 |
| arithmetic-10-short | 2.809889 | 0.001875–0.056 | 0.00049–0.01719 | 0.1465 | 0.1548 |
| arithmetic-10-correlation-80 | 8.683692 | 0.08339–0.207 | 0.0162–0.02751 | 0.2614 | 0.3041 |

The one joint pass had signed price error $`-0.002763`$ and delta error
$`+0.004032`$. The 16-interval terminal-loss ten-asset fit for seed 41003
had absolute price error about $`3.9\times10^{-5}`$ and maximum scaled
delta error 0.1171. The three vanilla value-gradient gamma estimates
were 0.00200–0.00468 against the analytical 0.01933. These are separate
price, risk and curvature checks.

### Time grid

The base ten-asset reference is shared across grids. Parameter counts
show the extra capacity that comes with a finer terminal-loss grid.
Each row contains three seeds.

| Intervals | Formulation | Parameters | Median absolute price error | Worst scaled delta error | Price passes | Joint passes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | Value-gradient | 481 | 0.01305 | 0.2534 | 0 | 0 |
| 10 | Terminal loss | 5,573 | 0.01468 | 0.1010 | 0 | 0 |
| 16 | Value-gradient | 481 | 0.02365 | 0.2860 | 0 | 0 |
| 16 | Terminal loss | 9,281 | 0.008786 | 0.1171 | 2 | 0 |
| 20 | Value-gradient | 481 | 0.111 | 0.2933 | 0 | 0 |
| 20 | Terminal loss | 11,753 | 0.008204 | 0.0914 | 2 | 0 |
| 40 | Value-gradient | 481 | 0.02932 | 0.3035 | 1 | 0 |
| 40 | Terminal loss | 24,113 | 0.01505 | 0.1078 | 1 | 0 |

Neither formulation showed monotone improvement in median price error
under the fixed update budget. The time grid changes control projection,
work per update, and the terminal-loss parameter count. The comparison
measures their combined effect.

## Learned controls and matched estimator cost

For a frozen predictable integrand, $`I=\sum_j\widehat{Z}_j^\top\Delta W_j`$
has mean zero. A coefficient $`\beta`$ fitted on independent pilot paths
preserves the mean of $`X-\beta I`$. Its variance is

```math
\begin{aligned}
\mathrm{Var}(X-\beta I) &= \mathrm{Var}(X)-2\beta\mathrm{Cov}(X,I) \\
&\quad+\beta^2\mathrm{Var}(I).
\end{aligned}
```

The optimal scalar for a nondegenerate control is
$`\mathrm{Cov}(X,I)/\mathrm{Var}(I)`$. For antithetic sampling,
the calculation uses pair averages. On the exact five-asset geometric
case, the analytical integrand gave variance reduction of about 11.5×
at 4 intervals and 165.9× at 64. The integral-mean and discrete
Itô-isometry diagnostics were within 1.2 standard errors of zero on
the tested grids. The notebook reports those checks separately from the
arithmetic cost experiment.

The ten-asset benchmark uses the 16-interval checkpoints. Each of three
seed cohorts has 16 independent repetitions. A method qualifies at the
smallest tested allocation where all 16 observed interval half-widths
meet the target. The allocations are 8,192, 32,768 and 131,072 terminal
states. Martingale controls also evaluate the intermediate grid.

```math
C(q)=C_{\mathrm{train}}+C_{\mathrm{load}}+C_{\mathrm{pilot}}+qC_{\mathrm{eval}}.
```

The median costs below are calculated across qualifying cohorts. The
fixed model and coefficient are reused across independent estimates of
the same contract. An em dash indicates an unavailable cost at the
target precision.

| Half-width | Method | Qualified cohorts | Terminal states | 1 query (s) | 16 queries (s) | 100 queries (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.025 | Antithetic geometric payoff | 3/3 | 8,192 | 0.003152 | 0.02414 | 0.1417 |
| 0.025 | Sobol geometric payoff | 3/3 | 8,192 | 0.03194 | 0.4842 | 3.017 |
| 0.025 | Geometric integral | 3/3 | 32,768 | 0.5379 | 8.606 | 53.79 |
| 0.025 | Value-gradient, unit | 3/3 | 32,768, 131,072 | 10.82 | 26.62 | 115.1 |
| 0.025 | Value-gradient, fitted | 3/3 | 32,768 | 10.48 | 26.25 | 114.6 |
| 0.025 | Terminal loss, unit | 3/3 | 32,768 | 16.69 | 24.53 | 68.36 |
| 0.025 | Terminal loss, fitted | 3/3 | 32,768 | 16.82 | 24.66 | 68.49 |
| 0.01 | Antithetic geometric payoff | 3/3 | 32,768 | 0.007705 | 0.09685 | 0.596 |
| 0.01 | Sobol geometric payoff | 3/3 | 8,192, 32,768 | 0.02683 | 0.4024 | 2.506 |
| 0.01 | Geometric integral | 3/3 | 131,072 | 2.169 | 34.7 | 216.9 |
| 0.01 | Value-gradient, unit | 0/3 | — | — | — | — |
| 0.01 | Value-gradient, fitted | 0/3 | — | — | — | — |
| 0.01 | Terminal loss, unit | 0/3 | — | — | — | — |
| 0.01 | Terminal loss, fitted | 0/3 | — | — | — | — |

At 0.025 both conventional geometric-payoff controls had lower
per-query cost in all three cohorts. Charging the upfront fit and pilot
costs leaves no sustained break-even for the learned controls in the
measured workload. At 0.01 the conventional controls and geometric
integral qualified in every cohort, and every learned variant missed.

## What held up

- Exact vanilla and geometric benchmarks agree with the small independent
  numerical checks. The arithmetic reference uncertainty is well below
  the declared fit tolerances.
- All 78 fits and 48 estimator repetitions completed. The retained Mac
  run took 33 minutes 7 seconds of guarded worker time. The Linux run
  selected the same checkpoints and reproduced every accuracy decision
  and matched-precision allocation.
- Conditional mean preservation gives a usable estimator construction
  even when a network has pointwise pricing or Greek errors. The
  analytical geometric experiment verifies the numerical control path
  and demonstrates its variance reduction.

## Where the methods fell short

- Seventeen fits passed on price, and only one met both the price and
  delta targets and the learned vanilla gamma was below
  the exact value for all three seeds.
- The equal-weight dimension ladder changes diversification as well as
  input size. At the common initial spot its relative variance is
  $`\sigma^2[\rho+(1-\rho)/d]`$. The results describe these equicorrelated
  synthetic baskets and finite training budgets.
- The 48 repetitions cover one ten-asset contract with frozen checkpoints.
  Timing and reuse conclusions are conditional on that contract,
  the allocated samples, and the recorded CPU environment. The accuracy evidence covers
  these synthetic inputs and individual contract fits.

## Further research

- Train over a range of initial states and compare objectives that
  explicitly weight initial spot risk. Validate delta and gamma on
  held-out states with independent references and predeclared tolerances.
- Separate grid error from finite-budget training effects by holding
  parameter count or measured optimization work fixed across grids.
  Vary the basket dimension while holding effective factor exposure fixed.
- Benchmark shared controls across a declared contract universe, charging
  training, setup and retraining when inputs change. A cost comparison
  over heterogeneous queries would test a different use case from the
  fixed-contract benchmark here.

## Run the project

From the repository root, create the pinned environment and run the
numerical and regression tests.

~~~sh
conda env create -f environment.yml
conda activate bsde-pricing
python -m unittest discover -s tests
~~~

[The executed notebook](notebooks/deep_bsde_study.ipynb) reads the retained
data and recomputes the small numerical diagnostics. It can run from the
root or from `notebooks/`. To enumerate the locked work, run it, and
produce a local summary.

~~~sh
python -m bsde_pricing experiment-design
python -m bsde_pricing final-suite --run-dir scratch/runs/final_suite
python -m bsde_pricing summarize scratch/runs/final_suite/report.json scratch/runs/final_suite/summary.json
~~~

The suite writes its ledger, logs, and checkpoints under the run
directory. Resume requires the original source, specification and runtime.
The retained result file records the source fingerprint of the
[v1.0.0 benchmark revision](https://github.com/KVInator/deep-bsde-pricing/releases/tag/v1.0.0)
and the raw Mac ledger hash.

| Path | Responsibility |
| --- | --- |
| [bsde_pricing/](bsde_pricing/) | Pricing, simulation, BSDE solvers, locked design and runner |
| [docs/](docs/) | Derivations, retained results and figure |
| [notebooks/](notebooks/) | Executed study using the package and retained results |
| [tests/](tests/) | Numerical identities, input contracts, checkpointing and suite behavior |

## References

- [Han, Jentzen and E (2018), Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568)
- [Raissi (2018), Forward-Backward Stochastic Neural Networks](https://arxiv.org/abs/1804.07010)
- [Han and Long (2020), Convergence of the Deep BSDE method for coupled FBSDEs](https://arxiv.org/abs/1811.01165)
