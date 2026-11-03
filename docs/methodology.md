# Mathematical specification

## Pricing model and units

Under the specified risk-neutral measure, the constant-parameter model is

$$
dS_t^i=(r-q_i)S_t^i\,dt+\sigma_iS_t^i\sum_{j=1}^{d}L_{ij}\,dW_t^j,
\qquad LL^\top=\rho.
$$

The components of $W$ are independent Brownian motions. Time is measured in years; rates and continuous dividend yields are per year; volatility is per square-root year. A rate of 3% is supplied as 0.03 and a volatility of 20% as 0.20. Brownian increments over an interval of length $h$ have variance $h$.

The risk-neutral drift $r-q_i$ follows the clean replication assumptions. It is not an estimate of the asset's realised expected return. The core assumes frictionless trading, continuous dividends and a common funding/discount rate. Parameters are fixed within each case. Rates may be negative. Volatilities are nonnegative so deterministic reference limits remain expressible; recovering neural deltas through the diffusion requires a separate nondegeneracy check.

The initial state is strictly positive. Correlation must be finite, symmetric, unit-diagonal and positive definite. Symmetry, diagonal and weight-normalisation checks allow an absolute tolerance of $10^{-12}$ for floating-point representation; this is an input-validation tolerance, not a pricing-accuracy target. Invalid matrices and weights are rejected without repair or renormalisation.

## European contracts

For positive weights summing to one, the supported terminal baskets are

$$
A_T=\sum_i w_iS_T^i,
\qquad G_T=\exp\left(\sum_i w_i\log S_T^i\right).
$$

A call pays $g(S_T)=(B_T-K)^+$, where $B$ is the selected basket. A single asset has weight one, so both definitions reduce to the ordinary European call. Strikes are strictly positive and maturities are nonnegative. Terminal states have array shape $(\ldots,d)$; payoff evaluation removes the final asset axis and preserves leading batch axes. A single state of shape $(d,)$ produces a zero-dimensional array.

Payoff evaluation also accepts zero terminal spots to represent the boundary limit needed by numerical references. A geometric basket is zero if any positive-weight component is zero; the implementation evaluates that limit explicitly without taking the logarithm of zero. Negative or non-finite terminal spots are rejected.

Clean valuation targets

$$
u(t,s)=e^{-r(T-t)}\mathbb E^{\mathbb Q}[g(S_T)\mid S_t=s].
$$

The terminal payoff and the pre-expiry price are distinct quantities. Payoff evaluation introduces neither discounting nor an expectation. Numerical price references will evaluate the displayed pricing relation independently.

## Base case

| Parameter | Value |
| --- | --- |
| Dimension | 1 |
| Initial spot | 100 |
| Strike | 100 |
| Maturity | 1 year |
| Continuous rate | 0.03 |
| Continuous dividend yield | 0 |
| Annualised volatility | 0.20 |
| Correlation matrix | $[1]$ |
| Weight | 1 |

The configuration is defined by `vanilla_case()` in `bsde_pricing/problems.py`. It is a controlled synthetic example. At terminal spots 90, 100 and 110, the respective payoffs are 0, 0 and 10 in the same currency unit as spot and strike.

## Value, delta and Brownian control

Use $u$ for the price function and $\Delta=\nabla_su$ for asset deltas. The Brownian-integral coefficient is

$$
Z=\Sigma(s)^\top\Delta,
\qquad \Sigma(s)=\operatorname{diag}(\sigma_i s_i)L.
$$

In one dimension, $Z=\sigma s\Delta$. Thus a learned Brownian control is not itself an asset delta. The full PDE/BSDE derivation and control recovery will be developed with the solvers.

`GBMModel` owns the law's parameters. `EuropeanCall` owns the payoff and expiry. `PricingCase` combines those objects with the initial spots. Separating initial state from model parameters permits spot sensitivities to hold rates, volatilities, dividends, correlation and contract terms fixed.

## Single-asset analytical reference

Let $\tau=T-t$. For constant coefficients and $\sigma>0$, $\tau>0$, Itô's formula for $\log S$ gives

$$
S_T=S_t\exp\left[(r-q-\tfrac12\sigma^2)\tau+\sigma\sqrt\tau\,N\right],
\qquad N\sim\mathcal N(0,1).
$$

The event $S_T>K$ is a normal-tail event with threshold

$$
d_2=\frac{\log(S/K)+(r-q-\tfrac12\sigma^2)\tau}{\sigma\sqrt\tau},
\qquad d_1=d_2+\sigma\sqrt\tau.
$$

Split the discounted call expectation into its stock and strike terms:

$$
C=e^{-r\tau}\mathbb E[S_T\mathbf1_{\{S_T>K\}}]
  -Ke^{-r\tau}\mathbb Q(S_T>K).
$$

The second term uses $\mathbb Q(S_T>K)=\Phi(d_2)$. For the first, completing the square in the normal density uses

$$
e^{a n}\phi(n)=e^{a^2/2}\phi(n-a),\qquad a=\sigma\sqrt\tau,
$$

which shifts the integration threshold by $a$. Thus

$$
C=Se^{-q\tau}\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),
\qquad P=Ke^{-r\tau}\Phi(-d_2)-Se^{-q\tau}\Phi(-d_1).
$$

Here $\Phi$ and $\phi$ are the standard normal CDF and density. $\Phi(d_2)$ is the risk-neutral exercise probability; $\Phi(d_1)$ is not the same probability. The [Black-Scholes chapter of Pricing and Hedging Derivative Securities](https://book.derivative-securities.org/Chapter_BlackScholes.html) provides a complementary derivation through digital claims and change of numeraire.

### Spot sensitivities

The identity $Se^{-q\tau}\phi(d_1)=Ke^{-r\tau}\phi(d_2)$ cancels the normal-density terms when differentiating the price with respect to spot. With all other inputs held fixed,

$$
\Delta_C=e^{-q\tau}\Phi(d_1),\qquad
\Delta_P=-e^{-q\tau}\Phi(-d_1),\qquad
\Gamma_C=\Gamma_P=\frac{e^{-q\tau}\phi(d_1)}{S\sigma\sqrt\tau}.
$$

Delta is measured in units of option value per unit of spot, and gamma in units of delta per unit of spot. For a small spot displacement $h$,

$$
C(S+h)-C(S)\approx\Delta_C h+\tfrac12\Gamma_C h^2.
$$

### Numerical conventions and deterministic limits

`BlackScholesReference` owns one single-asset model and one unit-weight contract; `evaluate(spot, time)` changes the state and valuation time while retaining those parameters. `BlackScholesResult` records call/put values, both deltas and their common gamma. The normal CDF uses $\Phi(x)=\tfrac12\operatorname{erfc}(-x/\sqrt2)$ to preserve the lower tail. Log-moneyness is formed as $\log S-\log K$ to avoid overflowing the ratio $S/K$.

At expiry, prices equal terminal payoffs. At zero volatility, prices equal

$$
C=(Se^{-q\tau}-Ke^{-r\tau})^+,\qquad
P=(Ke^{-r\tau}-Se^{-q\tau})^+.
$$

Away from a deterministic exercise boundary, delta is the corresponding discounted linear-payoff slope and gamma is zero. At the kink, the classical first and second spot derivatives do not exist and are returned as `None` (`null` in JSON). The implementation does not substitute a half-delta convention. Equality is assessed on the computed discounted spot and strike; no price-error tolerance widens the kink region.

Overflow or underflow of discount factors, unrepresentable integrated volatility, non-finite results and negative prices caused by cancellation fail explicitly. Positive-volatility results still have ordinary floating-point rounding and tail underflow; they are not arbitrary-precision values. Negative rates, continuous dividends, zero maturity and zero volatility are supported within the declared contracts.

### Verification and base results

The reference is checked using independently implemented numerical integration of the discounted lognormal payoff, call/put parity, economic price bounds and spot-bump convergence. Numerical integration is independent of the CDF implementation but shares the declared GBM model. A bump check assesses differentiation consistency; it does not independently verify the price model.

Price integration requests absolute and relative tolerances of $10^{-11}$. The tests require its estimated error below $10^{-9}$ and price agreement within $5\times10^{-10}$. These tolerances are for the small analytical verification cases, not the later neural solver's accuracy target.

Central spot differences use $h/S$ equal to 0.01, 0.005, 0.0025 and 0.00125 at spots 80, 100 and 120. At the finest bump, delta and gamma discrepancies must be below $3\times10^{-6}$ and $2\times10^{-7}$ respectively, with an error reduction exceeding a factor of eight from the coarsest bump. Finite differences have competing truncation and rounding errors; these checks do not imply that reducing the bump indefinitely improves accuracy.

The base-case command produces:

| Quantity | Value |
| --- | ---: |
| Call price | 9.413403383853 |
| Put price | 6.457956738704 |
| Call delta | 0.598706325683 |
| Put delta | -0.401293674317 |
| Common gamma | 0.019333405840 |

Reproduction: `python -m bsde_pricing reference --case vanilla`. These are analytical model values, not neural results or observations from a market.

## Exact-terminal Monte Carlo

The terminal distribution derived above permits direct sampling for a European payoff. `GBMModel.sample_terminal(initial_spots, horizon, n_paths, rng)` returns exact joint terminal states of shape $(N,d)$. In one dimension each stochastic sample uses one standard normal draw; the correlated extension below specifies the vector transform. There is no forward time grid or Euler approximation. `MonteCarloPricer` calls that model method and the existing contract payoff; its estimator does not call the analytical pricing engine.

For independent discounted payoff samples

$$
Y_i=e^{-r\tau}g(S_T^{(i)}),\qquad
\widehat C_N=\frac1N\sum_{i=1}^N Y_i,
$$

the price estimate is unbiased under the specified exact model. Its variance is $\operatorname{Var}(Y)/N$. The reported sample variance and estimated standard error are

$$
s_Y^2=\frac{1}{N-1}\sum_{i=1}^N(Y_i-\widehat C_N)^2,
\qquad \widehat{\mathrm{SE}}=\frac{s_Y}{\sqrt N}.
$$

The $N-1$ denominator makes the sample variance unbiased; it does not make the square-root standard deviation exactly unbiased. At least two samples are required. For a sufficiently informative IID sample, a central-limit normal approximation gives the nominal 95% interval

$$
\widehat C_N\pm\Phi^{-1}(0.975)\widehat{\mathrm{SE}}.
$$

The interval concerns repeated-simulation uncertainty under this model. It does not include uncertainty about model choice or calibrated inputs. Discounted call payoffs are not normally distributed, so the interval is an approximation. Its bounds are not clipped to enforce positive prices.

### Randomness and degenerate samples

The NumPy generator is passed explicitly. The CLI constructs it from the supplied seed and reports its bit-generator type. Reproducing a retained run also requires the environment and model configuration; a seed alone is not a guarantee of identical results across library changes or execution schemes. Pricing at expiry or with zero volatility consumes no random draws and returns the deterministic price with zero standard error.

For a stochastic contract, zero observed sample variance does not establish zero true variance. In particular, no positive payoffs in a finite sample can hide a rare exercise event. The report retains `positive_payoff_count` and withholds confidence bounds (`null`) when the observed stochastic standard error is zero. A known deterministic payoff instead has a point interval. Low positive-payoff counts also warrant caution about the normal approximation; an interval being present does not certify its coverage.

### Verification and interpretation

The scalar distribution checks use

$$
\mathbb E[S_T]=S_0e^{(r-q)\tau},\qquad
\operatorname{Var}(S_T)=S_0^2e^{2(r-q)\tau}(e^{\sigma^2\tau}-1),
$$

and the exact mean and variance of log returns. Fixed-seed checks use broad statistical tolerances for sampling variation, verify positive states and seeded repeatability, and test deterministic limits. Price comparisons use the independent Black-Scholes reference, including nonzero dividends and a negative-rate case. Standard-error scaling and repeated-seed dispersion are checked separately. No test requires every nominal 95% interval to contain the reference.

Increasing sample count by four reduces standard error by approximately two, on average. A realised absolute error need not decrease monotonically for a chosen seed. The convergence statement is about estimator variance, not a guarantee for each successive estimate. The [Monte Carlo chapter of Pricing and Hedging Derivative Securities](https://book.derivative-securities.org/Chapter_MonteCarlo.html) develops this pricing and uncertainty framework.

For the base case with 100,000 samples, seed 2026 and PCG64:

| Quantity | Value |
| --- | ---: |
| Estimated call price | 9.389657214926 |
| Estimated standard error | 0.044410690393 |
| Approximate 95% lower bound | 9.302613861228 |
| Approximate 95% upper bound | 9.476700568624 |
| Analytical call price | 9.413403383853 |
| Signed error / standard error | -0.534694883532 |
| Positive payoffs | 52,137 |

Reproduction: `python -m bsde_pricing monte-carlo --case vanilla --paths 100000 --seed 2026`. This is an introductory simulation check; its interval half-width is larger than the later reference-error budget. Stronger variance reduction is introduced before the high-dimensional reference study.

## Independent Monte Carlo delta

For fixed model parameters, write $S_T=S_0R_T$, where $R_T$ is the random GBM growth factor independent of initial spot. Therefore $\partial S_T/\partial S_0=S_T/S_0$. A call payoff has derivative $\mathbf1_{\{S_T>K\}}$ with respect to its terminal argument away from the strike. For positive volatility and time to expiry, the terminal law is continuous and the strike event has probability zero. The call is Lipschitz, and the relevant payoff difference quotients are bounded by the integrable $R_T$; dominated convergence justifies exchanging differentiation and expectation in this case.

Thus the pathwise delta estimator is

$$
\widehat\Delta_{\mathrm{PW}}=\frac1N\sum_iD_i,
\qquad D_i=e^{-r\tau}\mathbf1_{\{S_T^{(i)}>K\}}
\frac{S_T^{(i)}}{S_0}.
$$

Its estimated standard error is $s_D/\sqrt N$. It uses the terminal model states, not the analytical Greek. The model-specific derivative identity is essential; it must be rederived for different dynamics or parameter sensitivities. The [pathwise-Greeks section of Pricing and Hedging Derivative Securities](https://book.derivative-securities.org/Chapter_MonteCarlo.html#monte-carlo-greeks-ii-pathwise-estimates) provides the underlying estimator argument.

### Common-random-number check

A second calculation uses initial spots $S_0+h$ and $S_0-h$, retaining the same growth factors for both valuations. Under GBM, their terminal states are $(1+h/S_0)S_T$ and $(1-h/S_0)S_T$. The paired central-bump contributions are

$$
B_i(h)=e^{-r\tau}
\frac{g((1+h/S_0)S_T^{(i)})-g((1-h/S_0)S_T^{(i)})}{2h}.
$$

The estimate is the sample mean of $B_i(h)$, and its standard error comes from those paired contributions. Using independent random samples for the up/down values would introduce unnecessary difference noise. A finite bump targets the price secant $(C(S_0+h)-C(S_0-h))/(2h)$, rather than the exact derivative. For the smooth positive-volatility price, the central-bump bias is of order $h^2$ locally. Shrinking $h$ also exposes rounding and subtraction error, so an arbitrarily small bump is not an accuracy guarantee.

The pathwise and bump estimates are correlated because they share terminal draws. Consequently their discrepancy is assessed from $D_i-B_i(h)$ and its sample variance:

$$
\widehat{\mathrm{SE}}_{\mathrm{difference}}
=\frac{s_{D-B}}{\sqrt N}.
$$

Adding their separate variances as if independent would omit the covariance. The mean difference also contains the finite-bump bias, so its true expectation need not be zero. `bump_crossing_count` records how many paired terminal states change exercise status under the bump. Those paths explain the substantive difference between the estimators; elsewhere both calculations agree path by path up to floating-point rounding. With very few crossing paths, the normal approximation for the paired discrepancy can be unreliable even when the overall sample count is large.

### Scope and checks

`MonteCarloPricer.estimate_delta()` supports a single asset and unit contract weight, using the existing terminal sampler. One statistics implementation serves price, pathwise delta, central-bump delta and paired differences. The price command retains its previous output fields. Each delta report records sample size, seed, generator, time, relative/absolute bump and sampling uncertainty.

Zero observed stochastic variance continues to withhold confidence bounds. At expiry or zero volatility, an off-boundary pathwise delta is the deterministic discounted payoff slope; at the exercise kink it is undefined and rejected. A finite bump can cross a deterministic kink even when the base point is off-boundary, in which case the secant differs from the derivative. Bumps must be finite, strictly between zero and one as a fraction of spot, and large enough to produce distinct positive starting spots.

Tests compare pathwise delta with the analytical derivative, including dividends and a negative rate. Common-random-number bumps are checked against both separate repricing with matched seeds and the analytical price secant at several bump sizes. Paired discrepancies are compared with the analytical derivative-minus-secant, preserving the finite-bump bias. Deterministic limits, undefined kinks and invalid bump sizes are checked explicitly.

Naively differentiating the pathwise delta sample again does not provide a valid gamma estimator. The exercise indicator changes across a moving boundary; discarding its boundary contribution can give zero sample derivatives while the price has positive curvature. Gamma verification therefore retains the separate analytical reference and, later, controlled numerical checks.

For 100,000 samples, seed 2026 and relative bump 0.001 ($h=0.1$):

| Quantity | Estimate | Estimated standard error |
| --- | ---: | ---: |
| Pathwise delta | 0.599857759975 | 0.001849227795 |
| Central-bump delta | 0.599874155865 | 0.001847454149 |
| Paired difference, pathwise minus bump | -0.000016395890 | 0.000057981025 |
| Analytical delta | 0.598706325683 | Not applicable |

There are 424 bump-crossing paths in this run. Reproduction: `python -m bsde_pricing monte-carlo --quantity delta --case vanilla --paths 100000 --seed 2026`.

## Finite-difference reference

### From risk-neutral dynamics to the PDE

For one asset with constant parameters,

$$
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t,
\qquad V(S,T)=(S-K)^+.
$$

Itô's formula gives

$$
dV=\left(V_t+(r-q)SV_S+\tfrac12\sigma^2S^2V_{SS}\right)dt
+\sigma SV_S\,dW_t.
$$

Under the risk-neutral pricing assumptions, the discounted derivative value is a martingale. Setting the drift of $e^{-rt}V$ to zero gives

$$
V_t+\tfrac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV=0.
$$

The four terms represent calendar-time change, diffusion curvature, risk-neutral spot drift and discounting. With remaining time $\tau=T-t$ and $u(S,\tau)=V(S,T-\tau)$, the sign of the time derivative changes:

$$
u_\tau=\mathcal L u,
\qquad
\mathcal L=\tfrac12\sigma^2S^2\partial_{SS}+(r-q)S\partial_S-r,
\qquad u(S,0)=(S-K)^+.
$$

The numerical calculation therefore starts at the known terminal payoff and marches forward in remaining time to the requested valuation time. It approximates a whole spot slice, rather than following simulated paths. The model and payoff assumptions are identical to the analytical and Monte Carlo references.

### Finite domain and boundary conditions

The spot grid covers $[0,S_{\max}]$. At zero, GBM remains zero and a positive-strike call is worthless, so

$$u(0,\tau)=0.$$

At the upper boundary, the solver imposes the discounted linear asymptote

$$
u(S_{\max},\tau)\approx B(\tau)
=S_{\max}e^{-q\tau}-Ke^{-r\tau}.
$$

This condition is exact only asymptotically. Put–call parity identifies the omitted term as the residual put value at that boundary. A positive boundary gap alone does not establish that $S_{\max}$ is sufficiently far away. The notebook enlarges the domain while holding spot spacing and time resolution fixed. Enlarging the domain at fixed node count would coarsen the mesh and mix boundary and discretisation effects.

Discount factors and boundary values must remain representable. The upper gap must be positive throughout the horizon; its sign is controlled by a monotone exponential ratio, so the endpoint conditions suffice. Invalid boundary inputs raise an error. No analytical price, normal CDF or analytical Greek is used inside the PDE solver.

### Spot differences and matrix coefficients

Let $M$ count spot intervals, $\Delta S=S_{\max}/M$, and $S_i=i\Delta S$. There are $M+1$ price nodes and $M-1$ interior unknowns. Central differences are

$$
u_S(S_i)\approx\frac{u_{i+1}-u_{i-1}}{2\Delta S},
\qquad
u_{SS}(S_i)\approx\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta S^2}.
$$

The interior operator becomes

$$
(\mathcal L_h u)_i=a_i u_{i-1}+b_i u_i+c_i u_{i+1},
$$

with

$$
a_i=\tfrac12\sigma^2 i^2-\tfrac12(r-q)i,
\quad c_i=\tfrac12\sigma^2 i^2+\tfrac12(r-q)i,
\quad b_i=-a_i-c_i-r.
$$

When convection dominates diffusion near zero, a centered off-diagonal coefficient can be negative. On those rows only, the solver uses a forward derivative for positive $r-q$ or a backward derivative for negative $r-q$. Writing $d_i=\tfrac12\sigma^2i^2$ and $m_i=(r-q)i$, the replacement is

$$
a_i=d_i+\max(-m_i,0),\qquad
c_i=d_i+\max(m_i,0),\qquad b_i=-a_i-c_i-r.
$$

These one-sided rows have first-order local spatial accuracy. `upwind_node_count` exposes their use; the default model has none. Nonnegative spatial off-diagonals do not make Crank–Nicolson unconditionally positivity preserving in time. Values and Greeks are retained without clipping so instability remains visible.

### Time stepping and Rannacher startup

Let $\mathbf v$ contain the interior prices, $A$ the tridiagonal interior operator and $\mathbf g(\tau)$ the boundary source. Only the last source entry is nonzero: $g_{M-1}=c_{M-1}B(\tau)$. For a step of length $h$,

$$
(I-\theta hA)\mathbf v^{n+1}
=(I+(1-\theta)hA)\mathbf v^n
+h\left[(1-\theta)\mathbf g(\tau_n)
+\theta\mathbf g(\tau_{n+1})\right].
$$

Crank–Nicolson uses $\theta=1/2$; backward Euler uses $\theta=1$. The default replaces the first **two full time intervals** with **four backward-Euler half steps**, then resumes Crank–Nicolson. `rannacher_steps` counts replaced full intervals. Setting it to zero runs plain Crank–Nicolson for comparison; setting it to the full interval count uses backward-Euler half steps throughout.

The call payoff is continuous but its slope jumps at the strike. Plain Crank–Nicolson can leave high-frequency errors from that kink, particularly in gamma. Implicit startup damps these modes. This is a numerical treatment of terminal regularity, not a change to the payoff or financial model. See Giles and Carter, [Convergence analysis of Crank–Nicolson and Rannacher time-marching](https://people.maths.ox.ac.uk/gilesm/files/giles_carter.pdf), and Pooley, Vetzal and Forsyth, [Convergence remedies for non-smooth payoffs in option pricing](https://cs.uwaterloo.ca/~paforsyt/report.pdf).

Each step solves a tridiagonal system through SciPy's `solve_banded`, using three diagonal rows rather than a dense matrix. With $N$ full time intervals and $R$ startup intervals, there are $N+R$ linear solves. Work is $O(M(N+R))$ and working storage is $O(M)$; the implementation retains only the final spot slice. Rates may be negative. A conservative check requires $1+r\Delta\tau>0$ to retain diagonal dominance of the implicit matrices under negative discount rates; smaller time steps resolve a rejected configuration.

### Price and Greek evaluation

`FiniteDifferencePricer(model, contract).solve()` returns a `FiniteDifferenceResult`. Its read-only price and spot arrays have shape `(M+1,)`; central-difference delta and gamma arrays have shape `(M-1,)` and correspond to `spots[1:-1]`.

At the final time slice,

$$
\Delta_i=\frac{u_{i+1}-u_{i-1}}{2\Delta S},
\qquad
\Gamma_i=\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta S^2}.
$$

`result.evaluate(spot)` linearly interpolates prices and these nodal Greeks separately. Differentiating a piecewise-linear price interpolant would instead give a piecewise-constant delta and zero gamma between knots. The reported Greek interpolation is an approximation to the PDE derivative, not the literal derivative of the reported price interpolant. Evaluation is restricted to the first through last interior nodes; no boundary Greek extrapolation is supplied.

The solver supports a single asset, unit payoff weight, positive volatility and a valuation time strictly before expiry. Zero-volatility and expiry cases remain available through the separate analytical reference; the PDE interface rejects them rather than returning analytical results as numerical solutions.

### Convergence evidence and reproduction

The checks separate the following errors:

| Source | Experiment |
| --- | --- |
| Spot discretisation | Double spot intervals at fixed domain and fine time resolution. |
| Time discretisation | Refine time on one spot grid; compare with a fine-time solution on that same grid. |
| Terminal kink | Compare plain and smoothed Crank–Nicolson, including nodal gamma. |
| Upper boundary | Increase $S_{\max}$ at fixed $\Delta S$ and time resolution. |
| Interpolation | Compare an off-grid valuation with the analytical price and Greeks. |

With $S_{\max}=400$, $M=N=400$, and two Rannacher intervals, the base-case PDE price is **9.410950068925**, delta **0.598652232851**, and gamma **0.019340230898**. Their signed errors against Black–Scholes are approximately **−0.002453314928**, **−0.000054092832**, and **0.000006825058**. These are deterministic numerical errors, not standard errors or confidence intervals. Doubling both grids to 800 gives price **9.412790265519** and price error **−0.000613118334**.

Aligned spatial refinements with 100, 200 and 400 intervals and 2,048 time intervals show approximately factor-four error reductions for price, delta and gamma. Separate time checks compare 8, 16 and 32 time intervals with a 4,096-interval solution on the same 800-interval spot grid. Boundary checks include an intentionally inadequate upper spot of 120, whose base-case price error exceeds 0.1. Other tests cover dividends, negative rates, convection-dominated rows, evaluation time and invalid grids. On an 800-interval spot grid and 800 time intervals, tested prices, deltas and gammas at spots 80, 100 and 120 have absolute budgets of 0.002, 0.0001 and 0.00002 respectively. These budgets describe those checks, not a uniform error bound over all supported inputs.

The spatial accuracy evidence is specific to a smooth positive-time solution, an aligned strike, interior evaluations and a sufficiently distant boundary. Strike alignment, interpolation, locally one-sided rows and unresolved time or boundary error can change observed rates. Long maturities and high volatility require renewed domain and grid checks.

```bash
python -m bsde_pricing finite-difference --case vanilla --output scratch/vanilla_pde.json
python -m bsde_pricing finite-difference --case vanilla --space-steps 800 --time-steps 800
python -m unittest discover -s tests -v
```

The classical-reference notebook retains the plots and tables for these studies and runs them using the same core classes.

## Correlated terminal GBM

### Brownian dependence and exact transitions

For $d$ assets, the constant-parameter risk-neutral model is

$$
\frac{dS_i}{S_i}=(r-q_i)\,dt+\sigma_i\,dW_i,
\qquad d\langle W_i,W_j\rangle_t=\rho_{ij}\,dt.
$$

Correlation specifies the joint shocks; each marginal still follows its own GBM law. Itô's formula applied to $\log S_i$ yields the drift correction $-\sigma_i^2/2$. Over a horizon $\tau$,

$$
S_i(t+\tau)=S_i(t)\exp\left((r-q_i-\tfrac12\sigma_i^2)\tau
+\sigma_i\sqrt\tau\,Y_i\right),\qquad Y\sim N(0,\rho).
$$

The existing model method `GBMModel.sample_terminal()` implements this joint transition. It does not introduce a forward time grid. Outputs have shape `(n_paths, dimension)` with independent rows and dependent asset columns. All calculations use NumPy float64 on CPU. A seed determines the stream in a given numerical environment; cross-platform bitwise identity is not assumed.

### Cholesky orientation and random streams

For the lower Cholesky factor $\rho=LL^\top$ and an independent standard-normal column vector $z$, define $Y=Lz$. Then

$$\operatorname{Cov}(Y)=L I L^\top=\rho.$$

The implementation stores paths as rows, so the transform is `Z @ L.T`, followed by column-wise multiplication by $\sigma_i\sqrt\tau$. Using `Z @ L` instead would target $L^\top L$, which generally has different variances and covariances. NumPy's [Cholesky documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html) defines the lower-triangular factor used here.

The three-asset fixture uses

$$
\rho=\begin{pmatrix}1&0.6&-0.2\\0.6&1&0.3\\-0.2&0.3&1\end{pmatrix},
\qquad
L=\begin{pmatrix}1&0&0\\0.6&0.8&0\\-0.2&0.525&\sqrt{0.684375}\end{pmatrix}.
$$

A fixed-shock test computes these three linear combinations explicitly, independently of the production factorisation. Separate empirical moment checks then assess the generated distribution. Unequal volatilities and a negative pair correlation help expose asset-order and scaling mistakes.

One-dimensional sampling skips the matrix multiplication and retains the original arithmetic order and `(N,1)` normal draw shape. A regression test compares its states exactly with the earlier scalar transition. Zero horizon returns the initial spots exactly and consumes no random draws. An entirely zero-volatility model also consumes none. If only some volatilities are zero, the model draws a full `(N,d)` normal array, correlates it, and applies each coordinate's volatility. The zero-volatility coordinates are deterministic; stochastic neighbours still contribute price uncertainty. `MonteCarloPricer.estimate()` therefore classifies a positive-horizon basket as deterministic only when all coordinate volatilities are zero.

The correlation contract remains symmetric, unit diagonal and positive definite. Negative pair correlations are allowed if the whole matrix satisfies that condition. Singular perfect-correlation cases are rejected; the implementation does not silently repair matrices or choose a different factorisation. Near-singular valid matrices can amplify numerical sensitivity and require their own checks.

### Distribution diagnostics and their error scale

For $X_i=\log(S_i(t+\tau)/S_i(t))$,

$$
\mu_i=(r-q_i-\tfrac12\sigma_i^2)\tau,
\qquad \Sigma_{ij}=\sigma_i\sigma_j\rho_{ij}\tau.
$$

`TerminalSampleDiagnostics.from_sample()` compares theoretical moments with the supplied sample. It validates the state shape, strictly positive spots and at least two paths. Log returns are computed as a difference of logs to avoid overflow in the intermediate price ratio. The sample mean has theoretical standard error

$$\operatorname{SE}(\bar X_i)=\sqrt{\Sigma_{ii}/N}.$$

For the unbiased sample covariance with denominator $N-1$, the Gaussian identity is

$$
\operatorname{Var}(\widehat\Sigma_{ij})
=\frac{\Sigma_{ii}\Sigma_{jj}+\Sigma_{ij}^2}{N-1}.
$$

This follows by writing $(N-1)\widehat\Sigma$ as a sum of $N-1$ independent centered Gaussian outer products and using the fourth-moment relation $\mathbb E[X_i^2X_j^2]=\Sigma_{ii}\Sigma_{jj}+2\Sigma_{ij}^2$. Covariance entries are dependent. Individual standard errors are useful diagnostics, not a simultaneous confidence band for the full matrix. Zero-volatility or zero-horizon entries have zero theoretical sampling error, so a standardised residual for those entries is undefined; the report retains raw moments and standard errors without dividing by zero.

The terminal mean and its standard error follow from the marginal lognormal distribution:

$$
m_i=\mathbb E[S_i(t+\tau)]=S_i(t)e^{(r-q_i)\tau},
\qquad
\operatorname{SE}(\bar S_i)
=m_i\sqrt{\frac{e^{\sigma_i^2\tau}-1}{N}}.
$$

For positive volatilities and horizon, log-return correlation equals $\rho$. Pearson correlation of terminal spot levels generally differs because of exponentiation. Changing correlation leaves theoretical marginal distributions unchanged, while altering joint events and basket distributions. Matched seeds across correlation scenarios induce dependence between numerical estimates; they do not guarantee identical finite-sample marginal means.

### Implemented example and scope

`correlated_case()` defines spots $(80,100,120)$, volatilities $(0.15,0.25,0.35)$, dividend yields $(0,0.01,0.02)$, rate 0.03, maturity one year and the correlation above. Its arithmetic-call contract has strike 100 and weights $(0.2,0.3,0.5)$. Those weights do not enter the state-distribution diagnostics.

At one year, the target log-return mean is $(0.01875,-0.01125,-0.05125)$ and covariance is

$$
\Sigma=\begin{pmatrix}
0.0225&0.0225&-0.0105\\
0.0225&0.0625&0.02625\\
-0.0105&0.02625&0.1225
\end{pmatrix}.
$$

With 100,000 paths and seed 2026, the observed log-return means are approximately $(0.01851113,-0.01013352,-0.04985481)$. Their errors are approximately $(-0.504,1.412,1.261)$ theoretical standard errors. The notebook retains full covariance diagnostics, marginal checks, standardised scatter plots and a two-asset correlation sweep from -0.5 to 0.9. These checks support the state sampler; they do not establish basket-option price or delta accuracy.

```bash
python -m bsde_pricing describe --case correlated
python -m bsde_pricing sample --case correlated --paths 100000 --seed 2026 --output scratch/correlated_terminal.json
python -m bsde_pricing sample --case vanilla --paths 100000 --seed 2026
python -m unittest discover -s tests -v
```

The sampling report records seed, generator, financial inputs, horizon, theoretical moments, observed moments and theoretical sampling standard errors. It retains summary statistics rather than writing all paths into JSON. The sampler stores $O(Nd)$ numbers, with a dense $O(d^3)$ factorisation and $O(Nd^2)$ correlation transform; memory and runtime still need pilots before high-dimensional experiments.

The basket notebook begins with state validation. The geometric-basket reference below supplies analytical prices and coordinate deltas. Multivariate path grids and neural solvers remain separate work; the arithmetic reference below supplies price and delta estimates.

## Geometric-basket reference

### Reduction to one lognormal variable

For the fixed positive weights in the contract, define

$$G(S)=\prod_{i=1}^d S_i^{w_i},\qquad \log G=\sum_iw_i\log S_i.$$

This is a geometric average **across assets at one terminal time**, not an Asian average across observation dates. Under the joint GBM model, every linear combination of log prices is Gaussian. Let $a_i=w_i\sigma_i$ and

$$
\sigma_G^2=a^\top\rho a,
\qquad
\ell_G=\sum_iw_i(r-q_i-\tfrac12\sigma_i^2).
$$

Then

$$\log G_T\sim N(\log G_0+\ell_G\tau,\;\sigma_G^2\tau).$$

Itô's formula adds the quadratic-variation correction when returning from log $G$ to $G$:

$$
\frac{dG}{G}=b_G\,dt+\sum_iw_i\sigma_i\,dW_i,
\qquad b_G=\ell_G+\tfrac12\sigma_G^2.
$$

For weights summing to one,

$$
b_G=r-\sum_iw_iq_i-\tfrac12\left(\sum_iw_i\sigma_i^2-\sigma_G^2\right).
$$

The geometric index is not a self-financing constant-weight portfolio value, so its risk-neutral drift need not equal $r$. Even at zero dividends there is generally a drift correction. Changing correlation changes both its variance and its drift. Set $q_G=r-b_G$ as an effective carry coefficient for the scalar pricing reduction; it does not represent an additional cash dividend paid by the assets. The code retains the actual floating-point weight sum in the general expression rather than silently renormalising weights accepted within the contract's tolerance.

The lognormal geometric mean also provides a useful reference in basket control-variate work; see [New control variates for pricing basket options](https://academic.oup.com/imaman/article/36/1/111/7816702). The present calculation is the exact geometric contract, rather than a lognormal approximation of the arithmetic basket.

### Price and coordinate derivatives

Write $F_G=\mathbb E[G_T]=G_0e^{b_G\tau}$ and $v=\sigma_G\sqrt\tau$. For $v>0$,

$$
d_1=\frac{\log(F_G/K)+\tfrac12v^2}{v},\qquad d_2=d_1-v,
$$

$$
C=e^{-r\tau}\left(F_G\Phi(d_1)-K\Phi(d_2)\right),
\qquad
P=e^{-r\tau}\left(K\Phi(-d_2)-F_G\Phi(-d_1)\right).
$$

Consequently $C-P=e^{-r\tau}(F_G-K)$. Differentiating through the geometric starting value gives

$$
\frac{\partial G_0}{\partial S_k}=\frac{w_kG_0}{S_k},
\qquad
\Delta_k^C=\frac{w_kG_0}{S_k}e^{-q_G\tau}\Phi(d_1),
\qquad
\Delta_k^P=\frac{w_kG_0}{S_k}e^{-q_G\tau}(\Phi(d_1)-1).
$$

These are derivatives with respect to individual starting spots, with weights, other spots, model parameters and time fixed. The geometric basket is nonlinear in each spot: coordinate curvature contains both $C_{GG}(w_kG/S_k)^2$ and $C_Gw_k(w_k-1)G/S_k^2$. Their signs can differ, so a coordinate delta need not increase with its own spot. The scalar basket derivative is not the coordinate hedge vector. All deltas are expressed per unit of their corresponding asset spot; the common currency convention from the financial specification remains in force.

`GeometricBasketReference` derives the effective parameters once and composes the existing `BlackScholesReference`, keeping one implementation of the scalar price formula and its limiting conventions. The volatility is calculated as $\|L^\top a\|$ with $\rho=LL^\top$, avoiding subtraction in a covariance quadratic form. `EuropeanCall.basket_value()` is the shared basket calculation used by payoff evaluation and simulation risk. Equal-spot geometric baskets with an exactly unit weight sum preserve $G(s,\ldots,s)=s$ directly, preventing a log/exp round trip from moving an exactly specified payoff kink.

The analytical reference supports positive spots, matching dimensions, geometric contracts, expiry and zero effective volatility. At $v=0$, prices are the discounted deterministic payoffs. Off the deterministic exercise boundary, deltas follow the discounted payoff slope and chain rule; at the kink, all coordinate deltas are `null`. The single-asset, unit-weight case reduces to the existing Black–Scholes reference. Inputs or derived values outside floating-point range fail explicitly; no price or sensitivity clipping is used.

### Independent simulation and coordinate bumps

`MonteCarloPricer.estimate()` reuses the joint asset sampler and geometric payoff. Its price estimate does not call the analytical engine. With positive remaining variance, differentiation under the expectation gives pathwise contributions

$$D_{n,k}=e^{-r\tau}\mathbf1_{G_T^{(n)}>K}\frac{w_kG_T^{(n)}}{S_k(0)}.$$

`estimate_coordinate_delta(..., coordinate=k)` uses a **zero-based** coordinate. For a relative bump $\varepsilon$, set $h=\varepsilon S_k(0)$. The paired upper and lower terminal arrays scale only column $k$ by $(S_k(0)\pm h)/S_k(0)$, holding every other coordinate and all normal draws fixed. The existing payoff calculates each bumped contribution. The result contains pathwise, central-bump and paired-difference statistics, along with absolute/relative bump, positive-payoff and exercise-crossing counts. The same statistics kernel serves the earlier scalar and new coordinate interfaces.

The finite bump targets a price secant. Its analytical target is compared with that secant, while the paired pathwise-minus-bump discrepancy is compared with the analytical derivative-minus-secant. Combining separate estimator variances without covariance would give the wrong paired error scale. Estimates from matched seeds across coordinates are dependent; the notebook intervals are pointwise, not simultaneous portfolio-risk bands.

The coordinate Monte Carlo interface currently requires a geometric contract and positive remaining variance. It rejects expiry and entirely deterministic models; the analytical reference handles those limits. Individual zero-volatility assets remain allowed when other assets are stochastic: changing a deterministic coordinate still changes the geometric payoff. The original `estimate_delta()` retains its scalar scope, deterministic handling and report fields. Multivariate arithmetic-basket deltas are outside this interface's current scope.

### Numerical evidence and reproduction

The asymmetric fixture uses the earlier correlated model with weights $(0.2,0.3,0.5)$ and changes only the contract's basket type. At time zero,

| Quantity | Analytical value |
| --- | ---: |
| Geometric spot | 104.763167412292 |
| Effective volatility | 0.213600093633 |
| Effective carry $q_G$ | 0.032437500000 |
| Basket drift $b_G$ | -0.002437500000 |
| Call price | 10.805843052329 |
| Asset 1 delta | 0.157947844265 |
| Asset 2 delta | 0.189537413118 |
| Asset 3 delta | 0.263246407108 |

For 100,000 joint terminal samples and seed 2026, the call estimate is **10.854093879711**, with estimated standard error **0.050017218965**. The three pathwise deltas are approximately **(0.15853841, 0.19024609, 0.26423069)**. Their errors are each approximately 1.253 estimated standard errors in this matched-path example. A common indicator and common geometric terminal value make these coordinate estimates strongly dependent; agreement is not three independent statistical confirmations.

Tests independently integrate the Gaussian discounted payoff without using a normal CDF or the effective-dividend pricing reduction. Coordinate bump refinement checks the derivative formula, including delta parity. Joint Monte Carlo comparisons cover prices, pathwise deltas, central secants and paired gaps, with nonzero dividends, negative rates and nonzero valuation time. Matched-seed repricing verifies that only the selected initial spot is bumped. Scalar reduction, zero-volatility and expiry limits, exact equal-spot kinks, mixed stochastic/deterministic coordinates and invalid scope are checked separately. Quadrature price tolerance is $10^{-9}$; final spot-bump delta error is below $2\times10^{-6}$ on the tested fixture. Stochastic checks use six-standard-error budgets; they do not require every nominal 95% interval to contain its target.

```bash
python -m bsde_pricing reference --case geometric --output scratch/geometric_reference.json
python -m bsde_pricing monte-carlo --case geometric --paths 100000 --seed 2026 --output scratch/geometric_mc.json
python -m bsde_pricing monte-carlo --case geometric --quantity delta --coordinate 0 --paths 100000 --seed 2026 --output scratch/geometric_delta_0.json
python -m unittest discover -s tests -v
```

The basket notebook adds analytical/simulation comparisons, one-coordinate spot changes, equicorrelation scenarios and paired bump-size diagnostics. Its pricing maturity follows the editable simulation horizon. The geometric basket remains an analytically reducible control case at high dimension. These results do not establish a neural speed advantage or supply a general arithmetic-basket closed form; the arithmetic reference is developed below.

## Arithmetic-basket numerical reference

### Targets, pathwise deltas and price bounds

The arithmetic contract pays $(A_T-K)^+$ with $A_T=\sum_iw_iS_i(T)$. Its price is the discounted expectation under the same exact joint GBM transition. A weighted sum of correlated lognormal variables does not in general have a lognormal distribution. The reference therefore estimates price and all coordinate deltas numerically rather than applying the geometric closed form to $A$.

For strictly positive remaining variance, the pathwise contribution for coordinate $k$ is

$$D_{n,k}=e^{-r\tau}\mathbf1_{A_T^{(n)}>K}\,w_k\frac{S_k^{(n)}(T)}{S_k(0)}.$$

The code evaluates price and the full gradient on the same paths. Their errors are dependent. Mixed zero-volatility coordinates are allowed when at least one asset remains stochastic. Expiry and entirely deterministic models are rejected by this reference interface; their pricing limits remain available through earlier interfaces. The payoff kink has zero probability under the supported stochastic model.

For positive weights with exactly unit sum, the geometric mean and convexity give price bounds

$$C_G\le C_A\le\sum_iw_i C_i(S_i,K),$$

where each scalar call has the same strike, rate, maturity and its asset's volatility/dividend yield. These bounds constrain prices, not coordinate-delta ordering. The report supplies them only when the floating-point weight sum is exactly one; otherwise the fields are `null`. Estimates are retained without clipping to the bounds.

### Geometric controls and separate pilot fitting

Let $X$ collect the discounted arithmetic payoff and its pathwise delta contributions, and let $Y$ collect the corresponding geometric quantities. Their known expectation vector $\mu_Y$ comes from the analytical geometric reference. Price and each delta are separate expectation estimators; controlled deltas are not obtained by differentiating a fitted price estimate. Each component has a separately fitted coefficient:

$$\widehat X_j=X_j-\widehat\beta_j(Y_j-\mu_{Y,j}),
\qquad\beta_j^*=\frac{\operatorname{Cov}(X_j,Y_j)}{\operatorname{Var}(Y_j)}.$$

The pilot contains independent antithetic pair averages and is separate from every evaluation stream. The implementation estimates the covariance/variance ratio on those pilot units. Conditional on the pilot, $\widehat\beta$ is fixed and the evaluation expectation remains $\mathbb E[X]$. Thus fitting does not introduce the same-sample coefficient bias. Pilot cost is included in total terminal-state count and measured runtime. There is no claim that a short pilot discovers the optimal coefficient accurately.

If a pilot control has zero observed variance, the corresponding coefficient is set to zero and its index appears in `unfitted_controls`. This retains an unadjusted estimate instead of dividing by zero or claiming known variance. Index zero is price; following indices correspond to asset deltas. Negative or unhelpful fitted coefficients are not clipped. Zero observed evaluation variance withholds confidence bounds, as in the existing Monte Carlo statistics contract.

For IID evaluation pairs, the conditional residual variance determines the price/delta uncertainty. No extra unconditional pilot-variance term is needed because the conditional estimator mean is the same for every pilot. For Sobol evaluation, an IID-pair pilot coefficient need not minimise variance across scramble means. Controls can therefore worsen Sobol uncertainty, especially for discontinuous Greek contributions. Both plain and controlled results are always retained.

### Antithetic sampling units

For an independent normal vector $Z_n$, calculate terminal states under both $Z_n$ and $-Z_n$ and average each contribution:

$$X_n^{\mathrm{pair}}=\tfrac12(X(Z_n)+X(-Z_n)).$$

With $M$ pairs, there are $2M$ terminal states but only $M$ independent sampling units. Pilot fitting and evaluation statistics both use pair averages. Treating the legs as IID would omit their covariance and misstate uncertainty. The reported sample standard deviation is the standard deviation of pair contributions; standard error divides it by $\sqrt M$. Intervals use the existing normal-approximation multiplier.

`GBMModel.terminal_from_normals()` exposes the same transition kernel used by ordinary IID sampling. Antithetic and inverse-normal Sobol calculations therefore do not maintain separate GBM implementations. The supplied normal array has shape `(N,d)`, contains independent-coordinate shocks before correlation, and is not mutated. Existing scalar draws and seeded results are retained.

### Randomised Sobol sampling and uncertainty

Each evaluation replication uses a fresh SciPy Sobol engine with LMS+shift scrambling, 52 bits and no point optimisation. `random_base2(m)` generates a full $2^m$-point net without discarding initial points or thinning. Independent-coordinate normals are obtained with the inverse standard-normal CDF, then passed to the common transition kernel. Exact endpoints zero or one raise an error rather than receiving a silent tail clip.

Points within a net are dependent. For $R$ independent scrambles, first compute a price/gradient mean $\bar X_r$ for each complete net. The estimator and standard error are

$$\widehat\mu=\frac1R\sum_{r=1}^R\bar X_r,
\qquad\widehat{\mathrm{SE}}=\frac{s_{\bar X}}{\sqrt R}.$$

The approximate pointwise 95% interval uses $t_{R-1,0.975}\widehat{\mathrm{SE}}$. The report's sample standard deviation is the dispersion of **scramble means**, and `n_units` is $R$, not the total number of Sobol points. Scramble means are retained in the result for inspection. At least four independent scrambles are required; the default is 16. This is an uncertainty approximation, not a guaranteed error bound or simultaneous gradient band.

SciPy documents the [Sobol balance and scrambling requirements](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) and [independent RQMC replications](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html). The antithetic and control-variate principles are developed in the [Monte Carlo chapter of Pricing and Hedging Derivative Securities](https://book.derivative-securities.org/Chapter_MonteCarlo.html).

### Random streams, scope and computing cost

`BasketReferencePricer.antithetic()` uses `SeedSequence(seed, spawn_key=(0,))`, while `.sobol()` uses branch `(1,)`. Within each branch, spawned child zero supplies the pilot; remaining children supply evaluation. The methods therefore use disjoint branches and separate pilots. Repeatability requires the recorded numerical environment as well as the seed; identity across SciPy releases or platforms is not promised.

For pilot size $P$ pairs, evaluation size $M$ antithetic pairs costs $2P+2M$ terminal states. Sobol with $R$ scrambles and $2^m$ points costs $2P+R2^m$. Both arithmetic and geometric quantities are computed on each state. Runtime includes pilot fitting, analytical control expectations, both plain and controlled estimates and uncertainty calculations. No isolated plain-estimator runtime is measured. Arrays for one pair block or one scramble are held in memory, along with contribution arrays; memory scales with the chosen point/pair count and dimension. Scrambles are processed sequentially. The API validates integer allocations and caps net power at 30, but this is a generator bound, not a promise that such a net fits the MacBook. High-dimensional allocations require a separate memory/runtime pilot.

The geometric contract is also accepted for calibration of this machinery against its closed form. That reducible case does not supply an independent arithmetic price. Arithmetic delta tests instead compare pathwise contributions with selected-spot repricing on matched scrambled nets and use paired differences to measure discrepancy.

### Precision targets and retained findings

The notebook declares approximate 95% half-width targets of **0.005 price units** and **0.0005 for every coordinate delta**. These are statistical precision targets, not bounds on realised error. The initial matched-work comparison uses 65,536 evaluation pairs versus 16 scrambles of 8,192 points, with 4,096 pilot pairs for each method. Both spend 139,264 terminal states including the pilot.

For seed 2026, the controlled antithetic price is **13.337587776547**, with standard error **0.009102652276** and half-width about **0.01784**, which fails the price target. The controlled Sobol price is **13.356213457973**, with standard error **0.000519348009** and half-width about **0.001107**. Their discrepancy is approximately 2.04 combined standard errors. This initial result is retained rather than replaced or interpreted as a deterministic pricing discrepancy. The geometric control substantially reduces antithetic price variance, while some Sobol delta standard errors increase relative to plain Sobol. The fitted IID-pair control is not automatically an effective Sobol control.

A larger antithetic allocation of $2^{20}$ pairs is chosen from the measured standard error to meet the price target, with a fixed confirmation seed of 2027. The independent Sobol confirmation uses 16 scrambles of $2^{14}$ points. No further allocation or seed search is performed based on agreement. The controlled confirmation results are:

| Quantity | Antithetic estimate | Antithetic SE | Sobol estimate | Sobol SE |
| --- | ---: | ---: | ---: | ---: |
| Price | 13.35403144 | 0.00229101 | 13.35667441 | 0.00028312 |
| Asset 1 delta | 0.11627125 | 0.00003269 | 0.11618548 | 0.00004403 |
| Asset 2 delta | 0.18880762 | 0.00004607 | 0.18870710 | 0.00007094 |
| Asset 3 delta | 0.33951911 | 0.00006934 | 0.33953367 | 0.00012312 |

Both meet the declared half-width targets, with confirmation-method discrepancies below 1.6 combined standard errors for all four quantities. Arithmetic price bounds are approximately **[10.80584305, 17.03503764]**. The notebook additionally retains three-seed allocation comparisons and complete-call timings. Those small studies do not establish confidence-interval coverage or a universal speedup. Confirmation allocations differ in order to reach adequate precision; their runtime is not an equal-work comparison.

```bash
python -m bsde_pricing basket-reference --case correlated --method antithetic --pairs 1048576 --seed 2027 --output scratch/arithmetic_antithetic.json
python -m bsde_pricing basket-reference --case correlated --method sobol --power 14 --scrambles 16 --seed 2027 --output scratch/arithmetic_sobol.json
python -m unittest discover -s tests -v
```

The `correlated` case is the asymmetric arithmetic basket defined earlier. Reports include the financial problem, seed branch, allocations, uncertainty unit, pilot coefficients, missing-variance indices, plain/controlled results, available bounds and Sobol replication means. Unbiasedness statements refer to the ideal sampling model; finite-precision random-number generation and inverse-CDF transformation remain numerical approximations. The final reference remains numerical and carries its estimated uncertainty. Later learned-solver comparisons must preserve that distinction.

## Pricing PDE and BSDE

### Forward state and replication

The same constant-parameter risk-neutral GBM model is used throughout. Let \(B\) have independent Brownian coordinates, and let \(\rho=LL^\top\) with the lower-triangular Cholesky factor \(L\). Define

\[
\mu_i(s)=(r-q_i)s_i,\qquad A(s)=\operatorname{diag}(s_i\sigma_i)L.
\]

Then \(dS_t=\mu(S_t)dt+A(S_t)dB_t\). Asset rows and Brownian columns of \(A\) have distinct meanings. In particular,

\[
(AA^\top)_{ij}=s_i s_j\sigma_i\sigma_j\rho_{ij}.
\]

For a self-financing portfolio with value \(X\), holdings \(\Delta_i\) in dividend-paying assets and the balance in a bank account,

\[
dX=\sum_i\Delta_i(dS_i+q_iS_i\,dt)
  +r\left(X-\sum_i\Delta_iS_i\right)dt
  =rX\,dt+\Delta^\top A\,dB.
\]

This identity includes reinvested continuous dividends. The assumptions are frictionless trading, continuous rebalancing, one constant borrowing/lending rate, no transaction costs or default, and the existing European terminal payoff \(g\). Stochastic rates, funding asymmetry and early exercise are outside this model. Strictly positive volatilities and positive-definite correlation give a complete diffusion market; zero-volatility limits remain valid pricing cases but do not identify unique holdings in deterministic assets.

### Itô's formula and the driver sign

For \(Y_t=V(t,S_t)\), Itô's formula gives

\[
dY_t=\left[V_t+\mu\cdot\nabla V
  +\tfrac12\operatorname{tr}(AA^\top D^2V)\right]dt
  +(A^\top\nabla V)^\top dB_t.
\]

Matching replication with \(\Delta=\nabla V\) yields the pricing PDE

\[
V_t+\sum_i(r-q_i)s_iV_{s_i}
 +\tfrac12\sum_{i,j}\rho_{ij}\sigma_i\sigma_j s_i s_j V_{s_i s_j}
 -rV=0,\qquad V(T,s)=g(s).
\]

The BSDE convention is

\[
dY_t=-f(Y_t)dt+Z_t^\top dB_t,\qquad
f(y)=-ry,\qquad Z_t=A(S_t)^\top\nabla V(t,S_t).
\]

Consequently the forward value drift is **\(+rY\)**. The equivalent backward integral is

\[
Y_t=g(S_T)+\int_t^T f(Y_u)du-\int_t^T Z_u^\top dB_u.
\]

Dividends enter the state drift, not the discount driver. A forward Euler value step therefore reads \(Y_{n+1}\approx(1+r\Delta t)Y_n+Z_n^\top\Delta B_n\). Confusing the minus sign before \(f\) with the sign of \(f\) itself reverses discounting. This convention matches the semilinear PDE/BSDE formulation in [Han, Jentzen and E (2018), equations 1-5 and 18](https://pmc.ncbi.nlm.nih.gov/articles/PMC6112690/); the present driver is its linear pricing specialisation.

The call payoff is not differentiable at its exercise boundary. The classical derivation applies before maturity where the nondegenerate diffusion smooths the price, with the terminal condition understood by continuity. Terminal deltas at the payoff kink are not needed by the diagnostic.

### Spot delta and Brownian control

Write \(h_i=s_i\sigma_i\Delta_i\). With correlated Brownian motion \(W=LB\), the martingale term is \(h^\top dW\). With independent coordinates \(B\), it is \(Z^\top dB\), where

\[
Z=L^\top h.
\]

For NumPy row batches, shocks are transformed as `normals @ L.T`, while controls are transformed as `(deltas * spots * volatilities) @ L`. These transposes differ because a control pairs with a Brownian increment in an inner product. A hand-worked two-asset example in the tests distinguishes them. Independent Brownian coordinates are the convention for the BSDE contract.

In one dimension, \(Z=\sigma S\Delta\). For the vanilla base case, \(\Delta_0=0.5987063257\), but \(Z_0=11.9741265137\). Delta measures currency per unit of asset price; \(Z\) measures currency per square-root year. Where all \(s_i\sigma_i>0\), deltas can be recovered by solving \(L^\top h=Z\), then dividing by \(s_i\sigma_i\). Zero-volatility directions prevent that inversion.

`PricingBSDE` composes the existing model and payoff. It provides the drift, diffusion, driver, delta-to-control map and terminal condition. State and delta arrays have shape `(..., d)`, controls retain that shape, and diffusion arrays have shape `(..., d, d)`. Inputs are validated and not mutated. This NumPy contract fixes the financial conventions; framework-specific automatic differentiation is a separate implementation step.

### Discounted analytical-control diagnostic

Discounting removes the pricing driver:

\[
d(e^{-rt}Y_t)=e^{-rt}Z_t^\top dB_t,\qquad
V(0,S_0)=\mathbb E[e^{-rT}g(S_T)].
\]

For a uniform grid with \(m\) intervals, the diagnostic computes each path's residual

\[
\varepsilon_m=e^{-rT}g(S_T)-V(0,S_0)
 -\sum_{n=0}^{m-1}e^{-rt_n}Z(t_n,S_{t_n})^\top\Delta B_n.
\]

`AnalyticalBSDEDiagnostic` uses the already checked one-asset Black-Scholes value and delta. `BlackScholesReference.call_delta_array` evaluates batches before expiry; it shares the scalar normal-argument calculation and is checked against scalar deltas and independent price bumps. Positive maturity and volatility are required by this diagnostic.

The finest-grid increments are generated once. Coarser increments sum consecutive fine increments, so every grid follows the same Brownian paths and uses the same terminal payoff. At each left endpoint, the shared GBM transition evaluates the state exactly from the Brownian value at that time. Thus there is no Euler state bias in this experiment. The remaining approximation is the left-endpoint stochastic integral. The control never uses its current or a future increment.

Under the ideal sampling model, each adapted integral sum has zero expectation, so the residual mean is zero at every grid size. Its estimated standard error is the across-path sample standard deviation divided by \(\sqrt N\). The root mean square, \(\sqrt{N^{-1}\sum_k\varepsilon_{m,k}^2}\), measures the typical pathwise discretisation error. It need not vanish on a finite grid even with an exact analytical solution. Approximately square-root timestep behaviour is an empirical comparison guide, not a fitted convergence theorem for the nonsmooth terminal payoff. Grid estimates are dependent because the paths are coupled; six pointwise intervals do not establish coverage or six independent tests.

```bash
python -m bsde_pricing bsde-diagnostic --case vanilla --paths 8192 --seed 2026 --output scratch/vanilla_bsde.json
python -m unittest discover -s tests -v
```

The default grid sizes are 8, 16, 32, 64, 128 and 256. The largest must be divisible by every smaller size; `--steps` accepts a different increasing nested set. Memory scales as \(N m_{\max}\), with about 16 MiB for the default finest Brownian array, plus working arrays. Reports retain the problem, seed, generator, grid settings, residual means, standard errors and RMS values. Notebook sections 11-12 expose the path allocation and grid choices alongside the existing editable financial inputs.

For 8,192 paths and seed 2026, the base-case RMS falls from **2.791483** at eight intervals to **0.512118** at 256. The final mean is **-0.00276424**, with standard error **0.00565842**. Every grid's mean lies within one estimated standard error of zero in this run. These are analytical-control discretisation results, not neural training results or a demonstrated pricing speedup. Tests also cover negative rates, dividends, a shorter maturity and higher volatility. Local BSDE loss construction, PyTorch derivatives and learned controls are developed separately.

## Automatic derivatives and backend checks

### A smooth value map in physical coordinates

`ValueNetwork` is a pointwise two-hidden-layer tanh network, initially 16 units per hidden layer. For input rows \((t,S_1,\ldots,S_d)\), it represents

\[
V_\theta(t,S)=K_*N_\theta\left(t/T_*-1,S_1/s_{*,1}-1,\ldots,S_d/s_{*,d}-1\right).
\]

The scales are fixed positive buffers, moved with the network to its selected dtype and device. They are not trainable parameters. Smooth activations permit nonzero continuous second derivatives. Every layer acts on one sample independently; there is no batch normalisation or cross-sample operation. This architecture is a derivative pilot, not a trained option pricer or a selected final architecture.

With normalised coordinates \(x_0=t/T_*-1\) and \(x_i=S_i/s_{*,i}-1\), the chain rule gives

\[
V_t=\frac{K_*}{T_*}N_{x_0},\qquad
\Delta_i=\frac{K_*}{s_{*,i}}N_{x_i},\qquad
V_{S_iS_j}=\frac{K_*}{s_{*,i}s_{*,j}}N_{x_ix_j}.
\]

These factors are necessary for physical Greeks and for the BSDE control \(Z=A^\top\Delta\). Normalisation occurs inside `forward`, so differentiation with respect to physical inputs applies these factors automatically. The polynomial oracle uses different spot scales to expose a missing or reversed factor.

### Graph and tensor contracts

`value_derivatives` owns a detached copy of the input tensor and sets `requires_grad=True` on that leaf. It does not accumulate into the caller's input `.grad`. The first derivative uses `torch.autograd.grad(..., create_graph=True)`, so its computation remains differentiable. Differentiating each gradient column gives the full time/spot Hessian. Gradient columns are ordered `(time, asset 0, ..., asset d-1)`, and the Hessian has shape `(N, d+1, d+1)`.

The sum of row values can be differentiated in one call because rows are independent. A batch-coupled callable violates this contract and would require a different Jacobian calculation. `parameter_graph=True` retains the derivative graph, permitting a loss involving gradients or Hessians to backpropagate to the network weights. Otherwise returned values and derivatives are detached for inspection. Differentiating through a Hessian to parameters exercises mixed third derivatives, which a forward-pass device check does not test.

The network is constructed explicitly on CPU in float64, under a locally restored CPU random state. A supplied seed makes weights reproducible without changing the caller's CPU RNG. Diagnostics use a separate CPU generator for inputs. Network parameters, scale buffers and input tensors must share dtype and device. CPU float64 is the numerical baseline; CPU float32 and MPS float32 are compared against it. MPS float64 is not requested. No default dtype or global device setting is changed.

The graph behaviour follows the official [autograd gradient API](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad.html). Device trials use [MPS synchronisation](https://docs.pytorch.org/docs/2.15/generated/torch.mps.synchronize.html) around timed calls.

### Independent derivative checks

The scaled polynomial is

\[
P(t,S_1,S_2)=100\left[(t/2)^2+(S_1/80)^3+(S_2/120)^3
 +(S_1/80)(S_2/120)\right].
\]

Its time derivative, spot deltas, diagonal Hessian and both mixed spot derivatives are calculated directly. The second oracle is a narrow differentiable PyTorch expression for a vanilla call at three strictly pre-expiry states. Targets come from the existing scalar Black-Scholes implementation: price, delta, gamma and the time derivative implied by the pricing PDE. This diagnostic expression is not used by the application pricing commands. Unit tests also compare the smooth network with finite differences using `gradcheck`, `gradgradcheck` and direct second spot/time bumps. Affine and constant maps check zero curvature, and batch-versus-single-row evaluation checks independence.

Absolute tolerances are declared before the device trial:

| Quantity | CPU float64 | Float32 |
| --- | ---: | ---: |
| Polynomial gradient/Hessian, vanilla delta/gamma | 2e-10 | 2e-4 |
| Vanilla value/time derivative | 2e-9 | 5e-4 |
| Network value/gradient/Hessian and objective parameter gradient versus CPU float64 | 2e-9 | 1e-3 |

Each error retains its own units; these are bounds for this finite diagnostic set, not a common loss scale or final pricing targets. Float32 tolerances allow roundoff in higher derivatives. Passing them does not establish an error bound over an entire training domain.

### Backend workload and timing scope

```bash
python -m bsde_pricing autograd --output scratch/autograd.json
python -m bsde_pricing autograd --backend cpu --batch-size 256 --repeats 5
python -m unittest discover -s tests -v
```

The default workload has three assets, four input coordinates and 256 rows. It evaluates the network value, gradient and full Hessian, forms an objective from all three, and backpropagates to every network parameter. Derivative terms in this diagnostic objective are scaled to dimensionless units. There is no optimiser step and no loss calibration or training result.

For each backend the same fixed weights and inputs are used, followed by two warmup calls and five measured calls. Reports retain every call duration and its median. Timings include input-leaf allocation, derivative computation, gradient clearing, objective construction, parameter backward, gradient collection and numerical validation. Network setup and initial host/device copies are excluded. MPS calls are synchronised before and after measurement; validation itself may also synchronise. CPU uses one thread during the measured trial and restores the prior thread count afterwards. These are complete small diagnostic-call timings, not kernel-only timings or general CPU/GPU speed claims.

`auto` always checks CPU float64 and float32, then trials MPS float32 when available. An unavailable MPS backend is recorded explicitly; `--backend mps` makes its absence a failed command. Numerical failures and unsupported operations remain in the report and give a nonzero exit status. An oracle execution failure is retained while the network workload is still attempted. The MPS trial rejects `PYTORCH_ENABLE_MPS_FALLBACK=1`, so CPU fallback cannot masquerade as an MPS result. There is no automatic backend selection for subsequent training from availability alone.

Initial Linux workspace checks use PyTorch 2.15.0+cpu. Both CPU precisions pass the analytical and network comparisons. MPS is unavailable there, so Apple-silicon support, accuracy and timings require the MacBook report. The existing environment specification remains the verified MacBook specification. The next calculation is a known-solution BSDE loss check; the first trained-network notebook follows the first solver implementation.

### Apple-silicon verification

The default command was verified on an ARM64 MacBook with macOS 27.0.1, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0, using implementation [523ffb7](https://github.com/KVInator/deep-bsde-pricing/commit/523ffb715bd8b9ac97fd67cf98645c4a0f09feec). All analytical and network derivative checks pass on CPU float64, CPU float32 and MPS float32. The complete 56-test suite also passes on this machine.

| Backend | Precision | Median complete-call time |
| --- | --- | ---: |
| CPU, one thread | float64 | 1.4584 ms |
| CPU, one thread | float32 | 1.3577 ms |
| MPS | float32 | 6.7233 ms |

These are five measured calls after two warmups, at 256 rows, three assets and hidden width 16. MPS calls span 5.6155-7.9942 ms, while CPU float32 calls span 1.3553-1.3749 ms. MPS takes approximately 4.95 times the CPU float32 median for this small synchronized workload; this does not establish performance at larger batches or during full training.

CPU float64 vanilla price, time derivative, delta and gamma errors are at roundoff scale. MPS float32 absolute errors are approximately 5.85e-6 for network values, 3.29e-6 for the full input gradient, 1.04e-6 for the Hessian and 2.39e-8 for the objective's parameter gradient, all within the declared tolerances. The zero CPU float64 network discrepancies arise from comparison with the same deterministic CPU baseline; the independent evidence comes from the analytical oracles and finite-difference tests. The report verifies the tested operations without CPU fallback, not every possible MPS operator or future model. CPU float64 remains the baseline for the next small known-solution checks; backend choice for larger training workloads requires a fresh accuracy and cost comparison.

## Local discounted BSDE loss

### Residual and objective

For a candidate value map \(V_\theta\), define \(Y_n=V_\theta(t_n,S_n)\), spot deltas \(\Delta_n=\nabla_SV_\theta(t_n,S_n)\), and independent-Brownian controls \(Z_n=A(S_n)^\top\Delta_n\). The local residual follows from

\[
d(e^{-rt}Y_t)=-e^{-rt}(f(Y_t)+rY_t)dt+e^{-rt}Z_t^\top dB_t.
\]

The discrete residual on path \(k\) is

\[
R_{k,n}=e^{-rt_{n+1}}Y_{k,n+1}-e^{-rt_n}Y_{k,n}
 +e^{-rt_n}[f(Y_{k,n})+rY_{k,n}]\Delta t_n
 -e^{-rt_n}Z_{k,n}^\top\Delta B_{k,n}.
\]

For the pricing driver \(f(y)=-ry\), the bracket cancels exactly. Keeping it explicit fixes the BSDE convention and permits a driver-sign fault to be tested. The terminal residual is \(E_k=e^{-rT}[V_\theta(T,S_{k,M})-g(S_{k,M})]\).

With a fixed positive currency scale \(K_*\) and terminal weight \(\lambda>0\), the implemented loss is

\[
\mathcal L_{\rm local}=\frac1N\sum_{k=1}^N\sum_{n=0}^{M-1}
       \left(\frac{R_{k,n}}{K_*}\right)^2,\qquad
\mathcal L_{\rm terminal}=\frac1N\sum_{k=1}^N\left(\frac{E_k}{K_*}\right)^2,
\qquad \mathcal L=\mathcal L_{\rm local}+\lambda\mathcal L_{\rm terminal}.
\]

Both loss components are dimensionless. Residuals themselves retain discounted currency units. Local errors are summed over time, then averaged over paths; averaging over both axes would add another factor \(1/M\) and change the balance against the terminal term as the grid changes. Default \(K_*\) is strike and \(\lambda=1\), fixed before the diagnostic. There is no adaptive reweighting or division by the observed loss.

Local consistency plus terminal-value matching follows the formulation described by [Raissi's FBSNN methodology](https://maziarraissi.github.io/FBSNNs/). The discounting, exact GBM state transitions and normalisation above specify this implementation. The author's symbol \(Z\) denotes a spot gradient there; this project reserves \(Z\) for the Brownian control and uses \(\Delta\) for the spot gradient.

### Implementation and analytical substitution

`LocalBSDELoss` is the differentiable tensor adapter for the existing `PricingBSDE`. It owns fixed coefficient buffers, computes the same driver, control map and payoff in PyTorch, and returns local/terminal/total losses together with raw residuals. Tests compare those tensor financial operations with the existing NumPy contracts. Path tensors have shapes `(N,M+1,d)` for states, `(N,M,d)` for independent Brownian increments and `(M+1,)` for times. They are fixed finite data, without `requires_grad`; value-model parameters retain their graph.

Only first input derivatives are required by this loss. `value_derivatives(..., compute_hessian=False)` avoids constructing a full Hessian, while `parameter_graph=True` permits backpropagation through the deltas and controls. The Hessian checks remain available for derivative verification. A finite perturbation of a network weight independently checks the gradient of the complete BSDE loss. A hand-calculated one-step example checks signs, discount factors, time reduction, scale and terminal weight.

Input differentiation explicitly enables autograd, including during an outer `no_grad` evaluation. `parameter_graph=False` then detaches the results. The stronger `inference_mode` is rejected because it disables the required input derivative graph.

`GBMModel.path_from_increments` evaluates exact grid states from cumulative independent Brownian values using the existing terminal transition. Coarse increments sum consecutive increments from one finest-grid sample, coupling the paths across refinement levels. No Euler state approximation is introduced. Model drift, dividends and correlation remain those of the specified risk-neutral GBM.

`AnalyticalCall` supplies a differentiable vanilla/geometric known solution, sharing the tensor vanilla expression with the earlier autograd diagnostic. Its effective geometric parameters come from `GeometricBasketReference`; prices and coordinate deltas are checked against the existing scalar references. This is a diagnostic oracle, while application pricing commands continue to use those references. The oracle uses the exact payoff at maturity and differentiates controls strictly before maturity. Terminal supervision uses payoff values; no payoff Greek is required at the exercise kink. Positive maturity and variance are required by the oracle diagnostic.

### Refinement and fault detection

For the exact solution, local residuals need not be zero on individual finite-grid paths. For a sufficiently regular integrand, one-step RMS is of order \(\Delta t\); the sum of expected squared residuals is then of order \(\Delta t\) on a fixed horizon. These are comparison guides for the observed grids, not convergence guarantees at a nonsmooth call payoff. The earlier cumulative martingale diagnostic measures a different quantity: its terminal residual RMS is approximately of order \(\sqrt{\Delta t}\).

The diagnostic reports raw local RMS, both normalised loss terms and the mean of each path's summed residual divided by maturity. Under the ideal sampling model, the correct discounted analytical solution has zero expected signed residual at every grid size. Standard errors use those independent path sums. Time steps within a path and grid-level estimates on shared paths are dependent.

Two private diagnostic subclasses deliberately corrupt the canonical loss without changing its residual formula:

- Reversing the driver to \(+rY\) adds \(2r e^{-rt_n}Y_n\Delta t_n\). Its expected path-averaged residual-rate difference is \(2rV(0,S_0)\). The paired difference uses identical paths and must exceed five of its estimated standard errors to flag the fault. At \(r=0\), this fault is explicitly inapplicable.
- Transforming row controls with `L.T` instead of `L` introduces a stochastic exposure error. Its squared-increment contribution accumulates to a nonvanishing local-loss floor. The finest wrong-control loss must exceed three times the correct loss. A symmetric Cholesky factor makes the transpose fault inapplicable; the asymmetric three-asset geometric case supplies a meaningful check.

The other diagnostic checks require decreasing local loss, terminal loss at most 1e-24, and correct signed means within four estimated standard errors. These finite-sample checks apply to the declared cases, seed and grids. They are not simultaneous confidence-coverage statements. A sign fault can retain the same decreasing squared-loss order as the correct discretisation, so a falling loss alone cannot establish a correct BSDE implementation.

### Default known-solution results

```bash
python -m bsde_pricing bsde-loss --case vanilla --output scratch/vanilla_loss.json
python -m bsde_pricing bsde-loss --case geometric --output scratch/geometric_loss.json
python -m unittest discover -s tests -v
```

Both commands use CPU float64, one PyTorch CPU thread during evaluation, 2,048 paths, PCG64 seed 2026 and nested grids `(8,16,32,64)`. The report retains the financial specification, versions, seed, scale, weight, grid statistics, fault applicability and check outcomes. The previous PyTorch thread count is restored. Numerical failures raise explicitly; a failed diagnostic check still saves the report and returns a nonzero CLI status.

| Case | Local loss, 8 intervals | Local loss, 64 intervals | One-step RMS, 64 intervals | Terminal loss |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 8.0186436e-4 | 1.0545907e-4 | 0.12836659 | 0 |
| Geometric basket | 9.5442833e-4 | 1.2304068e-4 | 0.13865463 | 0 |

For vanilla, the finest correct summed-residual rate is -0.0081510 with SE 0.0226521. Reversing the driver gives a paired rate difference of 0.5651456 with SE 0.0096171. For the geometric basket, the corresponding paired difference is 0.6547961 with SE 0.0107733. The finest wrong-transpose geometric loss is 0.00380041, approximately 30.9 times the correct value. All declared default checks pass in the development workspace. Tests also cover a shorter maturity with negative rates, dividends and higher volatility.

The same default commands pass on the ARM64 MacBook with macOS 27.0.1, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0. Its 63-test suite passes without recorded warnings. The local-loss reductions from eight to 64 intervals are approximately 7.60-fold for vanilla and 7.76-fold for geometric. Finest signed residual means are within 0.36 and 0.24 estimated standard errors of zero, respectively; reversed-driver paired differences are about 58.8 and 60.8 standard errors from zero. The correlation-transpose ratio is 30.8874. These results agree with the workspace results to rounding precision.

These checks establish a usable finite-grid loss and its parameter-gradient calculation. They do not show that the analytical solution globally minimises the discrete objective, that an optimiser will recover it, or that a small training loss alone guarantees price/Greek accuracy. Neural approximation, optimisation, independent evaluation and grid bias remain separate parts of the first training experiment.

## First value-gradient training experiment

### Model, objective and fixed budget

`FBSNNSolver` fits the existing `ValueNetwork` with `LocalBSDELoss`. The first case is the scalar vanilla call, with fixed starting spot 100, strike 100, maturity one year, rate 3%, zero dividend yield and volatility 20%. The network has two hidden layers of 16 tanh units, 337 trainable parameters and the previously verified physical-input scaling. The output is an unconstrained call value; price positivity, monotonicity and Greek bounds are assessed through reference comparisons rather than imposed by the architecture.

The learned objects are \(V_\theta(t,S)\) and \(\Delta_\theta=\partial_SV_\theta\). The Brownian control is \(Z_\theta=\sigma S\Delta_\theta\), computed inside the differentiable loss. The network receives no analytical price or Greek labels in its fitting objective. The terminal payoff is the known contract condition. Analytical results are used only for monitoring and evaluation.

The run uses 16 time intervals, fresh batches of 128 exact GBM paths, 3,000 Adam updates and CPU float64. The learning rate is 0.001 for the first 70% of updates and 0.0003 for the remainder. Loss scale is strike and terminal weight is one. A 180-second training-time cap bounds the run; a capped run retains its actual completed-update count and a `time_budget_reached` status. The initial inspection targets, declared before fitting, are absolute errors no greater than 0.25 currency units for price and 0.02 for delta. Targets do not trigger extra updates, early stopping or checkpoint selection.

Adam follows the PyTorch implementation; checkpoints save module state as described in the official [model saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models). No optimiser or activation search is performed in this experiment.

### Streams and checkpoints

NumPy SeedSequence children of seed 2026 have explicit responsibilities: child 0 supplies fresh training paths; child 1 supplies 256 fixed monitoring paths; child 2 supplies 2,048 evaluation paths drawn only after fitting. CPU weight initialisation uses the same recorded seed under the network's locally restored Torch RNG context. No global NumPy or Torch RNG is consumed by path generation or network initialisation.

For this first recorded run, monitoring is descriptive. Its saved model is the last completed update, even if an earlier monitoring point had a smaller analytical error. Each logging window reports its mean pre-update training loss; monitoring values are evaluated after the logged update. History begins with the untrained model. The analytical local loss is also evaluated on the fixed monitoring paths as a discretisation comparison, not a lower bound on the objective.

The first run created a directory containing `model.pt` and `report.json`. Existing directories are rejected before fitting, so previous results are preserved. The checkpoint contains tensors and primitive configuration/financial metadata; loading uses `weights_only=True`, CPU mapping, strict state-dictionary matching and validation that saved scales agree with the financial metadata. Loaded checkpoints support inference. The original version 1 checkpoint has no optimiser/RNG continuation state. Version 2 continuation and validation selection are specified in the next section. A new fit requires a new solver instance. Numerical failures raise explicitly and retain a failure record instead of a successful-run report.

### Independent evaluation

The initial price and delta are deterministic outputs of the fixed trained model at \((0,S_0)\). Their signed errors are direct comparisons with Black-Scholes, without Monte Carlo error bars. Variation across trained seeds is not measured by this single run.

On the fresh child-2 sample, price and delta errors are evaluated against the scalar Black-Scholes reference at pre-expiry grid points. Price/delta RMSE averages squared pointwise errors over paths and those times. Terminal payoff RMSE is reported separately, and terminal deltas are excluded. Standard errors for mean signed errors use each independent path's time-averaged error, accounting for within-path dependence. They condition on the trained model and do not measure optimisation or training-seed uncertainty.

Training starts every path from the specified \(S_0\). The sampled distribution explores later states, but the experiment does not establish a global solution over initial spots, rates, volatilities, maturities or arbitrary states. Conditional slices in the notebook use sorted states sampled at fixed interior times, rather than treating changes of initial spot as already trained problems.

The notebook loads the checkpoint without fitting again and uses an additional descriptive sample: seed 2027, SeedSequence branch 3 and 512 paths. This sample is distinct from the fitting, monitoring and saved evaluation streams. The first three trajectories are plotted without selection by prediction quality. Figures retain the full learning trajectory, errors, sampled tails and terminal mismatch.

### Retained first-run results

In the Linux development workspace with Python 3.12.14, NumPy 2.3.5 and PyTorch 2.15.0+cpu, the fixed 3,000-update run finishes in approximately 12.7 seconds. This is one measured run on that environment, not a MacBook timing or a performance comparison. The timer includes batch generation, derivative evaluation, optimiser updates and monitoring; model/optimiser construction and the initial monitoring-path draw precede it. Post-training evaluation has its own reported duration. One PyTorch CPU thread is used during fitting and saved-run evaluation, and the previous thread count is restored.

| Initial quantity | Learned | Analytical | Signed error | Inspection target |
| --- | ---: | ---: | ---: | --- |
| Price | 9.5622976261 | 9.4134033839 | +0.1488942422 | Met: absolute error at most 0.25 |
| Delta | 0.5160791851 | 0.5987063257 | -0.0826271406 | Missed: absolute error at most 0.02 |

On the 2,048 independent evaluation paths, pre-expiry price RMSE is 0.92229567, delta RMSE is 0.11624185 and terminal payoff RMSE is 1.75547535. Mean signed price error is 0.08592612 (path-level SE 0.01316253); mean signed delta error is -0.01105461 (SE 0.00171941). A small average signed error can coexist with substantial pointwise errors through cancellation.

The held-out local loss is 0.0002490521 and terminal loss is 0.0002902230, giving total loss 0.0005392750. The analytical solution's local loss on those same paths is 0.0003959160. The trained local component being smaller than the analytical component does not establish better pricing: the learned model has a nonzero terminal error, a biased initial delta, and visible shape violations in sampled tails. Negative call values and deltas outside their analytical bounds remain in the plotted raw outputs. No output clipping or hard constraint projection is used.

The result meets the first price target and misses the delta target. It does not yet separate approximation error, optimiser error and finite-grid bias. The next controlled experiments can investigate stability and loss balance, while independent errors remain separate from the training objective. The run has not been extended or replaced to make both targets pass.

```bash
python -m bsde_pricing train --run-dir scratch/runs/vanilla_fbsnn
python -m unittest discover -s tests -v
```

The default directory must be new. An explicitly named alternative directory retains a separate experiment. `notebooks/03_neural_solvers.ipynb` loads its selected run and provides eight code cells with five retained figures. Core tests check parameter updates, independent stream ownership, analytical evaluation metrics, inference checkpoint round trips, directory protection, timeout reporting and explicit failure records.

### MacBook verification

The ARM64 MacBook run completes all 3,000 updates in 6.20594 seconds, with a separately recorded evaluation time of 0.11724 seconds, using Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0 on macOS 27.0.1. All 68 tests pass without recorded warnings. Initial price/delta and the recorded monitoring loss components match the workspace run within approximately 2e-14 over the 31 logged checkpoints. This is agreement for one configuration and seed, not a training-seed robustness study or a general cross-platform timing comparison.

The executed notebook at [4603398](https://github.com/KVInator/deep-bsde-pricing/commit/4603398e996089a62006b6cc491238933691c373) has eight sequential code cells, five saved figures, unchanged source and no recorded error or stderr outputs. It loads the saved checkpoint and reports the same initial price and delta as the training report. Its separate 512-path sample gives price RMSE 0.9037401, delta RMSE 0.1169896 and terminal payoff RMSE 1.720026. The initial price target is met and the delta target remains missed. State slices retain slope errors and sampled bound violations; the terminal fit remains imperfect. The first experiment is verified, while improving accuracy and assessing variability remain subsequent research questions.


## Training reproducibility and loss weighting

### Fixed comparison and selection rule

The comparison measures seed variability and one change to the objective. It uses seeds 2026, 2027 and 2028, terminal weights \(\lambda=1,5\), and the first experiment's 337-parameter network, 16 time intervals, 128 paths per update and 3,000-update Adam schedule. The six configurations are written to `design.json` before training. There is no architecture, activation, optimiser or additional weight search. The original run directory remains separate.

For a batch of exact GBM paths, write the previously defined discounted local residual as \(R_{n,m}(\theta)\), terminal error as \(E_n(\theta)=e^{-rT}[V_\theta(T,S_T^{(n)})-g(S_T^{(n)})]\), and strike scale as \(K\). The fitted objective is

\[
L_\lambda(\theta)=\underbrace{\frac1N\sum_{n=1}^N\sum_{m=0}^{M-1}(R_{n,m}(\theta)/K)^2}_{L_{\mathrm{local}}}
+\lambda\underbrace{\frac1N\sum_{n=1}^N(E_n(\theta)/K)^2}_{L_{\mathrm{terminal}}}.
\]

Both components are dimensionless. Increasing \(\lambda\) multiplies the terminal component of the parameter gradient by five, so \(\nabla_\theta L_5=\nabla_\theta L_{\mathrm{local}}+5\nabla_\theta L_{\mathrm{terminal}}\). This changes the compromise that Adam follows. It does not alter the payoff, supply analytical labels or impose exact terminal matching. Weighted totals for different \(\lambda\) values are different objectives. Cross-weight tables therefore retain raw local and terminal losses, their unweighted sum, and price/delta/terminal RMSE in physical units.

For each seed, the two weights share the initial parameters and the three SeedSequence streams: child 0 for fresh training paths, child 1 for 256 fixed validation paths and child 2 for 2,048 held-out evaluation paths. Separate seeds use separate initialisations and streams. The branching mechanism follows [NumPy's SeedSequence documentation](https://numpy.org/doc/stable/reference/random/parallel.html). Within each seed, a weight comparison is paired; the six runs are not six independent observations of one estimator.

Validation is measured at update zero, every 100 updates and the full configured update budget. The selected update is

\[
k^*=\underset{k\in\{0,100,\ldots,3000\}}{\operatorname{argmin}}\;L_\lambda^{\mathrm{validation}}(\theta_k),
\]

with the earlier update retained on an exact tie. Analytical errors are recorded for inspection but do not enter this rule. Selection compares candidates within one configured objective; a small validation objective does not guarantee the smallest analytical price or delta error. Each final and selected model is evaluated on identical fresh child-2 paths, drawn only after that run ends. Inspecting this exploratory study does not turn the same evaluation paths into a future untouched final test set.

### Early stopping, pauses and timing

`TrainingConfig.patience` optionally stops after that many consecutive scheduled validation checks without a strict new minimum. Its unit is validation checks, not optimiser updates. The best model is retained while the last model and stopping counter remain in continuation state. Early stopping is disabled in the six-run study so that all completed configurations have equal 3,000-update budgets. It is not triggered by an analytical accuracy target.

`fit(stop_after=k)` pauses at an absolute update count. A pause does not add an off-schedule validation check, clear the unfinished training-loss window or inspect held-out paths. The learning-rate change remains tied to 70% of `config.iterations`, not the length of the current invocation. A 180-second cap applies to each fitting invocation; cumulative fitting time is retained across continuations. A time-limited run is explicitly labelled and evaluated at its actual update count. Its selected model can only come from scheduled checks already reached. Completed and early-stopped runs are closed; a paused or time-limited run may continue from its saved state.

Training time includes path generation, differentiable losses, Adam updates, finite-parameter checks and scheduled validation. Model/optimiser construction and the initial fixed-validation draw precede that timer. Evaluation time includes both the final and selected model assessments, each with the analytical comparisons. Study wall time also includes the continuation verification and checkpoint/report writes. Timing is reported rather than used for checkpoint selection.

### Two checkpoint responsibilities

`model.pt` stores the validation-selected weights for inference. `resume.pt` stores the latest update, which may be later than the selected update. Its version 2 payload contains:

- Financial metadata and the complete training configuration.
- Current network parameters and Adam state, including moment estimates and update counters.
- Training and evaluation generator states, fixed validation states/increments and the locally owned Torch RNG state.
- Completed update count, the best model/loss/update, checks without improvement, history, accumulated time and unfinished training-loss window.

The fixed validation tensors have shapes \((256,17,1)\) for states and \((256,16,1)\) for independent Brownian increments under the default setup. The training stream's current state, rather than its starting seed alone, determines the next batch. Resetting Adam's moments or replaying the initial random stream would create another optimisation trajectory. The implementation follows the [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) for saving model and optimiser state and also retains the experiment-specific continuation state above. Loading uses CPU mapping and `weights_only=True`. The original version 1 inference checkpoint remains supported and cannot be resumed as if optimiser state had been saved.

The study first runs a bounded 80-update verification using four time intervals, width four and 16 training paths per update. It pauses after update 37, between 20-update logging checks and before the learning-rate change at update 57. The resumed and uninterrupted runs must match bitwise in current weights, Adam state, training/evaluation/Torch RNG state, fixed validation paths, selected model and update, stopping counter, unfinished window, numerical history and held-out evaluation. Elapsed times are excluded. All checks pass on the development runtime. Exact equality on this runtime does not promise bitwise agreement after changing library versions, hardware or numerical backends.

```bash
python -m bsde_pricing train --run-dir scratch/runs/pause_example --iterations 3000 --stop-after 731
python -m bsde_pricing train --resume scratch/runs/pause_example/resume.pt --run-dir scratch/runs/continued_example
```

Both output directories must be new. Resume uses the saved financial and training configuration; conflicting configuration flags are rejected. The files under `pause_example` remain intact. Loading `model.pt` with `FBSNNSolver.load` evaluates the selected model without fitting; `FBSNNSolver.resume` is the explicit continuation entry point.

### Measured comparison

The development-workspace study uses Linux, Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu, CPU float64 and one Torch thread. All six runs complete 3,000 updates with no recorded failures or time-limit stops. Study wall time is 99.25 seconds; individual fitting times range from 14.17 to 18.27 seconds. These timings include the current validation and finite-parameter checks and are not a speed comparison against the earlier implementation or the MacBook.

| Seed | Terminal weight | Selected update | Selected price error | Selected delta error | Final price error | Final delta error | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 1 | 2400 | +0.1537589 | -0.0827162 | +0.1488942 | -0.0826271 | 15.327 |
| 2026 | 5 | 1400 | -0.1967637 | -0.1093516 | +0.2311939 | -0.0861227 | 14.723 |
| 2027 | 1 | 3000 | +0.1561873 | -0.0276475 | +0.1561873 | -0.0276475 | 14.167 |
| 2027 | 5 | 3000 | +0.2583376 | -0.0588113 | +0.2583376 | -0.0588113 | 16.207 |
| 2028 | 1 | 3000 | -0.1541622 | +0.0044700 | -0.1541622 | +0.0044700 | 15.536 |
| 2028 | 5 | 2900 | +0.0244822 | +0.0448844 | -0.1747560 | +0.0500151 | 18.266 |

Weight-one selected-model delta error has mean -0.03530 and sample standard deviation 0.04409; weight five gives mean -0.04109 and sample standard deviation 0.07863. The sample standard deviation uses \(s=\sqrt{\sum_{i=1}^3(x_i-\bar{x})^2/(3-1)}\). It describes the observed seed dispersion. Three seeds do not estimate extreme failure probabilities reliably. The initial errors themselves are deterministic evaluations of each trained network and have no Monte Carlo standard error.

The initial price target is met by all three selected weight-one models and two selected weight-five models. The initial delta target is met by one selected weight-one model and no selected weight-five model. A mean signed error can conceal large errors with opposing signs. The full per-seed table remains the primary evidence.

At the final update, weight five improves all three paired pre-expiry price RMSEs, delta RMSEs and terminal payoff RMSEs. The mean terminal RMSE falls from 1.48912 to 1.39306. This does not consistently improve the initial delta: the final delta error is slightly larger in magnitude for seed 2026, larger for 2027 and larger for 2028. After validation selection, mean terminal RMSE is 1.51650 for weight one and 1.52509 for weight five. In particular, seed 2026 with weight five selects update 1,400 and has terminal RMSE 2.01726 versus 1.62291 at its final update. Validation selection is therefore functioning as specified without guaranteeing improved held-out fit.

Seed 2026 with weight one reproduces the original final-update price and delta errors, +0.14889424 and -0.08262714, and terminal RMSE 1.75547535. Its validation-selected update is 2,400 and has delta error -0.08271617. The original run is preserved; the study does not replace it with a favourable seed. The remaining error cannot be attributed uniquely to optimisation, network approximation, the terminal kink, validation sampling or the finite time grid from these comparisons.

```bash
python -m bsde_pricing training-study --run-dir scratch/runs/reproducibility
python -m unittest discover -s tests -v
```

`TrainingStudy` owns the fixed matrix, persistence, resume verification and across-seed summary; the CLI delegates to it. `study.json` is updated after every run and retains failed/time-limited entries. Means, sample standard deviations, ranges and sample counts are recorded rather than silently omitting failed runs. Directory protection rejects a repeated destination before training. The neural notebook retains the first experiment and adds the per-seed tables, selected/final comparisons, loss-component histories, continuation check and inference reload check. All 72 tests pass without warnings in the development workspace, including exact continuation across logging/schedule boundaries, selection, optional stopping, legacy checkpoint loading, stream independence and failed-run summaries. Notebook verification executes 13 code cells in a fresh IPython process and retains eight reviewed figures. MacBook verification is recorded below.

### MacBook reproducibility verification

The ARM64 MacBook study completes all six 3,000-update runs in 41.00595 seconds of study wall time, with individual fitting times of 6.20724-6.71159 seconds and paired final/selected evaluation times of 0.21623-0.30552 seconds. Its recorded environment is macOS 27.0.1, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0, using the prescribed CPU float64 configuration. All 72 tests pass in 2.522 seconds without recorded warnings. No runs fail or hit the time limit.

Every continuation check passes, including Adam state, current/selected weights, random streams, validation paths, stopping state, unfinished logging window, numerical history and held-out evaluation. The selected update counts are unchanged across the MacBook and workspace: 2,400 and 1,400 for seed 2026, 3,000 for both seed-2027 runs, and 3,000 and 2,900 for seed 2028, listed in terminal-weight order 1 then 5. The maximum absolute discrepancy across all reported selected/final numerical metrics is approximately 8.9e-15. This is observed agreement for these configurations; it does not establish a general cross-platform guarantee or a matched hardware speed comparison.

The executed notebook at [69b3cc5](https://github.com/KVInator/deep-bsde-pricing/commit/69b3cc59364e3316bf6da1ec118de15b97f13cc7) changes outputs only. Its 13 code cells have sequential execution counts, eight figures render completely, and no error or stderr output is recorded. All six inference reload checks report zero price/delta discrepancy against their saved evaluations. The original five figures are preserved byte for byte; the three new figures retain per-seed errors, selected/final comparisons and unweighted validation-loss components.

The observed accuracy limitations remain unchanged: one selected weight-one model meets the initial delta target and none of the selected weight-five models does. The larger terminal weight improves final-update fresh-path RMSEs without consistently improving initial sensitivities. Reproducibility and continuation are verified; the remaining pricing/risk errors remain research findings. No further tuning or additional seeds are introduced in this comparison.


## Terminal-loss Deep BSDE in one asset

### Learned quantities and discounted propagation

The terminal-loss formulation follows the initial-value and time-step-control construction of Han, Jentzen and E, [Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568). This implementation applies it to the existing clean scalar GBM call, uses exact forward GBM transitions and learns discounted Brownian exposures. It does not reproduce the paper's nonlinear examples or claim their accuracy/cost results.

For the already specified clean driver \(f=-rY\), write \(\bar Y_t=e^{-rt}Y_t\) and \(\bar Z_t=e^{-rt}Z_t\). The driver cancels under discounting:

\[
d\bar Y_t=\bar Z_t\,dW_t,
\qquad \bar Y_T=e^{-rT}g(S_T).
\]

For \(M\) uniform intervals, the trainable objects are a scalar \(y_0\), a scalar initial control \(z_0\), and independent control networks \(z_{\theta_m}(S_m)\) for \(m=1,\ldots,M-1\). All controls are discounted. The complete path calculation is

\[
\bar Y_0=y_0,\qquad
\bar Y_{m+1}=\bar Y_m+\bar Z_m\Delta W_m,
\qquad
\bar Y_M=y_0+\sum_{m=0}^{M-1}\bar Z_m\Delta W_m.
\]

The controls use the current spot and their fixed time index only. They cannot use the next Brownian increment or later state. This adaptedness makes each left-point stochastic increment have conditional mean zero. Coupling different batch rows would also change the intended state-only map; the networks contain no batch normalisation or cross-path layers.

`TerminalControlNetwork` owns these free parameters, time-step networks and the recurrence. `TerminalBSDESolver` owns fitting, validation, independent evaluation and persistence. The CLI and notebook call that implementation. `PathTrainingConfig` and `sample_training_paths` are shared with the value-gradient solver, preserving its original seeded calculation. The forward state kernel and terminal payoff remain owned by `GBMModel` and `EuropeanCall`; no second GBM or contract implementation is introduced.

### Units, architecture and objective

Use strike \(K\), initial spot \(S_0\) and maturity \(T\) as scales. For dimensionless free parameters \(a,b\) and dimensionless network outputs \(h_m\),

\[
y_0=Ka,\qquad \bar Z_0=\frac K{\sqrt T}b,
\qquad \bar Z_m=\frac K{\sqrt T}h_m(S_m/S_0-1).
\]

The factor \(K/\sqrt T\) gives controls price-per-square-root-year units; multiplying by \(\Delta W\), measured in square-root-years, gives price units. Discounting is already included in the learned control, so the recurrence does not apply another factor \(e^{-rt_m}\). Recovering the undiscounted control later requires \(Z_m=e^{rt_m}\bar Z_m\).

Both initial free parameters start at zero. The subnetworks use a locally seeded CPU initialisation that restores the caller's Torch random state. Each later time has two width-16 tanh layers and one scalar output. A scalar subnet has \(w^2+4w+1\) parameters. The full parameter count is

\[
2+(M-1)(w^2+4w+1)=2+15\times321=4817.
\]

The existing value-gradient network has 337 parameters. Equal update counts therefore do not represent equal parameter counts or equal computational work.

For each fresh training batch,

\[
E_n(\theta)=y_0+\sum_m\bar Z_m^{(n)}\Delta W_m^{(n)}-e^{-rT}g(S_T^{(n)}),
\qquad
L(\theta)=\frac1N\sum_n(E_n(\theta)/K)^2.
\]

There are no local value-consistency penalties or analytical price/Greek labels. The same known terminal payoff defines the training target. The loss differentiates the propagated terminal value with respect to all free parameters and subnet weights. No input gradient or Hessian of a price network is needed during this solver's training.

A useful population identity separates initial bias and residual dispersion. If \(P=e^{-rT}g(S_T)\) and \(I=\sum_m\bar Z_m\Delta W_m\), adaptedness gives \(\mathbb E[I]=0\). Thus

\[
\mathbb E[E]=y_0-\mathbb E[P],\qquad
\mathbb E[E^2]=(y_0-\mathbb E[P])^2+\operatorname{Var}(I-P).
\]

For fixed controls, the population-optimal initial scalar is the discounted expected payoff. A small observed initial bias can coexist with a large terminal residual variance. Conversely, a finite validation sample can select a checkpoint whose held-out error is worse. Neither the seed nor the training/selection objective guarantees accuracy.

### Fixed pilot, streams and saved models

The declared pilot uses the original vanilla financial inputs, 16 intervals, fresh batches of 128 paths, 3,000 Adam updates, width 16, seed 2026 and CPU float64. The learning rate is 0.001 until 70% of the configured updates and 0.0003 thereafter. A 180-second per-fit cap bounds execution. The existing 0.25-unit initial price inspection target is retained; no terminal-RMSE pass threshold is invented after observing the run. The budget is not extended to meet an accuracy target.

SeedSequence children 0, 1 and 2 supply training paths, 256 fixed validation paths and 2,048 held-out paths, respectively. Validation minimises terminal MSE at update zero, every 100 updates and the full configured budget. The earlier candidate wins an exact tie. Analytical prices are recorded for inspection but never select the checkpoint. A time-limited run retains its actual update count; only scheduled candidates reached before the cap can be selected. Early stopping and seed/architecture searches are not added to this first terminal-loss experiment.

Fitting time includes fresh path generation, network propagation, backpropagation, Adam updates, finite-gradient/parameter checks and scheduled validation. Network/optimiser construction and the fixed validation draw precede the timer. Evaluation time covers both final and selected models and the matched analytical-control residual calculations. The prior Torch thread count is restored after fitting and evaluation. Timing is from one measured development run, not a MacBook estimate or a matched performance comparison.

The new run directory contains selected inference weights in `model.pt`, final inference weights in `last_model.pt`, and numerical/environment/configuration evidence in `report.json`. Existing destinations are rejected before fitting. Numerical fitting failures retain `failure.json` and raise explicitly. Checkpoint loading maps tensors to CPU, uses `weights_only=True`, validates method/version and financial scales, and rejects nonfinite parameters. These first terminal-loss checkpoints support inference only; the separate value-gradient continuation implementation remains unchanged. No terminal-solver optimiser-resume claim is made.

### Independent price and terminal evaluation

The initial scalar price is compared directly with Black–Scholes. Its signed error is a deterministic model output and has no Monte Carlo standard error. The scalar does not depend on an input spot in a learned pricing function, so differentiating it with respect to spot cannot produce a delta. Recovering risk through the diffusion/control map belongs to the subsequent comparison. A saved fixed-start checkpoint also does not establish accuracy after changing initial spot or financial parameters.

For the same held-out paths, report terminal residual mean, its path-level standard error, RMSE, selected quantiles and the normalised terminal objective. Undiscounted terminal RMSE multiplies discounted RMSE by \(e^{rT}\); the two units are labelled separately. The reported mean stochastic integral and its standard error are an adapted-martingale diagnostic. Its finite-sample mean need not be exactly zero. Residual-mean uncertainty is not an uncertainty interval for the learned initial scalar or a training-seed variability estimate.

A separate analytical-control path calculation uses

\[
\bar Z_m^{\mathrm{BS}}=e^{-rt_m}\sigma S_m\Delta_{\mathrm{BS}}(t_m,S_m),\qquad
E^{\mathrm{BS}}=u_{\mathrm{BS}}(0,S_0)+\sum_m\bar Z_m^{\mathrm{BS}}\Delta W_m-P.
\]

It reuses `PricingBSDE.control` and the independent analytical reference on matched states and increments. Its residual is nonzero on a finite left-point grid even though the GBM states themselves are exact. It is a useful discretisation comparison, not a proven lower bound on the neural loss. Neither this calculation nor the analytical initial price enters training or selection.

Propagated \(\bar Y_m\) values generally depend on path history. They can be compared with analytical conditional prices along a path, but they are not outputs from a globally learned \(u_\theta(t,S)\). The notebook preserves raw terminal mismatches and any negative propagated values without clipping.

### Retained first terminal-loss results

The Linux development run uses Python 3.12.14, NumPy 2.3.5 and PyTorch 2.15.0+cpu. It completes 3,000 updates in 23.65706 seconds of fitting, with 0.02737 seconds for paired evaluation. The validation-selected update is 2,800. Fixed validation loss decreases from 0.09626809 before training to a selected minimum of 0.0006095010; the final validation loss is 0.0006099031.

| Quantity | Validation-selected update 2,800 | Final update 3,000 |
| --- | ---: | ---: |
| Initial price | 9.40694237 | 9.45653568 |
| Analytical price | 9.41340338 | 9.41340338 |
| Signed initial price error | -0.00646101 | +0.04313230 |
| Discounted initial Brownian control | 11.82480928 | 12.04941140 |
| Discounted terminal RMSE | 2.49502716 | 2.49221999 |
| Undiscounted terminal RMSE | 2.57101205 | 2.56811939 |
| Normalised terminal MSE | 0.0006225161 | 0.0006211160 |
| Terminal residual mean | +0.01404410 | +0.05974090 |
| SE of terminal residual mean | 0.05514542 | 0.05506842 |

Both models meet the pilot initial-price inspection target. For the selected model, the absolute price error is approximately 0.069% of analytical option value. Its stochastic-integral mean is -0.08629497 with standard error 0.30343400. The analytical-control discounted terminal RMSE on the same 2,048 paths is 2.03220071. The learned terminal mismatch therefore remains substantial despite the small initial-price error. The selected model's slightly higher held-out terminal RMSE than the final model is retained.

This is an encouraging scalar-price observation from one fixed seed, not a robust solver ranking, a Greek result or a speed advantage. Parameter counts, objectives and computational work differ between formulations. Seed variation, explicit control-to-delta recovery and a matched two-solver comparison have not been established for this new solver. The existing value-gradient runs and their mixed accuracy results remain intact.

```bash
python -m bsde_pricing train-terminal --run-dir scratch/runs/vanilla_terminal_bsde
python -m unittest discover -s tests -v
```

The neural notebook appends the saved-run configuration, parameter/cost table, selected/final initial prices, learning curves, terminal residuals and first three fresh propagated paths. Its extra descriptive evaluation uses 512 paths, seed 2027 and SeedSequence branch 3. It loads the saved checkpoints without fitting. MacBook verification is recorded below.

### MacBook terminal-loss verification

The ARM64 MacBook completes the same 3,000-update experiment in 12.71551 seconds of fitting and 0.03072 seconds of paired evaluation, using macOS 27.0.1, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0 with CPU float64. Its supplied test log records all 77 tests passing in 2.532 seconds without warnings. The report retains 384,000 training paths, 4,817 parameters and the same validation-selected update, 2,800.

The selected initial price is 9.4069423698, against 9.4134033839 analytically, giving signed error -0.0064610140. Selected discounted terminal RMSE remains 2.49502716, with residual mean 0.01404410 and standard error 0.05514542. The matched analytical-control RMSE is 2.03220071. Final-update price is 9.4565356798, with error +0.04313230 and discounted terminal RMSE 2.49221999. The maximum discrepancy across selected/final evaluation metrics and numerical training history, excluding elapsed times, is approximately 7.1e-15 between the MacBook and workspace. This is observed agreement for this one configuration and seed, not a general cross-platform reproducibility or timing claim.

Review of [ada17ea](https://github.com/KVInator/deep-bsde-pricing/commit/ada17ea8e594b624b3bce25085aa436b94495c89) confirms an outputs-only notebook commit with unchanged source, 17 sequential code cells, 11 complete figures and no recorded error or stderr output. The first eight figures are preserved byte for byte; the three terminal-solver figures are reviewed in full. Selected/final checkpoint reload checks complete without error. The separate 512-path descriptive sample has discounted terminal RMSE 2.543842, mean residual -0.138158 with standard error 0.112367, and matched analytical-control RMSE 2.096407.

The figures retain negative propagated values, terminal tail errors and differences from analytical conditional-value paths. The small initial-price error is confirmed, while accurate controls, recovered Greeks, seed robustness and a computational advantage remain unestablished. This completes verification of the first terminal-loss experiment without extending its budget or replacing its retained results.
