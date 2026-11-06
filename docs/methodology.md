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


## Common price and risk evaluation

### Frozen models and a common contract

The comparison uses two already selected checkpoints: seed 2026 with terminal weight one from the value-gradient reproducibility study, selected at update 2,400, and the first terminal-loss run, selected at update 2,800. Each training run completed 3,000 updates. Selection follows each formulation's previously declared validation objective. No new training, seed selection or tuning is performed for the comparison. The original final-update value-gradient experiment remains a separate retained result.

`SolverRiskComparison` requires identical model parameters, contract, starting states and time grids. `compare_saved_runs` loads the declared directories, checks initial outputs and financial/configuration metadata against their source reports, and retains model/report SHA-256 digests, selection rules, update counts, fitting/evaluation times and training runtimes. A changed report or checkpoint is therefore identifiable. Existing comparison destinations are rejected before loading models or evaluating paths.

The default fresh sample uses 2,048 paths, 16 time intervals and seed 2031 with SeedSequence branch 4. It is separate from the fitting, validation, saved post-training and descriptive branch-3 streams. Both solvers use the exact same spot states for every pre-expiry risk comparison. The report and `paths.npz` preserve these states, reference deltas, both predicted delta/control arrays and diffusion diagnostics. Loading the archive disables pickle and validates dimensions and finite float64 values. These exploratory results must not subsequently be described as an untouched confirmatory test after they have informed development choices.

### Control-to-delta recovery

For independent Brownian coordinates, the physical diffusion is

\[
\Sigma(s)=\operatorname{diag}(\sigma_i s_i)L,\qquad LL^\top=\rho,
\qquad Z=\Sigma(s)^\top\Delta.
\]

The domain-level inverse is `PricingBSDE.recover_deltas`. It solves the square transposed system with [NumPy's batched linear solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), rather than forming an explicit inverse. The right-hand side retains a trailing singleton column so batched vector semantics are explicit. Positive spots and strictly positive volatilities are required for this inverse; the model's correlation contract already requires positive definiteness. Singular maps are rejected without a pseudoinverse, clipping or regularisation.

The terminal solver learns \(\bar Z=e^{-rt}Z\), so its control must be undiscounted before solving:

\[
Z_t=e^{rt}\bar Z_t,\qquad
\Delta_t=\Sigma(S_t)^{-\top}e^{rt}\bar Z_t.
\]

For the scalar case this reduces to \(\Delta_t=e^{rt}\bar Z_t/(\sigma S_t)\). At inception, \(\sigma S_0=20\), and the learned terminal control 11.82480928 gives delta 0.59124046. Treating this control as a delta, omitting the diffusion scale or omitting discount conversion at later times would give different and incorrect sensitivities.

The value-gradient solver supplies \(\partial_S V_\theta\) through the existing physical-coordinate derivative helper. Its Brownian control is then reconstructed with the existing `PricingBSDE.control` map. Applying the common inverse to that control checks the map orientation and reconstruction. Neither numerical round-trip agreement nor a tiny reconstruction residual establishes that a learned control is close to its analytical target.

For recovered \(\widehat\Delta\), report Euclidean residual \(\|\Sigma^\top\widehat\Delta-Z\|_2\), its relative value when \(\|Z\|_2>0\), the smallest diffusion singular value, the 2-norm condition number and the inverse norm. A zero supplied control has zero delta and zero absolute/relative residual under the nonsingular contract. The singular values give

\[
\kappa_2(\Sigma)=s_{\max}/s_{\min},\qquad
\|\Sigma^{-\top}\|_2=1/s_{\min}.
\]

The [NumPy condition-number documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html) specifies the SVD-based 2-norm convention. For a fixed diffusion, a control perturbation satisfies

\[
\|\delta\Delta\|_2\leq\|\Sigma^{-\top}\|_2\|\delta Z\|_2.
\]

A scalar nonzero diffusion has condition number one at every scale. It can still have a large inverse norm, so the condition number alone does not describe absolute sensitivity. In this base case the initial inverse norm is 0.05. Across sampled pre-expiry states, the smallest diffusion singular value is 10.40827895 and the maximum inverse norm is 0.09607736. Condition numbers remain one. Unit checks separately cover a tiny positive scalar diffusion and a near-singular two-asset correlation, retaining their amplification diagnostics without repair.

### Independent risk errors and gamma

Initial price and delta are deterministic outputs from each frozen model. Both are compared with the same analytical price 9.4134033839 and delta 0.5987063257. Their errors do not have Monte Carlo standard errors. The existing pilot absolute-error bands, 0.25 for price and 0.02 for delta, remain inspection targets rather than checkpoint-selection criteria or a guarantee of practical precision.

The common sample measures delta RMSE over every path and pre-expiry time point. Mean signed delta errors average within each path first, then use the standard deviation of independent path averages divided by the square root of path count. The resulting standard error measures conditional evaluation sampling uncertainty, not training-seed variability. Call delta bounds are \(0\leq\Delta(t,S)\leq e^{-q(T-t)}\); violations use a 1e-10 allowance for roundoff. Counts and extrema retain tail failures. The denominator is 32,768 correlated path/time evaluations, including repeated initial states, rather than that many independent observations.

Only the value-gradient representation supplies a global differentiable price map. Its initial gamma is the physical spot entry of the Hessian from `value_derivatives`. Numerical consistency is checked with central bumps of the same network's delta at relative spot increments 0.01, 0.005 and 0.001:

\[
\Gamma_{\mathrm{bump}}(h)=\frac{\Delta_\theta(0,S_0+h)-\Delta_\theta(0,S_0-h)}{2h}.
\]

These bumps verify differentiation of this learned function; they do not establish accurate risk after retraining at another start or under changed financial parameters. Analytical gamma is an independent comparison. The terminal representation has a fixed-start initial scalar and time-step controls; no comparable global price Hessian is reported. Differentiating its free scalar with respect to spot would not recover gamma.

### Measured saved-model results

The development comparison uses Linux, Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu, CPU float64 and one Torch thread during model evaluation. Both source runs use the declared seed and full pilot budget; their parameter counts differ, 337 for the value-gradient network and 4,817 for the terminal-loss network.

| Quantity | Value-gradient selected model | Terminal-loss selected model |
| --- | ---: | ---: |
| Selected update | 2,400 | 2,800 |
| Initial price | 9.56716227 | 9.40694237 |
| Initial price error | +0.15375889 | -0.00646101 |
| Initial delta | 0.51599016 | 0.59124046 |
| Initial delta error | -0.08271617 | -0.00746586 |
| Initial Brownian control | 10.31980314 | 11.82480928 |
| Matched-state delta RMSE | 0.11278521 | 0.08397111 |
| Mean signed delta error | -0.01199854 | +0.00144648 |
| SE of mean signed delta error | 0.00163724 | 0.00104682 |
| Minimum sampled delta | -0.10486636 | -1.28194162 |
| Maximum sampled delta | 1.18282376 | 1.22801393 |
| Maximum absolute delta error | 0.88081526 | 1.28194162 |
| Bound violations / path-time evaluations | 1,541 / 32,768 | 3,136 / 32,768 |

The terminal model meets both initial inspection targets in this retained comparison; the value-gradient model meets the price target and misses the delta target. The terminal model also has lower average squared delta error on matched states, but more sampled bound violations and a larger worst absolute error. Its low-spot tails include strongly negative deltas. These observations rule out interpreting the smaller RMSE or the accurate initial point as uniformly accurate risk. Both methods' unprojected predictions remain visible.

Control reconstruction residuals are at floating-point scale. The maximum absolute residual across sampled states is 7.1e-15 for both models, with maximum relative residuals below 3.0e-16. The small numerical residual and moderate inverse scale indicate that the observed discrepancies concern the learned risks rather than an inaccurate linear solve in this base case.

Value-gradient initial gamma is 0.0057837158, against 0.0193334058 analytically, with error -0.0135496900. Central delta-bump gamma differs from its autograd value by -7.90e-8, -1.97e-8 and -7.86e-10 as the spot bump decreases from 1 to 0.5 to 0.1. Thus the network derivative is internally consistent while its curvature is substantially wrong relative to the independent solution. This gamma diagnostic has no newly invented pass target.

### Timing scope and reproduction

The report retains two initial-inference warmups and five measured repeats for each model. The workload includes price, delta, control mapping/recovery and diffusion diagnostics; it excludes checkpoint loading, references, path generation and training. A separate single-batch timer covers all matched-state model risk evaluations and recovery, excluding path generation and analytical references. Gamma evaluation has a separate timer and is excluded from the common initial price/delta workload. The caller's Torch thread count is restored.

In the recorded workspace comparison, median initial evaluation times are approximately 0.568 ms for value-gradient and 0.221 ms for terminal loss, with matched-state batches taking 0.05669 and 0.03909 seconds. All raw repeats remain in the report. These short measurements are descriptive, include diagnostic overhead and do not establish a stable latency ratio. Source fitting times, configurations, update counts and environment records are also retained. Different parameter counts, objectives, errors and fitting workloads prevent an unqualified end-to-end speedup claim. A broader cost/precision study remains later work.

```bash
python -m bsde_pricing compare-risk --run-dir scratch/runs/vanilla_risk_comparison
python -m unittest discover -s tests -v
```

`--value-run` and `--terminal-run` explicitly select existing run directories; defaults point to the two declared selected checkpoints. No training is performed. The comparison writes a new `report.json` and compressed `paths.npz`; it preserves all source checkpoints and reports. The notebook appends initial price/delta/gamma views, matched-state risk curves, tail-error and inverse-diffusion views, derivative checks and timing/provenance tables. All 84 tests pass without warnings, including a hand-worked correlated inverse, singular rejection, amplification diagnostics, analytical price/delta/gamma checks, discount unwinding, matched-state metrics, saved-array round trips and no-retraining/overwrite checks. The notebook executes 21 code cells in a fresh IPython process with 14 reviewed figures. MacBook verification is recorded below.

### MacBook common-risk verification

The ARM64 MacBook comparison uses the same saved seed-2026 checkpoints, 2,048 fresh paths, 16 intervals and seed 2031/branch 4, with no retraining. Its supplied test log records all 84 tests passing in 2.530 seconds without warnings. The recorded environment is macOS 27.0.1, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0, CPU float64 with one Torch thread. Numerical metrics, including initial risks, gamma, bump checks and violation counts, agree with the development workspace within approximately 5.4e-15, excluding timing and platform-specific artifact hashes.

Terminal initial price/delta errors remain -0.00646101 and -0.00746586, meeting both pilot inspection bands. Value-gradient errors remain +0.15375889 and -0.08271617. Matched-state delta RMSE is 0.08397111 for terminal loss versus 0.11278521 for value-gradient. The terminal model nevertheless retains more bound violations, 3,136 versus 1,541 of 32,768 path/time evaluations, and a larger worst absolute delta error, 1.28194162 versus 0.88081526. The value-gradient gamma remains 0.0057837158 against 0.0193334058; finite delta bumps agree with its automatic derivative.

Maximum absolute reconstruction residual is 7.1e-15 for both models; maximum relative residuals are below 3.7e-16. The initial diffusion is 20, inverse norm 0.05 and scalar condition number one. Sampled minimum singular value and maximum inverse norm remain 10.40827895 and 0.09607736. These diagnostics distinguish numerical recovery from learned-control accuracy.

Review of [55d58c8](https://github.com/KVInator/deep-bsde-pricing/commit/55d58c87252513fe7fa88b413005d80fa952db00) confirms an outputs-only notebook commit with unchanged source, 21 sequential code cells, 14 complete figures and no error or stderr output. The first 11 figures are preserved byte for byte; the three appended risk figures are reviewed in full. Tables and plots retain negative tail deltas, upper-bound breaches, curvature error, source checkpoint choices and training costs.

Recorded initial inference medians are 0.20546 ms for value-gradient and 0.07688 ms for terminal loss; the matched-state risk batches take 0.07334 and 0.06408 seconds. These short diagnostic-inclusive measurements do not establish a stable latency ratio, end-to-end speedup or robust solver ranking. This closes the saved-model risk comparison with its positive initial terminal-risk result and unresolved tail/curvature limitations retained.


## Five-asset geometric basket

### Declared case and exact reference

The geometric call uses strike 100, maturity one year and continuously compounded rate 3%. The five coordinates have different economic inputs.

| Asset | Spot | Weight | Volatility | Dividend yield | Correlation loading |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 90 | 0.10 | 0.16 | 0.005 | 0.65 |
| 2 | 100 | 0.15 | 0.19 | 0.010 | 0.50 |
| 3 | 110 | 0.20 | 0.23 | 0.015 | -0.35 |
| 4 | 95 | 0.25 | 0.26 | 0.020 | 0.45 |
| 5 | 105 | 0.30 | 0.31 | 0.025 | 0.55 |

With loadings \(b\), the correlation matrix is \(bb^\top+\operatorname{diag}(1-b_i^2)\). Every idiosyncratic variance is positive, so the matrix is positive definite. The negative loading makes asset 3 negatively correlated with each of the other four assets.

The existing geometric reduction prices \(G=\prod_i S_i^{w_i}\) as a scalar lognormal underlying. With weights summing to one,

\[
\sigma_G^2=(w\odot\sigma)^\top\rho(w\odot\sigma),\qquad
q_{\mathrm{eff}}=\sum_i w_iq_i+\tfrac12\sum_iw_i\sigma_i^2-\tfrac12\sigma_G^2.
\]

The initial geometric spot is 101.03933496, effective volatility 0.14116669 and effective dividend yield 0.03967848. The analytical call price is 5.49471932. Each coordinate delta uses the chain rule \(\Delta_i=\Delta_G w_iG/S_i\). The vector reference applies the same reduction to batches of states. The integration check compares the batched deltas with central price bumps in all five coordinates.

### Vector solver contracts

The value-gradient network takes time and five physical spots and returns a scalar. The existing input gradient yields five physical deltas. For row batches the Brownian control is \((\Delta\odot\sigma\odot S)L\), where \(LL^\top=\rho\). Exact GBM shocks use Brownian row increments multiplied by \(L^\top\).

The terminal model has one initial price and five initial controls. Each later network takes five spots and returns five discounted Brownian exposures. With controls and increments of shape \((N,M,5)\), the last-axis inner product gives one discounted value increment per path and step. Recovery first multiplies the discounted control by \(e^{rt}\), then solves \(\Sigma(S)^\top\Delta=Z\). In this case \(\Sigma(S)=\operatorname{diag}(\sigma_i S_i)L\). The test uses the transposed physical diffusion to reconstruct the discounted-model output at a positive time after accounting for discounting.

Both fits use CPU float64, seed 2026, width 16, 128 fresh paths per update, 16 intervals and 3,000 updates. The value-gradient terminal weight is one. Each selected checkpoint follows its existing validation rule. The separate risk run uses 2,048 matched paths with seed 2031 and SeedSequence branch 4. Reports retain price, all five coordinate risks, the same initial inspection bands of 0.25 for price and 0.02 for each delta, and the source fit metadata. Geometric archives use shape \((N,M,5)\) for pre-expiry risks and controls, and \((N,M+1,5)\) for states. Diffusion diagnostics use \((N,M)\).

### Observed prices, coordinate errors and tails

Both fits complete all 3,000 updates. The selected value-gradient checkpoint is at update 3,000 with 401 parameters and a recorded fit time of 15.40 seconds. Terminal loss selects update 2,700 with 6,801 parameters and a fit time of 24.56 seconds. The recorded runtime is Linux, Python 3.12.14, NumPy 2.3.5 and PyTorch 2.15.0+cpu.

| Model | Initial price | Signed price error | Matched-state delta RMSE | Maximum absolute delta error |
| --- | ---: | ---: | ---: | ---: |
| Value-gradient | 5.76936224 | +0.27464292 | 0.02791369 | 0.21941861 |
| Terminal loss | 5.52561574 | +0.03089642 | 0.02524980 | 0.45207228 |

| Asset | Analytical delta | Value-gradient initial error | Terminal initial error | Value-gradient path RMSE | Terminal path RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.05718615 | +0.00103829 | +0.00862142 | 0.01996655 | 0.01888358 |
| 2 | 0.07720131 | -0.00942098 | +0.00482740 | 0.02131964 | 0.01943079 |
| 3 | 0.09357734 | -0.02174397 | +0.00222855 | 0.02662496 | 0.02092674 |
| 4 | 0.13544089 | -0.00559817 | +0.00192072 | 0.03141863 | 0.02973996 |
| 5 | 0.14705011 | +0.00183190 | -0.00173087 | 0.03669690 | 0.03363367 |

The terminal model meets both initial inspection targets. The value-gradient model exceeds the price band by 0.02464292 and the asset 3 delta band by 0.00174397. At all five coordinates the terminal model has lower path RMSE and more violations of \(0\leq\Delta_i\leq e^{-q_{\mathrm{eff}}(T-t)}w_iG/S_i\). The respective totals are 14,107 for terminal loss and 7,484 for value-gradient across 163,840 correlated path/time/asset evaluations. The terminal model reaches a delta of -0.45207228. Its pooled delta RMSE exceeds the value-gradient RMSE from time 0.6875 onward on this grid. At the last pre-expiry step the two RMSEs are 0.05019 and 0.03634, respectively. The findings cover one fixed case and one training seed per solver.

The diffusion condition number is 2.93292 initially and ranges from 1.85138 to 6.63706 on the matched states. The minimum sampled singular value is 7.06947 and the maximum inverse norm is 0.14145331. Control reconstruction residuals are below 5.8e-15. The arithmetic recovery accuracy separates the observed model errors from rounding in the linear solve.

### Reproduction and regression

```bash
python -m bsde_pricing reference --case geometric-five
python -m bsde_pricing train --case geometric-five
python -m bsde_pricing train-terminal --case geometric-five
python -m bsde_pricing compare-risk \
  --value-run scratch/runs/geometric_five_fbsnn \
  --terminal-run scratch/runs/geometric_five_terminal_bsde \
  --run-dir scratch/runs/geometric_five_risk
python -m unittest discover -s tests -v
```

All 85 tests pass in the development workspace. The five-asset integration test fits and reloads both models, reloads the common archive, checks all coordinate references with central price bumps, and checks the vector reconstruction with an independently assembled diffusion. Re-running the vanilla comparison from the original checkpoints reproduces all archive arrays and the non-timing report exactly. The notebook adds three code cells and two figures covering coordinate accuracy, errors over time and bounds.

### MacBook verification

The MacBook run completes both 3,000-update fits. Value-gradient takes 9.0582 seconds and selects update 3,000. Terminal loss takes 15.0891 seconds and selects update 2,700. All 85 tests pass in 2.818 seconds. The run uses macOS 27.0.1 ARM64, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0 with one CPU thread and float64. The comparison report agrees with the terminal output. Its price, delta, tail and diffusion metrics agree with the workspace within approximately 1.2e-14.

Terminal loss meets the price band and all five initial delta bands. It also has 14,107 bound violations versus 7,484 for value-gradient, and a larger worst delta error. The two risk figures retain the crossover in per-time RMSE at 0.6875 years and the larger terminal-model RMSE near expiry. The maximum absolute reconstruction residual remains below 5.7e-15 on the MacBook.

The notebook has 24 sequential code cells and 16 figures with no error or stderr outputs. The two new figures and seven tables agree with the report, and the first 14 figures match the earlier reviewed outputs byte for byte. The notebook commit contains the executed results. [b294e4c](https://github.com/KVInator/deep-bsde-pricing/commit/b294e4c096fb0582f221e621a52509388199990f)


## Ten-asset arithmetic pilot

### Question and declared case

The pilot measures whether an independently resolved ten-asset arithmetic reference and two bounded neural fits are feasible before fixing the wider experiment schedule. It retains one development seed and the existing architectures. Accuracy misses, resource costs and numerical-reference uncertainty remain separate outcomes.

The call pays

\[
g(S_T)=(A_T-K)^+,\qquad A_T=\sum_{i=1}^{10}w_iS_T^i.
\]

All spots and the strike are 100, maturity is one year, the continuous rate is 3%, every volatility is 20%, dividends are zero and every weight is 0.10. Off-diagonal correlation is 0.30. The correlation eigenvalues are 0.70 with multiplicity nine and 3.70 once, so the diffusion has full rank. These are synthetic risk-neutral inputs under the existing constant-coefficient GBM model. The preceding unequal five-asset case separately checks coordinate orientation.

The forward process uses exact lognormal transitions. Neural controls still use a 16-interval grid, whose stochastic-integral approximation has a separate time-discretisation error. This pilot does not isolate that error from optimisation and network approximation.

### Independent price and delta contributions

Direct terminal simulation supplies the discounted price contribution and pathwise coordinate derivatives

\[
X_0=e^{-rT}(A_T-K)^+,\qquad
X_i=e^{-rT}\mathbf 1_{\{A_T>K\}}w_i\frac{S_T^i}{S_0^i}.
\]

The derivative follows from \(\partial S_T^i/\partial S_0^i=S_T^i/S_0^i\). Under this nondegenerate model the arithmetic basket has no atom at the strike, so the payoff kink has probability zero. The pathwise derivative estimates delta. Differentiating its indicator again would require a separate treatment of the strike boundary for gamma.

For the geometric basket \(G_T=\prod_i(S_T^i)^{w_i}\), the corresponding price and pathwise-delta contributions are \(Y_j\), with analytical expectations \(\mu_j\) from `GeometricBasketReference`. Each target has its own fitted coefficient

\[
\widehat\beta_j=
\frac{\widehat{\operatorname{Cov}}(X_j,Y_j)}{\widehat{\operatorname{Var}}(Y_j)},
\qquad H_j=X_j-\widehat\beta_j(Y_j-\mu_j).
\]

The coefficients use 4,096 independent antithetic pilot pairs. The coefficient pilot is disjoint from its method's evaluation sample. Conditional on a frozen coefficient, the control adjustment has zero mean. The same terminal-sampling and contribution kernels from the earlier basket references perform this calculation.

The primary reference uses 16 independent scrambled Sobol nets with \(2^{16}\) terminal states each, seed 2032 and reference branch one. A coarse level uses \(2^{12}\) states with the same scramble seeds and coefficient pilot. These two levels are dependent. The reference standard error is the sample standard deviation of the 16 controlled net means divided by four. The pointwise interval uses a Student-t critical value with 15 degrees of freedom. It is an approximate interval across randomised nets, without an IID-point assumption.

A separate check uses 131,072 IID antithetic pair averages at seed 2033, branch zero, with its own independent coefficient pilot. Its standard error counts pairs as sampling units. The training seed is 2026 with separate children zero, one and two for training, checkpoint validation and post-fit paths. Reference values never enter either training worker or select checkpoints.

### Reference precision and method agreement

The original roadmap targets are 0.01 currency units in price and 0.01 in every weight-scaled delta \(\Delta_i/w_i\). The reference allocation is 20% of each target. Thus the allowed 95% half-widths are 0.002 in price and 0.002 in scaled delta, equivalent to 0.0002 in each raw delta here. These targets are distinct from the earlier 0.25 price and 0.02 raw-delta inspection bands. Both sets retain their original meanings.

| Reference | Evaluation terminal states | Controlled price | Pointwise 95% half-width |
| --- | ---: | ---: | ---: |
| Sobol 2^12 | 65,536 | 6.40714337 | 0.00184860 |
| Sobol 2^16 | 1,048,576 | 6.40618406 | 0.00036192 |
| Antithetic IID | 262,144 | 6.40772167 | 0.00226616 |

The primary price is 6.40618406, with interval [6.40582214, 6.40654598]. The largest scaled-delta half-width is 0.00033689. Both precision allocations are met. The controlled price lies between the geometric-call lower bound 5.64841010 and the weighted single-asset-call upper bound 9.41340338.

The Sobol-minus-IID price difference is -0.00153761, or -1.31574 combined standard errors. Coordinate differences range from -1.84007 to -3.24880 combined standard errors, and some pointwise 95% intervals disagree. The coordinate contributions share terminal states and are correlated. The report retains all differences and a four-combined-standard-error diagnostic screen. That screen supplies no simultaneous confidence guarantee. The stated precision budget uses the primary interval's estimated uncertainty and does not prove its coverage from this one experiment.

For a fixed learned price or delta and reference interval [a,b], its signed error interval is [learned-b, learned-a]. Delta errors and intervals are divided by the coordinate weight. An entire interval inside the target band is classified within target, an entire interval outside is outside target, and a crossing interval is unresolved at reference precision.

### Neural fits and initial errors

Both solvers use 3,000 Adam updates, 128 fresh paths per update, width 16, 16 time intervals and a 180-second per-fit soft time cap. Validation has 256 fixed paths, post-fit evaluation has 2,048 paths, learning rate is 0.001 and the existing rate reduction applies after 70% of updates. Value-gradient uses terminal weight one. Validation selection rules retain the minimum scheduled total loss for value-gradient and minimum scheduled terminal loss for terminal fitting. Numerical-reference accuracy never selects the checkpoint.

| Selected model | Parameters | Selected / completed update | Price | Signed error | 0.01 target |
| --- | ---: | ---: | ---: | ---: | --- |
| Value-gradient | 481 | 3,000 / 3,000 | 6.54517829 | +0.13899423 | Missed |
| Terminal loss | 9,281 | 2,900 / 3,000 | 6.41770489 | +0.01152083 | Missed |

Both price-error intervals lie outside the 0.01 target. The terminal model is close to its boundary, with signed error interval [0.01115890, 0.01188275]. All ten value-gradient scaled-delta intervals lie outside their 0.01 bands. Terminal loss has nine outside and one unresolved, at coordinate two. No coordinate set meets all targets.

| Asset | Numerical delta | Value-gradient delta | Value-gradient scaled error | Terminal delta | Terminal scaled error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.06205644 | 0.06332829 | +0.01271847 | 0.05536382 | -0.06692621 |
| 2 | 0.06205976 | 0.07515089 | +0.13091129 | 0.06307560 | +0.01015848 |
| 3 | 0.06205965 | 0.05442991 | -0.07629742 | 0.06578262 | +0.03722972 |
| 4 | 0.06205727 | 0.06730738 | +0.05250114 | 0.06402795 | +0.01970688 |
| 5 | 0.06205440 | 0.06487484 | +0.02820446 | 0.05918592 | -0.02868474 |
| 6 | 0.06205936 | 0.05586342 | -0.06195940 | 0.06438542 | +0.02326057 |
| 7 | 0.06205971 | 0.05385863 | -0.08201079 | 0.05661401 | -0.05445697 |
| 8 | 0.06205793 | 0.05453418 | -0.07523751 | 0.06686803 | +0.04810103 |
| 9 | 0.06205698 | 0.06431697 | +0.02259994 | 0.05292142 | -0.09135559 |
| 10 | 0.06205671 | 0.05620806 | -0.05848653 | 0.06334506 | +0.01288345 |

The true coordinate deltas are equal by permutation symmetry of the financial case. The separate numerical coordinate estimates and learned predictions remain visible, without symmetry averaging. The largest absolute scaled-delta errors are 0.13091129 for value-gradient and 0.09135559 for terminal loss. Every learned initial delta lies inside the arithmetic-call bounds \(0\le\Delta_i\le w_i e^{-q_iT}=0.10\). Bound satisfaction supplies a weaker check than accuracy.

Initial diffusion condition number is 2.29907, smallest singular value is 16.73320 and inverse norm is 0.05976143. Maximum control reconstruction residual is below 1.1e-15. These algebraic checks establish correct recovery for the supplied controls, while the independent reference reveals their remaining financial errors.

Value-gradient terminal-payoff RMSE is 0.73003405 on its post-fit paths. Terminal-loss discounted terminal RMSE is 1.76839287. They measure different predictions and discounting conventions. The solver reports record missing conditional arithmetic reference values, path deltas and analytical-control comparisons as null. No arithmetic path-risk RMSE or global terminal-model gamma is inferred from the initial reference.

### Resource measurement and scheduling

`ArithmeticPilot` owns the declared design, sequential worker execution and final comparison. The reference and each solver run in fresh Python subprocesses with one-thread BLAS/OpenMP settings. PyTorch uses one CPU thread and float64. Solver fitting time covers the optimisation loop, paired evaluation covers selected and final checkpoints, and process wall time includes startup, imports, calculation and saving.

Peak resident memory includes libraries, data, autograd storage and allocator retention. Linux reads the executed image's `VmHWM` because `ru_maxrss` can retain a controller fork/exec startup floor. macOS uses `getrusage` with byte units. Each resource record states its source and baseline high-water mark. Controller memory is measured separately and other applications need additional headroom. This is resident memory, without a claim to count only tensor allocations or the whole machine's memory use.

The recorded workspace uses Linux x86-64, Python 3.12.14, NumPy 2.3.5 and PyTorch 2.15.0+cpu. The following costs are one quiet sequential measurement of the declared workload. They require separate MacBook verification and do not establish a hardware speed comparison.

| Task | Fit seconds | Paired evaluation seconds | Process wall seconds | Peak resident MiB |
| --- | ---: | ---: | ---: | ---: |
| Numerical reference | N/A | N/A | 1.8182 | 179.54 |
| Value-gradient | 13.5022 | 0.1051 | 15.9518 | 376.18 |
| Terminal loss | 20.9084 | 0.0471 | 23.3203 | 338.66 |

Controller peak resident memory is 256.54 MiB. The complete pilot takes 41.1304 seconds. Repeating both exact fits for three seeds projects 103.23 seconds of fitting or 103.69 seconds including the paired post-fit evaluations. References, process startup, tuning, higher dimensions and later stresses require additional allocation. Dimension scaling is not extrapolated from this single case. The later experiment lock will use the MacBook measurements to fix the actual schedule.

### Code contract and reproduction

`arithmetic_ten_case` owns the financial specification. `BasketReferencePricer` supplies the existing numerical contributions, geometric controls and sampling-unit uncertainty. `FBSNNSolver` and `TerminalBSDESolver` fit the payoff without an analytical oracle. `SolverRiskComparison.initial_risk` supplies the physical initial delta and diffusion diagnostics for the final independent check. Numerical reference values appear only in the pilot comparison report. Existing vanilla and geometric report shapes and analytical evaluations retain their established contracts.

| Array | Shape in this pilot |
| --- | --- |
| Training states | (128, 17, 10) |
| Independent Brownian increments | (128, 16, 10) |
| Learned value-gradient outputs | (128, 17) |
| Value-gradient coordinate deltas | (128, 16, 10) |
| Terminal discounted controls | (128, 16, 10) |
| Reference contribution vector per terminal state | (11,) for price and ten deltas |

```bash
python -m bsde_pricing describe --case arithmetic-ten
python -m bsde_pricing arithmetic-pilot
python -m unittest discover -s tests -v
```

The default protected directory is `scratch/runs/arithmetic_ten_pilot`. `design.json` is written before calculations. `reference.json` retains both Sobol sizes, the independent IID estimate, coefficients, replicate means, intervals and budget checks. Solver subdirectories retain selected checkpoints, final or continuation checkpoints and full histories. Resource records and logs describe each isolated task. The final `report.json` records source model/report digests, initial comparisons, runtime and budget projections. A failed child or comparison leaves a task-specific `failure.json` and retained evidence. A time-limited fit remains labelled time-limited. Existing outputs are never overwritten.

All 91 tests pass with user warnings treated as errors in the workspace. New checks cover all ten arithmetic pathwise deltas against matched price bumps, reference-free fitting, validation selection, checkpoint round trips, uncertainty propagation in weight-scaled units, isolated subprocess output, protected directories, failure records and platform memory units. Existing scalar, geometric, autograd, continuation and risk tests also pass. The notebook executes all 27 code cells from a fresh IPython process with 19 figures. Three new code cells and three figures cover this pilot, preserving the exact earlier source and MacBook output cells. The repository audit reports zero errors and three curated-notebook review items.

The implementation, workspace pilot and MacBook evidence are verified.

### MacBook verification

The MacBook pilot completes both 3,000-update fits and all 91 tests pass in 6.484 seconds. It uses macOS 27.0.1 ARM64, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0 on CPU float64. Validation selects updates 3,000 and 2,900, matching the workspace. Initial prices and delta errors agree to floating-point precision. Standardised independent-reference differences agree within 1.6e-12, including the retained coordinate discrepancies.

The primary price remains 6.40618406 with pointwise 95% half-width 0.00036192. Both reference precision allocations are met. Price errors remain +0.13899423 for value-gradient and +0.01152083 for terminal loss. Both miss the original 0.01 price target. All ten value-gradient weight-scaled delta intervals are outside their target bands. Terminal loss has nine outside and coordinate two unresolved. Initial bounds and diffusion reconstruction pass. These results retain the distinction between algebraic correctness and learned financial accuracy.

Fitting takes 8.9192 seconds for value-gradient and 17.1385 seconds for terminal loss. Worker peak resident memory is 339.05 MiB and 308.08 MiB respectively, with 336.56 MiB for the reference and a separate controller peak of 210.13 MiB. The complete pilot takes 31.8233 seconds. Repeating this exact pair of fits for three seeds projects 78.60 seconds including paired post-fit evaluation. Startup, references, tuning, higher dimensions and other applications require additional allocation. These are scheduling observations for this MacBook workload.

The supplied terminal summary agrees with the attached full report. The pushed notebook has unchanged source, 27 sequential executed code cells and 19 complete figures with no error or stderr outputs. Its first 16 figures match the previous verified outputs byte for byte. The three arithmetic figures and nine tables retain reference uncertainty, all coordinate errors, independent-method discrepancies and local resource costs. [Notebook 7700427](https://github.com/KVInator/deep-bsde-pricing/commit/7700427371d9fb921bc88f016dd28303bce3cabd)

## Locked experiment design

### Scope and workload

`experiments/design.json` is the authoritative machine-readable specification. `ExperimentDesign` validates its financial cases, configuration and stream separation, then enumerates the reserved jobs. The canonical sorted-JSON SHA-256 fingerprint identifies the exact design. Every later run must retain this fingerprint with its source commit and configuration. A changed fingerprint requires a documented protocol amendment before final evaluation. The present command validates and exports a plan. Training, sampling, resource enforcement and result collection are implemented in their later stages.

The six primary cases are vanilla, the existing unequal five-asset geometric basket, and equal-weight arithmetic baskets with 10, 25, 50 and 100 assets. Arithmetic base inputs retain spots and strike 100, maturity one year, rate 3%, volatility 20%, no dividends and correlation 0.30. The unequal geometric case retains its full existing specification. No Cartesian product of parameters is scheduled.

| Comparison | Cases and grids | Fits across both solvers and three seeds |
| --- | --- | ---: |
| Primary dimension ladder | Six cases, 16 intervals each | 36 |
| Time-grid sensitivity | Arithmetic 10, additional 10, 20 and 40 intervals | 18 |
| Financial stresses | Arithmetic 10 with strike 90, strike 110, maturity 0.25 or correlation 0.80, changed separately | 24 |
| Total | Ten financial cases and thirteen case/grid combinations | 78 |

The time-grid study reuses the 16-interval baseline. Numerical initial references are reused across grids of the same financial case. A fixed-start neural model is refitted for every changed financial case. Every row includes both formulations, every declared seed and any failure. An unavailable high-dimensional case stays in the results with its reason.

### Training and reserved streams

The common training configuration remains 3,000 updates, width 16, batch 128, learning rate 0.001, 256 validation paths and 2,048 post-fit evaluation paths. CPU float64 and one CPU thread remain the baseline. Value-gradient terminal weight is one, patience is disabled and the existing learning-rate schedule applies. Terminal controls retain separate time-step networks. Parameter counts vary with dimension and grid, and are reported explicitly. The lock allows zero further architecture or hyperparameter trials. Development investigates correctness and scaling under this configuration, retaining misses.

Development uses seeds 2026, 2027 and 2028. Final training reserves 41001, 41002 and 41003. Within each training seed, SeedSequence children zero, one and two retain training, validation and post-fit ownership. The two solvers share the declared forward stream for paired comparisons. A seed is reused across cases deliberately, while each financial case is a distinct fit. Across-seed spread is reported without selecting the best seed.

For zero-based case index j in the specification, final reference, independent reference check and common-path evaluation seeds are 42000+j, 42500+j and 43000+j. Coefficient-pilot and final estimator-evaluation seeds are 48000+j and 49000+j. Within estimator streams, method and repetition identifiers are explicit spawn-key components. Common evaluation paths may pair frozen solvers. A coefficient pilot must never share its evaluation increments. The validator checks disjoint top-level seed namespaces without instantiating a generator. Final namespaces remain unused until the final suite.

Checkpoints minimise the solver's scheduled validation objective. Held-out price or delta error never selects weights, updates, architecture or seeds. Selected and final-update diagnostics remain distinguishable. A bug correction records both the failed attempt and the corrected source, with the same declared configuration and seed. Accuracy misses do not authorise more training or a different seed.

### Targets, uncertainty and conclusions

For initial price p and coordinate deltas, the original targets remain

\[
|\widehat p-p_*|\le 0.01,\qquad
\left|\frac{\widehat\Delta_i-\Delta_{*,i}}{w_i}\right|\le0.01.
\]

Dividing by the basket weight avoids making a small raw coordinate error appear automatically accurate as dimension grows. Analytical references have no simulation interval. Numerical-reference pointwise 95% half-widths must be at most 0.002 for price and 0.002 for each weight-scaled delta. A learned estimate v and reference interval [a,b] produce signed error interval [v-b,v-a], with weight scaling applied to delta. Whole-interval inclusion is within target, whole-interval exclusion is outside target, and boundary crossing is unresolved. Reference precision failure makes that comparison inconclusive. Pointwise intervals supply no simultaneous coverage claim across coordinates or cases.

Report each seed, signed error, maximum absolute weight-scaled delta error, directional delta sum, initial bound violations and diffusion conditioning/reconstruction. Across seeds report mean, sample standard deviation and range, keeping individual failures. Price accuracy and delta accuracy are separate conclusions. Conditional path-risk RMSE is reported only where a justified conditional reference exists. Gamma remains a selected vanilla value-gradient diagnostic, with derivative consistency and financial accuracy assessed separately.

The arithmetic reference uses the existing geometric-control Sobol and antithetic contributions. The predeclared allocation ladder is 16 scrambles of 2^16 points followed, only if needed, by 2^18 points. The independent antithetic check has 131,072 then 524,288 pairs. Each method has a separate 4,096-pair coefficient pilot. At the first level meeting every required half-width and the four-combined-standard-error method-agreement screen, retain that reference. Otherwise advance once, then label the reference unresolved if requirements still fail. Retain both attempted levels, coefficients, scramble means and method differences. The second level uses the same scramble seeds and is dependent on the first. The screen is a discrepancy diagnostic, with no assertion that it proves confidence-interval coverage.

Larger reference allocations require bounded chunks of 4,096 terminal states and streaming statistics in the later implementation. Never allocate the whole largest high-dimensional sample merely because its point count is declared. Reference generation has its own time and memory limits and cannot borrow unused neural budgets to obtain a favourable conclusion.

### Resource rules and cost accounting

The MacBook ten-asset pilot measures 8.9192 and 17.1385 seconds of fitting, with peak worker memory below 340 MiB. Three repetitions of this exact pair project 78.60 seconds including paired evaluation. Those observations justify a small common baseline. They supply no linear extrapolation to 100 assets or 40 time intervals.

The declared fitting soft cap is 180 seconds per fit. Each isolated neural worker has a 300-second total wall limit and 2,048 MiB resident-memory limit. Reference workers have 180 seconds per allocation level and the same memory limit. Execution is sequential. Conditional path evaluation uses chunks of at most 128 paths to bound batched diffusion and derivative storage. A future runner must monitor worker RSS, stop a limit breach and retain its resource-limit status. Current pilot commands retain their existing behaviour and do not implement the newly declared memory guard.

The 78 scheduled fits give a maximum fitting allowance of 3.9 hours and a neural-worker wall allowance of 6.5 hours. Eight numerical financial references at two levels have at most 48 minutes of worker time. The full final study has an eight-hour wall ceiling across resumable sessions, including references and estimator comparisons. These are ceilings, rather than predicted runtimes. A ceiling reached before later rows produces explicit not-attempted-budget rows. The fixed matrix is retained. No case is quietly removed and no limit is raised after inspecting final accuracy.

Costs retain optimisation, checkpoint validation, final evaluation, reference construction, coefficient fitting, process startup and inference separately. Include the earlier development/tuning cost in any end-to-end efficiency claim. A single-query comparison includes training. An amortised comparison states the repeated workload and exactly which fixed-case checkpoints are reused. No hardware or precision speed ranking follows from unmatched workloads.

### Estimator comparison and failure reporting

The bounded estimator study uses arithmetic 10 at antithetic-pair allocations 4,096, 16,384 and 65,536 with 16 independent repetitions. Compare antithetic/geometric-control MC, scrambled Sobol/geometric-control estimates and frozen controls from both neural formulations. Known analytical martingale controls are first verified on the geometric-five case, whose price and controls are available exactly. They are not presented as exact arithmetic-basket controls. Sobol estimator allocations use equal total terminal-state counts, with 16 independent scrambles. Pair counts and scramble counts remain their appropriate uncertainty units.

Any fitted martingale coefficient uses an independent 4,096-pair pilot. The learned network is frozen before these paths. Report variance, standard error, mean error against the independent reference and total elapsed cost. Variance reduction that fails, increases variance or cannot amortise training stays visible. Left-endpoint adaptation and zero-mean-integral checks precede learned-control evaluation. The final repetitions use the reserved estimator namespace and remain untouched during development.

Every planned result row receives completed, accuracy-miss, reference-inconclusive, time-limit, memory-limit, numerical-failure or not-attempted-budget status, with raw diagnostics and provenance. Several labels can apply to a completed calculation, so execution status and accuracy assessment are separate fields. There is no aggregate success claim that excludes failures or unresolved references. Further architectures, full stress grids, nonlinear drivers and extra seed searches require separately scoped work after this core release.

### Validate the plan

```bash
python -m bsde_pricing experiment-design --output scratch/experiment_design.json
python -m unittest discover -s tests -v
```

The console prints the design fingerprint, counts and budgets. The JSON export additionally lists all reserved jobs. This step starts no training and draws no final random samples. It adds no notebook output because it introduces a protocol rather than another numerical experiment.


## Twenty-five-asset arithmetic development

### Financial scaling and the comparison question

This case increases the arithmetic basket from 10 to 25 assets under the version-one experiment specification. Spots and strike remain 100, maturity one year, rate 3%, volatility 20%, dividends zero and pairwise correlation 0.30. Every weight is 0.04. The basket retains a 100-unit initial level as dimension increases. Both formulations use the same locked width, batch, grid, update schedule and three development seeds. No extra architecture, terminal weighting or seed search is introduced.

Increasing dimension also changes diversification. For basket level \(A=\sum_iw_iS_i\), define instantaneous basket exposures \(b_i=w_iS_i/A\). The instantaneous relative variance is

\[
\frac{d\langle A\rangle_t}{A_t^2dt}
=(b\odot\sigma)^\top\rho(b\odot\sigma).
\]

At the equal-spot initial state with equal volatilities and weights, this reduces to

\[
v_0=\sigma^2\left(\rho+\frac{1-\rho}{d}\right).
\]

It falls from 0.0148 at ten assets to 0.01312 at 25 assets. This is an initial instantaneous arithmetic-basket variance and the geometric basket's log-variance rate under the same equal weights. It is not the arithmetic basket's terminal log variance. For the equal-carry arithmetic case the exact terminal relative variance is

\[
\frac{\operatorname{Var}(A_T)}{\mathbb E[A_T]^2}
=\frac{e^{\sigma^2T}-1}{d}
+\left(1-\frac1d\right)(e^{\rho\sigma^2T}-1).
\]

The report evaluates the joint-lognormal formula using the model's full covariance matrix. Correlation eigenvalues are 0.70 with multiplicity 24 and 8.20 once. Initial diffusion condition number is therefore \(\sqrt{8.20/0.70}\). Dimension, effective variance and conditioning are distinct characteristics of the problem.

### Streaming the existing reference

`BasketReferencePricer` accepts an optional chunk size for both reference methods. The default whole-sample API retains its previous results. Chunked Sobol evaluation draws consecutive blocks from each scrambled net and accumulates target/control sums. It preserves the complete net and its scramble-level sampling unit. It does not average independent chunks as if they were new scrambles.

For IID antithetic pair averages, each block contributes a count, mean and sum of squared deviations. Two groups with counts \(n_a,n_b\), means \(m_a,m_b\) and centred sums \(M_{2,a},M_{2,b}\) merge as

\[
\delta=m_b-m_a,\quad n=n_a+n_b,\qquad
m=m_a+\frac{n_b}{n}\delta,
\]
\[
M_2=M_{2,a}+M_{2,b}+\frac{n_an_b}{n}\delta^2.
\]

The sample variance is \(M_2/(n-1)\), and the standard error divides its square root by \(\sqrt n\). Antithetic pairs remain the independent units. The final partial block is included with its actual count. Neither method retains the full terminal sample. Independent coefficient pilots remain separate and the known geometric-control expectations retain their analytical calculation.

The development study uses chunks of 4,096 pair averages or Sobol states. Its first reference allocation is 16 independent scrambles of 2^16 points, checked against 131,072 antithetic pairs. If either precision or the declared method-agreement check fails, the next allocation is 2^18 points with 524,288 IID pairs. Coefficient pilots have 4,096 independent pairs. The two allocation levels share reference seeds and are dependent. Development reference seeds are 2032 and 2033, outside the reserved final namespaces.

The retained reference uses power 16. Its price is **6.13274065** with pointwise 95% half-width **0.00043547**. The largest weight-scaled delta half-width is **0.00049214**. Reference usability is `true` under the original 0.002 half-width allocations and four-combined-standard-error agreement screen. The largest absolute independent-method standardised difference is 1.42740. The report retains every coordinate and approximate pointwise interval, with no simultaneous coverage claim.

### Six development fits

Development seeds are 2026, 2027 and 2028. Every fit uses width 16, 128 paths per update, 16 intervals and 3,000 Adam updates, with 256 fixed validation paths and 2,048 post-fit paths. Initial states have shape (128,17,25), and independent Brownian increments have shape (128,16,25). The value-gradient network has 721 parameters. The separate terminal-control subnetworks have 16,721 parameters. Checkpoint selection retains each solver's minimum scheduled validation objective. Reference prices and deltas enter only the independent comparison process after fitting.

| Seed | Model | Selected / completed update | Initial price error | Price classification | Largest absolute scaled-delta error |
| --- | --- | ---: | ---: | --- | ---: |
| 2026 | Terminal loss | 2900 / 3000 | -0.02792348 | outside_target | 0.16524410 |
| 2026 | Value-gradient | 2900 / 3000 | +0.00140665 | within_target | 0.22682738 |
| 2027 | Terminal loss | 2500 / 3000 | -0.01343619 | outside_target | 0.22284046 |
| 2027 | Value-gradient | 2900 / 3000 | +0.00502464 | within_target | 0.19893607 |
| 2028 | Terminal loss | 2800 / 3000 | +0.03211951 | outside_target | 0.16015147 |
| 2028 | Value-gradient | 3000 / 3000 | +0.04720704 | outside_target | 0.07709504 |

Value-gradient meets the 0.01 price target at two seeds and misses it at the third. Terminal loss misses the price target at all three. Neither formulation meets all 25 weight-scaled delta targets at any seed. The mean terminal-model price error is close to zero because errors change sign across seeds. It does not establish accurate individual runs. Three seeds provide a descriptive development comparison, without a robust superiority claim.

| Model | Mean signed price error | Sample standard deviation | Price-error range | Worst-coordinate scaled-error range |
| --- | ---: | ---: | --- | --- |
| Terminal loss | -0.00308006 | 0.03133253 | [-0.02792348, +0.03211951] | [0.16015147, 0.22284046] |
| Value-gradient | +0.01787944 | 0.02546279 | [+0.00140665, +0.04720704] | [0.07709504, 0.22682738] |

Initial arithmetic delta bounds, directional delta sums, all coordinate errors and control-reconstruction diagnostics remain in each model record. A small reconstruction residual verifies the diffusion solve for the supplied controls. It does not resolve learned-delta errors. Conditional arithmetic path references are still unavailable, so the study reports no arithmetic path-risk RMSE or comparable global terminal-solver gamma.

### Resource budgets and retained evidence

The shared pilot worker now deserialises the declared financial case. `ArithmeticDevelopmentStudy` orchestrates one shared reference and the three paired fits using the specification's common configuration. Every run retains the specification fingerprint and SHA-256 digests of the package source files. Numerical kernels, training objectives and physical initial-risk recovery remain in their existing components.

The parent runs workers sequentially and polls resident memory every 0.2 seconds. Linux uses `/proc` resident memory and macOS uses `ps` RSS in KiB. It additionally checks the worker's own lifetime peak after exit. A 2,048 MiB breach or a wall-time limit terminates the worker and retains its guard and failure records. A rapid memory spike may only be identified from the completed worker's peak. This sampled RSS guard is a resource check, rather than an operating-system allocation quota.

Neural fits retain their 180-second soft fitting cap and a 300-second worker wall cap. Each reference allocation has a 180-second wall cap. Comparison has its own bounded worker. A timeout, memory failure, numerical failure or unusable reference remains in the report. A noisy reference marks model accuracy comparisons inconclusive. Raw point estimates remain available. Worker-wall cost includes all recorded attempts, including failures, and is separate from measured optimisation time.

| Seed | Model | Fit seconds | Paired evaluation seconds | Worker wall seconds | Peak resident MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026 | Terminal loss | 25.9822 | 0.0913 | 28.3590 | 362.00 |
| 2026 | Value-gradient | 19.1822 | 0.1694 | 21.6952 | 400.18 |
| 2027 | Terminal loss | 25.6876 | 0.0902 | 28.0746 | 362.52 |
| 2027 | Value-gradient | 18.8416 | 0.1734 | 21.3430 | 408.39 |
| 2028 | Terminal loss | 25.6931 | 0.0927 | 28.0510 | 362.90 |
| 2028 | Value-gradient | 19.0659 | 0.1679 | 21.4952 | 400.04 |

All six workspace fits complete 3,000 updates. The largest training-worker peak is 408.39 MiB, with controller peak 240.42 MiB recorded separately. Total measured study wall time is 155.8639 seconds. The runtime is Linux x86-64, Python 3.12.14, NumPy 2.3.5 and PyTorch 2.15.0+cpu, using CPU float64 and one thread. These are workspace measurements. MacBook results and resource use require separate verification.

The retained implementation-check and measured-reproduction runs use identical settings and seeds. Their combined cost belongs to development cost. The common configuration transfers computationally to this 25-asset case under the declared limits. Its accuracy remains mixed for price and insufficient for the coordinate-risk targets. A higher dimension is not automatically a more difficult economic instance because diversification changes as well.

### Reproduction and checks

```bash
python -m bsde_pricing development-study
python -m unittest discover -s tests -v
```

The protected default directory is `scratch/runs/arithmetic_25_development`. It contains the declared design, every attempted reference level, all three seed directories, full training histories, selected and final/continuation checkpoints, comparison records, task logs and resource guards. `report.json` is updated after each seed. Final training and estimator streams remain untouched. The locked experiment JSON and its fingerprint remain unchanged.

All 98 tests pass with user warnings treated as errors. New checks compare whole-sample and chunked estimates, replicate means and uncertainty, exercise a non-divisible final IID block, verify chunk allocation bounds, check macOS RSS units, stop actual time- and memory-limited processes, retain failed rows and run a small paired study through save/load. Existing vanilla, geometric, training and continuation regressions pass.

Notebook sections 26-28 load the saved study without fitting. Three new code cells and three figures cover reference precision, all six models' 25-coordinate errors, seed variation and resource cost. Earlier source and MacBook outputs are preserved exactly. The implementation, workspace study and MacBook evidence are verified.

### MacBook verification

The MacBook study completes all six 3,000-update fits under the declared limits. All 98 tests pass in 15.978 seconds. The run uses macOS 27.0.1 ARM64, Python 3.12.16, NumPy 2.5.4 and PyTorch 2.15.0 with CPU float64 and one thread. The protocol fingerprint and all 23 package-source digests match the workspace run. Selected steps and price/delta findings reproduce to floating-point precision. Standardised independent-reference diagnostics differ by at most approximately 1.1e-12.

The shared reference remains 6.13274065 with pointwise 95% half-width 0.00043547. The largest scaled-delta half-width is 0.00049214. Precision and agreement checks pass at the first allocation. Value-gradient meets the price target at seeds 2026 and 2027 and misses it at 2028. Terminal loss misses it at all three. Neither formulation meets every weight-scaled coordinate-delta target at any seed. All six initial bound checks pass. These outcomes preserve the distinction between an affordable fit and accurate coordinate risk.

| Seed | Value-gradient fit seconds | Terminal-loss fit seconds | Value-gradient worker peak MiB | Terminal-loss worker peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 12.6225 | 17.7081 | 382.95 | 372.08 |
| 2027 | 11.7282 | 18.3624 | 442.44 | 367.55 |
| 2028 | 11.8426 | 18.0666 | 466.45 | 389.48 |

Total study wall time is 107.2763 seconds. Reference-worker peak is 142.98 MiB and controller peak is separately 210.11 MiB on the 8 GiB machine. Every measured worker remains below the 2,048 MiB limit. These observations support the next bounded development attempt, with no extrapolated accuracy or resource guarantee for larger dimensions.

The supplied terminal summary matches the full report. The pushed notebook changes outputs only, with unchanged source, 30 sequential executed code cells and 22 complete figures without error or stderr outputs. The first 19 figures are byte-identical to the preceding verified notebook. The three new figures and eight tables retain all seeds, coordinate errors, reference precision, checkpoint selections and resource use. [Notebook ad0cde4](https://github.com/KVInator/deep-bsde-pricing/commit/ad0cde4c9c57b0c8b399c3113cc15c286f21ad8a)
