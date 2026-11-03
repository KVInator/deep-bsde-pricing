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

The terminal distribution derived above permits direct sampling for a European payoff. `GBMModel.sample_terminal(initial_spots, horizon, n_paths, rng)` initially implements the one-asset law and returns shape $(N,1)$. Each stochastic terminal sample uses one independent standard normal draw. There is no forward time grid or Euler approximation. `MonteCarloPricer` calls that model method and the existing contract payoff; its estimator does not call the analytical pricing engine.

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
