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
