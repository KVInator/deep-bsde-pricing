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
