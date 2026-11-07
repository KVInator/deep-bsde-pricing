# Pricing, risk and estimator costs

## Model

For basket value $B_T$ and payoff $g(S_T)=(B_T-K)^+$, the time-$t$ price is $u(t,s)=e^{-r(T-t)}\mathbb E[g(S_T)\mid S_t=s]$ under the risk-neutral measure. The weights are positive and sum to one. The basket is the arithmetic sum $A_t=\sum_i w_i S_t^i$ or geometric mean $G_t=\prod_i(S_t^i)^{w_i}$. Rates and dividends compound continuously. Time is in years and volatility is annualised.

Take $LL^\top=\rho$ and independent Brownian coordinates $W$. The diffusion is $\Sigma(s)=\operatorname{diag}(s_i\sigma_i)L$. Over an interval of length $h$ the exact transition is

$$
S_{t+h}^i=S_t^i\exp\left[(r-q_i-\tfrac12\sigma_i^2)h+\sigma_i(L\Delta W)_i\right],
\qquad \Delta W\sim N(0,hI).
$$

Terminal-payoff references sample the terminal state directly. The neural and martingale methods also simulate intermediate states to evaluate their controls. Grid refinement changes the stochastic-integral approximation and the learning problem while preserving exact forward transitions.

## Reference prices and Greeks

Black–Scholes gives the vanilla price, spot delta and gamma. The finite-difference solver uses Crank–Nicolson on a bounded spot interval and two backward-Euler half steps per startup interval to damp oscillations from the payoff kink. It uses the large-spot call asymptote at the upper boundary. The classical notebook refines space and time together with a fixed upper boundary.

For the geometric basket, write $a_i=w_i\sigma_i$, $v=a^\top\rho a$ and $m=\sum_i w_i(r-q_i-\sigma_i^2/2)$. The log return over remaining time $\tau$ is normal with mean $m\tau$ and variance $v\tau$. The scalar Black–Scholes price at $G_t$ uses volatility $\sqrt v$ and dividend yield $q_{\rm eff}=r-m-v/2$. Multiply its spot delta by $w_iG_t/S_t^i$ for coordinate $i$.

For an arithmetic call with positive remaining variance, the payoff kink has probability zero. The spot derivative is bounded by a constant times a lognormal factor with finite first moment. Differentiation under the expectation gives

$$
\Delta_i=e^{-r\tau}\mathbb E\!\left[\mathbf1_{\{A_T>K\}}\,w_i S_T^i/S_t^i\mid S_t=s\right].
$$

A second derivative also requires the boundary contribution from the indicator. The suite evaluates coordinate deltas and the vanilla value-gradient gamma. The sum of deltas is the derivative along an equal absolute shift in all spots.

The numerical arithmetic reference uses a geometric payoff control for price and geometric pathwise controls for deltas. Independent pilots fit the coefficients. Antithetic uncertainty uses independent pair averages. Randomised Sobol uncertainty uses Student-$t$ intervals across independently scrambled net means. An arithmetic reference qualifies when its price half-width and every weight-scaled delta half-width are at most 0.002, and its differences from a separate antithetic estimate are within four combined standard errors. A second allocation is available if needed. These intervals are pointwise.

## Neural formulations

The pricing PDE is $\partial_tu+\mathcal Lu-ru=0$ with terminal condition $u(T,s)=g(s)$ and risk-neutral generator $\mathcal L$. Itô's formula gives $dY_t=rY_t\,dt+Z_t^\top dW_t$ with $Y_t=u(t,S_t)$ and $Z_t=\Sigma(S_t)^\top\nabla_su(t,S_t)$. Discounting both $Y$ and $Z$ by $e^{-rt}$ removes the drift.

- **Value-gradient.** A shared smooth network represents $u_\theta(t,s)$. Autodifferentiation uses physical spot units. Each local residual is $R_j=\widetilde Y_{j+1}-\widetilde Y_j-\widetilde Z_j^\top\Delta W_j$. With terminal residual $R_T=\widetilde Y_T-e^{-rT}g(S_T)$ and terminal weight one, the loss averages $(\sum_j R_j^2+R_T^2)/K^2$ across paths.
- **Terminal loss.** A free initial scalar and initial control are fitted with one control network per later grid node. The objective is the mean squared difference between $Y_0+\sum_j\widetilde Z_j^\top\Delta W_j$ and the discounted payoff, divided by $K^2$. Additional time intervals add control networks.

Spot delta for the terminal solver solves $\Sigma(s)^\top\Delta=e^{rt}\widetilde Z_t$. A small reconstruction residual checks the linear algebra. Independent reference deltas assess economic accuracy. The value network supplies vanilla gamma through its second spot derivative. The terminal solver's scalar price is specific to its starting state.

## Locked evaluation

The design fixes width 16, batch size 128, up to 3,000 updates, and 16 base time intervals. Three extra grids at 10, 20 and 40 intervals apply to arithmetic ten. Checkpoints minimise the solver's own scheduled validation objective. Final price, risk and estimator evaluations use separate reserved streams.

A fit must meet a 0.01 absolute price target and a 0.01 absolute error target for each $\Delta_i/w_i$. For signed error $e$ and reference half-width $h$ in matching units, $|e|+h\le0.01$ establishes a pass and $|e|-h>0.01$ establishes a miss. Overlap is unresolved. Reference uncertainty and variation across training seeds measure different quantities. Execution status records limits and failures separately from accuracy.

For equal weights and fixed equicorrelation, the arithmetic basket's initial relative variance is $\sigma^2[\rho+(1-\rho)/d]$. The dimension ladder therefore changes diversification along with dimension. The grid study combines changes in discretisation, optimisation and terminal-network capacity at a fixed training budget.

## Martingale controls and matched cost

For a frozen square-integrable adapted control, $I=\sum_j\widetilde Z_j^\top\Delta W_j$ has mean zero. Averaging $X-\beta I$ for discounted payoff $X$ preserves the price when the fitted coefficient and model are independent of evaluation draws. A poor coefficient or integrand can increase variance. The known-control check reports the mean of $I$ and the mean of $I^2-\sum_j\|\widetilde Z_j\|^2\Delta t_j$ against zero.

The ten-asset arithmetic benchmark compares two learned formulations with the geometric Brownian integrand and with conventional geometric-payoff controls. It schedules 16 independent repetitions per pair of final checkpoints. The terminal-state budget is matched across antithetic pairs and scrambled nets. Martingale methods also evaluate the intermediate path grid. The smallest measured allocation whose interval half-width meets the target in all 16 repetitions qualifies. Targets are 0.025 and 0.01 price units.

The cost of $q$ fresh estimates of the same contract is $C(q)=C_{\rm upfront}+qC_{\rm eval}$. Upfront cost includes measured training, checkpoint loading and a coefficient pilot where applicable. The same fit and coefficient are reusable under that fixed-contract assumption. Break-even is assessed against each qualified conventional comparator using the recorded timings. The timed scope begins with final fitting. The recorded neural workload uses CPU float64 with one PyTorch thread.

## Method references

- Han, Jentzen and E (2018), [Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568).
- Raissi (2018), [Forward-backward stochastic neural networks](https://arxiv.org/abs/1804.07010).
