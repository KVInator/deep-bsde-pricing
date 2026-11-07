# Pricing, risk and estimator costs

## Model

The clean price of a European call is
$u(t,s)=e^{-r(T-t)}\mathbb E[(B_T-K)^+\mid S_t=s]$ under the risk-neutral measure. The basket is either the weighted arithmetic sum $A_t=\sum_i w_i S_t^i$ or the weighted geometric mean $G_t=\prod_i(S_t^i)^{w_i}$. The weights are positive and sum to one. Rates and dividends compound continuously. Time is in years and volatility is annualised.

With independent Brownian coordinates $W$ and $LL^\top=\rho$, the diffusion matrix is $\Sigma(s)=\operatorname{diag}(s_i\sigma_i)L$. The exact transition is

$$
S_{t+h}^i=S_t^i\exp\left[(r-q_i-\tfrac12\sigma_i^2)h
             +\sigma_i(L\Delta W)_i\right],
\qquad \Delta W\sim N(0,hI).
$$

This lets a terminal-payoff reference sample the terminal state directly. The neural and martingale methods use intermediate states because their stochastic integrals need left-endpoint controls. Changing their grid changes the integral approximation and the learning problem while preserving exact forward transitions.

## Reference prices and Greeks

### Vanilla and geometric calls

Black–Scholes gives the vanilla price, spot delta and gamma. The finite-difference reference solves the same PDE on a bounded spot interval using Crank–Nicolson with two backward-Euler half steps in each startup interval. This damps oscillations introduced by the payoff kink. The upper boundary is the large-spot call asymptote. The classical notebook jointly refines the time and space grids at fixed upper boundary, so residual errors include both discretisation and boundary effects.

For the geometric call, put $a_i=w_i\sigma_i$, $v=a^\top\rho a$ and $m=\sum_i w_i(r-q_i-\sigma_i^2/2)$. Over remaining time $\tau=T-t$, $\log(G_T/G_t)$ has mean $m\tau$ and variance $v\tau$. A scalar Black–Scholes formula with volatility $\sqrt v$ and dividend yield $q_{\rm eff}=r-m-v/2$ therefore prices the call at $G_t$. The coordinate delta is the scalar delta times $w_iG_t/S_t^i$.

### Arithmetic calls

At positive remaining variance, the payoff kink has probability zero. The spot derivative is bounded by a constant times a lognormal factor with finite first moment, so differentiation under the expectation gives

$$
\Delta_i=e^{-r\tau}\mathbb E\left[\mathbf1_{\{A_T>K\}}
                   w_i S_T^i/S_t^i\mid S_t=s\right].
$$

Differentiating the indicator a second time requires a boundary contribution. The final suite reports coordinate deltas and the vanilla value-gradient gamma. It also retains the sum of coordinate deltas, the directional derivative for an equal absolute bump to every spot.

Both numerical references use the geometric payoff and its pathwise delta contributions as controls with known expectations. Each coefficient is fitted on separate pilot draws. Antithetic uncertainty is calculated across independent pair averages. Randomised Sobol uncertainty is calculated across independently scrambled net means with a Student-$t$ interval. The final reference uses 16 scrambles and checks the price and each delta against independent antithetic estimates. An arithmetic reference qualifies when its price half-width and all weight-scaled delta half-widths are at most 0.002 and the two methods agree within four combined standard errors for each quantity. A second allocation is scheduled if needed. The intervals are pointwise.

## Neural formulations

The pricing PDE is $\partial_tu+\mathcal Lu-ru=0$, with terminal condition $u(T,s)=g(s)$ and $\mathcal L$ the risk-neutral GBM generator. Itô's formula gives $dY_t=rY_t\,dt+Z_t^\top dW_t$ with $Y_t=u(t,S_t)$ and $Z_t=\Sigma(S_t)^\top\nabla_s u(t,S_t)$. For $\widetilde Y_t=e^{-rt}Y_t$ and $\widetilde Z_t=e^{-rt}Z_t$, the discounted process has zero drift.

- **Value-gradient.** A smooth shared network represents $u_\theta(t,s)$. Autodifferentiation uses physical spot units. On each interval the loss penalises the discounted value increment minus the left-endpoint stochastic increment. A separate term penalises the terminal payoff error.
- **Terminal loss.** A free scalar $Y_0$ and Brownian control at time zero are trained jointly with one control network per later time node. The loss compares $Y_0+\sum_j\widetilde Z_j^\top\Delta W_j$ with $e^{-rT}g(S_T)$. More grid intervals increase the number of fitted networks.

For the value-gradient loss, let
$R_j=\widetilde Y_{j+1}-\widetilde Y_j-\widetilde Z_j^\top\Delta W_j$
and $R_T=\widetilde Y_T-e^{-rT}g(S_T)$. The path objective is
$(\sum_j R_j^2+R_T^2)/K^2$ at the locked terminal weight of one. The
terminal-loss objective squares its terminal residual and divides by
$K^2$. Both losses average over paths.

The controls live in independent Brownian coordinates. To recover spot deltas from the terminal solver at time zero, solve $\Sigma(s)^\top\Delta=\widetilde Z_0$. At later times the right-hand side is $e^{rt}\widetilde Z_t$. A small reconstruction residual checks the linear algebra. Reference deltas are needed to assess economic accuracy. The vanilla gamma is differentiated from the value network and compared with Black–Scholes. The terminal solver's scalar price is specific to the initial state.

## Locked evaluation

The [design](../experiments/design.json) fixes width 16, batch size 128, up to 3,000 updates and a 16-interval base grid. Three additional grids of 10, 20 and 40 intervals apply to the base ten-asset arithmetic case. Each fit uses the checkpoint with the lowest scheduled solver-specific validation objective. Price, risk and estimator results use separate reserved streams. Conditional path errors are reported only where the corresponding exact conditional benchmark exists.

A fit must meet the 0.01 price target and the 0.01 target for every $\Delta_i/w_i$. For a signed point error $e$ and reference half-width $h$ in matching units, $|e|+h\leq 0.01$ establishes a pass and $|e|-h>0.01$ establishes a miss. An overlap with the tolerance boundary is unresolved. The intervals describe reference uncertainty. Differences across the three fits describe sensitivity to training seed. Separate execution statuses retain numerical failures, time limits and unavailable dependencies.

The arithmetic dimension ladder uses equal weights and fixed equicorrelation. Its initial relative variance is $\sigma^2[\rho+(1-\rho)/d]$, so increasing dimension also changes diversification. The grid study compares finite-budget fits; it combines changes in stochastic-integral discretisation, optimisation and terminal-network capacity.

## Martingale controls and cost

For a frozen square-integrable adapted control, the left-endpoint sum $I=\sum_j\widetilde Z_j^\top\Delta W_j$ has mean zero. For discounted payoff $X$, the estimator averages $X-\beta I$. Conditional on independent training and coefficient-pilot data, its target remains $\mathbb E[X]$. Variance can rise if the coefficient is unsuitable. The known-control check compares the sample mean of $I$ with zero and the mean of $I^2-\sum_j\|\widetilde Z_j\|^2\Delta t_j$ with zero. Refinement improves the approximation of the analytical integrand in the two exact benchmark cases.

The ten-asset arithmetic benchmark compares both learned controls, the analytical geometric Brownian control and two terminal geometric-payoff controls. It uses 16 independent repetitions for each pair of final checkpoints. A common terminal-state budget matches $2n$ antithetic terminal states to the total states across 16 scrambled nets. Martingale methods additionally simulate and evaluate every time interval. The smallest measured allocation with half-width at most the target in all 16 repetitions qualifies. Half-width targets are 0.025 and 0.01 price units.

For repeated independent estimates of the same fixed contract, measured cost is $C(q)=C_{\rm upfront}+qC_{\rm eval}$. Upfront cost includes the recorded fit, checkpoint loading and one coefficient pilot where applicable. The benchmark records median setup and evaluation time and computes sustained break-even against each qualified conventional comparator. The same fitted model and coefficient are assumed reusable across queries. A new contract, set of inputs or starting point would require a separate cost assessment.

## Reading the results

The notebooks separate exact benchmarks, numerical-reference uncertainty, seed variation and interval precision. The [summary script](../experiments/summarize.py) derives the compact tables from the raw suite ledger. The recorded neural workloads use CPU float64 and one PyTorch thread. Runtime figures describe that platform and workload. The study covers clean European pricing under GBM and uses fixed cases and training budgets.

## Method references

- Han, Jentzen and E (2018), [Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568).
- Raissi (2018), [Forward-backward stochastic neural networks](https://arxiv.org/abs/1804.07010).
