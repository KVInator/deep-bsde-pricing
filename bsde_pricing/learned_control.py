"""Frozen Brownian controls and antithetic evaluation."""

import math
from dataclasses import replace

import numpy as np
import torch

from .bsde import PricingBSDE
from .geometric_basket import GeometricBasketReference
from .martingale import DiscountedControl, MartingaleEstimator
from .neural import value_derivatives
from .terminal_bsde import TerminalBSDESolver
from .training import FBSNNSolver


class FrozenControls:
    """Expose both trained formulations as discounted left-endpoint controls."""

    def __init__(self, value: FBSNNSolver, terminal: TerminalBSDESolver) -> None:
        if value.case != terminal.case or value.config.n_steps != terminal.config.n_steps:
            raise ValueError("Checkpoints must share a financial case and control grid.")
        self.value, self.terminal = value, terminal
        self.problem = PricingBSDE(value.case.model, value.case.contract)
        self.steps = terminal.config.n_steps
        self.dt = terminal.case.contract.maturity / self.steps
        for solver in (value, terminal):
            solver.network.eval().requires_grad_(False)

    def value_gradient(self, t: float, spots: np.ndarray) -> np.ndarray:
        points = torch.from_numpy(np.column_stack((np.full(len(spots), t), spots)))
        delta = value_derivatives(self.value.network, points, compute_hessian=False).gradient[:, 1:]
        return math.exp(-self.problem.model.rate*t) * self.problem.control(spots, delta.numpy())

    @torch.no_grad()
    def terminal_loss(self, t: float, spots: np.ndarray) -> np.ndarray:
        step = round(t/self.dt)
        if not 0 <= step < self.steps or not math.isclose(t, step*self.dt, abs_tol=1e-14, rel_tol=0):
            raise ValueError("Terminal control time must be a left endpoint on the checkpoint grid.")
        # The checkpoint output already incorporates the discount convention.
        return self.terminal.network.discounted_control_at(step, torch.tensor(spots)).numpy()

    def geometric_proxy(self) -> DiscountedControl:
        proxy = GeometricBasketReference(
            self.problem.model, replace(self.problem.contract, basket="geometric"),
        )

        def control(t: float, spots: np.ndarray) -> np.ndarray:
            delta = proxy.call_delta_array(spots, t)
            return math.exp(-self.problem.model.rate*t)*self.problem.control(spots, delta)

        return control


def _pair_samples(
    estimator: MartingaleEstimator, positive: np.ndarray,
    control: DiscountedControl | None, chunk_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average the two payoffs and integrals in each IID antithetic pair."""
    payoff = np.empty(len(positive))
    integral = np.empty_like(payoff)
    pair_chunk = chunk_paths//2
    for start in range(0, len(positive), pair_chunk):
        block = positive[start:start+pair_chunk]
        n = len(block)
        samples = estimator.sample(np.concatenate((block, -block)), discounted_control=control)
        payoff[start:start+n] = (samples.discounted_payoff[:n] + samples.discounted_payoff[n:])/2
        integral[start:start+n] = (samples.stochastic_integral[:n] + samples.stochastic_integral[n:])/2
    return payoff, integral


def _pilot_coefficient(payoff: np.ndarray, integral: np.ndarray) -> float | None:
    x, z = payoff-payoff.mean(), integral-integral.mean()
    variance = z @ z
    return None if variance == 0 else float((x @ z)/variance)
