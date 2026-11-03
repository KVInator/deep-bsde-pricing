"""Differentiable local discounted pricing-BSDE and terminal loss."""

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .bsde import PricingBSDE
from .neural import value_derivatives


@dataclass(frozen=True)
class BSDELossResult:
    total: torch.Tensor
    local: torch.Tensor
    terminal: torch.Tensor
    residuals: torch.Tensor
    terminal_residual: torch.Tensor


class LocalBSDELoss(nn.Module):
    """Tensor adapter for PricingBSDE, with one canonical residual formula.

    Residuals are discounted currency amounts. Losses are dimensionless:
    mean over paths of the sum of squared local residuals / value_scale^2,
    plus terminal_weight times mean squared terminal error / value_scale^2.
    Financial coefficients are fixed buffers, not optimisation parameters.
    """

    def __init__(
        self, problem: PricingBSDE, *, value_scale: float, terminal_weight: float = 1,
    ) -> None:
        super().__init__()
        if problem.contract.maturity <= 0:
            raise ValueError("The local BSDE loss requires positive maturity.")
        if not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("value_scale must be positive and finite.")
        if not math.isfinite(terminal_weight) or terminal_weight <= 0:
            raise ValueError("terminal_weight must be positive and finite.")
        self.geometric = problem.contract.basket == "geometric"
        for name, values in (("rate", problem.model.rate), ("sigma", problem.model.volatilities),
                             ("weights", problem.contract.weights), ("strike", problem.contract.strike),
                             ("maturity", problem.contract.maturity), ("value_scale", value_scale),
                             ("terminal_weight", terminal_weight)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64, device="cpu"))
        self.register_buffer("cholesky", torch.linalg.cholesky(torch.tensor(
            problem.model.correlation, dtype=torch.float64, device="cpu",
        )))

    def driver(self, values: torch.Tensor) -> torch.Tensor:
        return -self.rate*values

    def control(self, spots: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return (spots*self.sigma*deltas) @ self.cholesky

    def payoff(self, spots: torch.Tensor) -> torch.Tensor:
        basket = (torch.exp(torch.log(spots) @ self.weights) if self.geometric
                  else spots @ self.weights)
        return torch.relu(basket-self.strike)

    def forward(
        self, value: Callable[[torch.Tensor], torch.Tensor], times: torch.Tensor,
        states: torch.Tensor, increments: torch.Tensor,
        *, parameter_graph: bool = True,
    ) -> BSDELossResult:
        """Evaluate on paths (N,M+1,d) and independent increments (N,M,d).

        times is (M+1,) from zero to maturity. Path tensors must be fixed
        data; they are never differentiated through. Only left-endpoint
        input gradients enter controls; terminal payoff Greeks are unused.
        """
        tensors = (times, states, increments)
        if any(t.dtype != self.rate.dtype or t.device != self.rate.device for t in tensors):
            raise ValueError("Path tensors must match the loss dtype and device.")
        if (times.ndim != 1 or len(times) < 2 or states.ndim != 3
                or states.shape[0] < 1 or states.shape[1:] != (len(times), self.sigma.numel())
                or increments.shape != (states.shape[0], len(times)-1, self.sigma.numel())):
            raise ValueError("Inconsistent time, state or Brownian-increment shapes.")
        if any(t.requires_grad or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("Paths must be finite tensors without requires_grad.")
        if (bool((states <= 0).any()) or bool((times[1:] <= times[:-1]).any())
                or bool(times[0] != 0) or bool(times[-1] != self.maturity)):
            raise ValueError("States must be positive and times increase from zero to maturity.")
        n_paths, n_times, dimension = states.shape
        left_inputs = torch.cat((times[:-1].expand(n_paths, -1)[..., None], states[:, :-1]), dim=2)
        left = value_derivatives(
            value, left_inputs.reshape(-1, dimension+1),
            parameter_graph=parameter_graph, compute_hessian=False,
        )
        y = left.values.reshape(n_paths, n_times-1)
        deltas = left.gradient[:, 1:].reshape(n_paths, n_times-1, dimension)
        terminal_inputs = torch.cat((times[-1].expand(n_paths, 1), states[:, -1]), dim=1)
        with torch.set_grad_enabled(parameter_graph):
            y_terminal = value(terminal_inputs)
        if y_terminal.shape != (n_paths, 1) or not bool(torch.isfinite(y_terminal).all()):
            raise ValueError("Terminal values must be finite with shape (N,1).")
        discount = torch.exp(-self.rate*times)
        if not bool(torch.isfinite(discount).all()) or bool((discount <= 0).any()):
            raise ArithmeticError("Discount factors are outside floating-point range.")
        discounted_y = torch.cat((y, y_terminal), dim=1)*discount
        controls = self.control(states[:, :-1], deltas)
        stochastic = (controls*increments).sum(dim=2)*discount[:-1]
        # d(e^-rt Y) = -e^-rt (f+rY)dt + e^-rt Z.dB.
        residuals = (discounted_y[:, 1:]-discounted_y[:, :-1]
                     + discount[:-1]*(self.driver(y)+self.rate*y)*torch.diff(times)
                     - stochastic)
        terminal_residual = discount[-1]*(y_terminal[:, 0]-self.payoff(states[:, -1]))
        local = (residuals/self.value_scale).square().sum(dim=1).mean()
        terminal = (terminal_residual/self.value_scale).square().mean()
        total = local+self.terminal_weight*terminal
        if not bool(torch.isfinite(total)):
            raise ArithmeticError("Nonfinite BSDE loss.")
        return BSDELossResult(total, local, terminal, residuals, terminal_residual)
