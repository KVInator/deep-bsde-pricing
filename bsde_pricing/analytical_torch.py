"""Differentiable known-solution oracle for derivative and BSDE diagnostics."""

import math

import torch
from torch import nn

from .black_scholes import BlackScholesReference
from .geometric_basket import GeometricBasketReference
from .problems import EuropeanCall, GBMModel


class AnalyticalCall(nn.Module):
    """Vanilla/geometric call oracle; application prices use scalar references.

    Inputs are physical (time, spots) rows. Positive variance and maturity
    are required. At maturity only the payoff value is used: BSDE controls
    are differentiated strictly before maturity, never at the payoff kink.
    """

    def __init__(self, model: GBMModel, contract: EuropeanCall) -> None:
        super().__init__()
        if contract.basket == "geometric":
            parameters = GeometricBasketReference(model, contract).evaluate(
                (contract.strike,) * model.dimension,
            )
            sigma, dividend = parameters.effective_volatility, parameters.effective_dividend_yield
        else:
            BlackScholesReference(model, contract)
            sigma, dividend = model.volatilities[0], model.dividend_yields[0]
        if contract.maturity <= 0 or sigma <= 0:
            raise ValueError("The analytical tensor oracle requires positive maturity and variance.")
        self.geometric = contract.basket == "geometric"
        for name, values in (("weights", contract.weights), ("sigma", sigma),
                             ("dividend", dividend), ("rate", model.rate),
                             ("strike", contract.strike), ("maturity", contract.maturity)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64, device="cpu"))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.numel()+1:
            raise ValueError("Oracle inputs must have shape (N, dimension+1).")
        if inputs.dtype != self.weights.dtype or inputs.device != self.weights.device:
            raise ValueError("Oracle inputs must match its dtype and device.")
        time, spots = inputs[:, 0], inputs[:, 1:]
        if (not bool(torch.isfinite(inputs).all()) or bool((spots <= 0).any())
                or bool(((time < 0) | (time > self.maturity)).any())):
            raise ValueError("Oracle requires finite positive spots and times in [0,T].")
        basket = torch.exp(torch.log(spots) @ self.weights) if self.geometric else spots[:, 0]
        tau = self.maturity-time
        # Both branches of where are evaluated. A safe horizon keeps the
        # inactive BS branch finite; the expiry branch is the exact payoff.
        safe_tau = torch.where(tau > 0, tau, torch.ones_like(tau))
        deviation = self.sigma * torch.sqrt(safe_tau)
        d1 = (torch.log(basket/self.strike)
              + (self.rate-self.dividend+0.5*self.sigma**2)*safe_tau) / deviation
        d2 = d1-deviation
        cdf1 = 0.5*torch.erfc(-d1/math.sqrt(2))
        cdf2 = 0.5*torch.erfc(-d2/math.sqrt(2))
        call = (basket*torch.exp(-self.dividend*safe_tau)*cdf1
                - self.strike*torch.exp(-self.rate*safe_tau)*cdf2)
        return torch.where(tau > 0, call, torch.relu(basket-self.strike))[:, None]
