"""Smooth value maps and physical-coordinate automatic derivatives."""

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


class ValueNetwork(nn.Module):
    """Pointwise V(t, S) with inputs (N, d+1) in years and price units.

    Normalisation is inside the graph. No layer couples different rows.
    Construction uses a local CPU seed without changing the caller's RNG.
    """

    def __init__(
        self, spot_scales: tuple[float, ...], maturity: float, value_scale: float,
        *, width: int = 16, seed: int = 2026,
    ) -> None:
        super().__init__()
        scales = (float(maturity), *(float(s) for s in spot_scales))
        if len(scales) < 2 or any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("Time and spot scales must be positive and finite.")
        if not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("value_scale must be positive and finite.")
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.register_buffer("input_scales", torch.tensor(scales, dtype=torch.float64, device="cpu"))
        self.register_buffer("value_scale", torch.tensor(value_scale, dtype=torch.float64, device="cpu"))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.layers = nn.Sequential(
                nn.Linear(len(scales), width, dtype=torch.float64, device="cpu"), nn.Tanh(),
                nn.Linear(width, width, dtype=torch.float64, device="cpu"), nn.Tanh(),
                nn.Linear(width, 1, dtype=torch.float64, device="cpu"),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_scales.numel():
            raise ValueError("inputs must have shape (N, dimension + 1).")
        if inputs.dtype != self.input_scales.dtype or inputs.device != self.input_scales.device:
            raise ValueError("inputs must match the network's dtype and device.")
        return self.value_scale * self.layers(inputs / self.input_scales - 1)


@dataclass(frozen=True)
class ValueDerivatives:
    values: torch.Tensor
    gradient: torch.Tensor
    hessian: torch.Tensor | None


@torch.enable_grad()
def value_derivatives(
    value: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor,
    *, parameter_graph: bool = False, compute_hessian: bool = True,
) -> ValueDerivatives:
    """Return V, (V_t, deltas), and the full time/spot Hessian.

    Shapes are (N,1), (N,d+1), (N,d+1,d+1). The callable must act on rows
    independently and be twice differentiable. Inputs are copied to owned
    graph leaves; input .grad is untouched. parameter_graph retains the
    derivative graph for subsequent differentiation with respect to weights.
    compute_hessian=False omits curvature for gradient-only BSDE losses.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Input derivatives require autograd outside inference_mode.")
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 2:
        raise ValueError("inputs must have nonempty shape (N, dimension + 1).")
    if inputs.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite float32 or float64 tensors.")
    x = inputs.detach().clone().requires_grad_(True)
    y = value(x)
    if y.shape != (len(x), 1):
        raise ValueError("The value map must return shape (N, 1).")
    gradient = None
    if y.requires_grad:
        gradient = torch.autograd.grad(
            y.sum(), x, create_graph=compute_hessian or parameter_graph, allow_unused=True,
        )[0]
    if gradient is None:
        gradient = x * 0
    rows = []
    for coordinate in range(x.shape[1] if compute_hessian else 0):
        component = gradient[:, coordinate]
        row = (torch.autograd.grad(
            component.sum(), x, create_graph=parameter_graph, retain_graph=True,
            allow_unused=True,
        )[0] if component.requires_grad else None)
        rows.append(torch.zeros_like(x) if row is None else row)
    hessian = torch.stack(rows, dim=1) if compute_hessian else None
    if not all(bool(torch.isfinite(a).all()) for a in (y, gradient, *rows)):
        raise ArithmeticError("Nonfinite value or automatic derivative.")
    if not parameter_graph:
        y, gradient = y.detach(), gradient.detach()
        hessian = hessian.detach() if hessian is not None else None
    return ValueDerivatives(y, gradient, hessian)
