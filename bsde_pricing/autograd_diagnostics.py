"""Independent derivative oracles and a bounded CPU/MPS workload."""

import copy
import os
import platform
import statistics
import time

import numpy as np
import torch

from .black_scholes import BlackScholesReference
from .neural import ValueNetwork, value_derivatives
from .problems import EuropeanCall, GBMModel


def _check(actual, expected, tolerance: float) -> dict[str, object]:
    actual = actual.detach().cpu().double().numpy()
    error = float(np.max(np.abs(actual - np.asarray(expected))))
    return {"max_absolute_error": error, "absolute_tolerance": tolerance,
            "passed": bool(np.isfinite(error) and error <= tolerance)}


def analytical_checks(device: str, dtype: torch.dtype) -> dict[str, dict[str, object]]:
    """Polynomial mixed derivatives plus a differentiable BS oracle.

    The torch BS expression is an independent diagnostic, not a second
    application pricer. The existing scalar reference supplies its targets.
    """
    tight, loose = ((2e-10, 2e-9) if dtype == torch.float64 else (2e-4, 5e-4))
    x = torch.tensor([[0.1, 80, 120], [0.6, 110, 90]], dtype=dtype, device=device)
    def polynomial(a):
        u, v, w = a[:, 0] / 2, a[:, 1] / 80, a[:, 2] / 120
        return (100 * (u**2 + v**3 + w**3 + v*w))[:, None]
    d = value_derivatives(polynomial, x)
    a = x.detach().cpu().double().numpy()
    u, v, w = a[:, 0]/2, a[:, 1]/80, a[:, 2]/120
    expected_g = np.column_stack((100*u, 100*(3*v*v+w)/80, 100*(3*w*w+v)/120))
    expected_h = np.zeros((2, 3, 3))
    expected_h[:, 0, 0] = 50
    expected_h[:, 1, 1] = 600*v/80**2
    expected_h[:, 2, 2] = 600*w/120**2
    expected_h[:, 1, 2] = expected_h[:, 2, 1] = 100/(80*120)
    checks = {"polynomial_gradient": _check(d.gradient, expected_g, tight),
              "polynomial_hessian": _check(d.hessian, expected_h, tight)}
    inputs = torch.tensor([[0.1, 70], [0.4, 100], [0.8, 130]], dtype=dtype, device=device)
    # All oracle evaluations are strictly before maturity and away from kinks.
    def vanilla(a):
        tau, spot = 1-a[:, 0], a[:, 1]
        deviation = 0.2 * torch.sqrt(tau)
        d1 = (torch.log(spot/100) + 0.03*tau) / deviation
        d2 = d1-deviation
        cdf = lambda z: 0.5 * torch.erfc(-z / (2**0.5))
        return (spot*torch.exp(-0.02*tau)*cdf(d1)
                - 100*torch.exp(-0.03*tau)*cdf(d2))[:, None]
    d = value_derivatives(vanilla, inputs)
    reference = BlackScholesReference(GBMModel(0.03, (0.2,), (0.02,), ((1,),)), EuropeanCall(100, 1))
    targets = [reference.evaluate(float(s), float(t)) for t, s in inputs.detach().cpu().double().tolist()]
    spots = inputs[:, 1].detach().cpu().double().numpy()
    theta = [0.03*r.call_price - 0.01*s*r.call_delta - 0.5*0.2**2*s*s*r.gamma
             for r, s in zip(targets, spots)]
    checks.update({
        "vanilla_value": _check(d.values[:, 0], [r.call_price for r in targets], loose),
        "vanilla_time_derivative": _check(d.gradient[:, 0], theta, loose),
        "vanilla_delta": _check(d.gradient[:, 1], [r.call_delta for r in targets], tight),
        "vanilla_gamma": _check(d.hessian[:, 1, 1], [r.gamma for r in targets], tight),
    })
    return checks


def _workload(network: ValueNetwork, inputs: torch.Tensor):
    network.zero_grad(set_to_none=True)
    d = value_derivatives(network, inputs, parameter_graph=True)
    # Normalise derivative units for this diagnostic objective only.
    scales = network.input_scales
    value_scale = network.value_scale
    loss = ((d.values/value_scale).square().mean()
            + (d.gradient*scales/value_scale).square().mean()
            + (d.hessian*scales[:, None]*scales[None, :]/value_scale).square().mean())
    loss.backward()
    gradients = torch.cat([p.grad.reshape(-1) for p in network.parameters()])
    if not bool(torch.isfinite(gradients).all()):
        raise ArithmeticError("Nonfinite parameter gradients.")
    return d, gradients.detach().clone()


def run_autograd_diagnostics(
    *, backend: str = "auto", batch_size: int = 256, repeats: int = 5,
) -> dict[str, object]:
    if backend not in ("auto", "cpu", "mps"):
        raise ValueError("backend must be auto, cpu or mps.")
    for name, value in (("batch_size", batch_size), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if backend != "cpu" and torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Unset PYTORCH_ENABLE_MPS_FALLBACK for an actual MPS check.")
    network = ValueNetwork((80, 100, 120), 1, 100)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    inputs = torch.rand((batch_size, 4), dtype=torch.float64, generator=generator, device="cpu")
    inputs = inputs * torch.tensor([0.8, 40, 40, 40], dtype=torch.float64, device="cpu") + torch.tensor([0.1, 60, 80, 100], dtype=torch.float64, device="cpu")
    baseline, baseline_parameters = _workload(network, inputs)
    old_threads = torch.get_num_threads()
    reports = []
    configurations = [("cpu", torch.float64), ("cpu", torch.float32)]
    if backend != "cpu" and torch.backends.mps.is_available():
        configurations.append(("mps", torch.float32))
    try:
        torch.set_num_threads(1)
        for device, dtype in configurations:
            try:
                candidate = copy.deepcopy(network).to(device=device, dtype=dtype)
                batch = inputs.to(device=device, dtype=dtype)
                try:
                    checks = analytical_checks(device, dtype)
                except (RuntimeError, NotImplementedError, ArithmeticError) as exc:
                    # Retain oracle failures and still trial the actual network
                    # workload; an unsupported oracle op need not be a tanh op.
                    checks = {"analytical_oracle_execution": {
                        "passed": False, "error_type": type(exc).__name__, "error": str(exc),
                    }}
                result, parameters = _workload(candidate, batch)
                tolerance = 2e-9 if dtype == torch.float64 else 1e-3
                for name in ("values", "gradient", "hessian"):
                    checks["network_"+name] = _check(getattr(result, name), getattr(baseline, name).detach().cpu().numpy(), tolerance)
                checks["network_parameter_gradient"] = _check(parameters, baseline_parameters.cpu().numpy(), tolerance)
                # Two warmups; setup and host/device copies are outside timing.
                for _ in range(2):
                    _workload(candidate, batch)
                timings = []
                for _ in range(repeats):
                    if device == "mps": torch.mps.synchronize()
                    start = time.perf_counter()
                    _workload(candidate, batch)
                    if device == "mps": torch.mps.synchronize()
                    timings.append(time.perf_counter()-start)
                reports.append({"device": device, "dtype": str(dtype), "status": "passed" if all(c["passed"] for c in checks.values()) else "failed",
                                "checks": checks, "seconds": timings,
                                "median_seconds": statistics.median(timings)})
            except (RuntimeError, NotImplementedError, ArithmeticError) as exc:
                reports.append({"device": device, "dtype": str(dtype), "status": "failed",
                                "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        torch.set_num_threads(old_threads)
    if backend != "cpu" and not torch.backends.mps.is_available():
        reports.append({"device": "mps", "dtype": "torch.float32", "status": "unavailable"})
    return {
        "method": "physical_coordinate_autograd_diagnostic", "torch": torch.__version__,
        "python": platform.python_version(), "system": platform.platform(),
        "architecture": platform.machine(), "numpy": np.__version__,
        "batch_size": batch_size, "dimension": 3, "hidden_width": 16,
        "seed": 2026, "warmups": 2, "repeats": repeats, "cpu_threads": 1,
        "workload": "value_gradient_full_hessian_and_parameter_backward_no_optimizer_step",
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "passed": all(r["status"] != "failed" for r in reports) and not (backend == "mps" and not torch.backends.mps.is_available()),
        "backends": reports,
    }
