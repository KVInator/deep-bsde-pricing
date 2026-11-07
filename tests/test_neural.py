"""Independent finite differences and graph ownership before neural training."""

import unittest

import torch

from bsde_pricing.neural import ValueNetwork, value_derivatives


class NeuralDerivativeTests(unittest.TestCase):
    def test_smooth_network_finite_difference_derivatives(self):
        network = ValueNetwork((100,), 2, 100, width=4)
        x = torch.tensor([[0.3, 90.0], [1.2, 110.0]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(network, (x,), eps=1e-5, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(network, (x,), eps=1e-5, atol=1e-7, rtol=1e-5))
        d = value_derivatives(network, x)
        for j, h in enumerate((1e-4, 1e-2)):
            up, down = x.detach().clone(), x.detach().clone()
            up[:, j] += h
            down[:, j] -= h
            finite_gamma = (network(up)-2*network(x)+network(down))/h**2
            torch.testing.assert_close(d.hessian[:, j, j], finite_gamma[:, 0], atol=2e-5, rtol=2e-4)
        self.assertIsNone(x.grad)

        # Isolate differentiation through curvature: a value-only backward
        # would not pass this finite parameter bump of a Hessian-only loss.
        network.zero_grad(set_to_none=True)
        def curvature_loss():
            h = value_derivatives(network, x, parameter_graph=True).hessian
            scales = network.input_scales
            return (h*scales[:, None]*scales[None, :]/network.value_scale).square().sum()
        curvature_loss().backward()
        parameter = next(network.parameters())
        automatic = float(parameter.grad[0, 0])
        original = float(parameter.detach()[0, 0])
        estimates = []
        for sign in (1, -1):
            with torch.no_grad(): parameter[0, 0] = original + sign*1e-5
            estimates.append(float(curvature_loss().detach()))
        with torch.no_grad(): parameter[0, 0] = original
        self.assertAlmostEqual(automatic, (estimates[0]-estimates[1])/2e-5, delta=2e-8)
        self.assertFalse(d.values.requires_grad)
        # Pointwise layers make a batch row agree with evaluation in isolation.
        torch.testing.assert_close(d.gradient[:1], value_derivatives(network, x[:1]).gradient)

    def test_parameter_graph_through_hessian_and_seed_ownership(self):
        before = torch.get_rng_state().clone()
        model = ValueNetwork((100, 120), 1, 100, width=4)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        same = ValueNetwork((100, 120), 1, 100, width=4)
        for a, b in zip(model.parameters(), same.parameters()):
            torch.testing.assert_close(a, b)
        x = torch.tensor([[0.2, 80.0, 120.0]], dtype=torch.float64)
        d = value_derivatives(model, x, parameter_graph=True)
        loss = d.values.square().mean() + d.gradient.square().mean() + d.hessian.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
        self.assertIsNone(x.grad)

    def test_affine_zero_hessian_and_input_contracts(self):
        x = torch.ones((2, 2), dtype=torch.float64)
        d = value_derivatives(lambda a: (2*a[:, 0]+3*a[:, 1])[:, None], x)
        torch.testing.assert_close(d.gradient, torch.tensor([[2, 3], [2, 3]], dtype=x.dtype))
        torch.testing.assert_close(d.hessian, torch.zeros((2, 2, 2), dtype=x.dtype))
        for bad in (torch.ones(2), torch.ones((0, 2)), torch.ones((2, 2), dtype=torch.int64),
                    torch.full((2, 2), float('nan'))):
            with self.assertRaises(ValueError): value_derivatives(lambda a: a[:, :1], bad)
        for scales, maturity, value_scale, width in (((), 1, 100, 4), ((0,), 1, 100, 4),
                                                   ((100,), 0, 100, 4), ((100,), 1, -1, 4),
                                                   ((100,), 1, 100, True)):
            with self.assertRaises(ValueError): ValueNetwork(scales, maturity, value_scale, width=width)
        with self.assertRaises(ValueError): value_derivatives(lambda a: a, x)
        constant = value_derivatives(lambda a: torch.ones_like(a[:, :1]), x)
        torch.testing.assert_close(constant.hessian, torch.zeros_like(constant.hessian))

if __name__ == "__main__":
    unittest.main()
