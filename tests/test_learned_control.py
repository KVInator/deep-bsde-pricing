"""Independent checks for discounted adapters and paired pilot samples."""

import math
import unittest

import numpy as np
import torch

from bsde_pricing.learned_control import FrozenControls, _pair_samples, _pilot_coefficient
from bsde_pricing.martingale import MartingaleEstimator
from bsde_pricing.problems import vanilla_case
from bsde_pricing.terminal_bsde import TerminalBSDESolver
from bsde_pricing.training import FBSNNSolver, PathTrainingConfig, TrainingConfig


class LearnedControlTests(unittest.TestCase):
    def test_discounted_adapters_keep_independent_coordinates_and_checkpoint_time(self):
        case = vanilla_case()
        value = FBSNNSolver(case, TrainingConfig(n_steps=4))
        terminal = TerminalBSDESolver(case, PathTrainingConfig(n_steps=4))
        with torch.no_grad():
            terminal.network.initial_control.fill_(0.05)
            for layer in terminal.network.layers:
                for parameter in layer.parameters():
                    parameter.zero_()
            terminal.network.layers[0][-1].bias.fill_(0.08)
            terminal.network.layers[1][-1].bias.fill_(0.13)
        frozen = FrozenControls(value, terminal)
        spots = np.array([[90.0], [110.0]])
        t = 0.5
        points = torch.from_numpy(np.column_stack((np.full(2, t), spots)))
        bump = torch.tensor([0, 0.001], dtype=torch.float64)
        bumped_delta = ((value.network(points+bump)-value.network(points-bump))/0.002).numpy()
        expected = math.exp(-0.03*t)*spots*0.2*bumped_delta
        np.testing.assert_allclose(frozen.value_gradient(t, spots), expected, atol=2e-9, rtol=0)
        for time, amount in ((0, 5), (0.25, 8), (0.5, 13)):
            np.testing.assert_allclose(frozen.terminal_loss(time, spots), np.full((2, 1), amount))
        with self.assertRaisesRegex(ValueError, "left endpoint"):
            frozen.terminal_loss(0.375, spots)
        for solver in (value, terminal):
            self.assertFalse(solver.network.training)
            self.assertTrue(all(not p.requires_grad for p in solver.network.parameters()))

    def test_chunked_pairs_against_two_step_gbm_identity(self):
        shocks = np.array([[[0.25], [0.50]], [[-0.25], [0.25]], [[0.5], [-0.25]]])
        payoff, integral = _pair_samples(
            MartingaleEstimator(vanilla_case()), shocks, lambda t, spots: spots, chunk_paths=4,
        )
        b0, b1 = shocks[:, 0, 0], shocks[:, 1, 0]
        expected_integral = 100*np.exp(0.005)*np.sinh(0.2*b0)*b1
        expected_payoff = math.exp(-0.03)*50*(
            np.maximum(np.exp(0.01+0.2*(b0+b1))-1, 0)
            + np.maximum(np.exp(0.01-0.2*(b0+b1))-1, 0)
        )
        np.testing.assert_allclose(integral, expected_integral, atol=1e-13, rtol=0)
        np.testing.assert_allclose(payoff, expected_payoff, atol=1e-13, rtol=0)

    def test_coefficient_uses_centred_pilot_covariance(self):
        np.testing.assert_equal(_pilot_coefficient(np.array([3., 7., 11.]), np.array([1., 3., 5.])), 2)
        self.assertIsNone(_pilot_coefficient(np.array([3., 7., 11.]), np.ones(3)))
