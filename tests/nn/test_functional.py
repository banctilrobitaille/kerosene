import unittest

import torch
from hamcrest import assert_that, equal_to

import kerosene.nn.functional as F


class TestJensenShannonDivergence(unittest.TestCase):
    PROB_DIST1, PROB_DIST2, PROB_DIST3 = [1 / 2, 1 / 2, 0], [0, 1 / 10, 9 / 10], [1 / 3, 1 / 3, 1 / 3]

    def test_should_compute_jensen_shannon_divergence(self):
        prob_distributions = torch.tensor([[self.PROB_DIST1, self.PROB_DIST2, self.PROB_DIST3]])
        expected_results = torch.tensor([0.378889])

        assert_that(F.js_div(prob_distributions), equal_to(expected_results))

    def test_should_compute_jensen_shannon_divergence_of_same_distribution(self):
        prob_distributions = torch.tensor([[self.PROB_DIST1, self.PROB_DIST1, self.PROB_DIST1]])
        expected_results = torch.tensor([0.0])

        assert_that(F.js_div(prob_distributions), equal_to(expected_results))
