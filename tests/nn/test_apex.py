import unittest

import torch
from hamcrest import assert_that, equal_to

from kerosene.nn.apex import ApexLoss


class TestApexLoss(unittest.TestCase):

    def test_loss_addition(self):
        loss_id_1, loss_id_2 = 1, 2
        loss_value = torch.tensor([1.0, 2.0, 3.0])
        expected_result = ApexLoss(loss_id_1, torch.tensor([2.0, 4.0, 6.0]), None)

        loss1 = ApexLoss(loss_id_1, loss_value, None)
        loss2 = ApexLoss(loss_id_2, loss_value, None)

        assert_that(loss1 + loss2, equal_to(expected_result))

    def test_factor_multiplication(self):
        loss_id_1, loss_id_2 = 1, 2
        loss_value = torch.tensor([1.0, 2.0, 3.0])
        expected_result = ApexLoss(loss_id_1, torch.tensor([2.0, 4.0, 6.0]), None)

        loss = ApexLoss(loss_id_1, loss_value, None)

        assert_that(2 * loss, equal_to(expected_result))
