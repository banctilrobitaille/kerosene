import unittest

import torch
from hamcrest import assert_that, equal_to

from kerosene.nn.apex import ApexLoss


class TestApexLoss(unittest.TestCase):
    VALID_LOSS_VALUE = torch.tensor([1.0, 2.0, 3.0])
    INVALID_LOSS_VALUE = "Ford Fiesta"

    VALID_LOSS_MUL_FACTOR = 2.0
    INVALID_LOSS_MUL_FACTOR = "Ford Focus"

    VALID_LOSS_ID = 0
    INVALID_LOSS_ID = "Ford F150"

    def test_loss_addition(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, torch.tensor([2.0, 4.0, 6.0]), None)

        apex_loss1 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)
        apex_loss2 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(apex_loss1 + apex_loss2, equal_to(expected_result))

    def test_factor_multiplication(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_MUL_FACTOR * self.VALID_LOSS_VALUE, None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(self.VALID_LOSS_MUL_FACTOR * loss, equal_to(expected_result))
