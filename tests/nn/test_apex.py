import unittest

import torch
from hamcrest import assert_that, equal_to

from kerosene.nn.apex import ApexLoss


class TestApexLoss(unittest.TestCase):
    VALID_LOSS_VALUE = torch.tensor([1.0, 2.0, 3.0])
    INVALID_LOSS_VALUE = "Ford Fiesta"

    VALID_SCALAR = 2.0
    INVALID_SCALAR = "Ford Focus"

    VALID_LOSS_ID = 0
    INVALID_LOSS_ID = "Ford F150"

    def test_should_add_losses(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, torch.tensor([2.0, 4.0, 6.0]), None)

        apex_loss1 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)
        apex_loss2 = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(apex_loss1 + apex_loss2, equal_to(expected_result))

    def test_should_multiply_by_a_scalar(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR * self.VALID_LOSS_VALUE, None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(self.VALID_SCALAR * loss, equal_to(expected_result))
        assert_that(loss * self.VALID_SCALAR, equal_to(expected_result))

    # noinspection PyTypeChecker
    def test_should_divide_by_a_scalar(self):
        left_div_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE / self.VALID_SCALAR,
                                            None)
        right__div_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR / self.VALID_LOSS_VALUE,
                                              None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss / self.VALID_SCALAR, equal_to(left_div_expected_result))
        assert_that(self.VALID_SCALAR / loss, equal_to(right__div_expected_result))

    def test_should_compute_the_mean(self):
        expected_result = ApexLoss(self.VALID_LOSS_ID, torch.tensor([2.0]), None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss.mean(), equal_to(expected_result))

    # noinspection PyTypeChecker
    def test_should_substract_a_scalar(self):
        left_sub_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE - self.VALID_SCALAR, None)
        right_sub_expected_result = ApexLoss(self.VALID_LOSS_ID, self.VALID_SCALAR - self.VALID_LOSS_VALUE, None)

        loss = ApexLoss(self.VALID_LOSS_ID, self.VALID_LOSS_VALUE, None)

        assert_that(loss - self.VALID_SCALAR, equal_to(left_sub_expected_result))
        assert_that(self.VALID_SCALAR - loss, equal_to(right_sub_expected_result))
