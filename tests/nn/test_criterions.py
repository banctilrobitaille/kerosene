import unittest

import numpy as np
import torch
from hamcrest import *

from kerosene.nn.criterions import DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss, TverskyLoss
from kerosene.utils.tensors import to_onehot


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int)
    y_pred[5:15, 1:11] = 1
    y_pred[20:30, 20:30] = 2
    return y_true, y_pred


def compute_tensor_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    y_true_tensor = torch.from_numpy(y_true).unsqueeze(0).type(torch.long)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * 0.0
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 1.0
    y_logits = torch.from_numpy(y_probas).unsqueeze(0).type(torch.float32)
    return y_true_tensor, y_logits


def compute_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2.0 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())
    return true_res


def compute_generalized_dice_loss_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        weights = (1.0 / (np.sum(bin_y_true) * np.sum(bin_y_true) + 1e-15))
        intersection = (bin_y_true & bin_y_pred)
        true_res[index] = 2 * intersection.sum() * weights / (((bin_y_pred.sum() + bin_y_true.sum()) * weights) + 1e-15)
    return true_res


class TestDiceLoss(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_loss = np.subtract(1.0, np.mean(self.dice))

    def test_should_raise_exception_with_bad_values(self):
        dice_loss = DiceLoss()
        assert_that(calling(dice_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(dice_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))

    def test_should_compute_dice(self):
        dice_loss = DiceLoss(reduction=None)
        loss = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))

        np.testing.assert_almost_equal(loss.numpy(), np.subtract(1.0, self.dice))

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            dice_loss = DiceLoss(reduction=None, ignore_index=ignore_index)
            res = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.dice[:ignore_index] + self.dice[ignore_index + 1:])
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        dice_loss = DiceLoss(reduction="mean")
        loss = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))

        np.testing.assert_almost_equal(loss.numpy(), self.mean_dice_loss)

    def test_should_compute_mean_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            dice_loss = DiceLoss(ignore_index=ignore_index)
            res = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.dice[:ignore_index] + self.dice[ignore_index + 1:]).mean()
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)


class TestGeneralizedDiceLoss(unittest.TestCase):
    INVALID_REDUCTION = "sum"
    INVALID_INDEX = -1

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.generalized_dice_loss = compute_generalized_dice_loss_truth(self.y_true, self.y_pred)
        self.mean_generalized_dice_loss = np.subtract(1.0, np.mean(self.generalized_dice_loss))

    def test_should_raise_exception_with_bad_values(self):
        generalized_dice_loss = GeneralizedDiceLoss()
        assert_that(calling(GeneralizedDiceLoss).with_args(reduction=self.INVALID_REDUCTION),
                    raises(NotImplementedError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))

    def test_should_raise_exception_with_bad_ignore_index_values(self):
        generalized_dice_loss = GeneralizedDiceLoss(ignore_index=self.INVALID_INDEX)

        assert_that(calling(generalized_dice_loss.forward).with_args(inputs=self.y_logits,
                                                                     targets=to_onehot(self.y_true_tensor,
                                                                                       num_classes=3)),
                    raises(IndexError))

    def test_should_compute_generalized_dice(self):
        generalized_dice_loss = GeneralizedDiceLoss()
        loss = generalized_dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
        np.testing.assert_almost_equal(loss.numpy(), self.mean_generalized_dice_loss)

    def test_should_compute_generalized_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            generalized_dice_loss = GeneralizedDiceLoss(reduction=None, ignore_index=ignore_index)
            res = generalized_dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.generalized_dice_loss[:ignore_index] + self.generalized_dice_loss[
                                                                                    ignore_index + 1:])
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_generalized_dice(self):
        dice_loss = GeneralizedDiceLoss()
        loss = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))

        np.testing.assert_almost_equal(loss.numpy(), self.mean_generalized_dice_loss)

    def test_should_compute_mean_generalized_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            dice_loss = GeneralizedDiceLoss(ignore_index=ignore_index)
            res = dice_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.generalized_dice_loss[:ignore_index] + self.generalized_dice_loss[
                                                                                    ignore_index + 1:]).mean()
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)


class TestWeightedCrossEntropy(unittest.TestCase):
    WEIGHTED_CROSS_ENTROPY_LOSS_TRUTH = 1.0808

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)

    def test_should_raise_exception_with_bad_values(self):
        weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(weighted_cross_entropy_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))

    def test_should_compute_weights(self):
        weights = WeightedCrossEntropyLoss.compute_class_weights(self.y_logits)
        np.testing.assert_almost_equal(weights.numpy(), np.array([0.2857143, 8.0, 8.0]), decimal=7)

    def test_should_return_loss(self):
        weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
        loss = weighted_cross_entropy_loss.forward(self.y_logits, self.y_true_tensor)
        np.testing.assert_almost_equal(loss.numpy(), self.WEIGHTED_CROSS_ENTROPY_LOSS_TRUTH)


class TestTverskyLoss(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_loss = np.subtract(1.0, np.mean(self.dice))

    def test_should_raise_exception_with_bad_values(self):
        tversky_loss = TverskyLoss()
        assert_that(calling(tversky_loss.forward).with_args(inputs=None, targets=None),
                    raises(AttributeError))
        assert_that(calling(tversky_loss.forward).with_args(inputs=self.y_logits, targets=None),
                    raises(AttributeError))
        assert_that(calling(tversky_loss.forward).with_args(inputs=None, targets=self.y_true_tensor),
                    raises(AttributeError))

    def test_should_compute_tversky_index(self):
        tversky_loss = TverskyLoss(reduction=None)
        loss = tversky_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))

        np.testing.assert_almost_equal(loss.numpy(), np.subtract(1.0, self.dice))

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            tversky_loss = TverskyLoss(reduction=None, ignore_index=ignore_index)
            res = tversky_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.dice[:ignore_index] + self.dice[ignore_index + 1:])
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        tversky_loss = TverskyLoss(reduction="mean")
        loss = tversky_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))

        np.testing.assert_almost_equal(loss.numpy(), self.mean_dice_loss)

    def test_should_compute_mean_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            tversky_loss = TverskyLoss(ignore_index=ignore_index)
            res = tversky_loss.forward(self.y_logits, to_onehot(self.y_true_tensor, num_classes=3))
            true_res = np.subtract(1.0, self.dice[:ignore_index] + self.dice[ignore_index + 1:]).mean()
            np.testing.assert_almost_equal(res.numpy(), true_res), "{}: {} vs {}".format(ignore_index, res, true_res)
