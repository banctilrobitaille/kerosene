# -*- coding: utf-8 -*-
# Copyright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import numpy as np
import unittest

from kerosene.metrics.metrics import Dice, GeneralizedDice
from hamcrest import *


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
    y_probs = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probs[i, (y_pred == i)] = 720
    y_logits = torch.from_numpy(y_probs).unsqueeze(0)
    return y_true_tensor, y_logits


def compute_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        true_res[index] = 2 * intersection.sum() / (bin_y_pred.sum() + bin_y_true.sum())
    return true_res


def compute_generalized_dice_truth(y_true, y_pred):
    true_res = [0, 0, 0]
    weights = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        weights[index] = (1.0 / (np.sum(bin_y_true) * np.sum(bin_y_true) + 1e-15))
        intersection = (bin_y_true & bin_y_pred)
        true_res[index] = 2 * intersection.sum() * weights[index] / (
                ((bin_y_pred.sum() + bin_y_true.sum()) * weights[index]) + 1e-15)
    return true_res, weights


class TestDiceMetric(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_REDUCTION = "sum"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.dice_truth = compute_dice_truth(self.y_true, self.y_pred)
        self.mean_dice_truth = np.mean(self.dice_truth)

    def test_should_compute_dice_for_multiclass(self):
        dice_coefficient = Dice(num_classes=3, reduction=None)
        dice_coefficient.update((self.y_logits, self.y_true_tensor))
        res = dice_coefficient.compute().numpy()
        assert np.all(res == self.dice_truth)

    def test_should_compute_mean_dice_for_multiclass(self):
        dice_coefficient = Dice(num_classes=3, reduction="mean")
        dice_coefficient.update((self.y_logits, self.y_true_tensor))
        res = dice_coefficient.compute().numpy()
        truth = np.array(self.dice_truth).mean()
        assert res == truth

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            dice_coefficient = Dice(num_classes=3, ignore_index=ignore_index, reduction=None)
            dice_coefficient.update((self.y_logits, self.y_true_tensor))
            res = dice_coefficient.compute().numpy()
            true_res = self.dice_truth[:ignore_index] + self.dice_truth[ignore_index + 1:]
            assert np.all(res == true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        mean_dice_coefficient = Dice(num_classes=3, reduction="mean")
        mean_dice_coefficient.update((self.y_logits, self.y_true_tensor))
        res = mean_dice_coefficient.compute().numpy()
        assert_that(res, equal_to(self.mean_dice_truth))

    def test_should_compute_mean_dice_with_ignored_index(self):
        for ignore_index in range(3):
            mean_dice_coefficient = Dice(num_classes=3, reduction="mean", ignore_index=ignore_index)
            mean_dice_coefficient.update((self.y_logits, self.y_true_tensor))
            res = mean_dice_coefficient.compute().numpy()
            true_res = np.mean(self.dice_truth[:ignore_index] + self.dice_truth[ignore_index + 1:])
            assert_that(res, equal_to(true_res)), "{}: {} vs {}".format(ignore_index, res, true_res)


class TestGeneralizedDiceMetric(unittest.TestCase):
    INVALID_VALUE_1 = -1
    INVALID_VALUE_2 = "STEVE JOBS"
    INVALID_VALUE_3 = 10
    INVALID_VALUE_4 = 11

    def setUp(self):
        self.y_true, self.y_pred = get_y_true_y_pred()
        self.y_true_tensor, self.y_logits = compute_tensor_y_true_y_logits(self.y_true, self.y_pred)
        self.generalized_dice_truth, weights = compute_generalized_dice_truth(self.y_true, self.y_pred)
        self.weights = torch.from_numpy(np.array(weights))
        self.generalized_mean_dice_truth = np.mean(self.generalized_dice_truth)

    def test_should_compute_dice_for_multiclass(self):
        generalized_dice_coefficient = GeneralizedDice(num_classes=3)
        generalized_dice_coefficient.update((self.y_logits, self.y_true_tensor))
        res = generalized_dice_coefficient.compute().numpy()
        assert np.all(res == self.generalized_dice_truth)

    def test_should_compute_dice_for_multiclass_with_ignored_index(self):
        for ignore_index in range(3):
            generalized_dice_coefficient = GeneralizedDice(num_classes=3, ignore_index=ignore_index)
            generalized_dice_coefficient.update((self.y_logits, self.y_true_tensor))
            res = generalized_dice_coefficient.compute().numpy()
            true_res = self.generalized_dice_truth[:ignore_index] + self.generalized_dice_truth[ignore_index + 1:]
            assert np.all(res == true_res), "{}: {} vs {}".format(ignore_index, res, true_res)

    def test_should_compute_mean_dice(self):
        mean_generalized_dice_coefficient = GeneralizedDice(num_classes=3, reduction="mean")
        mean_generalized_dice_coefficient.update((self.y_logits, self.y_true_tensor))
        res = mean_generalized_dice_coefficient.compute().numpy()
        assert_that(res, equal_to(self.generalized_mean_dice_truth))

    def test_should_compute_mean_dice_with_ignored_index(self):
        for ignore_index in range(3):
            mean_generalized_dice_coefficient = GeneralizedDice(num_classes=3, reduction="mean", ignore_index=ignore_index)
            mean_generalized_dice_coefficient.update((self.y_logits, self.y_true_tensor))
            res = mean_generalized_dice_coefficient.compute().numpy()
            true_res = np.mean(
                self.generalized_dice_truth[:ignore_index] + self.generalized_dice_truth[ignore_index + 1:])
            assert_that(res, equal_to(true_res)), "{}: {} vs {}".format(ignore_index, res, true_res)