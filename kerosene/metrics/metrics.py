# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
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
from typing import Union, Tuple

import torch
from enum import Enum
from ignite.metrics import Accuracy, Average, Precision, MeanAbsoluteError, MeanPairwiseDistance, MeanSquaredError, \
    Recall, RootMeanSquaredError, TopKCategoricalAccuracy, IoU, mIoU, Metric, ConfusionMatrix, MetricsLambda, \
    Fbeta, GeometricAverage

from kerosene.utils.constants import EPSILON
from kerosene.utils.tensors import flatten, to_onehot


class MetricType(Enum):
    Accuracy = "Accuracy"
    Average = "Average"
    ConfusionMatrix = "ConfusionMatrix"
    DiceCoefficient = "Dice"
    Fbeta = "Fbeta"
    GeneralizedDice = "GeneralizedDice"
    GeometricAverage = "GeometricAverage"
    IoU = "IoU"
    mIoU = "mIoU"
    MeanAbsoluteError = "MeanAbsoluteError"
    MeanPairwiseDistance = "MeanPairwiseDistance"
    MeanSquaredError = "MeanSquaredError"
    Precision = "Precision"
    Recall = "Recall"
    RootMeanSquaredError = "RootMeanSquaredError"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    VariableAccumulation = "VariableAccumulation"

    def __str__(self):
        return self.value


class MetricFactory(object):
    def __init__(self):
        self._metrics = {
            "Accuracy": Accuracy,
            "Average": Average,
            "ConfusionMatrix": ConfusionMatrix,
            "Dice": Dice,
            "Fbeta": Fbeta,
            "GeneralizedDice": GeneralizedDice,
            "GeometricAverage": GeometricAverage,
            "IoU": IoU,
            "mIoU": mIoU,
            "MeanAbsoluteError": MeanAbsoluteError,
            "MeanPairwiseDistance": MeanPairwiseDistance,
            "MeanSquaredError": MeanSquaredError,
            "Precision": Precision,
            "Recall": Recall,
            "RootMeanSquaredError": RootMeanSquaredError,
            "TopKCategoricalAccuracy": TopKCategoricalAccuracy
        }

    def create(self, metric_type: Union[str, MetricType], params):
        return self._metrics[str(metric_type)](**params) if params is not None else self._metrics[str(metric_type)]()

    def register(self, metric: str, creator: Metric):
        """
        Add a new metric.
        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator


class Dice(Metric):
    """
    The Dice Metric.
    """
    SUPPORTED_REDUCTIONS = [None, "mean"]

    def __init__(self, num_classes: int, reduction: Union[None, str] = "mean", average: str = None,
                 weight: torch.Tensor = None, ignore_index: int = -100,
                 output_transform: callable = lambda x: x) -> None:
        """
        Metric initializer.
        Args:
            num_classes (int): The number of classes in the problem. In case of images, num_classes should also count the background index 0.
            average (str, optional): Confusion matrix values averaging schema: None, "samples", "recall", "precision".
                Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
                samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
                represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
                diagonal values represent class precisions.
            reduction (str): The type of reduction to apply (e.g. 'mean').
            ignore_index (int, optional): To ignore an index in Dice computation.
            output_transform (callable, optional): a callable that is used to transform the
                output into the form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise NotImplementedError("Reduction type not supported.")
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._weight = weight
        self._cm = ConfusionMatrix(num_classes=num_classes, average=average, output_transform=output_transform)
        self._metric = self.create_dice_metric(self._cm)
        super(Dice, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        """
        Reset the confusion matrix object.
        """
        self._cm.confusion_matrix = torch.zeros(self._num_classes, self._num_classes, dtype=torch.float64)

    def compute(self) -> torch.Tensor:
        """
        Compute the metric.
        Returns:
            :obj:`torch.Tensor`: The dice coefficient for each class.
        """
        return self._metric.compute()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Update the confusion matrix with output values.
        Args:
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth of the form `(y_pred, y)`.
        """
        self._cm.update(output)

    def create_dice_metric(self, cm: ConfusionMatrix):
        """
        Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)
        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
        Returns:
            array or float: The Sørensen–Dice Coefficient for each class or the mean Sørensen–Dice Coefficient.
        """
        # Increase floating point precision
        cm = cm.type(torch.float64)
        dice = 2 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + EPSILON)

        if self._ignore_index != -100:
            def remove_index(dice_vector):
                try:
                    indices = list(range(len(dice_vector)))
                    indices.remove(self._ignore_index)
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            self._ignore_index))

            dice = MetricsLambda(remove_index, dice)

        if self._weight is not None:
            def multiply_weights(dice_vector):
                return self._weight * dice_vector

            dice = MetricsLambda(multiply_weights, dice)

        if self._reduction == "mean":
            dice = dice.mean()

        return dice


class GeneralizedDice(Metric):
    """
    The Generalized Dice Metric.
    """
    SUPPORTED_REDUCTIONS = [None, "mean"]

    def __init__(self, num_classes: int, reduction: str = None, average: str = None,
                 ignore_index: int = -100, output_transform: callable = lambda x: x) -> None:
        """
        Metric initializer.
        Args:
            num_classes (int): The number of classes in the problem. In case of images, num_classes should also count the background index 0.
            average (str, optional): Confusion matrix values averaging schema: None, "samples", "recall", "precision".
                Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
                samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
                represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
                diagonal values represent class precisions.
            reduction (str): The type of reduction to apply (e.g. 'mean').
            ignore_index (int, optional): To ignore an index in Dice computation.
            output_transform (callable, optional): a callable that is used to transform the
                output into the form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise NotImplementedError("Reduction type not supported.")
        self._num_classes = num_classes
        self._average = average
        self._ignore_index = ignore_index
        self._metric = None
        self._reduction = reduction
        self._cm = ConfusionMatrix(num_classes=num_classes, average=average,
                                   output_transform=output_transform)
        super(GeneralizedDice, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        """
        Reset the confusion matrix object.
        """
        self._cm.confusion_matrix = torch.zeros(self._num_classes, self._num_classes, dtype=torch.float64)

    def compute(self) -> torch.Tensor:
        """
        Compute the metric.
        Returns:
            :obj:`torch.Tensor`: The dice coefficient for each class.
        """
        return self._metric.compute()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Update the confusion matrix with output values.
        Args:
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth of the form `(y_pred, y)`.
        """
        flattened_targets = flatten(to_onehot(output[1], output[0].size(1))).double()
        ones = torch.Tensor().new_ones((output[0].size(1),), dtype=torch.double, device=flattened_targets.device)
        weights = ones / torch.pow(flattened_targets.sum(-1), 2).clamp(min=EPSILON)

        self._metric = self.create_generalized_dice_metric(self._cm, weights)

        if self._reduction == "mean":
            self._metric = self._metric.mean()
        elif self._reduction is None:
            pass
        else:
            raise NotImplementedError("Reduction method not implemented.")

        self._cm.update(output)

    def create_generalized_dice_metric(self, cm: ConfusionMatrix, weight: torch.Tensor):
        """
        Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)
        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
            weight (:obj:`torch.Tensor`): A weight vector which length equals to the number of classes.
        Returns:
            ignite.Metric: The Generalized Dice Coefficient Metric object.
        """

        # Increase floating point precision
        cm = cm.type(torch.float64)
        dice = 2 * (cm.diag() * weight) / (((cm.sum(dim=1) + cm.sum(dim=0)) * weight) + EPSILON)

        if self._ignore_index != -100:
            def remove_index(dice_vector):
                try:
                    indices = list(range(len(dice_vector)))
                    indices.remove(self._ignore_index)
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            self._ignore_index))

            return MetricsLambda(remove_index, dice)
        else:
            return dice


def to_tensor(value):
    if not isinstance(value, torch.Tensor):
        return torch.tensor([value])
    else:
        return value
