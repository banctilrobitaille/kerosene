from enum import Enum
from typing import Union, Tuple

import abc
import torch

from ignite.metrics import Accuracy, Precision, MeanAbsoluteError, MeanPairwiseDistance, MeanSquaredError, Recall, \
    RootMeanSquaredError, TopKCategoricalAccuracy, IoU, mIoU, Metric, ConfusionMatrix, MetricsLambda


class MetricType(Enum):
    Dice = "Dice"
    GeneralizedDice = "GeneralizeDice"
    Accuracy = "Accuracy"
    Precision = "Precision"
    MeanAbsoluteError = "MeanAbsoluteError"
    MeanPairwiseDistance = "MeanPairwiseDistance"
    MeanSquaredError = "MeanSquaredError"
    Recall = "Recall"
    RootMeanSquaredError = "RootMeanSquaredError"
    TopKCategoricalAccuracy = "TopKCategoricalAccuracy"
    IoU = "IoU"
    mIoU = "mIoU"

    def __str__(self):
        return self.value


class MetricFactory(object):
    def __init__(self):
        self._metrics = {
            "Accuracy": Accuracy,
            "Precision": Precision,
            "MeanAbsoluteError": MeanAbsoluteError,
            "MeanPairwiseDistance": MeanPairwiseDistance,
            "MeanSquaredError": MeanSquaredError,
            "Recall": Recall,
            "RootMeanSquaredError": RootMeanSquaredError,
            "TopKCategoricalAccuracy": TopKCategoricalAccuracy,
            "IoU": IoU,
            "mIoU": mIoU,
        }

    def create(self, metric_type: Union[str, MetricType], params):
        return self._metrics[str(metric_type)](**params)

    def register(self, metric: str, creator: Metric):
        """
        Add a new activation layer.
        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator


class Dice(Metric):
    """
    The Dice Metric.
    """

    def __init__(self, num_classes: int, reduction: str = None, average: str = None, ignore_index: int = -100,
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
                :class:`~ignite.engine.Engine`'s `process_function`'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
        self._num_classes = num_classes
        self._average = average
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._cm = ConfusionMatrix(num_classes=num_classes, average=average,
                                   output_transform=output_transform)
        self._metric = self.compute_dice_coefficient(self._cm, self._ignore_index)

        if self._reduction == "mean":
            self._metric = self.compute_mean_dice_coefficient(self._cm, self._ignore_index)
        elif None:
            pass
        else:
            raise NotImplementedError("Reduction method not implemented.")

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
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth.
        """
        self._cm.update(output)

    @staticmethod
    def compute_dice_coefficient(cm: ConfusionMatrix, ignore_index: int = -100):
        """
        Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)

        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
            ignore_index (int): An index of a class to ignore for computation.

        Returns:
            array: The Sørensen–Dice Coefficient for each class.
        """
        # Increase floating point precision
        cm = cm.type(torch.float64)
        dice = 2 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + EPSILON)

        if ignore_index is not None:
            def remove_index(dice_vector):
                try:
                    indices = [x for x in range(len(dice_vector)) if x != ignore_index]
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            ignore_index))

            return MetricsLambda(remove_index, dice)
        else:
            return dice

    @staticmethod
    def compute_mean_dice_coefficient(cm: ConfusionMatrix, ignore_index: int = None):
        """
        Computes the mean Sørensen–Dice Coefficient.

        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
            ignore_index (int): An index of a class to ignore for computation.

        Returns:
            float: The mean Sørensen–Dice Coefficient.
        """
        return Dice.compute_dice_coefficient(cm=cm, ignore_index=ignore_index).mean()


class GeneralizedDice(Metric):
    """
    The Generalized Dice Metric.
    """

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
                :class:`~ignite.engine.Engine`'s `process_function`'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
        """
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

    def update(self, output: Tuple[torch.Tensor, torch.Tensor], weight: torch.Tensor = None) -> None:
        """
        Update the confusion matrix with output values.

        Args:
            output (tuple of :obj:`torch.Tensor`): A tuple containing predictions and ground truth.
            weight (:obj:`torch.Tensor`, optional): A weight vector which length equals to the number of classes.
        """

        self._metric = self.compute_generalized_dice_coefficient(self._cm, weight, self._ignore_index)

        if self._reduction == "mean":
            self._metric = self._metric.mean()
        else:
            raise NotImplementedError("Reduction method not implemented.")

        self._cm.update(output)

    @staticmethod
    def compute_mean_generalized_dice_coefficient(cm: ConfusionMatrix, weight: torch.Tensor,
                                                  ignore_index: int = -100):
        """
        Computes the mean Generalized Dice Coefficient.

        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
            weight (:obj:`torch.Tensor`): A tensor representing weights for each classes.
            ignore_index (int): An index of a class to ignore for computation.

        Returns:
            float: The mean Generalized Dice Coefficient.
        """
        return GeneralizedDice.compute_generalized_dice_coefficient(cm=cm, ignore_index=ignore_index,
                                                                    weight=weight).mean()

    @staticmethod
    def compute_generalized_dice_coefficient(cm: ConfusionMatrix, weight: torch.Tensor,
                                             ignore_index: int = -100):
        """
        Computes the Sørensen–Dice Coefficient (https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)

        Args:
            cm (:obj:`ignite.metrics.ConfusionMatrix`): A confusion matrix representing the classification of data.
            ignore_index (int): An index of a class to ignore for computation.
            weight (:obj:`torch.Tensor`): A weight vector which length equals to the number of classes.

        Returns:
            array: The Generalized Dice Coefficient for each class.
        """

        # Increase floating point precision
        cm = cm.type(torch.float64)
        dice = 2 * (cm.diag() * weight) / (((cm.sum(dim=1) + cm.sum(dim=0)) * weight) + EPSILON)

        if ignore_index != -100:
            def remove_index(dice_vector):
                try:
                    indices = [x for x in range(len(dice_vector)) if x != ignore_index]
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            ignore_index))

            return MetricsLambda(remove_index, dice)
        else:
            return dice
