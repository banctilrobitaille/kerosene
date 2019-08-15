from enum import Enum
from typing import Union

from ignite.metrics import Accuracy, Precision, MeanAbsoluteError, MeanPairwiseDistance, MeanSquaredError, Recall, \
    RootMeanSquaredError, TopKCategoricalAccuracy, IoU, mIoU, Metric


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


class MetricsFactory(object):
    def __init__(self):
        super(MetricsFactory, self).__init__()
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

    def create_metric(self, metric_type: Union[str, MetricType], params):
        return self._metrics[str(metric_type)](**params)

    def register(self, metric: str, creator: Metric):
        """
        Add a new activation layer.
        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator
