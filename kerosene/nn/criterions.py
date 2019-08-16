from enum import Enum
from typing import Union

import abc
import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from kerosene.utils import flatten
from ignite.metrics import MetricsLambda

SUPPORTED_REDUCTIONS = [None, "mean"]
EPSILON = 1e-15


class CriterionType(Enum):
    BCELoss = "BCELoss"
    BCEWithLogitsLoss = "BCEWithLogtisLoss"
    PoissonNLLLoss = "PoissonNLLLoss"
    CosineEmbeddingLoss = "CosineEmbeddingLoss"
    CrossEntropyLoss = "CrossEntropyLoss"
    CTCLoss = "CTCLoss"
    HingeEmbeddingLoss = "HingeEmbeddingLoss"
    KLDivLoss = "KLDivLoss"
    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    MarginRankingLoss = "MarginRankingLoss"
    MultiLabelMarginLoss = "MultiLabelMarginLoss"
    MultiLabelSoftMarginLoss = "MultiLabelSoftMarginLoss"
    MultiMarginLoss = "MultiMarginLoss"
    NLLLoss = "NLLLoss"
    SmoothL1Loss = "SmoothL1Loss"
    SoftMarginLoss = "SoftMarginLoss"
    TripletMarginLoss = "TripletMarginLoss"


class CriterionFactory(object):
    def __init__(self):
        super(CriterionFactory, self).__init__()
        self._criterion = {
            "DiceLoss": DiceLoss,
            "GeneralizeDiceLoss": GeneralizedDiceLoss,
            "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
            "BCELoss": nn.BCELoss,
            "BCEWithLogtisLoss": nn.BCEWithLogitsLoss,
            "PoissonNLLLoss": nn.PoissonNLLLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "CTCLoss": nn.CTCLoss,
            "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
            "KLDivLoss": nn.KLDivLoss,
            "L1Loss": nn.L1Loss,
            "MSELoss": nn.MSELoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
            "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
            "MultiMarginLoss": nn.MultiMarginLoss,
            "NLLLoss": nn.NLLLoss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "SoftMarginLoss": nn.SoftMarginLoss,
            "TripletMarginLoss": nn.TripletMarginLoss,
        }

    def create(self, criterion_type: Union[CriterionType, str], params):
        return self._criterion[str(criterion_type)](**params)

    def register(self, function: str, creator: _Loss):
        """
        Add a new criterion.
        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator


class AbstractCriterion(nn.Module):

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def validate_ignore_index(ignore_index: int, input_shape: int):
        """
        Validate the `ignore_index` variable.

        Args:
            ignore_index (int): An index of a class to ignore for computation.
            input_shape (int): Input tensor.
        """
        if ignore_index is not None:
            if not ignore_index >= 0:
                raise ValueError("'ignore_index' must be non-negative, but given {}".format(ignore_index))

        if ignore_index > input_shape:
            raise ValueError(
                "'ignore index' must be lower than the number of classes in confusion matrix, but {} was given.".format(
                    ignore_index))


class DiceLoss(AbstractCriterion):
    """
    The Sørensen-Dice Loss.
    """

    def __init__(self, reduction: Union[None, str] = "mean"):
        super(DiceLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction type not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None,
                ignore_index: int = None):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to
             be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.
            weights (:obj:`torch.Tensor`): A class weight vector of shape (C, ), one per class.

        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if ignore_index is not None:
            self.validate_ignore_index(ignore_index, inputs.shape[1])

        if not inputs.size() == targets.size():
            raise ValueError("'Inputs' and 'Targets' must have the same shape.")

        inputs = flatten(inputs)
        targets = flatten(targets)

        targets = targets.float()

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1)

        if weights is not None:
            if weights.requires_grad is not False:
                weights.requires_grad = False
            intersect = weights * intersect

        denominator = (inputs + targets).sum(-1)

        dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if ignore_index is not None:

            def ignore_index_fn(dice_vector):
                indices = list(range(len(dice_vector)))
                indices.remove(ignore_index)
                return dice_vector[indices]

            return MetricsLambda(ignore_index_fn, dice).compute()

        else:
            if self._reduction == "mean":
                dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON)).mean()
            elif self._reduction is None:
                pass
            else:
                raise NotImplementedError("Reduction method not implemented.")
            return dice


class GeneralizedDiceLoss(AbstractCriterion):
    """
      The Generalized Sørensen-Dice Loss.
      """

    def __init__(self, reduction: Union[None, str] = "mean"):
        super(GeneralizedDiceLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction type not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to
             be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.

        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if ignore_index is not None:
            self.validate_ignore_index(ignore_index, inputs.shape[1])

        if not inputs.size() == targets.size():
            raise ValueError("'Inputs' and 'Targets' must have the same shape.")

        inputs = flatten(inputs)
        targets = flatten(targets)

        targets = targets.float()

        class_weights = 1.0 / torch.pow(targets.sum(-1), 2).clamp(min=EPSILON)

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1) * class_weights

        denominator = (inputs + targets).sum(-1) * class_weights

        dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if ignore_index is not None:

            def ignore_index_fn(dice_vector):
                indices = list(range(len(dice_vector)))
                indices.remove(ignore_index)
                return dice_vector[indices]

            return MetricsLambda(ignore_index_fn, dice).compute()

        else:
            if self._reduction == "mean":
                dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON)).mean()
            elif self._reduction is None:
                pass
            else:
                raise NotImplementedError("Reduction method not implemented.")
            return dice


class WeightedCrossEntropyLoss(AbstractCriterion):

    def __init__(self, reduction="mean"):
        super(WeightedCrossEntropyLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction is not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = None) -> float:
        """
        Computes the Weighted Cross Entropy Loss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            inputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.
            targets (:obj:`torch.Tensor`): A tensor of shape (B, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.

       Returns:
           float: the weighted Cross-Entropy loss value.

        """
        if ignore_index is not None:
            self.validate_ignore_index(ignore_index, inputs.shape[1])
        else:
            ignore_index = -100

        class_weights = self.compute_class_weights(inputs)

        return torch.nn.functional.cross_entropy(inputs, targets, weight=class_weights,
                                                 ignore_index=ignore_index)

    @staticmethod
    def compute_class_weights(inputs: torch.Tensor):
        """
        Compute weights for each class as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            inputs: (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.

        Returns:
            :obj:`torch.Tensor`: A tensor containing class weights.
        """
        flattened_inputs = flatten(inputs)
        class_weights = (flattened_inputs.shape[1] - flattened_inputs.sum(-1)) / flattened_inputs.sum(-1)
        return class_weights
