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
from enum import Enum
from typing import Union

import torch
from ignite.metrics import MetricsLambda
from torch import nn
from torch.nn.modules.loss import _Loss

from kerosene.utils.constants import EPSILON
from kerosene.utils.tensors import flatten


class CriterionType(Enum):
    DiceLoss = "DiceLoss"
    GeneralizedDiceLoss = "GeneralizedDiceLoss"
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
            CriterionType.DiceLoss: DiceLoss,
            CriterionType.GeneralizedDiceLoss: GeneralizedDiceLoss,
            CriterionType.BCELoss: nn.BCELoss,
            CriterionType.BCEWithLogitsLoss: nn.BCEWithLogitsLoss,
            CriterionType.PoissonNLLLoss: nn.PoissonNLLLoss,
            CriterionType.CosineEmbeddingLoss: nn.CosineEmbeddingLoss,
            CriterionType.CrossEntropyLoss: nn.CrossEntropyLoss,
            CriterionType.CTCLoss: nn.CTCLoss,
            CriterionType.HingeEmbeddingLoss: nn.HingeEmbeddingLoss,
            CriterionType.KLDivLoss: nn.KLDivLoss,
            CriterionType.L1Loss: nn.L1Loss,
            CriterionType.MSELoss: nn.MSELoss,
            CriterionType.MarginRankingLoss: nn.MarginRankingLoss,
            CriterionType.MultiLabelMarginLoss: nn.MultiLabelMarginLoss,
            CriterionType.MultiLabelSoftMarginLoss: nn.MultiLabelSoftMarginLoss,
            CriterionType.MultiMarginLoss: nn.MultiMarginLoss,
            CriterionType.NLLLoss: nn.NLLLoss,
            CriterionType.SmoothL1Loss: nn.SmoothL1Loss,
            CriterionType.SoftMarginLoss: nn.SoftMarginLoss,
            CriterionType.TripletMarginLoss: nn.TripletMarginLoss,
        }

    def create(self, criterion_type: CriterionType, params):
        return self._criterion[str(criterion_type)](**params) if params is not None else self._criterion[
            str(criterion_type)]()

    def register(self, function: str, creator: _Loss):
        """
        Add a new criterion.
        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator


class DiceLoss(_Loss):
    """
    The Sørensen-Dice Loss.
    """
    SUPPORTED_REDUCTIONS = [None, "mean"]

    def __init__(self, reduction: Union[None, str] = "mean", ignore_index: int = -100, weight: torch.Tensor = None):
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise NotImplementedError("Reduction type not supported.")
        super(DiceLoss, self).__init__(reduction=reduction)
        self._ignore_index = ignore_index
        self._weight = weight

        if self._weight is not None:
            if self._weight.requires_grad is not False:
                self._weight.requires_grad = False

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to
             be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.

        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if not inputs.size() == targets.size():
            raise ValueError("'Inputs' and 'Targets' must have the same shape.")

        inputs = flatten(inputs)
        targets = flatten(targets).float()

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1)

        if self._weight is not None:
            intersect = self._weight * intersect

        denominator = (inputs + targets).sum(-1)

        ones = torch.Tensor().new_ones((inputs.size(0),), dtype=torch.float, device=inputs.device)

        dice = ones - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if self._ignore_index != -100:
            def ignore_index_fn(dice_vector):
                try:
                    indices = list(range(len(dice_vector)))
                    indices.remove(self._ignore_index)
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            self._ignore_index))

            dice = MetricsLambda(ignore_index_fn, dice).compute()

        if self.reduction == "mean":
            dice = dice.mean()

        return dice


class GeneralizedDiceLoss(_Loss):
    """
      The Generalized Sørensen-Dice Loss.
      """
    SUPPORTED_REDUCTIONS = [None, "mean"]

    def __init__(self, reduction: Union[None, str] = "mean", ignore_index: int = -100):
        if reduction not in self.SUPPORTED_REDUCTIONS:
            raise NotImplementedError("Reduction type not supported.")
        super(GeneralizedDiceLoss, self).__init__(reduction=reduction)
        self._ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to
             be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.

        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """
        if not inputs.size() == targets.size():
            raise ValueError("'Inputs' and 'Targets' must have the same shape.")

        inputs = flatten(inputs)
        targets = flatten(targets).float()
        ones = torch.Tensor().new_ones((inputs.size(0),), dtype=torch.float, device=inputs.device)
        class_weights = ones / torch.pow(targets.sum(-1), 2).clamp(min=EPSILON)

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1) * class_weights

        denominator = (inputs + targets).sum(-1) * class_weights

        dice = ones - (2.0 * intersect / denominator.clamp(min=EPSILON))

        if self._ignore_index != -100:
            def ignore_index_fn(dice_vector):
                try:
                    indices = list(range(len(dice_vector)))
                    indices.remove(self._ignore_index)
                    return dice_vector[indices]
                except ValueError as e:
                    raise IndexError(
                        "'ignore_index' must be non-negative, and lower than the number of classes in confusion matrix, but {} was given. ".format(
                            self._ignore_index))

            dice = MetricsLambda(ignore_index_fn, dice).compute()

        if self.reduction == "mean":
            dice = dice.mean()

        return dice


class WeightedCrossEntropyLoss(_Loss):

    def __init__(self, ignore_index: int = -100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes the Weighted Cross Entropy Loss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf

        Args:
            inputs (:obj:`torch.Tensor`): A tensor of shape (B, C, ..). The model's prediction on which the loss has to be computed.
            targets (:obj:`torch.Tensor`): A tensor of shape (B, ..). The ground truth.

       Returns:
           float: the weighted Cross-Entropy loss value.

        """
        class_weights = self.compute_class_weights(inputs)

        return torch.nn.functional.cross_entropy(inputs, targets, weight=class_weights,
                                                 ignore_index=self._ignore_index)

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
