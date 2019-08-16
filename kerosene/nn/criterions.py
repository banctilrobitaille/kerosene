from enum import Enum
from typing import Union

from torch import nn
from torch.nn.modules.loss import _Loss


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

    def create(self, criterion_type: Union[CriterionType, str], **params):
        return self._criterion[str(criterion_type)](**params)

    def register(self, function: str, creator: _Loss):
        """
        Add a new criterion.
        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator
