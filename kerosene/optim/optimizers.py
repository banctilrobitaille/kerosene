from enum import Enum
from typing import Union

from torch import optim


class OptimizerType(Enum):
    SGD = "SGD"
    Adam = "Adam"
    Adagrad = "Adagrad"
    Adadelta = "Adadelta"
    SparseAdam = "SparseAdam"
    Adamax = "Adamax"
    Rprop = "Rprop"
    RMSprop = "RMSprop"
    ASGD = "ASGD"

    def __str__(self):
        return self.value


class OptimizerFactory(object):

    def __init__(self):
        super(OptimizerFactory, self).__init__()

        self._optimizers = {
            "Adam": optim.Adam,
            "Adagrad": optim.Adagrad,
            "SGD": optim.SGD,
            "Adadelta": optim.Adadelta,
            "SparseAdam": optim.SparseAdam,
            "Adamax": optim.Adamax,
            "Rprop": optim.Rprop,
            "RMSprop": optim.RMSprop,
            "ASGD": optim.ASGD
        }

    def create(self, optimizer_type: Union[str, OptimizerType], params):
        return self._optimizers[str(optimizer_type)](**params)

    def register(self, function: str, creator: torch.optim.Optimizer):
        """
        Add a new activation layer.
        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._optimizers[function] = creator
