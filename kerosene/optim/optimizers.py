from enum import Enum
from typing import Union

from torch import optim
from torch.optim import Optimizer


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
        self._optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
        }

    def create(self, optimizer_type: Union[str, OptimizerType], model_params, params):
        return self._optimizers[str(optimizer_type)](model_params, **params) if params is not None else \
            self._optimizers[str(optimizer_type)](model_params)

    def register(self, function: str, creator: Optimizer):
        """
        Add a new activation layer.
        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._optimizers[function] = creator
