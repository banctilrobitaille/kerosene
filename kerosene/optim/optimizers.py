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

from torch import optim
from torch.optim import Optimizer

try:
    from apex.optimizers import FusedSGD, FusedAdam

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
except ImportError as e:
    APEX_AVAILABLE = False
    print("Unable to import apex optimizers please upgrade your apex version".format(e))


class OptimizerType(Enum):
    FusedSGD = "FusedSGD"
    FusedAdam = "FusedAdam"
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

        if APEX_AVAILABLE:
            self.register("FusedSGD", FusedSGD)
            self.register("FusedAdam", FusedAdam)

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
