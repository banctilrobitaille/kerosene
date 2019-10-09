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
from typing import Union, Callable

from torch.nn import init


class InitializerType(Enum):
    Uniform = "Uniform"
    Normal = "Normal"
    Constant = "Constant"
    Ones = "Ones"
    Zeros = "Zeros"
    Eye = "Eye"
    Dirac = "Dirac"
    XavierUniform = "Xavier_Uniform"
    XavierNormal = "Xavier_Normal"
    KaimingUniform = "Kaiming_Uniform"
    KaimingNormal = "Kaiming_Normal"
    Orthogonal = "Orthogonal"
    Sparse = "Sparse"

    def __str__(self):
        return self.value


class InitializerFactory(object):
    def __init__(self):
        self._initializers = {
            "Uniform": init.uniform_,
            "Normal": init.normal_,
            "Constant": init.constant_,
            "Ones": init.ones_,
            "Zeros": init.zeros_,
            "Eye": init.eye_,
            "Dirac": init.dirac_,
            "XavierUniform": init.xavier_uniform_,
            "XavierNormal": init.xavier_normal_,
            "KaimingUniform": init.kaiming_uniform_,
            "KaimingNormal": init.kaiming_normal_,
            "Orthogonal": init.orthogonal_,
            "Sparse": init.sparse_,
        }

    def create(self, init_type: Union[str, InitializerType], params):
        return self._initializers[str(init_type)](**params) if params is not None else self._initializers[
            str(init_type)]()

    def register(self, initializer: str, creator: Callable):
        """
        Add a new initializer.
        Args:
           initializer (str): Initializer's name.
           creator: A Callable wrapping the new custom Initializer function.
        """
        self._initializers[initializer] = creator
