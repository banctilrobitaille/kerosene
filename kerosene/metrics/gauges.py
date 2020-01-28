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
from abc import ABC, abstractmethod


class Gauge(ABC):

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class AverageGauge(Gauge):

    def __init__(self):
        self._value = 0.0
        self._sum = 0.0
        self._count = 0

    @property
    def count(self):
        return self._count

    def has_been_updated(self):
        return True if self._count > 0 else False

    def update(self, value, n=1):
        self._value = value
        self._sum += value * n
        self._count += n

    def compute(self):
        return self._sum / self._count if self._count != 0 else 0.0

    def reset(self):
        self._value = 0.0
        self._sum = 0.0
        self._count = 0
