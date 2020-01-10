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

from kerosene.events import Moment


class EventHandler(ABC):

    def __init__(self, every=1):
        self._every = every

    @property
    def every(self):
        return self._every

    def should_handle(self, moment: Moment):
        if iter == 0 and self._every != 1:
            return False
        else:
            return moment.iteration % self._every == 0

    @abstractmethod
    def __call__(self, temporal_event, monitors, sender):
        raise NotImplementedError()
