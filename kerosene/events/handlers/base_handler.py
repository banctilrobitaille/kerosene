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
from typing import List, Union

from kerosene.events import TemporalEvent, BaseEvent
from kerosene.events.exceptions import UnsupportedEventException


class EventHandler(ABC):

    def __init__(self, supported_events: List[Union[BaseEvent, TemporalEvent]] = None, every=1):
        self._every = every
        self._supported_events = supported_events

    @property
    def every(self):
        return self._every

    @property
    def supported_events(self):
        return self._supported_events

    def should_handle(self, event: TemporalEvent):
        if (self.supported_events is not None) and (event not in self.supported_events):
            raise UnsupportedEventException(self.supported_events, event)

        if iter == 0 and self._every != 1:
            return False
        else:
            return event.iteration % self._every == 0

    @abstractmethod
    def __call__(self, temporal_event: TemporalEvent, monitors, sender):
        raise NotImplementedError()
