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

from kerosene.events import BaseEvent, TemporalEvent


class EventPublisher(ABC):
    def __init__(self):
        self._event_handlers = {}

    @property
    @abstractmethod
    def sender(self):
        raise NotImplementedError()

    @abstractmethod
    def with_event_handler(self, handler, event: BaseEvent):
        raise NotImplementedError()

    def fire(self, temporal_event: TemporalEvent, monitors: dict = None):
        if temporal_event.event in self._event_handlers.keys():
            for handler in self._event_handlers[temporal_event.event]:
                handler(temporal_event, monitors, self.sender)
