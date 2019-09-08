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
from typing import Callable

from kerosene.events import Event
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.trainers import Trainer


class EventPreprocessor(ABC):

    @abstractmethod
    def __call__(self, event: Event, state: Trainer):
        raise NotImplementedError()


class Identity(EventPreprocessor):

    def __call__(self, event: Event, state: Trainer):
        return state


class HandlerPreprocessor(EventPreprocessor):
    def __init__(self, handler: EventHandler, preprocessor: Callable = Identity()):
        self._handler = handler
        self._preprocessor = preprocessor

    def __call__(self, event: Event, state: Trainer):
        processed_state = self._preprocessor(event, state)
        self._handler(processed_state)
