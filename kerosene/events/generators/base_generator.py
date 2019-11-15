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

from kerosene.events import BaseEvent


class EventGenerator(ABC):
    def __init__(self):
        self._event_handlers = {}

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError()

    @abstractmethod
    def with_event_handler(self, handler, event: BaseEvent):
        raise NotImplementedError()

    def fire(self, event: BaseEvent):
        if event in self._event_handlers.keys():
            state = self.state

            for handler in self._event_handlers[event]:
                handler(event, state)

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_valid_batch_begin(self):
        pass

    def on_valid_batch_end(self):
        pass

    def on_test_batch_begin(self):
        pass

    def on_test_batch_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_begin(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def on_test_epoch_begin(self):
        pass

    def on_test_epoch_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass

    def on_valid_begin(self):
        pass

    def on_valid_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self):
        pass

    def finalize(self):
        pass
