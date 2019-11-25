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

from kerosene.events import Event, BaseEvent


class EventHandler(ABC):

    def __init__(self, every=1):
        self._every = every

    @property
    def every(self):
        return self._every

    def should_handle_epoch_data(self, event):
        return event in Event.epoch_events()

    def should_handle_train_step_data(self, event):
        return event in Event.train_batch_events()

    def should_handle_valid_step_data(self, event):
        return event in Event.valid_batch_events()

    @abstractmethod
    def __call__(self, event: BaseEvent, trainer, *args, **kwargs):
        raise NotImplementedError()


class PeriodicEventHandler(EventHandler, ABC):

    def __init__(self, every=1):
        super().__init__(every)
        self._every = every

    @property
    def every(self):
        return self._every

    def should_handle_iteration(self, iter):
        if iter == 0 and self._every != 1:
            return False
        else:
            return iter % self._every == 0
