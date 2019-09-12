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

from kerosene.events import Event


class EventHandler(ABC):

    def __init__(self, every=1):
        self._every = every

    @property
    def every(self):
        return self._every

    def should_handle_epoch_data(self, event, trainer):
        return (event in [Event.ON_EPOCH_BEGIN, Event.ON_EPOCH_END, Event.ON_TRAIN_EPOCH_BEGIN,
                          Event.ON_TRAIN_EPOCH_END, Event.ON_VALID_EPOCH_BEGIN, Event.ON_VALID_EPOCH_END]) and (
                           trainer.epoch % self._every == 0)

    def should_handle_step_data(self, event, trainer):
        if event in [Event.ON_TRAIN_BATCH_BEGIN, Event.ON_TRAIN_BATCH_END, Event.ON_BATCH_END]:
            return trainer.current_train_step % self._every == 0
        elif event in [Event.ON_VALID_BATCH_BEGIN, Event.ON_VALID_BATCH_END, Event.ON_BATCH_END]:
            return trainer.current_valid_step % self._every == 0

    @abstractmethod
    def __call__(self, *inputs):
        raise NotImplementedError()
