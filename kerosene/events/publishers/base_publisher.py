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

from kerosene.events import BaseEvent, Event
from kerosene.training import Status


class EventPublisher(ABC):
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


class BatchEventPublisherMixin(object):
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

    def _on_batch_begin(self):
        self.on_batch_begin()
        self.fire(Event.ON_BATCH_BEGIN)

    def _on_batch_end(self):
        self.on_batch_end()
        self.fire(Event.ON_BATCH_END)

    def _on_train_batch_begin(self):
        self.on_train_batch_begin()
        self.fire(Event.ON_TRAIN_BATCH_BEGIN)

    def _on_train_batch_end(self):
        self.on_train_batch_end()
        self.fire(Event.ON_TRAIN_BATCH_END)

    def _on_valid_batch_begin(self):
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_BEGIN)

    def _on_valid_batch_end(self):
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_END)

    def _on_test_batch_begin(self):
        self.on_test_batch_begin()
        self.fire(Event.ON_TEST_BATCH_BEGIN)

    def _on_test_batch_end(self):
        self.on_test_batch_end()
        self.fire(Event.ON_TEST_BATCH_END)


class EpochEventPublisherMixin(object):
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

    def _on_epoch_begin(self):
        self._reset_model_trainers()
        self.on_epoch_begin()
        self.fire(Event.ON_EPOCH_BEGIN)

    def _on_epoch_end(self):
        self.scheduler_step()
        self.on_epoch_end()
        self.fire(Event.ON_EPOCH_END)

    def _on_train_epoch_begin(self):
        self._status = Status.TRAINING
        self.on_train_epoch_begin()
        self.fire(Event.ON_TRAIN_EPOCH_BEGIN)

    def _on_train_epoch_end(self):
        self.on_train_epoch_end()
        self.fire(Event.ON_TRAIN_EPOCH_END)

    def _on_valid_epoch_begin(self):
        self.on_valid_epoch_begin()
        self.fire(Event.ON_VALID_EPOCH_BEGIN)

    def _on_valid_epoch_end(self):
        self._current_valid_batch = 0
        self.on_valid_epoch_end()
        self.fire(Event.ON_VALID_EPOCH_END)

    def _on_test_epoch_begin(self):
        self.on_test_epoch_begin()
        self.fire(Event.ON_TEST_EPOCH_BEGIN)

    def _on_test_epoch_end(self):
        self._current_test_batch = 0
        self.on_test_epoch_end()
        self.fire(Event.ON_TEST_EPOCH_END)


class TrainingPhaseEventPublisherMixin(object):
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

    def on_before_zero_grad(self):
        pass

    def on_after_backward(self):
        pass

    def finalize(self):
        pass

    def _on_training_begin(self):
        self.on_training_begin()
        self._status = Status.READY
        self.fire(Event.ON_TRAINING_BEGIN)

    def _on_training_end(self):
        self.on_training_end()
        self.fire(Event.ON_TRAINING_END)

    def _on_valid_begin(self):
        self.on_valid_begin()
        self._status = Status.VALIDATING
        self.fire(Event.ON_TRAINING_BEGIN)

    def _on_valid_end(self):
        self.on_valid_end()
        self.fire(Event.ON_TRAINING_END)

    def _on_test_begin(self):
        self.on_test_begin()
        self._status = Status.TESTING
        self.fire(Event.ON_TEST_BEGIN)

    def _on_test_end(self):
        self.on_test_end()
        self.fire(Event.ON_TEST_END)

    def _on_before_zero_grad(self):
        self.on_before_zero_grad()
        self.fire(Event.ON_BEFORE_ZERO_GRAD)

    def _on_after_backward(self):
        self.on_after_backward()
        self.fire(Event.ON_AFTER_BACKWARD)

    def _finalize(self):
        self._status = Status.FINALIZING
        self.finalize()
        self.fire(Event.ON_FINALIZE)
        self._status = Status.FINALIZED
