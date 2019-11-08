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
from kerosene.events import Event
from kerosene.training import Status
from kerosene.training.trainers import Trainer


class BatchEventMixin(object):
    def _on_batch_begin(self):
        assert isinstance(self, Trainer)
        self.on_batch_begin()
        self.fire(Event.ON_BATCH_BEGIN)

    def _on_batch_end(self):
        assert isinstance(self, Trainer)
        self.on_batch_end()
        self.fire(Event.ON_BATCH_END)

    def _on_train_batch_begin(self):
        assert isinstance(self, Trainer)
        self.on_train_batch_begin()
        self.fire(Event.ON_TRAIN_BATCH_BEGIN)

    def _on_train_batch_end(self):
        assert isinstance(self, Trainer)
        self.on_train_batch_end()
        self.fire(Event.ON_TRAIN_BATCH_END)

    def _on_valid_batch_begin(self):
        assert isinstance(self, Trainer)
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_BEGIN)

    def _on_valid_batch_end(self):
        assert isinstance(self, Trainer)
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_END)

    def _on_test_batch_begin(self):
        assert isinstance(self, Trainer)
        self.on_test_batch_begin()
        self.fire(Event.ON_TEST_BATCH_BEGIN)

    def _on_test_batch_end(self):
        assert isinstance(self, Trainer)
        self.on_test_batch_end()
        self.fire(Event.ON_TEST_BATCH_END)


class EpochEventMixin(object):
    def _on_epoch_begin(self):
        assert isinstance(self, Trainer)
        self._reset_model_trainers()
        self.on_epoch_begin()
        self.fire(Event.ON_EPOCH_BEGIN)

    def _on_epoch_end(self):
        assert isinstance(self, Trainer)
        self.scheduler_step()
        self.on_epoch_end()
        self.fire(Event.ON_EPOCH_END)

    def _on_train_epoch_begin(self):
        assert isinstance(self, Trainer)
        self._status = Status.TRAINING
        self.on_train_epoch_begin()
        self.fire(Event.ON_TRAIN_EPOCH_BEGIN)

    def _on_train_epoch_end(self):
        assert isinstance(self, Trainer)
        self.on_train_epoch_end()
        self.fire(Event.ON_TRAIN_EPOCH_END)

    def _on_valid_epoch_begin(self):
        assert isinstance(self, Trainer)
        self.on_valid_epoch_begin()
        self.fire(Event.ON_VALID_EPOCH_BEGIN)

    def _on_valid_epoch_end(self):
        assert isinstance(self, Trainer)
        self.on_valid_epoch_end()
        self.fire(Event.ON_VALID_EPOCH_END)

    def _on_test_epoch_begin(self):
        assert isinstance(self, Trainer)
        self.on_test_epoch_begin()
        self.fire(Event.ON_TEST_EPOCH_BEGIN)

    def _on_test_epoch_end(self):
        assert isinstance(self, Trainer)
        self.on_test_epoch_end()
        self.fire(Event.ON_TEST_EPOCH_END)


class PhaseEventMixin(object):
    def _on_training_begin(self):
        assert isinstance(self, Trainer)
        self.on_training_begin()
        self._status = Status.READY
        self.fire(Event.ON_TRAINING_BEGIN)

    def _on_training_end(self):
        assert isinstance(self, Trainer)
        self.on_training_end()
        self.fire(Event.ON_TRAINING_END)

    def _on_valid_begin(self):
        assert isinstance(self, Trainer)
        self.on_valid_begin()
        self._status = Status.VALIDATING
        self.fire(Event.ON_TRAINING_BEGIN)

    def _on_valid_end(self):
        assert isinstance(self, Trainer)
        self.on_valid_end()
        self.fire(Event.ON_TRAINING_END)

    def _on_test_begin(self):
        assert isinstance(self, Trainer)
        self.on_test_begin()
        self._status = Status.TESTING
        self.fire(Event.ON_TEST_BEGIN)

    def _on_test_end(self):
        assert isinstance(self, Trainer)
        self.on_test_end()
        self.fire(Event.ON_TEST_END)
