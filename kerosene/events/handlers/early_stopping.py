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

import logging

from kerosene.events.handlers.base_monitor_watcher import MonitorWatcher, MonitorPatienceExceeded
from kerosene.training.trainers import Trainer


class EarlyStopping(MonitorWatcher):
    LOGGER = logging.getLogger("EarlyStopping")

    def __call__(self, trainer: Trainer):
        for model_trainer in trainer.model_trainers:
            try:
                value = getattr(model_trainer, str(self._monitor))
                self.watch(model_trainer.name, value)
            except MonitorPatienceExceeded as e:
                model_trainer._finalize()
