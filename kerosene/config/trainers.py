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
import multiprocessing
from typing import Union, Iterable

import torch

from kerosene.config.exceptions import InvalidConfigurationError


class RunConfiguration(object):
    def __init__(self, use_amp: bool = True, amp_opt_level: str = 'O2', local_rank: int = 0,
                 num_workers: int = multiprocessing.cpu_count()):
        self._use_amp = use_amp
        self._amp_opt_level = amp_opt_level
        self._devices = ([torch.device("cuda:{}".format(device_id)) for device_id in
                          range(torch.cuda.device_count())]) if torch.cuda.is_available() else [torch.device("cpu")]
        self._local_rank = local_rank
        self._device = self._devices[self._local_rank]
        self._num_workers = num_workers

    @property
    def use_amp(self):
        return self._use_amp

    @property
    def amp_opt_level(self):
        return self._amp_opt_level

    @property
    def devices(self):
        return self._devices

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def device(self):
        return self._device

    @property
    def num_workers(self):
        return self._num_workers


class TrainerConfiguration(object):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Training Configuration</h2> \n {}".format(configuration_values)


class ModelTrainerConfiguration(object):
    def __init__(self, model_name, model_type, model_params, optimizer_type, optimizer_params, scheduler_type,
                 scheduler_params, criterion_type, criterion_params, metric_type, metric_params):
        self._model_name = model_name
        self._model_type = model_type
        self._model_params = model_params

        self._optimizer_type = optimizer_type
        self._optimizer_params = optimizer_params

        self._scheduler_type = scheduler_type
        self._scheduler_params = scheduler_params

        self._criterion_type = criterion_type
        self._criterion_params = criterion_params

        self._metric_type = metric_type
        self._metric_params = metric_params

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_params(self):
        return self._model_params

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @property
    def optimizer_params(self):
        return self._optimizer_params

    @property
    def scheduler_type(self):
        return self._scheduler_type

    @property
    def scheduler_params(self):
        return self._scheduler_params

    @property
    def criterion_type(self):
        return self._criterion_type

    @property
    def criterion_params(self):
        return self._criterion_params

    @property
    def metric_type(self):
        return self._metric_type

    @property
    def metric_params(self):
        return self._metric_params

    @classmethod
    def from_dict(cls, model_name, config_dict):
        try:
            return cls(model_name, config_dict["type"], config_dict.get("params"), config_dict["optimizer"]["type"],
                       config_dict["optimizer"].get("params"), config_dict["scheduler"]["type"],
                       config_dict["scheduler"].get("params"), config_dict["criterion"]["type"],
                       config_dict["criterion"].get("params"), config_dict["metric"]["type"],
                       config_dict["metric"].get("params"))
        except KeyError as e:
            raise InvalidConfigurationError(
                "The provided model configuration is invalid. The section {} is missing.".format(e))

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h4>{}</h4> \n {}".format(self._model_name, configuration_values)
