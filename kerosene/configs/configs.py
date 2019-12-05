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
import abc
import os
from socket import socket

import torch

try:
    from torch.distributed import is_nccl_available

    NCCL_AVAILABLE = is_nccl_available()
except ImportError:
    NCCL_AVAILABLE = False

from kerosene.configs.exceptions import InvalidConfigurationError
from kerosene.utils.devices import get_devices, on_multiple_gpus, num_gpus, on_cpu


class Configuration(object):

    @abc.abstractmethod
    def to_html(self):
        raise NotImplementedError()


class DatasetConfiguration(Configuration):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Dataset Configuration</h2> \n {}".format(configuration_values)


class ModelTrainerConfiguration(Configuration):
    def __init__(self, model_name, model_type, path, model_params, optimizer_types, optimizer_params, scheduler_types,
                 scheduler_params, criterion_type, criterion_params, metric_types, metric_params,
                 gradient_clipping_strategy, gradient_clipping_params):

        self._model_name = model_name
        self._model_type = model_type
        self._path = path
        self._model_params = model_params

        self._optimizer_types = optimizer_types
        self._optimizer_params = optimizer_params

        self._scheduler_types = scheduler_types
        self._scheduler_params = scheduler_params

        self._criterion_type = criterion_type
        self._criterion_params = criterion_params

        self._metric_types = metric_types
        self._metric_params = metric_params

        self._gradient_clipping_strategy = gradient_clipping_strategy
        self._gradient_clipping_params = gradient_clipping_params

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
    def optimizer_types(self):
        return self._optimizer_types

    @property
    def path(self):
        return self._path

    @property
    def optimizer_params(self):
        return self._optimizer_params

    @property
    def scheduler_types(self):
        return self._scheduler_types

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
    def metric_types(self):
        return self._metric_types

    @property
    def metric_params(self):
        return self._metric_params

    @property
    def gradient_clipping_strategy(self):
        return self._gradient_clipping_strategy

    @property
    def gradient_clipping_params(self):
        return self._gradient_clipping_params

    @classmethod
    def from_dict(cls, model_name, config_dict):
        try:
            return cls(model_name, config_dict["type"], config_dict.get("path"), config_dict.get("params"),
                       [optimizer["type"] for optimizer in config_dict["optimizers"]],
                       [optimizer.get("params") for optimizer in config_dict["optimizers"]],
                       [scheduler["type"] for scheduler in config_dict["schedulers"]],
                       [scheduler.get("params") for scheduler in config_dict["schedulers"]],
                       config_dict["criterion"]["type"], config_dict["criterion"].get("params"),
                       [metric["type"] for metric in config_dict["metrics"]],
                       [metric.get("params") for metric in config_dict["metrics"]],
                       config_dict.get("gradients", {}).get("clipping_strategy"),
                       config_dict.get("gradients", {}).get("params"))
        except KeyError as e:
            raise InvalidConfigurationError(
                "The provided model configuration is invalid. The section {} is missing.".format(e))

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Model Configuration \n </h2><h4>{}</h4> \n {}".format(self._model_name, configuration_values)


class RunConfiguration(Configuration):

    def __init__(self, use_amp: bool = True, amp_opt_level: str = 'O1', local_rank: int = 0,
                 world_size: int = num_gpus()):
        self._use_amp = use_amp
        self._amp_opt_level = amp_opt_level
        self._local_rank = local_rank
        self._world_size = world_size

        self._devices = get_devices()
        self._device = self._devices[self._local_rank]

        if not on_cpu(self._device):
            torch.cuda.set_device(self._device)
            self._initialize_ddp_process_group()

    @property
    def use_amp(self):
        return self._use_amp

    @property
    def amp_opt_level(self):
        return self._amp_opt_level

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def devices(self):
        return self._devices

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @staticmethod
    def _get_random_free_port():
        with socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _initialize_ddp_process_group(self):
        if on_multiple_gpus(self._devices):
            if NCCL_AVAILABLE:
                if os.environ.get("MASTER_ADDR") is None:
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                if os.environ.get("MASTER_PORT") is None:
                    os.environ["MASTER_PORT"] = str(self._get_random_free_port())
                if os.environ.get("WORLD_SIZE") is None:
                    os.environ["WORLD_SIZE"] = str(self._world_size)
                torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                                     world_size=os.environ["WORLD_SIZE"], rank=self._local_rank)
            else:
                raise Exception("NCCL not available and required for multi-GPU training.")

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Run Configuration</h2> \n {}".format(configuration_values)


class TrainerConfiguration(Configuration):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Training Configuration</h2> \n {}".format(configuration_values)
