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
from typing import List

import torch

try:
    from torch.distributed import is_nccl_available

    NCCL_AVAILABLE = is_nccl_available()
except ImportError:
    NCCL_AVAILABLE = False

from kerosene.configs.exceptions import InvalidConfigurationError
from kerosene.utils.devices import get_devices, on_multiple_gpus, num_gpus, on_cpu


class HtmlConfiguration(object):

    @abc.abstractmethod
    def to_html(self):
        raise NotImplementedError()


class DatasetConfiguration(HtmlConfiguration):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Dataset Configuration</h2> \n {}".format(configuration_values)


class Configuration(object):
    def __init__(self, name, type, params):
        self._name = name
        self._type = type
        self._params = params

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value


class ModelConfiguration(Configuration, HtmlConfiguration):
    def __init__(self, model_name, model_type, model_params, optimizer_type, optimizer_params, scheduler_type,
                 scheduler_params, criterions_configs: List[Configuration], metrics_configs: List[Configuration],
                 gradient_clipping_strategy, gradient_clipping_params):
        super().__init__(model_name, model_type, model_params)
        self._optimizer_type = optimizer_type
        self._optimizer_params = optimizer_params

        self._scheduler_type = scheduler_type
        self._scheduler_params = scheduler_params

        self._criterions_configs = criterions_configs
        self._metrics_configs = metrics_configs

        self._gradient_clipping_strategy = gradient_clipping_strategy
        self._gradient_clipping_params = gradient_clipping_params

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
    def criterions_configs(self):
        return self._criterions_configs

    @property
    def metrics_configs(self):
        return self._metrics_configs

    @property
    def gradient_clipping_strategy(self):
        return self._gradient_clipping_strategy

    @property
    def gradient_clipping_params(self):
        return self._gradient_clipping_params

    @classmethod
    def from_dict(cls, model_name, config_dict):
        try:
            return cls(model_name, config_dict["type"], config_dict.get("params"),
                       config_dict["optimizer"]["type"], config_dict["optimizer"].get("params"),
                       config_dict["scheduler"]["type"], config_dict["scheduler"].get("params"),
                       list(map(lambda criterion: Configuration(criterion, config_dict["criterion"][criterion]["type"],
                                                                config_dict["criterion"][criterion].get("params")),
                                config_dict["criterion"].keys())),
                       list(map(lambda metric: Configuration(metric, config_dict["metrics"][metric]["type"],
                                                             config_dict["metrics"][metric].get("params")),
                                config_dict["metrics"].keys())),
                       config_dict.get("gradients", {}).get("clipping_strategy"),
                       config_dict.get("gradients", {}).get("params"))
        except KeyError as e:
            raise InvalidConfigurationError(
                "The provided model configuration is invalid. The section {} is missing.".format(e))

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Model Configuration \n </h2><h4>{}</h4> \n {}".format(self.name, configuration_values)


class RunConfiguration(HtmlConfiguration):

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
                                                     world_size=int(os.environ["WORLD_SIZE"]), rank=self._local_rank)
            else:
                raise Exception("NCCL not available and required for multi-GPU training.")

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Run Configuration</h2> \n {}".format(configuration_values)


class TrainerConfiguration(HtmlConfiguration):
    def __init__(self, config_dict):
        for key in config_dict:
            setattr(self, key, config_dict[key])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Training Configuration</h2> \n {}".format(configuration_values)
