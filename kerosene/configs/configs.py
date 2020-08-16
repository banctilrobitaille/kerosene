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

    def update(self, config_dict):
        if "type" in config_dict:
            self._type = config_dict["type"]

        if "params" in config_dict:
            self._params.update(config_dict["params"])

    @classmethod
    def from_dict(cls, name, config_dict):
        return cls(name, config_dict["type"], config_dict.get("params", {}))


class ConfigurationList(Configuration):
    def __init__(self, configurations: List[Configuration]):
        super().__init__(list(map(lambda config: config.name, configurations)),
                         list(map(lambda config: config.type, configurations)),
                         list(map(lambda config: config.params, configurations)))
        self._configurations = configurations

    def update(self, config_dict):
        for configuration in self._configurations:
            if configuration.name in config_dict:
                configuration.update(config_dict[configuration.name])

    def __len__(self):
        return len(self._configurations)

    def __getitem__(self, index):
        return self._configurations[index]

    def __iter__(self):
        return iter(self._configurations)


class ModelConfiguration(Configuration, HtmlConfiguration):
    def __init__(self, model_name, model_type, model_params, path, optimizer_config: Configuration,
                 scheduler_config: Configuration, criterions_configs: ConfigurationList,
                 metrics_configs: ConfigurationList, gradient_clipping_config: Configuration):
        super().__init__(model_name, model_type, model_params)
        self._path = path
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._criterions_configs = criterions_configs
        self._metrics_configs = metrics_configs
        self._gradient_clipping_config = gradient_clipping_config

    @property
    def path(self):
        return self._path

    @property
    def optimizer_config(self):
        return self._optimizer_config

    @property
    def scheduler_config(self):
        return self._scheduler_config

    @property
    def criterions_configs(self):
        return self._criterions_configs

    @property
    def metrics_configs(self):
        return self._metrics_configs

    @property
    def gradient_clipping_config(self):
        return self._gradient_clipping_config

    def update(self, config_dict):
        if "type" in config_dict:
            self._type = config_dict["type"]

        if "params" in config_dict:
            self._params.update(config_dict["params"])

        if "optimizer" in config_dict:
            self._optimizer_config.update(config_dict["optimizer"])

        if "scheduler" in config_dict:
            self._scheduler_config.update(config_dict["scheduler"])

        if "criterions" in config_dict:
            self._criterions_configs.update(config_dict["criterions"])

        if "metrics" in config_dict:
            self._metrics_configs.update(config_dict["metrics"])

    @classmethod
    def from_dict(cls, model_name, config_dict):
        try:
            optimizer_config = Configuration.from_dict("", config_dict["optimizer"])
            scheduler_config = Configuration.from_dict("", config_dict["scheduler"])
            criterion_configs = ConfigurationList(
                [Configuration.from_dict(criterion, config_dict["criterion"][criterion]) for criterion
                 in config_dict["criterion"].keys()])

            if "metrics" in config_dict.keys():
                metric_configs = ConfigurationList(
                    [Configuration.from_dict(metric, config_dict["metrics"][metric]) for metric in
                     config_dict["metrics"].keys()])
            else:
                metric_configs = None

            if "gradients" in config_dict.keys():
                gradient_clipping_config = Configuration.from_dict("", config_dict["gradients"])
            else:
                gradient_clipping_config = None

            return cls(model_name, config_dict["type"], config_dict.get("params"), config_dict.get("path"),
                       optimizer_config, scheduler_config, criterion_configs, metric_configs, gradient_clipping_config)
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
