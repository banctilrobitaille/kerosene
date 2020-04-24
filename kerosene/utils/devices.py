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
from typing import List

import torch


def on_cpu(device: torch.device):
    return str(device) == "cpu"


def on_gpu(device: torch.device):
    return device.type == "cuda"


def on_gpus(devices: List[torch.device]):
    return all([device.type == "cuda" for device in devices])


def on_single_device(devices: List[torch.device]):
    return len(devices) == 1


def on_multiple_devices(devices: List[torch.device]):
    return len(devices) > 1


def on_single_gpu(devices: List[torch.device]):
    return on_single_device(devices) and on_gpus(devices)


def on_multiple_gpus(devices: List[torch.device]):
    return on_multiple_devices(devices) and on_gpus(devices)


def get_devices():
    return [torch.device("cuda:{}".format(device_id)) for device_id in
            range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device("cpu")]


def num_gpus():
    return len(list(([device.type == "cuda" for device in get_devices()])))
