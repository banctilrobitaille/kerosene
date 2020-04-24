import unittest

import torch
from hamcrest import *

from kerosene.utils.devices import on_multiple_gpus, on_single_gpu, get_devices


class DeviceTests(unittest.TestCase):

    def setUp(self) -> None:
        self._multiple_gpus_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        self._single_gpu_device = [torch.device("cuda:0")]
        self._single_cpu_device = [torch.device("cpu")]

    def test_on_single_gpu_should_return_true_with_single_GPU_device(self):
        assert_that(on_single_gpu(self._single_gpu_device), is_(True))

    def test_on_single_gpu_should_return_false_with_multiple_GPU_devices(self):
        assert_that(on_single_gpu(self._multiple_gpus_devices), is_(False))

    def test_on_multiple_gpu_should_return_false_with_single_GPU_device(self):
        assert_that(on_multiple_gpus(self._single_gpu_device), is_(False))

    def test_on_multiple_gpu_should_return_true_with_multiple_GPU_devices(self):
        assert_that(on_multiple_gpus(self._multiple_gpus_devices), is_(True))

    def test_get_devices_should_return_at_least_one_cuda_enabled_device(self):
        if torch.cuda.is_available():
            assert_that(get_devices(), has_item(torch.device("cuda:0")))
            assert_that(get_devices(), has_length(torch.cuda.device_count()))
            assert_that(get_devices(), is_(
                [torch.device("cuda:{}".format(id)) for id in range(int(torch.cuda.device_count()))]))
        else:
            assert_that(get_devices(), is_([torch.device("cpu")]))
