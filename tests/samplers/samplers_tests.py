import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed
from hamcrest import *
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

from kerosene.samplers.samplers import SamplerType, SamplerFactory


class SamplersTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(SamplersTest, cls).setUpClass()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "65000"
        torch.distributed.init_process_group("nccl", init_method="env://", world_size=1, rank=0)

    def setUp(self) -> None:
        self._dataset = MNIST('./files/', train=True, download=True)

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_sampler_with_one_device(self, mocked_devices):
        mocked_devices.return_value = False
        sampler = SamplerFactory().create(SamplerType.Sampler, self._dataset, None)
        assert_that(sampler, instance_of(torch.utils.data.Sampler))

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_distributed_sampler_with_one_device(self, mocked_devices):
        mocked_devices.return_value = False
        sampler = SamplerFactory().create(SamplerType.DistributedSampler, self._dataset, None)
        assert_that(sampler, instance_of(torch.utils.data.DistributedSampler))