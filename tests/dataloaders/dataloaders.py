import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed
from hamcrest import *
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

from kerosene.dataloaders.dataloaders import DataLoaderFactory
from kerosene.exceptions.exceptions import IncorrectSamplerTypeException
from kerosene.samplers.samplers import SamplerType, SamplerFactory


class DataLoaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(DataLoaderTest, cls).setUpClass()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "65000"
        torch.distributed.init_process_group("nccl", init_method="env://", world_size=1, rank=0)

    def setUp(self) -> None:
        self._dataset = MNIST('./files/', train=True, download=True)
        self._distributed_sampler = SamplerFactory().create(SamplerType.DistributedSampler, self._dataset, None)
        self._normal_sampler = SamplerFactory().create(SamplerType.Sampler, self._dataset, None)

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_dataloader_without_sampler_with_one_device(self, mocked_devices):
        mocked_devices.return_value = False
        dataloader = DataLoaderFactory().create(self._dataset, None)
        assert_that(dataloader, instance_of(torch.utils.data.DataLoader))

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_dataloader_with_distributed_sampler(self, mocked_devices):
        mocked_devices.return_value = True
        dataloader = DataLoaderFactory().create(self._dataset, sampler=self._distributed_sampler)
        assert_that(dataloader, instance_of(torch.utils.data.DataLoader))

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_dataloader_with_normal_sampler_with_one_device(self, mocked_devices):
        mocked_devices.return_value = False
        dataloader = DataLoaderFactory().create(self._dataset, sampler=self._normal_sampler)
        assert_that(dataloader, instance_of(torch.utils.data.DataLoader))

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_create_dataloader_with_distributed_sampler_with_one_device(self, mocked_devices):
        mocked_devices.return_value = False
        dataloader = DataLoaderFactory().create(self._dataset, sampler=self._normal_sampler)
        assert_that(dataloader, instance_of(torch.utils.data.DataLoader))

    @patch("kerosene.dataloaders.dataloaders.is_on_multiple_devices")
    def test_should_raise_exception_with_normal_sampler_and_multiple_devices(self, mocked_devices):
        mocked_devices.return_value = True
        assert_that(calling(DataLoaderFactory().create).with_args(self._dataset, sampler=self._normal_sampler),
                    raises(IncorrectSamplerTypeException))
