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

from typing import Callable, Optional

import torch

from kerosene.dataloaders.dataloaders import DataloaderFactory
from kerosene.datasets.datasets import DatasetFactory
from kerosene.training.trainers import ModelTrainerFactory
from kerosene.utils.devices import on_single_device
from kerosene.training.trainers import Trainer


class Experiment(object):
    def __init__(self, name=None, running_config=None, training_config=None, dataset_config=None,
                 model_trainer_configs=None, model_trainer_factory=None, dataset_factory=None, train_dataset=None,
                 valid_dataset=None, test_dataset=None, train_dataloader=None, valid_dataloader=None,
                 test_dataloader=None, model_trainers=None, trainer=None, collate_fn=None):
        self._name = name
        self._running_config = running_config
        self._training_config = training_config
        self._dataset_config = dataset_config
        self._model_trainer_configs = model_trainer_configs
        self._model_trainer_factory = model_trainer_factory
        self._dataset_factory = dataset_factory
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader
        self._model_trainers = model_trainers
        self._collate_fn = collate_fn
        self._trainer = trainer

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def running_config(self):
        return self._running_config

    @running_config.setter
    def running_config(self, config):
        self._running_config = config

    @property
    def training_config(self):
        return self._training_config

    @training_config.setter
    def training_config(self, config):
        self._training_config = config

    @property
    def dataset_config(self):
        return self._dataset_config

    @dataset_config.setter
    def dataset_config(self, config):
        self._dataset_config = config

    @property
    def model_trainer_configs(self):
        return self._model_trainer_configs

    @model_trainer_configs.setter
    def model_trainer_configs(self, configs):
        self._model_trainer_configs = configs

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @valid_dataset.setter
    def valid_dataset(self, dataset):
        self._valid_dataset = dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset):
        self._test_dataset = dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, dataloader):
        self._train_dataloader = dataloader

    @property
    def valid_dataloader(self):
        return self._valid_dataloader

    @valid_dataloader.setter
    def valid_dataloader(self, dataloader):
        self._valid_dataloader = dataloader

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @test_dataloader.setter
    def test_dataloader(self, dataloader):
        self._test_dataloader = dataloader

    @property
    def model_trainers(self):
        return self._model_trainers

    @model_trainers.setter
    def model_trainers(self, model_trainers):
        self._model_trainers = model_trainers

    @property
    def model_trainer_factory(self):
        return self._model_trainer_factory

    @model_trainer_factory.setter
    def model_trainer_factory(self, factory):
        self._model_trainer_factory = factory

    @property
    def dataset_factory(self):
        return self._dataset_factory

    @dataset_factory.setter
    def dataset_factory(self, factory):
        self._dataset_factory = factory

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn):
        self._collate_fn = collate_fn

    @classmethod
    def from_experiment(cls, running_config=None, training_config=None, dataset_config=None, model_trainer_configs=None,
                        train_dataset=None, valid_dataset=None, test_dataset=None, train_dataloader=None,
                        valid_dataloader=None, test_dataloader=None, model_trainers=None, trainer=None):
        return cls(running_config, training_config, dataset_config, model_trainer_configs, train_dataset, valid_dataset,
                   test_dataset, train_dataloader, valid_dataloader, test_dataloader, model_trainers, trainer)

    def run(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def resume(self):
        pass

    class ExperimentBuilder(object):
        def __init__(self, name=None, running_config=None, training_config=None, dataset_config=None,
                     model_trainer_configs=None, model_trainer_factory=None, dataset_factory=None, train_dataset=None,
                     valid_dataset=None, test_dataset=None, train_dataloader=None, valid_dataloader=None,
                     test_dataloader=None, model_trainers=None, trainer=None, collate_fn=None, event_handlers=None):
            self._name = name
            self._running_config = running_config
            self._training_config = training_config
            self._dataset_config = dataset_config
            self._model_trainer_configs = model_trainer_configs
            self._model_trainer_factory = model_trainer_factory
            self._dataset_factory = dataset_factory
            self._train_dataset = train_dataset
            self._valid_dataset = valid_dataset
            self._test_dataset = test_dataset
            self._train_dataloader = train_dataloader
            self._valid_dataloader = valid_dataloader
            self._test_dataloader = test_dataloader
            self._model_trainers = model_trainers
            self._trainer = trainer
            self._collate_fn = collate_fn
            self._event_handlers = event_handlers

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, name):
            self._name = name

        @property
        def running_config(self):
            return self._running_config

        @running_config.setter
        def running_config(self, config):
            self._running_config = config

        @property
        def training_config(self):
            return self._training_config

        @training_config.setter
        def training_config(self, config):
            self._training_config = config

        @property
        def dataset_config(self):
            return self._dataset_config

        @dataset_config.setter
        def dataset_config(self, config):
            self._dataset_config = config

        @property
        def model_trainer_configs(self):
            return self._model_trainer_configs

        @model_trainer_configs.setter
        def model_trainer_configs(self, configs):
            self._model_trainer_configs = configs

        @property
        def train_dataset(self):
            return self._train_dataset

        @train_dataset.setter
        def train_dataset(self, dataset):
            self._train_dataset = dataset

        @property
        def valid_dataset(self):
            return self._valid_dataset

        @valid_dataset.setter
        def valid_dataset(self, dataset):
            self._valid_dataset = dataset

        @property
        def test_dataset(self):
            return self._test_dataset

        @test_dataset.setter
        def test_dataset(self, dataset):
            self._test_dataset = dataset

        @property
        def train_dataloader(self):
            return self._train_dataloader

        @train_dataloader.setter
        def train_dataloader(self, dataloader):
            self._train_dataloader = dataloader

        @property
        def valid_dataloader(self):
            return self._valid_dataloader

        @valid_dataloader.setter
        def valid_dataloader(self, dataloader):
            self._valid_dataloader = dataloader

        @property
        def test_dataloader(self):
            return self._test_dataloader

        @test_dataloader.setter
        def test_dataloader(self, dataloader):
            self._test_dataloader = dataloader

        @property
        def model_trainers(self):
            return self._model_trainers

        @model_trainers.setter
        def model_trainers(self, model_trainers):
            self._model_trainers = model_trainers

        @property
        def trainer(self):
            return self._trainer

        @trainer.setter
        def trainer(self, trainer):
            self._trainer = trainer

        @property
        def model_trainer_factory(self):
            return self._model_trainer_factory

        @model_trainer_factory.setter
        def model_trainer_factory(self, factory):
            self._model_trainer_factory = factory

        @property
        def dataset_factory(self):
            return self._dataset_factory

        @dataset_factory.setter
        def dataset_factory(self, factory):
            self._dataset_factory = factory

        @property
        def collate_fn(self):
            return self._collate_fn

        @property
        def event_handlers(self):
            return self._event_handlers

        @event_handlers.setter
        def event_handlers(self, event_handlers):
            self._event_handlers = event_handlers

        @collate_fn.setter
        def collate_fn(self, collate_fn):
            self._collate_fn = collate_fn

        def _create_model_trainers(self, model_trainer_factory: ModelTrainerFactory = ModelTrainerFactory):
            return list(
                map(lambda model_trainer_config: model_trainer_factory.create(model_trainer_config,
                                                                              self.running_config),
                    self.model_trainer_configs))

        def _create_train_valid_test_dataloaders(self, train_dataset, valid_dataset, test_dataset, running_config,
                                                 training_config, collate_fn: Callable):
            if not on_single_device(self.running_config.devices):
                torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                                     rank=self.running_config.local_rank)

            return DataloaderFactory(train_dataset, valid_dataset, test_dataset).create(running_config,
                                                                                        training_config,
                                                                                        collate_fn)

        def _create_train_valid_test_datasets(self, dataset_config, dataset_factory):
            return dataset_factory(dataset_config).create_train_valid_test_datasets()

        def with_name(self, name):
            self.name = name
            return self

        def with_running_config(self, running_config):
            self.running_config = running_config
            return self

        def with_training_config(self, training_config):
            self.training_config = training_config
            return self

        def with_dataset_config(self, dataset_config):
            self.dataset_config = dataset_config
            return self

        def with_model_trainer_configs(self, model_trainer_configs):
            self.model_trainer_configs = model_trainer_configs
            return self

        def with_model_trainer_factory(self, model_trainer_factory):
            self.model_trainer_factory = model_trainer_factory
            return self

        def with_dataset_factory(self, dataset_factory):
            self.dataset_factory = dataset_factory
            return self

        def with_collate_fn(self, collate_fn):
            self.collate_fn = collate_fn
            return self

        def with_event_handlers(self, event_handlers):
            self.event_handlers = event_handlers
            return self

        def build(self):
            experiment = Experiment()
            experiment.running_config = self.running_config
            experiment.training_config = self.training_config
            experiment.dataset_config = self.dataset_config
            experiment.dataset_factory = self._dataset_factory
            experiment.model_trainer_configs = self.model_trainer_configs

            experiment.model_trainers = self._create_model_trainers(self.model_trainer_factory)
            experiment.train_dataset, experiment.valid_dataset, experiment.test_dataset = \
                self._create_train_valid_test_datasets(self.dataset_config, self.dataset_factory)
            experiment.train_dataloader, experiment.valid_dataloader, experiment.test_dataloader = \
                self._create_train_valid_test_dataloaders(experiment.train_dataset,
                                                          experiment.valid_dataset,
                                                          experiment.test_dataset,
                                                          experiment.running_config,
                                                          experiment.training_config,
                                                          self.collate_fn)

            experiment.trainer = Trainer(experiment.name, experiment.train_dataloader, experiment.valid_dataloader,
                                         experiment.test_dataloader, experiment.model_trainers,
                                         experiment.running_config).with_event_handler()

            return experiment
