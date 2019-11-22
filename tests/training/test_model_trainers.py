import unittest

from hamcrest import *
from ignite.metrics import Accuracy, Metric
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.trainers import ModelTrainerConfiguration, RunConfiguration
from mockito import mock, verify, spy
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from kerosene.nn.utils.gradients import GradientClippingStrategy
from kerosene.training.trainers import ModelTrainer, ModelTrainerFactory
from tests.constants import *


# noinspection PyUnresolvedReferences
class ModelTrainerTest(unittest.TestCase):
    MODEL_NAME = "Harry Potter"

    def setUp(self) -> None:
        self._model_mock = mock(nn.Module)
        self._criterion_mock = spy(nn.CrossEntropyLoss())
        self._optimizer_mock = mock(Optimizer)
        self._scheduler_mock = mock(lr_scheduler)
        self._metric_computer_mock = spy(Accuracy())
        self._gradient_clipping_strategy = None

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model_mock, self._criterion_mock,
                                           self._optimizer_mock, self._scheduler_mock,
                                           {"Accuracy": self._metric_computer_mock}, self._gradient_clipping_strategy)

        self._gradient_clipping_strategy = mock(GradientClippingStrategy)

    def tearDown(self) -> None:
        super().tearDown()

    def test_should_compute_train_metric_and_update_state(self):
        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_train_metrics(metric)
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ONE))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_train_metrics(metric)
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ONE / 2))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_valid_metric_and_update_state(self):
        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_valid_metrics(metric)
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ONE))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_valid_metrics(metric)
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ONE / 2))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_test_metric_and_update_state(self):
        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_test_metric(metric)
        assert_that(self._model_trainer.test_metric, equal_to(ONE))
        assert_that(self._model_trainer.step_test_metric, equal_to(ONE))
        verify(self._metric_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metric(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_test_metric(metric)
        assert_that(self._model_trainer.test_metric, equal_to(ONE / 2))
        assert_that(self._model_trainer.step_test_metric, equal_to(ZERO))

        verify(self._metric_computer_mock, times=2).compute()

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_train_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_train_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_train_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_valid_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_valid_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_valid_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))

        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_test_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.test_loss, equal_to(ZERO))

    def test_should_compute_test_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_test_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_test_loss(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss, close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to(ZERO))
        assert_that(self._model_trainer.train_loss, equal_to(ZERO))
        assert_that(self._model_trainer.valid_loss, equal_to(ZERO))


class ModelTrainerFactoryTest(unittest.TestCase):
    VALID_CONFIG_FILE_PATH = "tests/config/valid_config.yml"
    MODELS_CONFIG_YML_TAG = "models"
    SIMPLE_NET_NAME = "SimpleNet"

    def setUp(self) -> None:
        config_dict = YamlConfigurationParser.parse_section(self.VALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)
        self._model_trainer_config = ModelTrainerConfiguration.from_dict(self.SIMPLE_NET_NAME,
                                                                         config_dict[self.SIMPLE_NET_NAME])
        self._run_config = RunConfiguration()
        self._model = nn.Conv2d(1, 32, (3, 3))

    def test_should_create_model_trainer_with_config(self):
        model_trainer = ModelTrainerFactory(model=self._model).create(self._model_trainer_config)

        assert_that(model_trainer, instance_of(ModelTrainer))
        assert_that(model_trainer.name, is_(self.SIMPLE_NET_NAME))
        assert_that(model_trainer.criterion, instance_of(torch.nn.CrossEntropyLoss))
        assert_that(model_trainer.optimizer, instance_of(torch.optim.SGD))
        assert_that(model_trainer.scheduler, instance_of(torch.optim.lr_scheduler.ReduceLROnPlateau))
        assert_that(len(model_trainer._metric_computers.keys()), is_(2))
        [assert_that(model_trainer._metric_computers[metric], instance_of(Metric)) for metric in
         model_trainer._metric_computers.keys()]
