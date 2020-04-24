import os
import unittest

from hamcrest import *
from ignite.metrics import Accuracy, Metric, Recall
from mockito import mock, verify, spy
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.optim import lr_scheduler

from kerosene.configs.configs import ModelConfiguration, RunConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.nn.utils.gradients import GradientClippingStrategy
from kerosene.training.trainers import ModelTrainer, ModelTrainerFactory
from tests.constants import *
from tests.functionals.models import SimpleNet


# noinspection PyUnresolvedReferences
class ModelTrainerTest(unittest.TestCase):
    MODEL_NAME = "Harry Potter"
    LOSS_NAME = "CrossEntropy"

    def setUp(self) -> None:
        self._model = SimpleNet()
        self._criterion_mock = spy(nn.CrossEntropyLoss())
        self._optimizer_mock = spy(SGD(self._model.parameters(), lr=0.001))
        self._scheduler_mock = mock(lr_scheduler)
        self._accuracy_computer_mock = spy(Accuracy())
        self._recall_computer_mock = spy(Recall())
        self._gradient_clipping_strategy = None

        self._model_trainer = ModelTrainer(self.MODEL_NAME, self._model, {self.LOSS_NAME: self._criterion_mock},
                                           self._optimizer_mock, self._scheduler_mock,
                                           {"Accuracy": self._accuracy_computer_mock,
                                            "Recall": self._recall_computer_mock}, self._gradient_clipping_strategy)

        self._gradient_clipping_strategy = mock(GradientClippingStrategy)

    def tearDown(self) -> None:
        super().tearDown()

    def test_should_compute_train_metric_and_update_state(self):
        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_train_metrics(metric)
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ONE))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ONE))
        verify(self._accuracy_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_train_metrics(metric)
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ONE / 2))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))

        verify(self._accuracy_computer_mock, times=2).compute()

        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_test_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.test_loss, equal_to({self.LOSS_NAME: ZERO}))

    def test_should_compute_valid_metric_and_update_state(self):
        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_valid_metrics(metric)
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ONE))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ONE))
        verify(self._accuracy_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_valid_metrics(metric)
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ONE / 2))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))

        verify(self._accuracy_computer_mock, times=2).compute()
        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_test_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.test_loss, equal_to({self.LOSS_NAME: ZERO}))

    def test_should_compute_test_metric_and_update_state(self):
        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0)
        self._model_trainer.update_test_metrics(metric)
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ONE))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ONE))
        verify(self._accuracy_computer_mock).update((MODEL_PREDICTION_CLASS_0, TARGET_CLASS_0))

        metric = self._model_trainer.compute_metrics(MODEL_PREDICTION_CLASS_1, TARGET_CLASS_0)
        self._model_trainer.update_test_metrics(metric)
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ONE / 2))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))

        verify(self._accuracy_computer_mock, times=2).compute()

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_test_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.test_loss, equal_to({self.LOSS_NAME: ZERO}))

    def test_should_compute_train_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_train_loss(self.LOSS_NAME, MODEL_PREDICTION_CLASS_0,
                                                                 TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss[self.LOSS_NAME],
                    close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss[self.LOSS_NAME], close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_train_loss(self.LOSS_NAME, MODEL_PREDICTION_CLASS_0,
                                                                 TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_train_loss[self.LOSS_NAME],
                    close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.train_loss[self.LOSS_NAME], close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_test_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.test_loss, equal_to({self.LOSS_NAME: ZERO}))

    def test_should_compute_valid_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_valid_loss(self.LOSS_NAME, MODEL_PREDICTION_CLASS_0,
                                                                 TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss[self.LOSS_NAME],
                    close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss[self.LOSS_NAME], close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_valid_loss(self.LOSS_NAME, MODEL_PREDICTION_CLASS_0,
                                                                 TARGET_CLASS_1)
        assert_that(loss._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_valid_loss[self.LOSS_NAME],
                    close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.valid_loss[self.LOSS_NAME], close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))

        assert_that(self._model_trainer.step_train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_test_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.test_loss, equal_to({self.LOSS_NAME: ZERO}))

    def test_should_compute_test_loss_and_update_state(self):
        loss = self._model_trainer.compute_and_update_test_loss(self.LOSS_NAME, MODEL_PREDICTION_CLASS_0,
                                                                TARGET_CLASS_0)

        assert_that(loss._loss, close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss[self.LOSS_NAME],
                    close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss[self.LOSS_NAME], close_to(MINIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        loss = self._model_trainer.compute_and_update_test_losses(MODEL_PREDICTION_CLASS_0, TARGET_CLASS_1)
        assert_that(loss[self.LOSS_NAME]._loss, close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.step_test_loss[self.LOSS_NAME],
                    close_to(MAXIMUM_BINARY_CROSS_ENTROPY_LOSS, DELTA))
        assert_that(self._model_trainer.test_loss[self.LOSS_NAME], close_to(AVERAGED_BINARY_CROSS_ENTROPY_LOSS, DELTA))

        assert_that(self._model_trainer.train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_valid_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_test_metrics["Accuracy"], equal_to(ZERO))
        assert_that(self._model_trainer.step_train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.step_valid_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.train_loss, equal_to({self.LOSS_NAME: ZERO}))
        assert_that(self._model_trainer.valid_loss, equal_to({self.LOSS_NAME: ZERO}))


class ModelTrainerFactoryTest(unittest.TestCase):
    VALID_CONFIG_FILE_PATH = "tests/configs/valid_config.yml"
    MODELS_CONFIG_YML_TAG = "models"
    SIMPLE_NET_NAME = "SimpleNet"

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def setUp(self) -> None:
        config_dict = YamlConfigurationParser.parse_section(self.VALID_CONFIG_FILE_PATH, self.MODELS_CONFIG_YML_TAG)
        self._model_trainer_config = ModelConfiguration.from_dict(self.SIMPLE_NET_NAME,
                                                                  config_dict[self.SIMPLE_NET_NAME])
        self._run_config = RunConfiguration()
        self._model = nn.Conv2d(1, 32, (3, 3))

    def test_should_create_model_trainer_with_config(self):
        model_trainer = ModelTrainerFactory(model=self._model).create(self._model_trainer_config)
        assert_that(model_trainer, instance_of(ModelTrainer))
        assert_that(model_trainer.name, is_(self.SIMPLE_NET_NAME))
        assert_that(len(model_trainer.criterions), is_(2))
        [assert_that(model_trainer.criterions[criterion], instance_of(_Loss)) for criterion in
         model_trainer.criterions.keys()]
        assert_that(model_trainer.optimizer, instance_of(torch.optim.SGD))
        assert_that(model_trainer.scheduler, instance_of(torch.optim.lr_scheduler.ReduceLROnPlateau))
        assert_that(len(model_trainer._metric_computers.keys()), is_(2))
        [assert_that(model_trainer._metric_computers[metric], instance_of(Metric)) for metric in
         model_trainer._metric_computers.keys()]
