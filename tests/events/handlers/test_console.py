import logging
import unittest

import torch
from mockito import verify, spy

from kerosene.events import Phase, Monitor
from kerosene.events.handlers.console import PrintTrainingStatus, StatusConsoleColorPalette, MonitorsTable


class ConsoleHandlerTest(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self._logger_mock = spy(logging.getLogger())

    def test_print_status_with_colors(self):
        expected_string = "\nCurrent states: [92mTraining[0m |  Epoch: 0 | Training step: 0 | Validation step: 0 | Test step: 0\n"
        handler = PrintTrainingStatus(colors=StatusConsoleColorPalette.DEFAULT)
        handler.LOGGER = self._logger_mock
        handler.print_status(Phase.TRAINING, 0, 0, 0, 0)
        verify(self._logger_mock).info(expected_string)

    def test_print_status_without_colors(self):
        expected_string = "\nCurrent states: Training\x1b[0m |  Epoch: 0 | Training step: 0 | Validation step: 0 | Test step: 0\n"
        handler = PrintTrainingStatus()
        handler.LOGGER = self._logger_mock
        handler.print_status(Phase.TRAINING, 0, 0, 0, 0)
        verify(self._logger_mock).info(expected_string)

    def test_print_monitors_table(self):
        monitors = {
            "Model 1": {Phase.TRAINING: {Monitor.METRICS: {"Accuracy": torch.tensor([0.6])},
                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.6])}},
                        Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.6])},
                                           Monitor.LOSS: {"MSELoss": torch.tensor([0.6])}},
                        Phase.TEST: {Monitor.METRICS: {"Accuracy": torch.tensor([0.6])},
                                     Monitor.LOSS: {"MSELoss": torch.tensor([0.6])}}}}

        training_values = {**monitors["Model 1"][Phase.TRAINING][Monitor.METRICS],
                           **monitors["Model 1"][Phase.TRAINING][Monitor.LOSS]}
        validation_values = {**monitors["Model 1"][Phase.VALIDATION][Monitor.METRICS],
                             **monitors["Model 1"][Phase.VALIDATION][Monitor.LOSS]}

        table = MonitorsTable("Model 1")
        table.update(training_values, validation_values)
        table.show()

        monitors = {
            "Model 1": {Phase.TRAINING: {Monitor.METRICS: {"Accuracy": torch.tensor([0.7])},
                                         Monitor.LOSS: {"MSELoss": torch.tensor([0.7])}},
                        Phase.VALIDATION: {Monitor.METRICS: {"Accuracy": torch.tensor([0.4])},
                                           Monitor.LOSS: {"MSELoss": torch.tensor([0.6])}},
                        Phase.TEST: {Monitor.METRICS: {"Accuracy": torch.tensor([0.4])},
                                     Monitor.LOSS: {"MSELoss": torch.tensor([0.8])}}}}

        training_values = {**monitors["Model 1"][Phase.TRAINING][Monitor.METRICS],
                           **monitors["Model 1"][Phase.TRAINING][Monitor.LOSS]}
        validation_values = {**monitors["Model 1"][Phase.VALIDATION][Monitor.METRICS],
                             **monitors["Model 1"][Phase.VALIDATION][Monitor.LOSS]}

        table.update(training_values, validation_values)
        table.show()
