import logging
import unittest

from mockito import verify, spy

from kerosene.events import Phase
from kerosene.events.handlers.console import PrintTrainingStatus, StatusConsoleColorPalette


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
