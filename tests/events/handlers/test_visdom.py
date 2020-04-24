import unittest

from mockito import mock, spy

from kerosene.events.handlers.visdom import PlotLosses
from kerosene.loggers.visdom.visdom import VisdomLogger
from tests.base_test import BaseTestEnvironment


class VisdomTest(BaseTestEnvironment):

    def setUp(self):
        self._visdom_logger_mock = mock(VisdomLogger)
        self._plot_losses_handler = spy(PlotLosses)

    def test_should_plot_losses(self):
        pass
