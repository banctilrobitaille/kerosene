from typing import Union, List, Dict

from visdom import Visdom

from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.data import VisdomData
from kerosene.loggers.visdom.plots import VisdomPlotFactory, VisdomPlot


class VisdomLogger(object):
    def __init__(self, visdom_config: VisdomConfiguration, plot_factory: VisdomPlotFactory = VisdomPlotFactory()):
        self._visdom_config = visdom_config
        self._visdom = Visdom(server=visdom_config.server, port=visdom_config.port, env=visdom_config.env)
        self._plots: Dict[int, VisdomPlot] = {}
        self._plot_factory = plot_factory

    @property
    def visdom_config(self):
        return self._visdom_config

    @property
    def plots(self):
        return self._plots

    def __call__(self, visdom_data: Union[List[VisdomData], VisdomData]):
        visdom_data = [visdom_data] if not hasattr(visdom_data, '__iter__') else visdom_data

        for visdom_datum in visdom_data:
            if visdom_datum.id not in self._plots.keys():
                self._plots[visdom_datum.id] = self._plot_factory.create_plot(self._visdom, visdom_datum.plot_type)

            self._plots[visdom_datum.id].update(visdom_datum)
