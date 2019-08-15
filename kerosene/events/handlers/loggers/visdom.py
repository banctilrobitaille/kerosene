from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Dict

from visdom import Visdom

from kerosene.config.loggers import VisdomConfiguration
from kerosene.events.handlers.base_handler import EventHandler


class PlotType(Enum):
    LINE_PLOT = "Line Plot"
    IMAGES_PLOT = "Images Plot"
    IMAGE_PLOT = "Image Plot"

    def __str__(self):
        return self.value


class PlotFrequency(Enum):
    EVERY_STEP = "Step"
    EVERY_EPOCH = "Epoch"

    def __str__(self):
        return self.value


class VisdomData(object):

    def __init__(self, source_name, variable_name: str, plot_type: PlotType, plot_frequency: PlotFrequency, x, y):
        self._source_name = source_name
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._plot_frequency = plot_frequency
        self._x = x
        self._y = y

    @property
    def id(self):
        return hash(self)

    @property
    def source_name(self):
        return self._source_name

    @property
    def variable_name(self):
        return self._variable_name

    @property
    def plot_type(self):
        return self._plot_type

    @property
    def plot_frequency(self):
        return self._plot_frequency

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __hash__(self):
        return hash(self._source_name + self._variable_name)


class VisdomPlot(ABC):

    @abstractmethod
    def update(self, visdom_data: VisdomData):
        raise NotImplementedError()


class LinePlot(VisdomPlot):
    def __init__(self, visdom):
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.line(X=visdom_data.x,
                                             Y=visdom_data.y,
                                             opts=dict(xlabel=visdom_data.plot_frequency,
                                                       ylabel=visdom_data.variable_name,
                                                       title="{} {} per {}".format(visdom_data.source_name,
                                                                                   visdom_data.variable_name,
                                                                                   visdom_data.plot_frequency),
                                                       legend=[visdom_data.source_name]))
        else:
            self._visdom.line(
                X=visdom_data.x,
                Y=visdom_data.y,
                win=self._window,
                update='append')


class VisdomPlotFactory(object):

    @staticmethod
    def create_plot(visdom, plot_type):

        if plot_type == PlotType.LINE_PLOT:
            plot = LinePlot(visdom)
        else:
            raise NotImplementedError("Uuable to create a plot for {}".format(plot_type))

        return plot


class VisdomLogger(EventHandler):
    def __init__(self, visdom_config: VisdomConfiguration):
        self._visdom_config = visdom_config
        self._visdom = Visdom(server=visdom_config.server, port=visdom_config.port, env=visdom_config.env)
        self._plots: Dict[int, VisdomPlot] = {}

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
                self._plots[visdom_datum.id] = VisdomPlotFactory.create_plot(self._visdom, visdom_datum.plot_type)

            self._plots[visdom_datum.id].update(visdom_datum)
