from enum import Enum


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
        return hash(self._source_name + self._variable_name + str(self._plot_frequency) + str(self._plot_type))
