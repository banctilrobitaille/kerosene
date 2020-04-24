from typing import Union, List


class VisdomData(object):

    def __init__(self, source_name, variable_name: Union[List[str], str], plot_type, plot_frequency, x, y,
                 params: dict = None):
        self._source_name = source_name
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._plot_frequency = plot_frequency
        self._x = x
        self._y = y
        self._params = params

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

    @property
    def params(self):
        return self._params

    def __hash__(self):
        return hash(self._source_name + self._variable_name + str(self._plot_frequency) + str(self._plot_type))

    def __str__(self):
        return "Source: {} Variable: {} Plot Type: {} Frequency: {} x: {} y: {}".format(self.source_name,
                                                                                        self.variable_name,
                                                                                        str(self.plot_type),
                                                                                        str(self.plot_frequency),
                                                                                        self.x, self.y)
