from abc import ABC, abstractmethod

from visdom import Visdom

from kerosene.events.handlers.visdom.data import VisdomData, PlotType


class VisdomPlot(ABC):

    def __init__(self, visdom: Visdom):
        self._visdom = visdom

    @property
    def visdom(self):
        return self._visdom

    @abstractmethod
    def update(self, visdom_data: VisdomData):
        raise NotImplementedError()


class ImagePlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.image(img=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.image(img=visdom_data.y, win=self._window, **visdom_data.params)


class ImagesPlot(VisdomPlot):
    def __init__(self, visdom: Visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.images(tensor=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.images(tensor=visdom_data.y, win=self._window, **visdom_data.params)


class LinePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.line(X=[visdom_data.x], Y=visdom_data.y, **visdom_data.params)
        else:
            self._visdom.line(X=[visdom_data.x], Y=visdom_data.y, win=self._window, update='append')


class PiePlot(VisdomPlot):
    def __init__(self, visdom):
        super().__init__(visdom)
        self._visdom = visdom
        self._window = None

    def update(self, visdom_data: VisdomData):
        if self._window is None:
            self._window = self._visdom.pie(X=visdom_data.x, **visdom_data.params)
        else:
            self._visdom.pie(X=visdom_data.x, win=self._window, **visdom_data.params)


class VisdomPlotFactory(object):

    @staticmethod
    def create_plot(visdom, plot_type):

        if plot_type == PlotType.LINE_PLOT:
            plot = LinePlot(visdom)
        elif plot_type == PlotType.IMAGE_PLOT:
            plot = ImagePlot(visdom)
        elif plot_type == PlotType.IMAGES_PLOT:
            plot = ImagesPlot(visdom)
        elif plot_type == PlotType.PIE_PLOT:
            plot = PiePlot(visdom)
        else:
            raise NotImplementedError("Unable to create a plot for {}".format(plot_type))

        return plot
