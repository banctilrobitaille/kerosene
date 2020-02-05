from enum import Enum


class PlotType(Enum):
    LINE_PLOT = "Line Plot"
    IMAGES_PLOT = "Images Plot"
    IMAGE_PLOT = "Image Plot"
    PIE_PLOT = "Pie Plot"
    TEXT_PLOT = "Text Plot"
    HISTOGRAM_PLOT = "Histogram Plot"
    SCATTER_PLOT = "Scatter Plot"
    STEM_PLOT = "Stem Plot"
    HEATMAP_PLOT = "Heatmap Plot"
    BAR_PLOT = "Bar Plot"
    BOX_PLOT = "Box Plot"
    SURFACE_PLOT = "Surface Plot"
    CONTOUR_PLOT = "Contour Plot"
    QUIVER_PLOT = "Quiver Plot"
    MESH_PLOT = "Mesh Plot"
    MATPLOTLIB_PLOT = "Matplotlib Plot"

    def __str__(self):
        return self.value


class PlotFrequency(Enum):
    EVERY_STEP = "Step"
    EVERY_EPOCH = "Epoch"

    def __str__(self):
        return self.value
