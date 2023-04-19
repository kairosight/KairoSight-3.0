

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget

from pyqtgraph.widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from pyqtgraph import ImageItem, HistogramLUTItem, HistogramLUTWidget


class GraphicsWidget(QWidget):
    """Widget defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of widget
        super(GraphicsWidget, self).__init__(parent)

        # Create a central Graphics Layout Widget
        self.widget = GraphicsLayoutWidget()

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.widget.addPlot()
        # Item for displaying image data
        self.img_item = ImageItem()
        self.img_item.axisOrder = 'row-major'
        self.p1.addItem(self.img_item)
        self.p1.getViewBox().invertY(True)

        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add widget to vertical box
        self.vbl.addWidget(self.widget)
        # set the layout to the vertical box
        self.setLayout(self.vbl)

        # Levels/color control with a histogram
        # self.hist = HistogramLUTWidget()
        # self.hist.setImageItem(self.img)
        # parent.horizontalLayout_View.addWidget(self.hist)
        # # self.widget.addWidget(self.hist, 0, 1)
        # self.hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier

        # Create histogram
        # Levels/color control with a histogram
        self.histogram = HistogramLUTItem()
        # TODO Halve histogram width
        self.histogram.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
        self.histogram.setImageItem(self.img_item)
        self.widget.addItem(self.histogram)
