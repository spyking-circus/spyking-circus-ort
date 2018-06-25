import sys

from multiprocessing import Process
from PyQt4.QtGui import QApplication

from circusort.block.qt_displayer.qt_window import QtWindow


class QtProcess(Process):

    def __init__(self, params_pipe, number_pipe, data_pipe):

        Process.__init__(self)

        self._params_pipe = params_pipe
        self._number_pipe = number_pipe
        self._data_pipe = data_pipe

    def run(self):

        app = QApplication(sys.argv)
        screen_resolution = app.desktop().screenGeometry()
        window = QtWindow(self._params_pipe, self._number_pipe, self._data_pipe, screen_resolution)
        window.show()
        app.exec_()

        return
