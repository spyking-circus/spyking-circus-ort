import sys

from multiprocessing import Process
try:
    from PyQt4.QtGui import QApplication  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from PyQt5.QtWidgets import QApplication  # Python 3 compatibility.

from circusort.block.gui.gui_window import GUIWindow

class GUIProcess(Process):

    def __init__(self, all_pipes):

        Process.__init__(self)

        self.pipes = {}

        for key, value in all_pipes.items():
            self.pipes[key] = value

    def run(self):

        app = QApplication(sys.argv)
        screen_resolution = app.desktop().screenGeometry()
        window = GUIWindow(self.pipes, screen_resolution=screen_resolution)
        #window.show()
        sys.exit(app.exec_())

        return
