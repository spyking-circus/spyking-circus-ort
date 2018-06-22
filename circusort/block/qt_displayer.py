import numpy as np
import sys

from multiprocessing import Process, Pipe
from PyQt4.QtCore import QThread, SIGNAL, Qt, pyqtSignal
from PyQt4.QtGui import QApplication, QWidget, QLabel, QVBoxLayout
from vispy import app

from circusort.block.block import Block


__classname__ = "QtDisplayer"


class QtDisplayer(Block):
    """Displayer"""

    name = "Displayer"

    params = {}

    def __init__(self, **kwargs):
        """Initialization"""

        Block.__init__(self, **kwargs)
        self.add_input('data', structure='dict')

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._sampling_rate = None

        self._pipe = Pipe()
        self._qt_process = QtProcess(self._pipe)

    def _initialize(self):

        self._qt_process.start()

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels
        self._sampling_rate = sampling_rate

        return

    def _process(self):

        data_packet = self.get_input('data').receive()
        number = data_packet['number']
        batch = data_packet['payload']

        self._measure_time(label='start', frequency=10)

        # TODO remove the 2 following lines.
        self._pipe[1].send(number)
        _ = batch

        self._measure_time(label='end', frequency=10)

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self._sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return


class QtProcess(Process):

    def __init__(self, pipe):

        Process.__init__(self)

        self._pipe = pipe

    def run(self):

        app = QApplication(sys.argv)
        window = QtWindow(self._pipe)
        window.show()
        app.exec_()

        return


class QtWindow(QWidget):

    def __init__(self, pipe):

        QWidget.__init__(self)

        self._label = QLabel()
        self._label.setText("<number>")
        self._label.setAlignment(Qt.AlignCenter)

        self._canvas = app.Canvas(title="Vispy canvas", keys='interactive')

        self._vbox = QVBoxLayout()
        self._vbox.addWidget(self._label)
        self._vbox.addWidget(self._canvas.native)

        self.setLayout(self._vbox)

        self.setWindowTitle("Qt Displayer")
        self.resize(600, 400)

        self._thread = MyQtThread(pipe)

        self._thread.signal.connect(self.number_received)

        self._thread.start()

    def number_received(self, number):

        self._label.setText(number)

        return


class MyQtThread(QThread):

    signal = pyqtSignal(object)

    def __init__(self, pipe):

        QThread.__init__(self)

        self._pipe = pipe

    def __del__(self):

        self.wait()

    def run(self):

        while True:

            try:
                msg = self._pipe[0].recv()
                if msg == 'TERM':
                    break
                self.signal.emit(str(msg))
                self.usleep(1)
            except EOFError:
                pass

        return
