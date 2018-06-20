import numpy as np
import sys

from multiprocessing import Process
from PyQt4.QtGui import QApplication, QWidget

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

        self._qt_process = QtProcess()

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
        print(number)
        print(batch)

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

    def __init__(self):

        Process.__init__(self)

        self._app = QApplication(sys.argv)
        self._widget = QWidget()

        self._widget.resize(320, 240)
        self._widget.setWindowTitle("Qt Displayer")

    def run(self):

        self._widget.show()
        self._app.exec_()

        return
