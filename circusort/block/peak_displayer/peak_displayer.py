import numpy as np

from multiprocessing import Pipe

from circusort.block.block import Block
from circusort.block.peak_displayer.gui_process import GUIProcess


__classname__ = "PeakDisplayer"


class PeakDisplayer(Block):
    """Peak displayer"""

    name = "Peak displayer"

    params = {
        'probe_path': None,
    }

    def __init__(self, **kwargs):
        """Initialization"""

        Block.__init__(self, **kwargs)
        self.add_input('data', structure='dict')
        self.add_input('mads', structure='dict')
        self.add_input('peaks', structure='dict')

        # The following line is useful to avoid PyCharm warnings.
        self.probe_path = self.probe_path

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._sampling_rate = None

        self._params_pipe = Pipe()
        self._number_pipe = Pipe()
        self._data_pipe = Pipe()
        self._mads_pipe = Pipe()
        self._peaks_pipe = Pipe()
        self._qt_process = GUIProcess(self._params_pipe, self._number_pipe, self._data_pipe, self._mads_pipe,
                                      self._peaks_pipe, probe_path=self.probe_path)

        self._is_mad_reception_blocking = False
        self._is_peak_reception_blocking = False

    def _initialize(self):

        self._qt_process.start()

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels
        self._sampling_rate = sampling_rate

        self._params_pipe[1].send({
            'nb_samples': self._nb_samples,
            'sampling_rate': self._sampling_rate,
        })

        return

    def _process(self):

        data_packet = self.get_input('data').receive()
        number = data_packet['number']
        data = data_packet['payload']

        mads_packet = self.get_input('mads').receive(blocking=self._is_mad_reception_blocking, number=number)
        if not self._is_mad_reception_blocking and mads_packet is not None:
            self._is_mad_reception_blocking = True

        peaks_packet = self.get_input('peaks').receive(blocking=self._is_peak_reception_blocking, number=number)
        if not self._is_peak_reception_blocking and peaks_packet is not None:
            self._is_peak_reception_blocking = True

        self._measure_time(label='start', period=10)

        self._number_pipe[1].send(number)
        self._data_pipe[1].send(data)
        if mads_packet is not None:
            mads = mads_packet['payload']
            self._mads_pipe[1].send(mads)
        else:
            self._mads_pipe[1].send(None)
        if peaks_packet is not None:
            peaks = peaks_packet['payload']
            self._peaks_pipe[1].send(peaks)
        else:
            self._peaks_pipe[1].send(None)

        self._measure_time(label='end', period=10)

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
