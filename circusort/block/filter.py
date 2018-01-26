import numpy as np

from scipy import signal

from circusort.block.block import Block


class Filter(Block):
    """Filtering of the voltage traces of the recording channels

    Attributes:
        sampling_rate: float
            The sampling rate used to record the signal [Hz]. The default value is 20e+3.
        cut_off: float
            The cutoff frequency used to define the high-pass filter [Hz]. The default value is 500.0.
        remove_median: boolean
            The option to remove the median over all the channels for each time step. The default value is False.
    See also:
        circusort.block.Block
    """
    # TODO complete docstring.

    name = "Filter"

    params = {
        'cut_off': 500.0,  # Hz
        'sampling_rate': 20000.0,  # Hz
        'remove_median': False,
    }

    def __init__(self, **kwargs):
        """Initialization.

        Parameters:
            sampling_rate: float
                The sampling rate used to record the signal [Hz]. The default value is 20e+3.
            cut_off: float
                The cutoff frequency used to define the high-pass filter [Hz]. The default value is 500.0.
            remove_median: boolean
                The option to remove the median over all the channels for each time step. The default value is False.

        """

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

        # Line useful to solve PyCharm warnings.
        self.cut_off = self.cut_off
        self.sampling_rate = self.sampling_rate
        self.remove_median = self.remove_median

    def _initialize(self):

        cut_off = np.array([self.cut_off, 0.95 * (self.sampling_rate / 2.0)])
        filter_ = signal.butter(3, cut_off / (self.sampling_rate / 2.0), 'pass')
        self.b = filter_[0]
        self.a = filter_[1]
        self.z = {}

        return

    @property
    def nb_channels(self):

        return self.input.shape[1]

    @property
    def nb_samples(self):

        return self.input.shape[0]

    def _guess_output_endpoints(self):

        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)
        self.z = {}
        m = max(len(self.a), len(self.b)) - 1
        for i in xrange(self.nb_channels):
            self.z[i] = np.zeros(m, dtype=np.float32)

    def _process(self):

        # Receive input data.
        batch = self.input.receive()

        self._measure_time('start', frequency=100)

        # Process data.
        for i in xrange(self.nb_channels):
            batch[:, i], self.z[i] = signal.lfilter(self.b, self.a, batch[:, i], zi=self.z[i])
            batch[:, i] -= np.median(batch[:, i])
        if self.remove_median:
            global_median = np.median(batch, 1)
            for i in xrange(self.nb_channels):
                batch[:, i] -= global_median

        # Send output data.
        self.output.send(batch)

        self._measure_time('end', frequency=100)

        return

    def _introspect(self):
        # TODO add docstring.

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
