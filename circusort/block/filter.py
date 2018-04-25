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
        'use_gpu': False,
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
        self.add_input('data', structure='dict')

        # Lines useful to solve PyCharm warnings.
        self.cut_off = self.cut_off
        self.sampling_rate = self.sampling_rate
        self.remove_median = self.remove_median
        self.use_gpu = self.use_gpu

        if self.cut_off < 0.1:
            # Check that the cut off frequency is at least 0.1 Hz.
            self.cut_off = 0.1  # Hz

    def _initialize(self):

        cut_off = np.array([self.cut_off, 0.95 * (self.sampling_rate / 2.0)])
        filter_ = signal.butter(3, cut_off / (self.sampling_rate / 2.0), 'pass')
        self.b = filter_[0]
        self.a = filter_[1]
        self.z = {}
        return

    @property
    def nb_channels(self):

        # return self.input.shape[1]
        return 9

    @property
    def nb_samples(self):

        # return self.input.shape[0]
        return 1024

    def _guess_output_endpoints(self):

        # TODO clean the following code replacement.
        # self.output.configure(dtype=self.input.dtype, shape=self.input.shape)
        dtype = 'float32'
        shape = (self.nb_samples, self.nb_channels)
        self.output.configure(dtype=dtype, shape=shape)
        self.z = {}
        m = max(len(self.a), len(self.b)) - 1
        for i in xrange(self.nb_channels):
            self.z[i] = np.zeros(m, dtype=np.float32)

        if self.use_gpu:
            from scipy.signal import iirfilter
            import pyopencl
            from circusort.utils.gpu.filter import GpuFilter
            from circusort.utils.gpu.utils import get_first_gpu_device
            platform_name = 'NVIDIA'
            # platform_name = 'Intel'
            device = get_first_gpu_device(platform_name)
            assert device is not None, 'No GPU devices for this platform'
            context = pyopencl.Context([device])
            coefficients = iirfilter(3, [self.cut_off / (self.sampling_rate / 2.0), 0.95],
                                     btype='bandpass', ftype='butter', output='sos')
            self.filter_engine = GpuFilter(context, coefficients, self.nb_channels, 'float32', self.nb_samples)

    def _process(self):

        # Receive input data.
        data_packet = self.get_input('data').receive()
        batch = data_packet['payload']

        self._measure_time('start', frequency=100)

        # Preallocation of filtered data.
        filtered_batch = np.empty(batch.shape, dtype=batch.dtype)

        # Process data.
        if self.use_gpu:
            filtered_batch = self.filter_engine.compute_one_chunk(batch)
        else:
            for i in range(0, self.nb_channels):
                # Filter data (locally).
                filtered_batch[:, i], self.z[i] = signal.lfilter(self.b, self.a, batch[:, i], zi=self.z[i])
                # Center data (locally).
                local_median = np.median(filtered_batch[:, i])
                filtered_batch[:, i] -= local_median
            if self.remove_median:
                # Center data (globally).
                global_median = np.median(filtered_batch, 1)
                filtered_batch[:, :] -= global_median

        # Send output data.
        self.output.send(filtered_batch)

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
