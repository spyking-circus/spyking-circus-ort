import numpy as np

from scipy import signal

from circusort.block.block import Block


class Filter(Block):
    """Filtering of the voltage traces of the recording channels

    Attributes:
        sampling_rate: float
            The sampling rate used to record the signal [Hz].
        cut_off: float
            The cutoff frequency used to define the high-pass filter [Hz].
        order: integer
            The order used to define the high-pass filter.
        remove_median: boolean
            The option to remove the median over all the channels for each time step.
        use_gpu: boolean
            The option to use the GPU.
    See also:
        circusort.block.Block
    """

    name = "Filter"

    params = {
        'cut_off': 500.0,  # Hz
        'order': 1,
        'sampling_rate': 20e+3,  # Hz  # TODO allow None value, i.e. receive sampling rate from input block.
        'remove_median': False,
        'use_gpu': False,
    }

    def __init__(self, **kwargs):
        """Initialization.

        Arguments:
            sampling_rate: float (optional)
                The sampling rate used to record the signal [Hz].
                The default value is 20e+3.
            cut_off: float (optional)
                The cutoff frequency used to define the high-pass filter [Hz].
                The default value is 500.0.
            order: integer (optional)
                The order used to define the high-pass filter.
                The default value is 1.
            remove_median: boolean (optional)
                The option to remove the median over all the channels for each time step.
                The default value is False.
            use_gpu: boolean (optional)
                The option to use the GPU.
                The default value is False.
        """

        Block.__init__(self, **kwargs)
        self.add_output('data', structure='dict')
        self.add_input('data', structure='dict')

        # Lines useful to solve PyCharm warnings.
        self.cut_off = self.cut_off
        self.order = self.order
        self.sampling_rate = self.sampling_rate
        self.remove_median = self.remove_median
        self.use_gpu = self.use_gpu

        # Check that the cut off frequency is at least 0.1 Hz.
        if self.cut_off < 0.1:
            self.cut_off = 0.1  # Hz

        self.dtype = None
        self.nb_samples = None
        self.nb_channels = None
        # self.sampling_rate = None  # already defined above

        self._b = None
        self._a = None
        self._z = None
        self._filter_engine = None

    def _initialize(self):

        if self.use_gpu:
            pass
        else:
            cut_offs = np.array([self.cut_off, 0.95 * (self.sampling_rate / 2.0)])
            critical_frequencies = cut_offs / (self.sampling_rate / 2.0)  # half-cycles / samples
            filter_ = signal.butter(self.order, critical_frequencies, btype='bandpass', analog=False, output='ba')
            self._b = filter_[0]
            self._a = filter_[1]
            self._z = {}

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        if dtype is not None:
            self.dtype = dtype
        if nb_samples is not None:
            self.nb_samples = nb_samples
        if nb_channels is not None:
            self.nb_channels = nb_channels
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

        return

    def _update_initialization(self):

        # TODO integrate the 2 following line properly.
        # TODO Should it be a default pattern to update the initialization of blocks?
        input_parameters = self.get_input('data').get_input_parameters()
        self.configure_input_parameters(**input_parameters)

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
            self._filter_engine = GpuFilter(context, coefficients, self.nb_channels, 'float32', self.nb_samples)
        else:
            self._z = {}
            m = max(len(self._a), len(self._b)) - 1
            for i in range(0, self.nb_channels):
                self._z[i] = np.zeros(m, dtype=np.float32)

        return

    def _get_output_parameters(self):

        params = {
            'dtype': self.dtype,
            'nb_samples': self.nb_samples,
            'nb_channels': self.nb_channels,
            'sampling_rate': self.sampling_rate,
        }

        return params

    def _process(self):

        # Receive input data.
        data_packet = self.get_input('data').receive()
        batch = data_packet['payload']

        self._measure_time('start')

        # Preallocate filtered data.
        filtered_batch = np.empty(batch.shape, dtype=batch.dtype)

        # Process data.
        if self.use_gpu:
            filtered_batch = self._filter_engine.compute_one_chunk(batch)
        else:
            for j in range(0, self.nb_channels):
                # Filter data.
                filtered_batch[:, j], self._z[j] = signal.lfilter(self._b, self._a, batch[:, j], zi=self._z[j])
            # Center data channel by channel.
            channel_medians = np.median(filtered_batch, axis=0)
            for j in range(0, self.nb_channels):
                filtered_batch[:, j] -= channel_medians[j]
            # Center data sample by sample (if necessary).
            if self.remove_median:
                sample_medians = np.median(filtered_batch, axis=1)
                for i in range(0, self.nb_samples):
                    filtered_batch[i, :] -= sample_medians[i]

        # Prepare output data packet.
        packet = {
            'number': data_packet['number'],
            'payload': filtered_batch,
        }

        # Send output data packet.
        self.get_output('data').send(packet)

        self._measure_time('end')

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
