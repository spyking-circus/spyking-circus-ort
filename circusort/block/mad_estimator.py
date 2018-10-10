import numpy as np

from circusort.block.block import Block


__classname__ = 'MADEstimator'


class MADEstimator(Block):
    """MAD estimator block

    Attributes:
        sampling_rate: float
            The sampling rate [Hz].
        time_constant: float
            Size of the averaging time window [s]. The default value is 1.0.
        epsilon: float
            Trigger parameter of the broadcast of MAD values. The default value is 5e-3.

    Input:
        data

    Output:
        mads
    """

    name = "MAD Estimator"

    params = {
        'sampling_rate': 20e+3,
        'time_constant': 1.0,
        'epsilon': 5e-3,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads', structure='dict')
        self.add_input('data', structure='dict')

        # The following lines are useful to avoid some PyCharm warnings.
        self.sampling_rate = self.sampling_rate
        self.time_constant = self.time_constant
        self.epsilon = self.epsilon

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._n = 0
        self._medians = None
        self._mads = None
        self._last_mads = None

    def _initialize(self):

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels

        return

    def _update_initialization(self):

        # Define shape.
        shape = (1, self._nb_channels)
        # Define data type.
        if self._dtype in ['float32', 'float64']:
            dtype = self._dtype
        elif self._dtype in ['int16', 'uint16']:
            dtype = np.float64
        else:
            string = "MAD is not supported for data of type '{}'"
            message = string.format(self._dtype)
            raise TypeError(message)

        self._medians = np.zeros(shape, dtype=dtype)
        self._mads = np.zeros(shape, dtype=dtype)
        self._last_mads = np.zeros(shape, dtype=dtype)

        self._tau = self.time_constant * self.sampling_rate / self._nb_samples
        self._gamma = np.exp(- 1.0 / self._tau)

        return

    def _check_if_active(self):
        # Compute test value.

        # TODO check the statistics behind this test value.
        test = np.zeros_like(self._mads)
        i, j = np.nonzero(self._last_mads)
        test[i, j] = self._mads[i, j] / self._last_mads[i, j]
        test = np.mean(np.abs(test - 1.0))
        if test < self.epsilon:
            # Log info message.
            min_mad = np.amin(self._mads)
            median_mad = np.median(self._mads)
            max_mad = np.amax(self._mads)
            string = "{} has converged (min={}, median={}, max={})."
            message = string.format(self.name_and_counter, min_mad, median_mad, max_mad)
            self.log.info(message)
            # Set block as active.
            self._set_active_mode()

        return

    @staticmethod
    def _alpha(gamma, n):

        alpha = (1.0 - gamma ** n) / (1.0 - gamma ** (n + 1))

        return alpha

    @staticmethod
    def _beta(gamma, n):

        beta = gamma ** n * (1.0 - gamma) / (1.0 - gamma ** (n + 1))

        return beta

    def _process(self):

        # Receive input data.
        data_packet = self.get_input('data').receive()
        batch = data_packet['payload']

        self._measure_time('start')

        # Update the weights.
        alpha = self._alpha(self._gamma, self._n)
        beta = self._beta(self._gamma, self._n)
        # Compute the medians.
        medians = np.median(batch, axis=0)
        self._medians = alpha * self._medians + beta * medians
        # Compute the MADs.
        mads = np.median(np.abs(batch - self._medians), axis=0)
        self._mads = alpha * self._mads + beta * mads
        # Increment counter.
        self._n += 1
        # Check state (if necessary).
        if not self.is_active:
            self._check_if_active()
        # Prepare and send output data packet (if necessary).
        if self.is_active:
            # Prepare output data packet.
            packet = {
                'number': data_packet['number'],
                'payload': self._mads,
            }
            # Send output data packet.
            self.get_output('mads').send(packet)
        # Update last seen MADs.
        self._last_mads = self._mads

        self._measure_time('end')

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
