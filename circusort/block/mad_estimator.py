import numpy as np

from circusort.block.block import Block


class Mad_estimator(Block):
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
    # TODO complete docstring.

    name = "MAD Estimator"

    params = {
        'sampling_rate': 20e+3,
        'time_constant': 1.0,
        'epsilon': 5e-3,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads')
        self.add_input('data')

        # The following lines are useful to avoid some PyCharm warnings.
        self.sampling_rate = self.sampling_rate
        self.time_constant = self.time_constant
        self.epsilon = self.epsilon

        self._n = 0
        self._medians = None
        self._mads = None
        self._last_mads = None

    def _initialize(self):

        return

    @property
    def nb_channels(self):

        return self.input.shape[1]

    @property
    def nb_samples(self):

        return self.input.shape[0]

    def _guess_output_endpoints(self):

        shape = (1, self.nb_channels)

        self._medians = np.zeros(shape, dtype=np.float32)
        self._mads = np.zeros(shape, dtype=np.float32)
        self._last_mads = np.zeros(shape, dtype=np.float32)

        self._tau = self.time_constant * self.sampling_rate / self.nb_samples
        self._gamma = np.exp(- 1.0 / self._tau)

        self.outputs['mads'].configure(dtype='float32', shape=shape)

        return

    def _check_if_active(self):
        # Compute test value.

        # TODO check the statistics behind this test value.
        test = self._mads / self._last_mads
        test[np.isnan(test)] = 0.0
        test = np.mean(np.abs(test - 1.0))
        if test < self.epsilon:
            message = "{} has converged".format(self.name_and_counter)
            self.log.info(message)
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
        batch = self.input.receive()

        self._measure_time('start', frequency=100)

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
        # Send output data (if necessary).
        if self.is_active:
            self.get_output('mads').send(self._mads)
        # Update last seen MADs.
        self._last_mads = self._mads

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
