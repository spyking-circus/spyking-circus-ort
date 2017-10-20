from .block import Block
import numpy as np


class Mad_estimator(Block):
    """MAD estimator block

    Attributes:
        time_constant: float
            Size of the averaging time window [s]. The default value is 1.0.
        epsilon: float
            Trigger parameter of the broadcast of MAD values. The default value is 5e-3.
        sampling_rate: float
            Sampling rate [Hz]. The default value is 20e+3

    Input:
        data

    Output:
        mads

    """
    # TODO complete docstring.

    name = "MAD Estimator"

    params = {
        'time_constant': 1.0,
        'epsilon': 5e-3,
        'sampling_rate': 20e+3,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads')
        self.add_input('data')

    def _initialize(self):

        return

    @property
    def nb_channels(self):

        return self.input.shape[1]
        
    @property
    def nb_samples(self):

        return self.input.shape[0]

    def _guess_output_endpoints(self):

        self.mads = np.zeros(self.nb_channels, dtype=np.float32)
        self.median_means = np.zeros(self.nb_channels, dtype=np.float32)
        self.dt = float(self.nb_samples) / self.sampling_rate
        self.factor = self.dt / float(self.time_constant)
        self.decay_time = np.exp(-self.factor)
        self.outputs['mads'].configure(dtype='float32', shape=(self.nb_channels, 1))
        self.last_mads_mean = np.zeros(self.nb_channels, dtype=np.float32)

        return

    def _check_if_active(self):
        # Compute test value.

        # TODO check the statistics behind this test value.
        test = self.mads / self.last_mads_mean
        test[np.isnan(test)] = 0.0
        test = np.median(np.abs(test - 1.0))
        if (test < self.epsilon):
            self.log.info('{n} has converged'.format(n=self.name_and_counter))
            self._set_active_mode()

        return

    def _process(self):

        batch = self.input.receive()

        self.median_means = self.median_means * self.decay_time + np.median(batch, 0) * self.factor
        self.mads = self.mads * self.decay_time + np.median(np.abs(batch) - self.median_means, 0) * self.factor

        if not self.is_active:
            self._check_if_active()

        if self.is_active:
            self.get_output('mads').send(self.mads)

        self.last_mads_mean = self.mads

        return
