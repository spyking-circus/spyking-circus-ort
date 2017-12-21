import numpy as np

from circusort.block.block import Block


class Mad_estimator(Block):
    """MAD estimator block

    Attributes:
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
        'time_constant': 1.0,
        'epsilon': 5e-3,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads')
        self.add_input('data')

        self.time_constant = self.time_constant if isinstance(self.time_constant, float) else 1.0  # s
        self.epsilon = self.epsilon if isinstance(self.epsilon, float) else 5e-3  # arb. unit

        self._n = 0
        self._tau = self.time_constant
        self._gamma = np.exp(- 1.0 / self._tau)
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

        self.outputs['mads'].configure(dtype='float32', shape=shape)

        return

    def _check_if_active(self):
        # Compute test value.

        # TODO check the statistics behind this test value.
        test = self._mads / self._last_mads
        test[np.isnan(test)] = 0.0
        test = np.median(np.abs(test - 1.0))
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

        batch = self.input.receive()

        alpha = self._alpha(self._gamma, self._n)
        beta = self._beta(self._gamma, self._n)

        medians = np.median(batch, axis=0)
        self._medians = alpha * self._medians + beta * medians
        mads = np.median(np.abs(batch - self._medians), axis=0)
        self._mads = alpha * self._mads + beta * mads

        self._n += 1

        if not self.is_active:
            self._check_if_active()

        if self.is_active:
            self.get_output('mads').send(self._mads)

        self._last_mads = self._mads

        return
