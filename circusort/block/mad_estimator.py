from .block import Block
import numpy


class Mad_estimator(Block):
    '''TODO add docstring'''

    name = "MAD Estimator"

    params = {'time_constant' : 1.,
              'epsilon'       : 1e-5,
              'threshold'     : 5,
              'sampling_rate' : 20000.}

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
        self.mads         = numpy.zeros(self.nb_channels, dtype=numpy.float32)
        self.median_means = numpy.zeros(self.nb_channels, dtype=numpy.float32)
        self.dt           = float(self.nb_samples)/self.sampling_rate
        self.factor       = self.dt/float(self.time_constant)
        self.decay_time   = numpy.exp(-self.factor)
        self.outputs['mads'].configure(dtype='float32', shape=(self.nb_channels, 1))
        self.last_mads_mean = numpy.zeros(self.nb_channels, dtype=numpy.float32)

    def _check_if_active(self):
        test = numpy.mean(numpy.abs(self.mads/self.last_mads_mean) - 1)
        if (test < self.epsilon):
            self.log.info('{n} has converged'.format(n=self.name_and_counter))
            self._set_active_mode()

    def _process(self):
        batch             = self.input.receive()
        self.median_means = self.median_means*self.decay_time + numpy.median(batch, 0)*self.factor
        self.mads         = self.mads*self.decay_time + numpy.median(numpy.abs(batch) - self.median_means, 0)*self.factor

        if not self.is_active:
            self._check_if_active()

        if self.is_active:
            self.get_output('mads').send(self.threshold*self.mads)

        self.last_mads_mean = self.mads
        return