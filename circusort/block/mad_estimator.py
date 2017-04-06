from .block import Block
import numpy


class Mad_estimator(Block):
    '''TODO add docstring'''

    name = "MAD Estimator"

    params = {'time_constant' : 10.,
              'epsilon'       : 1e-5,
              'threshold'     : 6,
              'sampling_rate' : 20000.}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('mads')
        self.add_input('data')

    def _initialize(self):
        self.is_ready = False
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]
        
    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _guess_output_endpoints(self):
        self.mads         = numpy.zeros((self.nb_channels, 1), dtype=numpy.float32)
        self.median_means = numpy.zeros((self.nb_channels, 1), dtype=numpy.float32)
        self.decay_time   = numpy.exp(-(self.nb_samples/self.sampling_rate)/float(self.time_constant))
        self.outputs['mads'].configure(dtype='float32', shape=(self.nb_channels, 1))
        self.is_ready     = False

    def _process(self):
        batch     = self.input.receive(blocking=False)
        if batch is not None:
            self.median_means[:,0] = self.median_means[:,0]*self.decay_time + numpy.median(batch, 1)/self.nb_samples
            
            before = self.mads.mean()
            self.mads = self.mads*self.decay_time + numpy.median(numpy.abs(batch) - self.median_means, 1)/self.nb_samples
            
            if not self.is_ready and (numpy.abs(self.mads.mean() - before) < self.epsilon):
                self.is_ready = True
                self.log.info('{n} has converged'.format(n=self.name_and_counter))

            if self.is_ready:
                self.get_output('mads').send(self.threshold*self.median_means.flatten())
        return