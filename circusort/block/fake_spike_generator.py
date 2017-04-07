from .block import Block
import numpy


class Fake_spike_generator(Block):
    '''TODO add docstring'''

    name = "Noise Generator"

    params = {'dtype'         : 'float32',
              'nb_channels'   : 10,
              'sampling_rate' : 20000, 
              'nb_samples'    : 1024, 
              'time_constant' : 60.,
              'rate'          : 10,
              'nb_cells'      : 100}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))
        self.dt         = 1./self.sampling_rate
        self.decay_time = numpy.exp(-self.dt/float(self.time_constant))
        self.noise      = numpy.zeros(self.nb_channels)
        self.result     = numpy.zeros((self.nb_channels, self.nb_samples), dtype=self.dtype)
        self.positions  = numpy.random.randint(0, self.nb_channels, self.nb_cells)
        self.amplitudes = 0.1*numpy.random.randn(self.nb_cells)
        return

    def _process(self):

        for i in xrange(0, self.nb_samples-1):
            self.result[:, i+1] = self.result[:, i]*self.decay_time + 2*numpy.random.randn(self.nb_channels)*self.dt

        ## Add fake spikes
        nb_spikes = 2
        src   = numpy.random.randint(0, self.nb_cells, nb_spikes)
        times = numpy.random.randint(0, self.nb_samples-1, nb_spikes) 
        for a, b in zip(src, times):
            self.result[self.positions[a], b] = self.amplitudes[a]

        self.output.send(self.result)
        self.result[:, 0] = self.result[:, -1]*self.decay_time + numpy.random.randn(self.nb_channels)*self.dt

        
        return
