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
              'rate'          : 5,
              'nb_cells'      : 100,
              'refractory'    : 5}

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
        self.refrac     = int(self.refractory * 1e-3 * self.sampling_rate)
        return

    def _process(self):

        for i in xrange(0, self.nb_samples-1):
            self.result[:, i+1] = self.result[:, i]*self.decay_time + 2*numpy.random.randn(self.nb_channels)*self.dt


        ## Add fake spikes
        for i in xrange(self.nb_cells):
            spikes = numpy.random.rand(self.nb_samples) < self.rate / float(self.sampling_rate)
            spikes = numpy.where(spikes == True)[0]
            pos    = self.positions[i]
            amp    = self.amplitudes[i]
            t_last = - self.refrac
            for scount, spike in enumerate(spikes):
                if (spike - t_last) > self.refrac:
                    self.result[pos, spike] = amp

        self.output.send(self.result)
        self.result[:, 0] = self.result[:, -1]*self.decay_time + numpy.random.randn(self.nb_channels)*self.dt

        
        return
