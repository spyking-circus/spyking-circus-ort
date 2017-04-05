from .block import Block
import numpy
from scipy import signal

class Filter(Block):
    '''TODO add docstring'''

    name = "Filter"

    params = {'cut_off'       : 500,
              'sampling_rate' : 20000,
              'remove_median' : True}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    def _initialize(self):
        cut_off = numpy.array([self.cut_off, 0.95*(self.sampling_rate/2.)])
        b, a   = signal.butter(3, cut_off/(self.sampling_rate/2.), 'pass')
        self.b = b
        self.a = a
        self.z = {}
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        
        self.z = {}
        m = max(len(self.a), len(self.b)) - 1
        for i in xrange(self.nb_channels):
            self.z[i] = numpy.zeros(m, dtype=numpy.float32)


    def _process(self):
        batch = self.input.receive()
        for i in xrange(self.nb_channels):
            batch[i], self.z[i]  = signal.lfilter(self.b, self.a, batch[i], zi=self.z[i])
            batch[i] -= numpy.median(batch[i]) 

        if self.remove_median:
            global_median = numpy.median(batch, 0)
            for i in xrange(self.nb_channels):
                batch[i] -= global_median
        self.output.send(batch.flatten())
        return