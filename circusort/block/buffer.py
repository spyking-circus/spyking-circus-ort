from .block import Block
import numpy
from scipy import signal

class Buffer(Block):
    '''TODO add docstring'''

    name = "Buffer"

    params = {'sampling_rate' : 500,
              'time_before'   : 5,
              'time_after'    : 5}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    def _initialize(self):
        self._buffer_pre_width_ = int(self.sampling_rate*self.time_before*1e-3)
        self._buffer_pre_width_ = int(self.sampling_rate*self.time_after*1e-3)
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2        
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