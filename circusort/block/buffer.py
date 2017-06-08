from .block import Block
import numpy
from scipy import signal

class Buffer(Block):
    '''TODO add docstring'''

    name = "Buffer"

    params = {'sampling_rate' : 20000,
              'time_buffer'   : 5}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    def _initialize(self):
        self._buffer_width_ = int(self.sampling_rate*self.time_buffer*1e-3)
        return

    @property
    def nb_channels(self):
        return self.input.shape[1]

    @property
    def nb_samples(self):
        return self.input.shape[0]

    def _guess_output_endpoints(self):
        shape       = (self.nb_samples + self._buffer_width_, self.nb_channels)
        self.buffer = numpy.zeros(shape, dtype=self.input.dtype)
        self.output.configure(dtype=self.input.dtype, shape=shape)

    def _process(self):
        batch = self.input.receive()
        self.buffer[self._buffer_width_:, :] = batch
        self.output.send(self.buffer)
        self.buffer[:self._buffer_width_, :] = self.buffer[-self._buffer_width_:, :]
        return