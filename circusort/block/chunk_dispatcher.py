from .block import Block
import numpy

class Channel_dispatcher(Block):
    '''TODO add docstring'''

    name   = "Chunk Dispatcher"

    params = {'nb_groups' : 1}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_input('data')
        for i in xrange(self.nb_groups):
            self.add_output('data_%d' %i)    
        
    @property
    def nb_channels(self):
        return self.input.shape[0]

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _initialize(self):
        return

    def _guess_output_endpoints(self):
        for i in xrange(self.nb_groups):
            self.get_output('data_%d' %i).configure(dtype=self.input.dtype, shape=(self.nb_channels, self.nb_samples))
        self.buffer = numpy.zeros((self.nb_channels, self.nb_groups*self.nb_samples), dtype=self.input.dtype)

    def _process(self):
        batch = self.input.receive()
        self.buffer[:nb_samples] = batch
        for i in xrange(self.nb_groups):
            self.get_output('data_%d' %i).send(self.buffer[i*self.nb_groups:(i+1)*self.nb_groups].flatten())
        self.buffer = numpy.roll(self.buffer, self.nb_samples)
        return