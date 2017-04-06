from .block import Block
import numpy

class Chunk_dispatcher(Block):
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
        self.shape = self.nb_samples // self.nb_groups
        for i in xrange(self.nb_groups):
            self.get_output('data_%d' %i).configure(dtype=self.input.dtype, shape=(self.nb_channels, self.shape))

    def _process(self):
        batch = self.input.receive()
        for i in xrange(self.nb_groups):
            self.get_output('data_%d' %i).send(batch[:, i*self.shape:(i+1)*self.shape].flatten())
        return