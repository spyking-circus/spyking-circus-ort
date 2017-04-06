from .block import Block
import numpy

class Chunk_grouper(Block):
    '''TODO add docstring'''

    name   = "Chunk Grouper"

    params = {'nb_groups' : 1}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('data')
        for i in xrange(self.nb_groups):
            self.add_input('data_%d' %i)    
        
    @property
    def nb_samples(self):
        shape = 0
        for i in xrange(self.nb_groups):
            shape += self.get_input('data_%d' %i).shape[1]
        return shape

    @property
    def nb_channels(self):
        return self.get_input('data_0').shape[0]

    @property
    def dtype(self):
        return self.get_input('data_0').dtype

    @property
    def shape(self):
        return self.get_input('data_0').shape[1]

    def _initialize(self):
        return

    def _guess_output_endpoints(self):
        try:
            self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))
            self.result = numpy.zeros((self.nb_channels, self.nb_samples), dtype=self.dtype)
        except Exception:
            self.log.debug('Not all input connections have been established!')


    def _process(self):

        for i in xrange(self.nb_groups):
            batch = self.get_input('data_%i' %i).receive()
            self.result[:, i*self.shape:(i+1)*self.shape] = batch

        self.output.send(self.result.flatten())
        return