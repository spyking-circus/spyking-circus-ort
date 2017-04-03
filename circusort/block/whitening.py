from .block import Block
import numpy
from numpy.linalg import eigh


class Whitening(Block):
    '''TODO add docstring'''

    name   = "Whitening"

    params = {'spike_width'   : 5,
              'radius'        : 'auto', 
              'fudge'         : 1e-18, 
              'sampling_rate' : 20000,
              'chunks'        : 10}

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)
        self.add_output('data')
        self.add_input('data')

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    def _initialize(self):
        self.is_ready = False
        self.duration = self.chunks*self.sampling_rate
        return

    def _guess_output_endpoints(self):
        self.output.configure(dtype='float32', shape=(self.nb_channels, self.nb_samples))
        self.silences = numpy.zeros((self.nb_channels, 0), dtype=self.input.dtype)

    def _get_whitening_matrix(self):
        Xcov = numpy.dot(self.silences, self.silences.T)/self.silences.shape[1]
        d,V  = eigh(Xcov)
        D    = numpy.diag(1./numpy.sqrt(d+self.fudge))
        self.whitening_matrix = numpy.dot(numpy.dot(V,D), V.T).astype(numpy.float32)
        self.is_ready = True

    def _process(self):
        batch = self.input.receive()
        if self.is_ready:
            batch = numpy.dot(self.whitening_matrix, batch)
            self.output.send(batch.flatten())
        else:
            self.silences = numpy.hstack((self.silences, batch))
            if self.silences.shape[1] > self.duration:
                self._get_whitening_matrix()
                self.log.info("{n} computes whitening matrix".format(n=self.name))
        return