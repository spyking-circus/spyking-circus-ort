from .block import Block
import zmq
import numpy
from scipy import signal
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid


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
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]

    @property
    def nb_samples(self):
        return self.input.shape[1]

    def _connect(self, key):
        self.get_output(key).socket = self.context.socket(zmq.PAIR)
        self.get_output(key).socket.connect(self.get_output(key).addr)
        return

    def _guess_output_endpoints(self):
        self.output.configure(dtype=self.input.dtype, shape=self.input.shape)        

    def _process(self):
        batch = self.input.receive()
        for i in xrange(self.nb_channels):
            batch[i]  = signal.filtfilt(self.b, self.a, batch[i])
            batch[i] -= numpy.median(batch[i]) 

        if self.remove_median:
            global_median = numpy.median(batch, 0)
            for i in xrange(self.nb_channels):
                batch[i] -= global_median
        self.output.send(batch.flatten())
        return