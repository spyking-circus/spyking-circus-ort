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
        self.inputs['data']  = Endpoint(self)
        self.outputs['data'] = Endpoint(self)

    def _initialize(self):
        self.cut_off = numpy.array([self.cut_off, 0.95*(self.sampling_rate/2.)])
        b, a   = signal.butter(3, self.cut_off/(self.sampling_rate/2.), 'pass')
        self.b = b
        self.a = a
        return

    @property
    def nb_channels(self):
        return self.input.shape[0]

    def _connect(self):
        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)
        return

    def _run(self):
        while self.running:
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