from .block import Block
import os
import numpy
from circusort.io.generate import synthetic_grid



class Reader(Block):
    '''TODO add docstring'''

    name = "File reader"

    params = {'data_path'     : '/tmp/input.dat', 
              'dtype'         : 'float32',
              'repeat'        : False,
              'nb_channels'   : 10,
              'nb_samples'    : 1024, 
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        '''TODO add docstring'''
        self.data  = numpy.memmap(self.data_path, dtype=self.dtype, mode='r')
        self.shape = (self.data.size/self.nb_channels, self.nb_channels)
        self.data  = None
        self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))
        return

    def _process(self):
        self.data = numpy.memmap(self.data_path, dtype=self.dtype, mode='r', shape=self.shape)

        i_min = self.nb_samples * self.counter
        i_max = self.nb_samples * (self.counter + 1)

        if self.repeat:
            i_min = i_min % self.shape[0]
            i_max = i_max % self.shape[0]

        if i_min < self.shape[0]:
          self.output.send(self.data[i_min:i_max, :])
  
        self.data = None
        return
