from .block import Block
import os
import numpy
from circusort.io.generate import synthetic_grid



class Reader(Block):
    '''TODO add docstring'''

    name = "File reader"

    params = {'data_path'     : '/tmp/input.dat', 
              'force'         : False,
              'dtype'         : 'float32',
              'size'          : 10,
              'sampling_rate' : 20000, 
              'nb_buffers'    : 1000, 
              'nb_samples'    : 1024,
              'duration'      : 60}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_output('data')

    @property
    def nb_channels(self):
        return self.size**2

    def _initialize(self):
        '''TODO add docstring'''

        if not os.path.exists(self.data_path) or self.force:
            # Generate input file
            synthetic_grid(self.data_path,
                           size=self.size,
                           duration=self.duration,
                           sampling_rate=self.sampling_rate)
        # Create input memory-map
        self.data = numpy.memmap(self.data_path, dtype=self.dtype, mode='r')
        self.batch_size = self.nb_channels * self.nb_samples
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))
        return

    def _process(self):
        i_min = self.batch_size * self.counter
        i_max = self.batch_size * (self.counter + 1)
        batch = self.data[i_min:i_max]
        self.output.send(batch.flatten())
        return
