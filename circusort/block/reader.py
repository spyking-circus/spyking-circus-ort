from .block import Block
import zmq
import numpy
import os
from circusort.base.endpoint import Endpoint
from circusort.base import utils
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
        self.outputs['data'] = Endpoint(self)


    @property
    def shape(self):
        return (self.nb_channels,)

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
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))
        return

    def _connect(self):
        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def _run(self):
        sample_size = numpy.product(self.shape)
        batch_shape = (self.nb_samples,) + self.shape
        batch_size = numpy.product(batch_shape)

        for i in range(0, self.nb_buffers):

            # TODO get data sample from input
            i_min = batch_size * i
            i_max = batch_size * (i + 1)
            batch = self.data[i_min:i_max]
            batch = batch.reshape(batch_shape)

            # TODO set data sample to output
            self.output.send(batch)

        return
