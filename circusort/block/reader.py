import numpy
import os
import threading
import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils
from circusort.io.generate import synthetic_grid



class Reader(threading.Thread):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        threading.Thread.__init__(self)

        self.log_address = log_address

        self.name = "Reader's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.path = "/tmp/input.dat"
        self.force = False
        self.dtype = 'float32'
        self.size = 2
        self.duration = 60.0 # s
        self.sampling_rate = 20.0e+3 # Hz

        self.nb_buffers = 1000
        self.nb_samples = 100

        self.data = None
        self.context = zmq.Context()
        self.output = Endpoint(self)

        self.log.info("reader created")

    @property
    def nb_channels(self):
        '''TODO add docstring'''

        return self.size ** 2

    def initialize(self):
        '''TODO add docstring'''

        self.log.info("initialize {n}".format(n=self.name))

        shape = (self.nb_channels,)

        if not os.path.exists(self.path) or self.force:
            # Generate input file
            synthetic_grid(self.path,
                           size=self.size,
                           duration=self.duration,
                           sampling_rate=self.sampling_rate)
        # Create input memory-map
        self.data = numpy.memmap(self.path, dtype=self.dtype, mode='r')

        self.output.dtype = self.dtype
        self.output.shape = (self.nb_channels, self.nb_samples)

        return

    def connect(self):
        '''TODO add docstring'''

        self.log.info("connect {n}".format(n=self.name))

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def run(self):
        '''TODO add dosctring'''

        self.log.info("run {n}".format(n=self.name))

        sample_shape = (self.nb_channels,)
        sample_size = numpy.product(sample_shape)
        batch_shape = (self.nb_samples,) + sample_shape
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
