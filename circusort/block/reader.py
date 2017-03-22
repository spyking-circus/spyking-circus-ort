import numpy
import threading
import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils



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
        self.dtype = 'float32'
        self.nb_channels = 4

        # # TODO remove following lines
        # self.file = None
        self.data = None
        self.context = zmq.Context()
        self.output = Endpoint(self)

    def initialize(self):
        '''TODO add docstring'''

        self.log.debug("initialization")
        # Create input file object

        shape = (self.nb_channels,)

        # # TODO remove following lines
        # self.file = open(self.path, mode='r')
        self.data = numpy.memmap(self.path, dtype=self.dtype, mode='r')

        # TODO check correctness
        self.output.dtype = self.dtype
        self.output.shape = (self.nb_channels, 100)

        return

    def connect(self):
        '''TODO add docstring'''

        self.log.debug("connection")

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def run(self):
        '''TODO add dosctring'''

        self.log.debug("run")

        sample_shape = (self.nb_channels,)
        sample_size = numpy.product(sample_shape)
        batch_shape = (100,) + sample_shape
        batch_size = numpy.product(batch_shape)

        i = 0
        while i < 1000:

            # TODO get data sample from input
            i_min = batch_size * i
            i_max = batch_size * (i + 1)
            batch = self.data[i_min:i_max]
            batch = batch.reshape(batch_shape)

            # TODO set data sample to output
            self.output.socket.send(batch)

            # TODO increment counter
            i = i + 1

        return
