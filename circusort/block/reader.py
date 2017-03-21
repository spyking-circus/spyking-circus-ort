import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Reader(object):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        object.__init__(self)

        self.log_address = log_address

        self.name = "Reader's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.path = "/tmp/input.dat"
        self.dtype = 'float32'
        self.n_electrodes = 1

        self.file = None
        self.context = zmq.Context()
        self.output = Endpoint(self)

    def initialize(self):
        '''TODO add docstring'''

        # Create input file object
        self.file = open(self.path, mode='r')

        # TODO check correctness
        self.output.dtype = self.dtype
        self.output.shape = self.n_electrodes

        return

    def connect(self):
        '''TODO add docstring'''

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def start(self):
        '''TODO add dosctring'''

        i = 0
        while i < 1000:
            print(i)
            msg = b"a"
            self.output.socket.send(msg)
            i = i + 1

        return
