import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Computer(object):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        object.__init__(self)

        self.log_address = log_address

        self.name = "Computer's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.context = zmq.Context()
        self.input = Endpoint(self)
        self.output = Endpoint(self)

    def initialize(self):
        '''TODO add docstring'''

        # Bind socket for input data
        transport = 'tcp'
        host = '127.0.0.1'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        self.input.socket = self.context.socket(zmq.PAIR)
        self.input.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.input.socket.bind(address)
        self.input.addr = self.input.socket.getsockopt(zmq.LAST_ENDPOINT)
        # # TODO remove following line
        # print("\033[91m{}\033[0m".format(self.input.addr))

        return

    def connect(self):
        '''TODO add docstring'''

        self.output.socket = self.context.socket(zmq.PAIR)
        self.output.socket.connect(self.output.addr)

        return

    def configure(self):
        '''TODO add docstring'''

        self.output.dtype = self.input.dtype
        self.output.shape = self.input.shape

        return

    def start(self):
        '''TODO add dosctring'''

        i = 0
        while i < 1000:
            msg = self.input.socket.recv()
            self.output.socket.send(msg)
            i = i + 1

        return
