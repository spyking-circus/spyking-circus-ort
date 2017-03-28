import threading
import time
import zmq

from circusort.base.endpoint import Endpoint
from circusort.base import utils



class Writer(threading.Thread):
    '''TODO add docstring'''

    def __init__(self, log_address=None):

        threading.Thread.__init__(self)

        self.log_address = log_address

        self.name = "Writer's name (original)"
        if self.log_address is None:
            raise NotImplementedError("no logger address")
        self.log = utils.get_log(self.log_address, name=__name__)

        self.nb_buffers = 1000

        self.path = "/tmp/output.dat"

        self.context = zmq.Context()
        self.input = Endpoint(self)
        self.t_start = None
        self.t_comp = None

        self.log.info("writer created")

    def initialize(self):
        '''TODO add docstring'''

        self.log.debug("initialization")

        # Bind socket for input data
        transport = 'tcp'
        host = '127.0.0.1'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        self.input.socket = self.context.socket(zmq.PAIR)
        # self.input.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.input.socket.bind(address)
        self.input.addr = self.input.socket.getsockopt(zmq.LAST_ENDPOINT)

        # TODO create output file object
        self.file = open(self.path, mode='wb')

        return

    def connect(self):
        '''TODO add docstring'''

        self.log.debug("connection")

        return

    def process(self):

        # self.log.debug("process") # commented to reduce logging

        # TODO receive first batch of data
        batch = self.input.receive()

        # TODO set batch of data to the output
        batch = batch.tobytes()
        self.file.write(batch)

        return

    def run(self):
        '''TODO add dosctring'''

        self.log.debug("run")

        self.process()

        self.t_start = time.time()

        for i in range(1, self.nb_buffers):
            self.process()

        self.t_comp = time.time() - self.t_start

        return
