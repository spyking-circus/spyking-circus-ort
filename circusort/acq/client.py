import time
import zmq
import logging
from circusort.nodes.nodes import Node

logger = logging.getLogger(__name__)


class DataReceiverNode(Node):

    def __init__(self, config):
        Node.__init__(self, config, name="data_receiver")
        self.interface = self.config.acquisition.server_ip
        self.port      = self.config.acquisition.port
        self.protocol  = self.config.acquisition.protocol

    def _start(self):
        '''TODO add docstring...'''
        self.address = "{}://{}:{}".format(self.protocol, self.interface, self.port)
        context = zmq.Context()
        socket  = context.socket(zmq.PAIR)
        socket.connect(self.address)

        logger.debug("Client socket listens on network port:\n  {}".format(self.address))

        while True:
            message = socket.recv()
            print("message: {}".format(message))
            socket.send("client message to server1")
