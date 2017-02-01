import time
import zmq
import logging
from circusort.nodes.nodes import Node


logger = logging.getLogger(__name__)


class DataServerNode(Node):

    def __init__(self, config):
        Node.__init__(self, config, name="data_server")
        self.interface = self.config.acquisition.server_ip
        self.port      = self.config.acquisition.port
        self.protocol  = self.config.acquisition.protocol

    def _start(self):
        '''TODO add docstring...'''
        self.address = "{}://{}:{}".format(self.protocol, self.interface, self.port)
        context = zmq.Context()
        socket  = context.socket(zmq.PAIR)
        socket.bind(self.address)

        logger.debug("Server's socket Send data on network port:\n  {}".format(self.address))

        while True:
            message = "Hello world!"
            socket.send(message)
            message = socket.recv()
            print("message: {}".format(message))
            time.sleep(1)
