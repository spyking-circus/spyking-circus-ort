import time
import zmq
import logging
import numpy
from circusort.nodes.nodes import Node

logger = logging.getLogger(__name__)

class DataSource(object):

    def __init__(self, config):
        self.config      = config
        self.buffer      = int(self.config.acquisition.buffer_size)
        self.nb_channels = self.config.nb_channels 
        self._time       = 0

    def get_next_buffer(self):
        self._time += self.buffer
        return self._get_buffer()

    def _get_buffer(self):
        raise NotImplementedError



class RNGSource(DataSource):

    def __init__(self, config):
        DataSource.__init__(self, config)

    def _get_buffer(self):
        return numpy.random.randn((self.nb_channels, self.buffer), dtype=numpy.float32)


class FileSource(DataSource):

    def __init__(self, config):
        DataSource.__init__(self, config)
        self.file       = open(self.config.acquisition.file, 'wb')
        self.data_dtype = open(self.config.acquisition.data_dtype, 'wb')
        self._offset    = 0

    def _get_buffer(self):
        return 

    def __del__(self):
        self.file.close()


class SocketSource(DataSource):

    def __init__(self, config):
        DataSource.__init__(self, config)


class BufferSource(DataSource):

    def __init__(self, config):
        DataSource.__init__(self, config)



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
        socket  = context.socket(zmq.PUB)
        socket.bind(self.address)

        logger.debug("Server's socket Send data on network port:\n  {}".format(self.address))

        while True:
            message = "Hello world!"
            #message = self.source.get_next_buffer()
            socket.send(message)
            #message = socket.recv()
            #print("message: {}".format(message))
            time.sleep(1)

    def set_data_source(self, source):
        assert isinstance(source, DataSource), "source should be a DataSource object"
        self.source = source
