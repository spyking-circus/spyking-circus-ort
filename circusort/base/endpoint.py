import numpy
import zmq

class Connection(object):

    defaults = {'dtype' : None,
                'shape' : None}

    params   = {}

    def __init__(self, block, name, **kwargs):

        self.block = block
        self.name  = name
        self.initialized = False
        self.defaults.update(self.params)
        self.defaults.update(kwargs)
        self.configure(**self.defaults)

    def configure(self, **kwargs):
        '''TODO add docstring'''

        for key, value in kwargs.items():
            self.defaults[key] = kwargs[key]
            self.__setattr__(key, value)

        #self.log.debug("{n} is configured".format(n=self.name))

        return

    def initialize(self, **kwargs):
        if not self.initialized:
            self._initialize(**kwargs)
            self.initialized = True

    def receive(self):
        '''TODO add docstring'''
        return self._get_data()

    def send(self, batch):
        '''TODO add docstring'''

        self._send_data(batch)
        return


class Endpoint(Connection):
    '''TODO add docstring'''

    params = {'addr'  : None, 
              'socket' : None}

    def __init__(self, block, name, **kwargs):

        Connection.__init__(self, block, name, **kwargs)


    def __del__(self):

        if self.socket is not None:
            self.socket.close()

    def _get_data(self):
        '''TODO add docstring'''

        batch = self.socket.recv()
        batch = numpy.fromstring(batch, dtype=self.dtype)
        batch = numpy.reshape(batch, self.shape)
        return batch

    def _send_data(self, batch):
        self.socket.send(batch)


    def _initialize(self, protocol='tcp', host='127.0.0.1', port='*'):
        endpoint  = '{h}:{p}'.format(h=host, p=port)
        address   = '{t}://{e}'.format(t=protocol, e=endpoint)
        self.socket = self.block.context.socket(zmq.PAIR)
        self.socket.bind(address)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT)
