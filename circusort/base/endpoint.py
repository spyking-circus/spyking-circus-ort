import numpy
import zmq
import tempfile
import os
import json

class Connection(object):

    _defaults_structure = {'array' : {'dtype': None, 'shape' : None},
                          'dict'  : {}}

    params   = {}

    def __init__(self, block, name, structure, **kwargs):

        self.block     = block
        self.structure = structure
        self.name      = name
        self.initialized = False
        params = self._defaults_structure[self.structure]
        self.params.update(params)
        self.params.update(kwargs)
        self.configure(**self.params)

    def configure(self, **kwargs):
        '''TODO add docstring'''

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        #self.log.debug("{n} is configured".format(n=self.name))

        return

    def initialize(self, **kwargs):
        if not self.initialized:
            self._initialize(**kwargs)
            self.initialized = True

    def receive(self):
        return self._get_data()

    def get_description(self):
        return self._get_description()

    def send(self, batch):
        if self.initialized:
            self._send_data(batch)
        return


class Endpoint(Connection):
    '''TODO add docstring'''

    params = {'addr'  : None, 
              'socket' : None}

    def __init__(self, block, name, structure, **kwargs):

        Connection.__init__(self, block, name, structure, **kwargs)

    def __del__(self):

        if self.socket is not None:
            self.socket.close()

    def _get_data(self):
        '''TODO add docstring'''

        batch = self.socket.recv()
        if self.structure == 'array':
            batch = numpy.fromstring(batch, dtype=self.dtype)
            batch = numpy.reshape(batch, self.shape)
        elif self.structure == 'dict':
            json.loads(batch)
        return batch

    def _send_data(self, batch):
        if self.structure == 'array':
            self.socket.send(batch)
        elif self.structure == 'dict':
            self.socket.send(json.dumps(batch))

    def _get_description(self):
        description = {'addr' : self.addr}
        if self.structure == 'array':
            description.update({'dtype' : self.dtype, 'shape' : self.shape})
        return description

    def _initialize(self, protocol='tcp', host='127.0.0.1', port='*'):
        if protocol == 'ipc':
            tmp_file = tempfile.NamedTemporaryFile()
            tmp_name = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".ipc"
            tmp_file.close()
            address = '{t}://{e}'.format(t=protocol, e=tmp_name)
        else:
            endpoint = '{h}:{p}'.format(h=host, p=port)
            address  = '{t}://{e}'.format(t=protocol, e=endpoint)
        self.socket = self.block.context.socket(zmq.PUB)
        self.socket.bind(address)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT)