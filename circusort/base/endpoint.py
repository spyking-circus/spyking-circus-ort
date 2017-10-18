import numpy
import zmq
import tempfile
import os
import json


TERM_MSG = "END"


class Connection(object):
    """Connection"""
    # TODO complete docstring.

    _defaults_structure = {
        'array': {
            'dtype': None,
            'shape': None
        },
        'dict': {},
        'boolean': {},
    }

    params = {}

    def __init__(self, block, name, structure, **kwargs):

        self.block = block
        self.structure = structure
        self.name = name
        self.initialized = False
        params = self._defaults_structure[self.structure]
        self.params.update(params)
        self.params.update(kwargs)
        self.configure(**self.params)

    def configure(self, **kwargs):
        """Configure connection"""
        # TODO complete docstring.

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        return

    def initialize(self, **kwargs):
        if not self.initialized:
            self._initialize(**kwargs)
            self.initialized = True

    def receive(self, blocking=True):
        return self._get_data(blocking)

    def get_description(self):
        return self._get_description()

    def send(self, batch):
        if self.initialized:
            self._send_data(batch)
        return

    def send_end_connection(self):
        if self.initialized:
            self._send_end_connection()
        return


class Encoder(json.JSONEncoder):

    def default(self, obj):
        if obj is None:
            obj = json.JSONEncoder.default(obj)
        else:
            if isinstance(obj, numpy.ndarray):
                obj = obj.tolist()
            else:
                raise TypeError("Type {t} is not serializable.".format(t=type(obj)))
        return obj


class EOCError(Exception):
    """End of connection error."""

    def __init__(self, msg=None):

        if msg is None:
            msg = "End of connection"

        super(EOCError, self).__init__(msg)


class Endpoint(Connection):
    """Endpoint"""
    # TODO complete docstring.

    params = {
        'addr': None,
        'socket': None,
    }

    def __init__(self, block, name, structure, **kwargs):

        Connection.__init__(self, block, name, structure, **kwargs)

    def __del__(self):

        if self.socket is not None:
            self.socket.close()

    def _get_data(self, blocking=True):
        """Get batch of data."""
        # TODO complete docstring.

        if not blocking:
            try:
                batch = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
        else:
            batch = self.socket.recv()

        if batch == TERM_MSG:
            raise EOCError()

        if self.structure == 'array':
            batch = numpy.fromstring(batch, dtype=self.dtype)
            batch = numpy.reshape(batch, self.shape)
        elif self.structure == 'dict':
            batch = json.loads(batch)
        elif self.structure == 'boolean':
            batch = bool(batch)

        return batch

    def _send_data(self, batch):

        if self.structure == 'array':
            self.socket.send(batch.tostring())
        elif self.structure == 'dict':
            self.socket.send(json.dumps(batch, cls=Encoder))
        elif self.structure == 'boolean':
            self.socket.send(str(batch))

        return

    def _send_end_connection(self):

        self.socket.send(TERM_MSG)

        return

    def _get_description(self):
        description = {
            'addr': self.addr,
            'structure': self.structure,
        }
        if self.structure == 'array':
            description.update({'dtype': self.dtype, 'shape': self.shape})
        return description

    def _initialize(self, protocol='tcp', host='127.0.0.1', port='*'):
        if protocol == 'ipc':
            tmp_file = tempfile.NamedTemporaryFile()
            self.tmp_name = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name)) + ".ipc"
            tmp_file.close()
            address = '{t}://{e}'.format(t=protocol, e=self.tmp_name)
        else:
            self.tmp_name = None
            endpoint = '{h}:{p}'.format(h=host, p=port)
            address = '{t}://{e}'.format(t=protocol, e=endpoint)
        self.socket = self.block.context.socket(zmq.PUB)
        self.socket.bind(address)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT)

    def __del__(self):
        self.socket.close()
        if self.tmp_name is not None:
            os.remove(self.tmp_name)
