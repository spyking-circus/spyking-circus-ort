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

    def _initialize(self, **kwargs):
        """Abstract method to initialize the connection."""

        raise NotImplementedError()

    def initialize(self, **kwargs):
        """Initialize the connection."""
        # TODO complete docstring.

        if not self.initialized:
            self._initialize(**kwargs)
            self.initialized = True

        return

    def _get_data(self, blocking=True):
        """Abstract method to get data from this connection."""

        raise NotImplementedError()

    def receive(self, blocking=True, discarding_eoc=False):
        """Receive data.

        Parameter:
            blocking: boolean (optional)
                If true then this waits until data arrives. The default value is True.
            discarding_eoc: boolean (optional)
                If true then this discards any end of connection (EOC) signal received. The default value is False.
        Return:
            data: np.ndarray | dictionary | boolean | string
                The data to receive.
        """

        data = self._get_data(blocking=blocking, discarding_eoc=discarding_eoc)

        return data

    def _get_description(self):
        """Abstract method to get a description of the connection."""

        raise NotImplementedError()

    def get_description(self):
        """Get a description of the connection."""

        description = self._get_description()

        return description

    def _send_data(self, batch):
        """Abstract method to send data through this connection."""

        raise NotImplementedError()

    def send(self, batch):
        """Send data.

        Parameter:
            batch: ?
                The data to send.
        """
        # TODO complete docstring.

        if self.initialized:
            self._send_data(batch)

        return

    def _send_end_connection(self):
        """Abstract method to send end of connection message through this connection."""

        raise NotImplementedError()

    def send_end_connection(self):
        """Send end of connection message."""

        if self.initialized:
            self._send_end_connection()

        return


class Encoder(json.JSONEncoder):

    def default(self, obj):
        if obj is None:
            obj = json.JSONEncoder.default(self, obj)
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

        # The following lines are useful to remove some PyCharm warnings.
        if self.structure == 'array':
            self.dtype = self.dtype
            self.shape = self.shape

    def __del__(self):

        if self.socket is not None:
            self.socket.close()
        if self.tmp_name is not None:
            os.remove(self.tmp_name)

    def _get_data(self, blocking=True, discarding_eoc=False):
        """Get batch of data from this endpoint.

        Parameter:
            blocking: boolean (optional)
                If true then this waits until a batch of data arrives. The default value is True.
            discarding_eoc: boolean (optional)
                If true then this discards any end of connection (EOC) signal received. The default value is False.
        Return:
            batch: numpy.ndarray | dictionary | boolean | string
                The batch of data to get.
        """

        if blocking:
            try:
                batch = self.socket.recv()
            except zmq.Again:
                # Resource temporarily unavailable (with respect to a 5 s timeout).
                raise EOCError()
        else:
            try:
                batch = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None

        if batch == TERM_MSG:
            if discarding_eoc:
                return None
            else:
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
        """Send a batch of data from this endpoint.

        Parameter:
            batch: numpy.ndarray | dictionary | boolean | string
                The batch of data to send.
        """

        if self.structure == 'array':
            self.socket.send(batch.tostring())
        elif self.structure == 'dict':
            self.socket.send(json.dumps(batch, cls=Encoder))
        elif self.structure == 'boolean':
            self.socket.send(str(batch))

        return

    def _send_end_connection(self):
        """Send end of connection message from this endpoint."""

        self.socket.send(TERM_MSG)

        return

    def _get_description(self):
        """Get a description of this endpoint."""

        description = {
            'addr': self.addr,
            'structure': self.structure,
        }
        if self.structure == 'array':
            description.update({'dtype': self.dtype, 'shape': self.shape})

        return description

    def _initialize(self, protocol='tcp', host='127.0.0.1', port='*'):
        """Initialize this endpoint.

        Parameters:
            protocol: string (optional)
                The default value is 'tcp'.
            host: string (optional)
                The default value is '127.0.0.1'
            port: string (optional)
                The default value is '*'.
        """

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

        return
