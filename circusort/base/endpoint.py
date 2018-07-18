import base64
import numpy
import zmq
import tempfile
import os
import json


TERM_MSG = b"END"


class Connection(object):
    """Connection"""

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

        for key, value in kwargs.items():
            self.params[key] = kwargs[key]
            self.__setattr__(key, value)

        return

    def _initialize(self, **kwargs):
        """Abstract method to initialize the connection."""

        raise NotImplementedError()

    def initialize(self, **kwargs):
        """Initialize the connection."""

        if not self.initialized:
            self._initialize(**kwargs)
            self.initialized = True

        return

    def _get_data(self, blocking=True, number=None, discarding_eoc=False):
        """Abstract method to get data from this connection."""

        raise NotImplementedError()

    def receive(self, blocking=True, number=None, discarding_eoc=False):
        """Receive data.

        Parameter:
            blocking: boolean (optional)
                If true then this waits until data arrives.
                The default value is True.
            number: none | integer (optional)
                If specified then this receives the packet with this number.
                The default value is None.
            discarding_eoc: boolean (optional)
                If true then this discards any end of connection (EOC) signal received.
                The default value is False.
        Return:
            data: np.ndarray | dictionary | boolean | string
                The data to receive.
        """

        data = self._get_data(blocking=blocking, number=number, discarding_eoc=discarding_eoc)

        return data

    def _has_received(self):

        raise NotImplementedError()

    def has_received(self):

        return self._has_received()

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
        """Encode numpy arrays.

        See also:
            https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
        """

        if isinstance(obj, numpy.ndarray):
            # Prepare data.
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = numpy.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            # Prepare serialized object.
            ser_obj = {
                '__ndarray__': base64.b64encode(obj_data),
                '__dtype__': str(obj.dtype),
                '__shape__': obj.shape,
            }
        elif isinstance(obj, bytes):
            ser_obj = {
                '__bytes__': obj.decode('utf-8'),
                '__encoding__': 'utf-8',
            }
        else:
            ser_obj = json.JSONEncoder.default(self, obj)

        return ser_obj


def object_hook(ser_obj):
    """Decode numpy arrays.

    See also:
        https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
    """

    if isinstance(ser_obj, dict) and '__ndarray__' in ser_obj and '__dtype__' in ser_obj and '__shape__' in ser_obj:
        data = base64.b64decode(ser_obj['__ndarray__'])
        obj = numpy.frombuffer(data, dtype=ser_obj['__dtype__'])
        obj = obj.reshape(ser_obj['__shape__'])
    elif isinstance(ser_obj, dict) and '__bytes__' in ser_obj and '__encoding__' in ser_obj:
        data = ser_obj['__bytes__']
        obj = data.encode(ser_obj['__encoding__'])
    else:
        obj = ser_obj

    return obj


class LOCError(Exception):
    """Loss of connection error."""

    def __init__(self, msg=None):

        if msg is None:
            msg = "Loss of connection"

        super(LOCError, self).__init__(msg)


class EOCError(Exception):
    """End of connection error."""

    def __init__(self, msg=None):

        if msg is None:
            msg = "End of connection"

        super(EOCError, self).__init__(msg)


class Endpoint(Connection):
    """Endpoint"""

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

        self.socket = None
        self.tmp_name = None

        self._has_received_flag = False
        self._cached_batch = None

    def __del__(self):

        if self.socket is not None:
            self.socket.close()
        if self.tmp_name is not None:
            os.remove(self.tmp_name)

    def _has_cached_batch(self):

        return self._cached_batch is not None

    def _pop_cached_batch(self):

        batch = self._cached_batch
        self._cached_batch = None

        return batch

    def _put_cached_batch(self, batch):

        self._cached_batch = batch

        return

    def _get_data(self, blocking=True, number=None, discarding_eoc=False):
        """Get batch of data from this endpoint.

        Parameter:
            blocking: boolean (optional)
                If true then this waits until a batch of data arrives.
                The default value is True.
            number: none | integer (optional)
                If specified then gets the batch of data with this number.
                The default value is None.
            discarding_eoc: boolean (optional)
                If true then this discards any end of connection (EOC) signal received.
                The default value is False.
        Return:
            batch: numpy.ndarray | dictionary | boolean | string
                The batch of data to get.
        """

        # Find next batch.
        if self._has_cached_batch():
            # Use cached batch.
            batch = self._pop_cached_batch()
        else:
            # Receive batch.
            batch = self._get_data_aux(blocking=blocking, discarding_eoc=discarding_eoc)
        # Seek targeted batch.
        if number is not None:
            while batch is not None and batch['number'] < number:
                batch = self._get_data_aux(blocking=blocking, discarding_eoc=discarding_eoc)
            if batch is not None and batch['number'] > number:
                self._put_cached_batch(batch)
                batch = None

        return batch

    def _get_data_aux(self, blocking=True, discarding_eoc=False):

        # Try to receive batch.
        if blocking:
            try:
                batch = self.socket.recv()
            except zmq.Again:
                # Loss of connection (because of the socket timeout).
                raise LOCError()
        else:
            try:
                batch = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                # Batch not available yet.
                batch = None
        # Check if termination message has been received.
        if batch == TERM_MSG:
            if discarding_eoc:
                batch = None
            else:
                raise EOCError()
        # Load data from batch.
        batch = self._load_data(batch) if batch is not None else None

        return batch

    def _load_data(self, batch):

        if batch is not None:
            if self.structure == 'array':
                batch = numpy.fromstring(batch, dtype=self.dtype)
                batch = numpy.reshape(batch, self.shape)
            elif self.structure == 'dict':
                batch = json.loads(batch, object_hook=object_hook)
            elif self.structure == 'boolean':
                batch = bool(batch)
            else:
                # Raise value error.
                string = "Unexpected structure value: {}"
                message = string.format(self.structure)
                raise ValueError(message)

        return batch

    def _has_received(self):

        if not self._has_received_flag:
            batch = self._get_data_aux(blocking=False, discarding_eoc=False)
            if batch is not None:
                self._has_received_flag = True
                self._put_cached_batch(batch)

        return self._has_received_flag

    def _send_data(self, batch):
        """Send a batch of data from this endpoint.

        Parameter:
            batch: numpy.ndarray | dictionary | boolean | string
                The batch of data to send.
        """

        if self.structure == 'array':
            self.socket.send(batch.tostring())
        elif self.structure == 'dict':
            self.socket.send_string(json.dumps(batch, cls=Encoder))
        elif self.structure == 'boolean':
            self.socket.send(str(batch))
        else:
            # Raise value error.
            string = "Unexpected structure value: {}"
            message = string.format(self.structure)
            raise ValueError(message)

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

    def _initialize(self, protocol='tcp', host='127.0.0.1', port='*', timeout=60):
        """Initialize this endpoint.

        Parameters:
            protocol: string (optional)
                The default value is 'tcp'.
            host: string (optional)
                The default value is '127.0.0.1'.
            port: string (optional)
                The default value is '*'.
            timeout: none | integer( optional)
                Timeout in seconds.
                The default value is 60.
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
        if timeout is not None and timeout > 0:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        self.socket.bind(address)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        self.addr = self.addr.decode('utf-8')

        return
