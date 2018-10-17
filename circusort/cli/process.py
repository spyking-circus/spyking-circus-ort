import json
import sys
import traceback
import zmq

from circusort.base.utils import get_log
from circusort.base.proxy import Proxy
from circusort.block.block import Block

if sys.version_info.major == 3:
    unicode = str  # Python 3 compatibility.


class Process(object):
    """Process object.

    Attributes:
        logger
        encoder
        context
        socket
        address: string
        last_obj_id: integer
        objs
        poller
        running: boolean
    """

    def __init__(self, host, address, log_address):
        """Initialize process.

        Arguments:
            host: string
            address: string
            log_address: string
        """

        object.__init__(self)

        if log_address is None:
            raise NotImplementedError()
            # TODO remove
        self.logger = get_log(log_address, name=__name__)

        # TODO find proper space to define following class.
        class Encoder(json.JSONEncoder):

            def default(self_, obj):
                if obj is None:
                    obj = json.JSONEncoder.default(obj)
                else:
                    if isinstance(obj, Proxy):
                        obj = obj.encode()
                    else:
                        obj = self.wrap_proxy(obj)
                        obj = obj.encode()
                return obj

        self.encoder = Encoder

        self.context = zmq.Context()
        # Log debug message.
        string = "connect temporary socket at {}"
        message = string.format(address)
        self.logger.debug(message)
        # Connect temporary socket.
        socket = self.context.socket(zmq.PAIR)
        socket.connect(address)
        # Bind RPC socket.
        transport = 'tcp'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        # Log debug message
        string = "bind RPC socket at {}"
        message = string.format(address)
        self.logger.debug(message)
        # Bind RPC socket (bis).
        self.socket = self.context.socket(zmq.PAIR)
        # self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.socket.bind(address)
        # Get RPC address.
        self.address = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        self.address = self.address.decode('utf-8')
        # Log debug message.
        string = "RPC socket binded at {}"
        message = string.format(self.address)
        self.logger.debug(message)
        # Send RPC address.
        message = {
            'address': self.address,
        }
        socket.send_json(message)

        # Define additional attributes.
        self.last_obj_id = -1
        self.objs = {}
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = False

    def get_attr(self, obj, name):

        raise NotImplementedError("name: {n}".format(n=name))

    def run(self):

        # Log debug message.
        message = "run process"
        self.logger.debug(message)

        # Run main loop.
        self.running = True
        while self.running:
            socks = self.poller.poll(timeout=1)
            if len(socks):
                message = self.receive()
                self.process(message)
            else:
                for obj in self.objs.values():
                    if isinstance(obj, Block) and obj.mpl_display is True:
                        obj._plot()
                        import matplotlib.pyplot as plt
                        plt.pause(0.01)

        return

    def unwrap_proxy(self, proxy):

        # Log debug message.
        message = "unwrap proxy"
        self.logger.debug(message)

        obj_id = proxy.obj_id
        obj = self.objs[obj_id]

        for attr in proxy.attributes:
            # Log debug message.
            string = "object's type {}"
            message = string.format(type(obj))
            self.logger.debug(message)
            # Log debug message.
            string = "object's representation {}"
            message = string.format(repr(obj))
            self.logger.debug(message)
            # Log debug message.
            string = "object's attributes {}"
            message = string.format(dir(obj))
            self.logger.debug(message)
            # Log debug message.
            string = "attribute {}"
            message = string.format(attr)
            self.logger.debug(message)
            # Get attribute.
            obj = getattr(obj, attr)

        return obj

    def unwrap_object(self, obj):

        # Log debug message.
        message = "unwrap object"
        self.logger.debug(message)

        # Unwrap object.
        if isinstance(obj, list):
            obj = [self.unwrap_object(v) for v in obj]
        elif isinstance(obj, dict):
            obj = {k: self.unwrap_object(v) for k, v in obj.items()}
        elif isinstance(obj, Proxy):
            obj = self.unwrap_proxy(obj)
        else:
            obj = obj

        return obj

    def decode(self, dct):

        # Log debug message.
        message = "decode"
        self.logger.debug(message)

        if isinstance(dct, dict):
            obj_type = dct.get('__type__', None)
            if obj_type is None:
                return dct
            elif obj_type == 'proxy':
                dct['attributes'] = tuple(dct['attributes'])
                dct['process'] = self  # TODO correct.
                proxy = Proxy(**dct)
                if self.address == proxy.address:
                    return self.unwrap_proxy(proxy)
                else:
                    return proxy
            else:
                # Log debug message.
                string = "unknown object type {}"
                message = string.format(obj_type)
                self.logger.debug(message)
                # Raise error.
                raise NotImplementedError()
        else:
            # Log debug message.
            string = "invalid type {t}"
            message = string.format(t=type(dct))
            self.logger.debug(message)
            # Raise error.
            raise NotImplementedError()

    def loads(self, options):

        # Log debug message.
        message = "loads"
        self.logger.debug(message)

        options = json.loads(options.decode(), object_hook=self.decode)

        return options

    def receive(self):

        # Log debug message.
        message = "receive message"
        self.logger.debug(message)

        request_id, request, serialization_type, options = self.socket.recv_multipart()

        request_id = int(request_id.decode())
        request = request.decode()
        serialization_type = serialization_type.decode()
        if options == b'':
            options = None
        else:
            options = self.loads(options)

        data = {
            'request_id': request_id,
            'request': request,
            'serialization_type': serialization_type,
            'options': options,
        }

        return data

    def new_object_identifier(self):

        obj_id = self.last_obj_id + 1
        self.last_obj_id += 1

        return obj_id

    def wrap_proxy(self, obj):

        # Log debug message.
        message = "wrap proxy"
        self.logger.debug(message)

        for t in [type(None), str, unicode, int, float, tuple, list, dict]:
            if isinstance(obj, t):
                proxy = obj
                return proxy

        obj_id = self.new_object_identifier()
        obj_type = str(type(obj))
        proxy = Proxy(self.address, obj_id, obj_type)

        self.objs[obj_id] = obj

        return proxy

    def process(self, data):

        # Log debug message.
        message = "process message"
        self.logger.debug(message)

        request_id = data['request_id']
        request = data['request']
        options = data['options']

        try:

            if request == 'get_proxy':
                result = self
            elif request == 'get_module':
                self.logger.debug("request of module")
                name = options['name']
                parts = name.split('.')
                result = __import__(parts[0])
                for part in parts[1:]:
                    result = getattr(result, part)
            elif request == 'call_obj':
                self.logger.debug("request of object call")
                obj = options['obj']
                args = options['args']
                kwds = options['kwds']
                self.logger.debug("obj: {o}".format(o=obj))
                self.logger.debug("args: {a}".format(a=args))
                self.logger.debug("kwds: {k}".format(k=kwds))
                result = obj(*args, **kwds)
            elif request == 'get_attr':
                self.logger.debug("request to get object attribute")
                obj = options['obj']
                name = options['name']
                result = getattr(obj, name)
            elif request == 'set_attr':
                self.logger.debug("request to set object attribute")
                obj = options['obj']
                name = options['name']
                value = options['value']
                setattr(obj, name, value)
                result = None
            elif request == 'finish':
                self.running = False
                result = None
            else:
                self.logger.debug("unknown request {r}".format(r=request))
                raise NotImplementedError()

            exception = None

        except Exception as e:

            result = traceback.format_exc()  # Exception trace as a string.

            exception = e.__class__.__name__  # Exception name as a string.

        result = self.wrap_proxy(result)

        # Send result or exception back to proxy.
        data = {
            'response': 'return',
            'request_id': request_id,
            'serialization_type': 'json',
            'result': result,
            'exception': exception,
        }
        data = self.dumps(data)
        self.socket.send_multipart([data])

        return

    def get_module(self, name, **kwargs):

        # Log debug message.
        string = "get module {}"
        message = string.format(name)
        self.logger.debug(message)

        _ = kwargs  # i.e. discard keyword arguments

        parts = name.split('.')
        result = __import__(parts[0])
        for part in parts[1:]:
            result = getattr(result, part)

        return result

    def dumps(self, obj):

        # Log debug message.
        message = "dumps"
        self.logger.debug(message)

        dumped_obj = json.dumps(obj, cls=self.encoder)
        data = dumped_obj.encode()

        return data


def main(args):

    host = args['host']
    address = args['address']
    log_address = args['log_address']

    process = Process(host, address, log_address=log_address)
    process.run()

    return
