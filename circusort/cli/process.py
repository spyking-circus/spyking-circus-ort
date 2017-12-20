import json
import zmq

from circusort.utils.base import get_log
from circusort.base.proxy import Proxy
from circusort.block.block import Block



class Process(object):
    '''TODO add docstring'''

    def __init__(self, host, address, log_address):

        object.__init__(self)

        if log_address is None:
            raise NotImplementedError()
            # TODO remove
        self.logger = get_log(log_address, name=__name__)

        # TODO find proper space to define following class
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
        # TODO connect tmp socket
        self.logger.debug("connect tmp socket at {a}".format(a=address))
        socket = self.context.socket(zmq.PAIR)
        socket.connect(address)
        # TODO bind rpc socket
        transport = 'tcp'
        port = '*'
        endpoint = '{h}:{p}'.format(h=host, p=port)
        address = '{t}://{e}'.format(t=transport, e=endpoint)
        self.logger.debug("bind rpc socket at {a}".format(a=address))
        self.socket = self.context.socket(zmq.PAIR)
        # self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.socket.bind(address)
        self.address = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        self.logger.debug("rpc socket binded at {a}".format(a=self.address))
        # TODO send rpc address
        self.logger.debug("send back rpc address")
        message = {
            'address': self.address,
        }
        socket.send_json(message)

        self.last_obj_id = -1
        self.objs = {}
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def get_attr(self, obj, name):
        '''TODO add docstring'''

        raise NotImplementedError("name: {n}".format(n=name))

    def run(self):
        '''TODO add docstring'''

        self.logger.debug("run process")

        self.running = True
        while self.running:
            socks = self.poller.poll(timeout=1)
            if len(socks):
                message = self.receive()
                self.process(message)
            else:
                # print self.objs
                for obj in self.objs.itervalues():
                    if isinstance(obj, Block) and obj.mpl_display == True:
                        obj._plot()
                        import matplotlib.pyplot as plt
                        plt.pause(0.01)
        return

    def unwrap_proxy(self, proxy):
        '''TODO add docstring'''

        self.logger.debug("unwrap proxy")

        obj_id = proxy.obj_id
        obj = self.objs[obj_id]

        for attr in proxy.attributes:
            self.logger.debug("object's type {t}".format(t=type(obj)))
            self.logger.debug("object's representation {r}".format(r=repr(obj)))
            self.logger.debug("object's attributes {d}".format(d=dir(obj)))
            self.logger.debug("attribute {a}".format(a=attr))
            obj = getattr(obj, attr)

        # # TODO remove following line
        # self.logger.debug(dir(obj))

        return obj

    def unwrap_object(self, obj):
        '''TODO add docstring'''

        self.logger.debug("unwrap object")

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
        '''TODO add docstring'''

        self.logger.debug("decode")

        if isinstance(dct, dict):
            # TODO retrieve obj_type
            obj_type = dct.get('__type__', None)
            if obj_type is None:
                return dct
            elif obj_type == 'proxy':
                dct['attributes'] = tuple(dct['attributes'])
                dct['process'] = self # TODO correct
                proxy = Proxy(**dct)
                if self.address == proxy.address:
                    return self.unwrap_proxy(proxy)
                else:
                    return proxy
            else:
                self.logger.debug("unknown object type {t}".format(t=obj_type))
                raise NotImplementedError()
        else:
            self.logger.debug("invalid type {t}".format(t=type(dct)))
            raise NotImplementedError()

    def loads(self, options):
        '''TODO add docstring'''

        self.logger.debug("loads")

        options = json.loads(options.decode(), object_hook=self.decode)

        return options

    def receive(self):
        '''TODO add docstring'''

        self.logger.debug("receive message")

        request_id, request, serialization_type, options = self.socket.recv_multipart()

        request_id = int(request_id.decode())
        request = request.decode()
        serialization_type = serialization_type.decode()
        if options == b'':
            options = None
        else:
            options = self.loads(options)

        message = {
            'request_id': request_id,
            'request': request.decode(),
            'serialization_type': serialization_type.decode(),
            'options': options,
        }

        return message

    def new_object_identifier(self):
        '''TODO add docstring'''

        obj_id = self.last_obj_id + 1
        self.last_obj_id +=1

        return obj_id

    def wrap_proxy(self, obj):
        '''TODO add docstring'''

        self.logger.debug("wrap proxy")

        for t in [type(None), str, unicode, int, float, tuple, list, dict]:
            if isinstance(obj, t):
                proxy = obj
                return proxy

        obj_id = self.new_object_identifier()
        obj_type = str(type(obj))
        proxy = Proxy(self.address, obj_id, obj_type)

        self.objs[obj_id] = obj

        return proxy

    def process(self, message):
        '''TODO add docstring'''

        self.logger.debug("process message")

        request_id = message['request_id']
        request = message['request']
        options = message['options']

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
            result = setattr(obj, name, value)
        elif request == 'finish':
            self.running = False
            result = None
        else:
            self.logger.debug("unknown request {r}".format(r=request))
            raise NotImplementedError()
            # TODO correct

        result = self.wrap_proxy(result)

        # TODO send result or exception back to proxy
        message = {
            'response': 'return',
            'request_id': request_id,
            'serialization_type': 'json',
            'result': result,
            'exception': None,
        }
        message = self.dumps(message)
        self.socket.send_multipart([message])

        return

    def get_module(self, name, **kwds):
        '''TODO add docstring'''

        self.logger.debug("get module {n}".format(n=name))

        parts = name.split('.')
        result = __import__(parts[0])
        for part in parts[1:]:
            result = getattr(result, part)

        return result

    # def dumps(self, message):
    #     '''TODO add docstring'''
    #
    #     dumped_response = str(message['response']).encode()
    #     dumped_request_id = str(message['request_id']).encode()
    #     dumped_serialization_type = str('json').encode()
    #     if message['result'] is None:
    #         dumped_result = b""
    #     else:
    #         dumped_result = json.dumps(message['result'], cls=self.encoder).encode()
    #     if message['exception'] is None:
    #         dumped_exception = b""
    #     else:
    #         raise NotImplementedError()
    #
    #     message = [
    #         dumped_response,
    #         dumped_request_id,
    #         dumped_serialization_type,
    #         dumped_result,
    #         dumped_exception,
    #     ]
    #
    #     return message

    def dumps(self, obj):
        '''TODO add docstring'''

        self.logger.debug("dumps")

        dumped_obj = json.dumps(obj, cls=self.encoder)
        message = dumped_obj.encode()

        # # TODO remove following line
        # self.logger.debug("message: {m}".format(m=message))

        return message


def main(args):

    host = args['host']
    address = args['address']
    log_address = args['log_address']

    process = Process(host, address, log_address=log_address)
    process.run()

    return
