import json
import subprocess
import sys
import zmq

from circusort.base import utils
from circusort.base.proxy import Proxy
from circusort.base.serializer import Serializer



def create_process(host='localhost', log_address=None):
    '''TODO add docstring'''

    process = Process(host=host, log_address=log_address)
    # TODO correct
    # proxy = process.get_proxy()

    # return proxy
    return process


class Process(object):
    '''Spawn a process (local or remote host) and return a proxy.

    Parameter
    ---------
    host: string
        Host (i.e. machine) {'localhost', '127.0.0.1', 'X.X.X.X'}
    log_address: None or string
        Log address {None, 'tcp://X.X.X.X:X'}
    '''

    def __init__(self, host='localhost', log_address=None):

        object.__init__(self)

        if log_address is None:
            raise NotImplementedError()
            # TODO remove
        self.logger = utils.get_log(log_address, name=__name__)

        self.host = host

        self.last_request_id = -1
        self.serializer = Serializer()

        self.context = zmq.Context()
        if host in ['localhost', '127.0.0.1']:
            self.logger.debug("create remote process on {h}".format(h=host))
            # TODO bind tmp socket
            transport = 'tcp'
            host = '127.0.0.1'
            port = '*'
            endpoint = '{h}:{p}'.format(h=host, p=port)
            address = '{t}://{e}'.format(t=transport, e=endpoint)
            self.logger.debug("bind tmp socket at {a}".format(a=address))
            socket = self.context.socket(zmq.PAIR)
            socket.setsockopt(zmq.RCVTIMEO, 10000)
            socket.bind(address)
            address = socket.getsockopt(zmq.LAST_ENDPOINT)
            self.logger.debug("tmp socket binded at {a}".format(a=address))
            # TODO spawn remote process on local host
            command = [sys.executable]
            command += ['-m', 'circusort.cli.spawn_process']
            command += ['-a', address]
            command += ['-l', log_address]
            self.logger.debug("spawn remote process with: {c}".format(c=' '.join(command)))
            subprocess.Popen(command)
            # TODO ensure connection
            self.logger.debug("ensure connection")
            message = socket.recv_json()
            address = message['address']
            # TODO connect rpc socket
            self.logger.debug("connect rpc socket to {a}".format(a=address))
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.connect(address)
            # TODO close tmp socket
            self.logger.debug("close tmp socket")
            socket.close()
        else:
            # TODO bind tmp socket
            transport = 'tcp'
            port = '*'
            endpoint = '{h}:{p}'.format(h=host, p=port)
            address = '{t}://{e}'.format(t=transport, e=endpoint)
            socket = self.context.socket(zmq.PAIR)
            socket.bind(address)
            address = socket.getsockopt(zmq.LAST_ENDPOINT)
            # TODO spawn remote process on remote host
            # TODO ensure connection
            # TODO close tmp socket
            socket.close()
            # TODO correct socket definition
            self.socket = None

        # TODO create process
        # TODO return proxy

    def __del__(self):

        request = 'finish'
        response = self.send(request)

        self.socket.close()

    # # TODO correct or remove
    # def get_proxy(self):
    #     '''TODO add docstring'''
    #
    #     raise NotImplementedError()

    def get_module(self, name, **kwds):
        '''TODO add docstring'''

        self.logger.debug("get module {n}".format(n=name))

        request = 'get_module'
        options = {
            'name': name,
        }
        response = self.send(request, options=options, **kwds)

        return response

    def call_obj(self, obj, args, kwds):
        '''TODO add docstring'''

        self.logger.debug("call object {o}".format(o=obj))

        request = 'call_obj'
        options = {
            'obj': obj,
            'args': args,
            'kwds': kwds,
        }
        response = self.send(request, options=options)

        return response

    def get_attr(self, obj, name):
        ''' TODO add docstring'''

        self.logger.debug("get attribute {n} of object {o}".format(n=name, o=obj))

        request = 'get_attr'
        options = {
            'obj': obj,
            'name': name,
        }
        response = self.send(request, options=options)

        return response

    def set_attr(self, obj, name, value):
        '''TODO add docstring'''

        self.logger.debug("set attribute {n} of object {o}".format(n=name, o=obj))

        request = 'set_attr'
        options = {
            'obj': obj,
            'name': name,
            'value': value,
        }
        response = self.send(request, options=options)

        return response

    def new_request_id(self):
        '''TODO add docstring...'''

        self.logger.debug("generate new request identifier")

        request_id = self.last_request_id + 1
        self.last_request_id += 1

        return request_id

    def serialize_options(self, options):
        '''TODO add docstring...'''

        self.logger.debug("serialize options")

        if options is None:
            serialized_options = b""
        else:
            raise NotImplementedError()

        return serialized_options

    def send(self, request, options=None, **kwds):
        '''Send a request to the process and return the response result.

        Parameters
        ----------
        request: string
            The request to invoke on the process.
        options: dictionary
            The options to be sent with the request.

        TODO complete...
        '''

        message = {
            'request_id': self.new_request_id(),
            'request': request,
            'options': options,
        }
        message = self.serializer.dumps(message)

        self.logger.debug("send request")
        self.socket.send_multipart(message)

        result = self.process_until_result()

        return result

    def decode(self, dct):
        '''TODO add docstring'''

        self.logger.debug("decode")

        if isinstance(dct, dict):
            # TODO retrieve obj_type
            obj_type = dct.get('__type__', None)
            # TODO process obj according to type
            if obj_type is None:
                return dct
            elif obj_type == 'proxy':
                # TODO check if correct
                self.logger.debug("dct: {d}".format(d=dct))
                dct['attributes'] = tuple(dct['attributes'])
                dct['process'] = self
                proxy = Proxy(**dct)
                return proxy
            else:
                self.logger.debug("unknown object type {t}".format(t=obj_type))
                raise NotImplementedError()
        else:
            self.logger.debug("invalid type {t}".format(t=type(dct)))
            raise NotImplementedError()

    def loads(self, message):
        '''TODO add docstring'''

        self.logger.debug("loads")

        s = message.decode()
        obj = json.loads(s, object_hook=self.decode)

        return obj

    def receive(self):
        '''TODO add docstring'''

        self.logger.debug("receive")

        message = self.socket.recv()
        message = self.loads(message)

        return message

    def process_until_result(self):
        '''TODO add docstring'''

        self.logger.debug("process until result")

        # TODO receive and read message
        message = self.receive()
        self.logger.debug("message received and read: {m}".format(m=message))

        # TODO process message
        if message['response'] == 'return':
            if message['exception'] is None:
                # TODO set and return result
                result = message['result']
                self.logger.debug("result: {r}".format(r=result))
                return result
            else:
                self.logger.debug("message: {m}".format(m=message))
                raise NotImplementedError()
                # TODO set and raise exception
                exception = message['exception']
                raise exception
        elif message['response'] == 'disconnect':
            raise NotImplementedError()
            # TODO remote process asks for disconnection
        else:
            raise NotImplementedError()
            # TODO complete
