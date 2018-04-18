import exceptions
import json
import paramiko
import subprocess
import zmq
import logging

from circusort.base.utils import get_log, find_interface_address_towards
from circusort.base.proxy import Proxy
from circusort.base.serializer import Serializer


def create_process(host=None, log_address=None, name=None, log_level=logging.INFO):
    """Create a process (local or remote host)

    Arguments:
        host: none | string (optional)
            The default value is None.
        log_address: none | string (optional)
            The default value is None.
        name: none | string (optional)
            The default value is None.
        log_level: integer (optional)
            The default value is logging.INFO.
    """
    # TODO complete docstring.

    process = Process(host=host, log_address=log_address, name=name, log_level=log_level)
    proxy = process.get_proxy()

    return proxy


class Process(object):
    """Spawn a process (local or remote host) and return a proxy.

    Attributes:
        log_level: integer
        logger
        host: string
        name: string
        last_request_id: int
        serializer
        context
        socket
    """
    # TODO complete docstring.

    def __init__(self, host=None, log_address=None, name=None, log_level=logging.INFO):
        """Initialize Process.

        Arguments:
            host: string (optional)
                Host (i.e. machine) {None, '127.0.0.1', 'X.X.X.X'}
            log_address: none | string (optional)
                Log address {None, 'tcp://X.X.X.X:X'}
            name: none | string (optional)
                The default value is None.
            log_level: integer
                The default value is logging.INFO.
        """
        # TODO complete docstring.

        object.__init__(self)

        if log_address is None:
            raise NotImplementedError()
            # TODO remove
        self.log_level = log_level
        self.logger = get_log(log_address, name=__name__, log_level=self.log_level)

        if host is None:
            self.host = '127.0.0.1'
        else:
            self.host = host
        self.name = name

        self.last_request_id = -1
        self.serializer = Serializer()

        self.context = zmq.Context()
        if self.host in ['127.0.0.1']:
            self.logger.debug("create local process on {h}".format(h=self.host))
            # 1. Bind temporary socket
            transport = 'tcp'
            localhost = find_interface_address_towards(self.host)
            port = '*'
            endpoint = '{h}:{p}'.format(h=localhost, p=port)
            address = '{t}://{e}'.format(t=transport, e=endpoint)
            self.logger.debug("bind tmp socket at {a}".format(a=address))
            socket = self.context.socket(zmq.PAIR)
            socket.setsockopt(zmq.RCVTIMEO, 60 * 1000)
            socket.bind(address)
            address = socket.getsockopt(zmq.LAST_ENDPOINT)
            self.logger.debug("tmp socket binded at {a}".format(a=address))
            # 2. Spawn remote process on local host
            command = ['/usr/bin/python2']
            command += ['-m', 'circusort.cli.spawn_process']
            command += ['--host', self.host]
            command += ['--address', address]
            command += ['--log-address', log_address]
            self.logger.debug("spawn remote process with: {c}".format(c=' '.join(command)))
            subprocess.Popen(command)
            # 3. Ensure connection
            self.logger.debug("ensure connection")
            message = socket.recv_json()
            address = message['address']
            # 4. Connect RPC socket
            self.logger.debug("connect rpc socket to {a}".format(a=address))
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.connect(address)
            # 5. Close temporary socket
            self.logger.debug("close tmp socket")
            socket.close()
        else:
            self.logger.debug("create remote process on {h}".format(h=self.host))
            # 1. Bind temporary socket
            transport = 'tcp'
            localhost = find_interface_address_towards(self.host)
            port = '*'
            endpoint = '{h}:{p}'.format(h=localhost, p=port)
            address = '{t}://{e}'.format(t=transport, e=endpoint)
            self.logger.debug("bind tmp socket at {a}".format(a=address))
            socket = self.context.socket(zmq.PAIR)
            socket.setsockopt(zmq.RCVTIMEO, 60 * 1000)
            socket.bind(address)
            address = socket.getsockopt(zmq.LAST_ENDPOINT)
            self.logger.debug("tmp socket binded at {a}".format(a=address))
            # 2. Spawn remote process on remote host
            command = ['/usr/bin/python2']
            command += ['-m', 'circusort.cli.spawn_process']
            command += ['--host', self.host]
            command += ['--address', address]
            command += ['--log-address', log_address]
            self.logger.debug("spawn remote process with: {c}".format(c=' '.join(command)))
            command = ' '.join(command)
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(self.host)
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=60)
            # Check if everything went well.
            try:
                stderr = stderr.readlines()
                stderr = ''.join(stderr)
                if stderr != "":
                    # Log the standard input.
                    try:
                        stdin = stdin.readlines()
                        stdin = ''.join(stdin)
                    except IOError:
                        stdin = ""
                    if stdin != "":
                        self.logger.debug("stdin:\n{}".format(stdin))
                    # Log the standard output.
                    try:
                        stdout = stdout.readlines()
                        stdout = ''.join(stdout)
                    except IOError:
                        stdout = ""
                    if stdout != "":
                        self.logger.debug("stdout:\n{}".format(stdout))
                    # Log the standard error.
                    self.logger.debug("stderr:\n{}".format(stderr))
            except IOError:
                pass
            # 3. Ensure connection
            self.logger.debug("ensure connection")
            message = socket.recv_json()
            address = message['address']
            # 4. Connect RPC socket
            self.logger.debug("connect rpc socket to {a}".format(a=address))
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.connect(address)
            # 5. Close temporary socket
            self.logger.debug("close tmp socket")
            socket.close()

    def __del__(self):
        # TODO add docstring.

        # Log debug message.
        if self.name is None:
            string = "delete process on {}"
            message = string.format(self.host)
        else:
            string = "delete process {} on {}"
            message = string.format(self.name, self.host)
        self.logger.debug(message)

        request = 'finish'
        response = self.send(request)

        self.socket.close()

    def get_proxy(self):
        # TODO add docstring.

        # Log debug message.
        message = "get proxy"
        self.logger.debug(message)

        # Send request to get the proxy.
        request = 'get_proxy'
        response = self.send(request)

        return response

    def get_module(self, name, **kwargs):
        # TODO add docstring.

        # Log debug message.
        string = "get module {}"
        message = string.format(name)
        self.logger.debug(message)

        # Sen request to get the module.
        request = 'get_module'
        options = {
            'name': name,
        }
        response = self.send(request, options=options, **kwargs)

        return response

    def call_obj(self, obj, args, kwargs):
        # TODO add docstring.

        # Log debug message.
        string = "call object {}"
        message = string.format(obj)
        self.logger.debug(message)

        # Send request to get the object.
        request = 'call_obj'
        options = {
            'obj': obj,
            'args': args,
            'kwds': kwargs,
        }
        response = self.send(request, options=options)

        return response

    def get_attr(self, obj, name):
        # TODO add docstring.

        # Log debug message.
        string = "get attribute {} of object {}"
        message = string.format(name, obj)
        self.logger.debug(message)

        # Send request to get the attribute.
        request = 'get_attr'
        options = {
            'obj': obj,
            'name': name,
        }
        response = self.send(request, options=options)

        return response

    def set_attr(self, obj, name, value):
        # TODO add docstring.

        # Log debug message.
        string = "set attribute {} of object {}"
        message = string.format(name, obj)
        self.logger.debug(message)

        # Send request to set the attribute.
        request = 'set_attr'
        options = {
            'obj': obj,
            'name': name,
            'value': value,
        }
        response = self.send(request, options=options)

        return response

    def new_request_id(self):
        # TODO add docstring.

        # Log debug message.
        message = "generate new request identifier"
        self.logger.debug(message)

        # Compute the new request identifier.
        request_id = self.last_request_id + 1
        self.last_request_id += 1

        return request_id

    def serialize_options(self, options):
        # TODO add docstring.

        # Log debug message.
        message = "serialize options"
        self.logger.debug(message)

        if options is None:
            serialized_options = b""
        else:
            raise NotImplementedError()

        return serialized_options

    def send(self, request, options=None, **kwargs):
        """Send a request to the process and return the response result.

        Arguments:
            request: string
                The request to invoke on the process.
            options: dictionary
                The options to be sent with the request.
        """
        # TODO complete docstring.

        _ = kwargs  # i.e. discard additional keyword arguments

        # Set request.
        data = {
            'request_id': self.new_request_id(),
            'request': request,
            'options': options,
        }
        data = self.serializer.dumps(data)

        # Lod debug message.
        string = "send request {}"
        message = string.format(data)
        self.logger.debug(message)

        # Send request and receive result.
        self.socket.send_multipart(data)
        result = self.process_until_result()

        return result

    def decode(self, dct):
        # TODO add docstring.

        # Log debug message.
        message = "decode"
        self.logger.debug(message)

        if isinstance(dct, dict):
            # Retrieve object type.
            obj_type = dct.get('__type__', None)
            # Process object according to type.
            if obj_type is None:
                return dct
            elif obj_type == 'proxy':
                dct['attributes'] = tuple(dct['attributes'])
                dct['process'] = self
                proxy = Proxy(**dct)
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
            string = "invalid type {}"
            message = string.format(type(dct))
            self.logger.debug(message)
            # Raise error.
            raise NotImplementedError()

    def loads(self, data):
        # TODO add docstring.

        # Log debug message.
        message = "loads"
        self.logger.debug(message)

        s = data.decode()
        obj = json.loads(s, object_hook=self.decode)

        return obj

    def receive(self):
        # TODO add docstring.

        # Log debug message.
        message = "receive"
        self.logger.debug(message)

        # Receive and load data.
        data = self.socket.recv()
        data = self.loads(data)

        return data

    def process_until_result(self):
        # TODO add docstring.

        # Log debug message.
        message = "process until result"
        self.logger.debug(message)

        # Receive and set data.
        data = self.receive()

        # Log debug message.
        string = "data received and read: {}"
        message = string.format(data)
        self.logger.debug(message)

        # Process data.
        if data['response'] == 'return':
            if data['exception'] is None:
                # Extract result.
                result = data['result']
                # Log debug message.
                string = "result: {}"
                message = string.format(result)
                self.logger.debug(message)
                # Return result.
                return result
            else:
                # Extract exception name and trace.
                exception_name = data['exception']
                exception_trace = data['result']  # i.e. exception trace
                # Log debug message.
                string = "exception: {} {}"
                message = string.format(exception_name, exception_trace)
                self.logger.error(message)
                # Raise exception.
                exception_class = getattr(exceptions, exception_name)
                raise exception_class(exception_trace)
        elif data['response'] == 'disconnect':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
