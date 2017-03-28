import paramiko
import re
import subprocess
import sys
import zmq

from circusort.base import utils
from circusort.base.process import Process
from circusort.io import load_configuration



class RPCClient(object):
    '''Connect to a RPC server.

    TODO complete...

    '''

    @staticmethod
    def get_client(address):
        '''Return the RPC client for this thread and a given RPC server.

        TODO complete...
        '''

        # # TODO correct...
        # key = (thread_id, address)
        # if key in RPCClient.clients:
        #     client = RPCClient.clients[key] # return existing client
        # else:
        #     client = RPCClient(address) # create new client
        client = RPCClient(address)
        return client

    def __init__(self, address):
        object.__init__(self)
        self.address = address

    def get_object(self, name, **kwds):
        return self.send('get_object', options={'name': name}, **kwds)

    def send(self, request, options=None):
        '''Send a request to the remote process.

        Parameters
        ----------
        request: string
            The request to make to the remote process.
        options: dictionary
            The options to be sent with the request.

        TODO complete...
        '''

        # TODO prepare request...
        req_id = 0
        req_type = request
        res_type = 'auto'
        ser_type = 'json'
        # # TODO serialize options...
        if options is None:
            options = b""
        else:
            options = self.serializer.dumps(options)
        # TODO create message...
        message = [req_id, req_type, res_type, ser_type, options]
        # TODO send message...
        # TODO receive response...
        # TODO return result...
        raise NotImplementedError("\nrequest: {r}\noptions: {o}".format(r=request, o=options))


class ObjectProxy(object):
    '''Proxy to a remote object'''

    def __init__(self, rpc_addr, obj_id, ref_id, type_str='', names=()):
        object.__init__(self)
        namespace = {
            '_rpc_addr': rpc_addr,
            '_obj_id': obj_id,
            '_ref_id': ref_id,
            '_type_str': type_str,
            '_names': names,
            '_parent_proxy': None,
        }
        self.__dict__.update(namespace)
        namespace = {
            '_client_': None,
            '_server_': None,
        }
        self.__dict__.update(namespace)

    def _client(self):
        if self._client_ is None:
            self.__dict__['_client_'] = RPCClient.get_client(self._rpc_addr)
        return self._client_

    def _server(self):
        raise NotImplementedError()

    def _undefer(self):
        '''Undefer attribute lookups and return attributes.'''
        if len(self._names) == 0:
            return self
        else:
            client = self._client()
            return client.get_object(self)
        return



class Manager(object):
    '''Proxy connected to a manager process.

    Note: this is a RPC client connected to a RPC server.

    TODO complete...
    '''

    def __init__(self, interface=None, log_addr=None):

        object.__init__(self)

        if log_addr is None:
            raise NotImplementedError("no logger address")

        # Get logger instance
        self.log = utils.get_log(log_addr, name=__name__)

        self.log.debug("start manager at {i}".format(i=interface))

        # TODO uncomment
        self.interface = interface
        self.context = zmq.Context()
        if self.interface is None:
            self.log.debug("start new manager on local machine")
            # 1. create temporary socket
            tmp_interface = utils.find_loopback_interface()
            tmp_address = 'tcp://{h}:*'.format(h=tmp_interface)
            self.log.debug("bind tmp socket at {a}".format(a=tmp_address))
            tmp_socket = self.context.socket(zmq.PAIR)
            tmp_socket.setsockopt(zmq.RCVTIMEO, 10000) # ?
            tmp_socket.bind(tmp_address)
            # # TODO remove or adapt following line...
            # tmp_socket.linger = 1000 # ?
            tmp_address = tmp_socket.getsockopt(zmq.LAST_ENDPOINT)
            self.log.debug("tmp socket binded at {a}".format(a=tmp_address))
            tmp_port = utils.extract_port(tmp_address)
            # 2. spawn manager locally
            command = [sys.executable]
            command += ['-m', 'circusort.cli.launch_manager']
            command += ['-i', tmp_interface]
            command += ['-p', tmp_port]
            command += ['-l', log_addr]
            self.log.debug("spawn manager locally with: {c}".format(c=' '.join(command)))
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
            # 3. receive greetings from the manager process
            message = tmp_socket.recv_json()
            kind = message['kind']
            assert kind == 'greetings', "kind: {k}".format(k=kind)
            self.rpc_interface = message['rpc interface']
            self.rpc_port = message['rpc port']
            self.log.debug("receive greetings from manager via {a}".format(a=self.rpc_address))
            # 4. send greetings to the manager process
            self.log.debug("connect rpc socket to {a}".format(a=self.rpc_address))
            self.rpc_socket = self.context.socket(zmq.PAIR)
            self.rpc_socket.connect(self.rpc_address)
            # # TODO: remove or adapt the following line...
            # self.rpc_socket.linger = 1000 # ?
            self.log.debug("send greetings to manager via {a}".format(a=self.rpc_address))
            message = {
                'kind': 'greetings',
            }
            self.rpc_socket.send_json(message)
            # 5. close temporary socket
            tmp_socket.close()
            # TODO save RPC socket somewhere...
            # TODO create manager client...
            self.client = None
        else:
            self.log.debug("start new manager on remote machine")
            # 1. create temporary socket
            tmp_interface = utils.find_ethernet_interface()
            tmp_address = 'tcp://{}:*'.format(tmp_interface)
            self.log.debug("bind tmp socket at {a}".format(a=tmp_address))
            tmp_socket = self.context.socket(zmq.PAIR)
            tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
            tmp_socket.bind(tmp_address)
            tmp_socket.linger = 1000
            tmp_address = tmp_socket.getsockopt(zmq.LAST_ENDPOINT)
            self.log.debug("tmp socket created at {a}".format(a=tmp_address))
            tmp_port = utils.extract_port(tmp_address)
            # 2. spawn manager remotely
            command = ['/usr/bin/python']
            command += ['-m', 'circusort.cli.launch_manager']
            command += ['-i', tmp_interface]
            command += ['-p', tmp_port]
            command += ['-l', log_addr]
            command = ' '.join(command)
            self.log.debug("spawn manager remotely with: {c}".format(c=command))
            configuration = load_configuration()
            self.log.debug("create SSH connnection via paramiko")
            ssh_client = paramiko.SSHClient() # basic interface to instantiate server connections and file transfers
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # auto-accept inbound host keys
            ssh_client.connect(self.interface, username=configuration.ssh.username) # connect to the local SSH server
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=10.0) # run command synchronously
            # # TODO: remove or adapt the three following lines...
            # ssh_transport = client.get_transport()
            # ssh_channel = ssh_transport.open_session()
            # ssh_channel.exec_command(commande) # run command asynchronously
            # 3. receive greetings from the manager process
            message = tmp_socket.recv_json()
            kind = message['kind']
            assert kind == 'greetings', "kind: {k}".format(k=kind)
            self.rpc_interface = message['rpc interface']
            self.rpc_port = message['rpc port']
            self.log.debug("receive greetings from manager via {a}".format(a=self.rpc_address))
            # 4. send greetings to the manager process
            self.log.debug("connect rpc socket to {a}".format(a=self.rpc_address))
            self.rpc_socket = self.context.socket(zmq.PAIR)
            self.rpc_socket.connect(self.rpc_address)
            # # TODO: remove or adapt the following line...
            # self.rpc_socket.linger = 1000 # ?
            self.log.debug("send greetings to manager via {a}".format(a=self.rpc_address))
            message = {
                'kind': 'greetings',
            }
            self.rpc_socket.send_json(message)
            # 5. close temporary socket
            tmp_socket.close()
            # TODO save RPC socket somewhere...
            # TODO create manager client...
            self.client = None
        self.workers = {}

    def __del__(self):

        if hasattr(self, 'rpc_socket'):
            message = {
                'kind': 'order',
                'action': 'stop',
            }
            self.rpc_socket.send_json(message)
            message = self.rpc_socket.recv_json()
            kind = message['kind']
            assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
            self.rpc_socket.close()

        return

    def __getattr__(self, name):
        '''Lookup attribute on the remote manager and return attribute.

        Parameter
        ---------
        name: string
            Attribute name.
        '''
        proxy = self._deferred_attr(name)
        return proxy._undefer()

    def _deferred_attr(self, name):
        '''Return a proxy to an attribute of this object.

        Parameters
        ----------
        name: string
            Attribute name.
        '''
        _rpc_addr = None
        _obj_id = None
        _ref_id = None
        _type_str = None
        _names = ()
        proxy = ObjectProxy(_rpc_addr, _obj_id, _ref_id, _type_str, _names + (name,))
        proxy.__dict__['_parent_proxy'] = self # reference so that remote object cannot be released
        return proxy

    @property
    def rpc_address(self):
        return "tcp://{i}:{p}".format(i=self.rpc_interface, p=self.rpc_port)

    @property
    def nb_workers(self):
        return len(self.workers)

    def set_name(self, name):
        '''TODO add docstring'''

        self.name = name

        return

    def create_reader(self):
        self.log.info("manager at {i} creates new reader".format(i=self.rpc_interface))
        # # TODO remove the two following lines...
        # reader = Reader(self)
        # self.register_worker(reader)
        # TODO create a reader and attach it to this manager...
        message = {
            'kind': 'order',
            'action': 'create_reader',
            'path': '/tmp/spyking-circus-ort/synthetic_data.raw',
        }
        self.rpc_socket.send_json(message)
        message = self.rpc_socket.recv_json()
        kind = message['kind']
        assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        # TODO complete...
        reader = None
        return reader

    def create_computer(self):
        self.log.info("manager at {i} creates new computer".format(i=self.rpc_interface))
        # # TODO remove the following two lines...
        # computer = Computer()
        # self.register_worker(computer)
        # TODO create a computer and attach it to this manager...
        message = {
            'kind': 'order',
            'action': 'create_computer',
        }
        self.rpc_socket.send_json(message)
        message = self.rpc_socket.recv_json()
        kind = message['kind']
        assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        # TODO complete...
        computer = None
        return computer

    def create_writer(self):
        self.log.info("manager at {i} creates new writer".format(i=self.rpc_interface))
        # # TODO remove the following two lines...
        # writer = Writer()
        # self.register_worker(writer)
        # TODO create a writer and attach it to this manager...
        message = {
            'kind': 'order',
            'action': 'create_writer',
            'path': '/tmp/spyking-circus-ort/synthetic_data_copy.raw',
        }
        self.rpc_socket.send_json(message)
        message = self.rpc_socket.recv_json()
        kind = message['kind']
        assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        # TODO complete...
        writer = None
        return writer

    def register_worker(self, worker, name=None):
        if name is None:
            identifier = 1 + self.nb_workers
            name = "worker_{}".format(identifier)
        self.workers.update({name: worker})
        return

    def initialize_all(self):
        for worker in self.workers.itervalues():
            worker.initialize()
        return

    def start_all(self):
        for worker in self.workers.itervalues():
            worker.start()
        return

    def stop_all(self):
        for worker in self.workers.itervalues():
            worker.stop()
        return
