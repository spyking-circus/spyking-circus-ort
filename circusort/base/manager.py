import json
import paramiko
import re
from subprocess import Popen, PIPE
import sys
from time import sleep, time
import zmq

from .reader import Reader
from .computer import Computer
from .writer import Writer
from . import utils
from circusort.io import load_configuration



class Manager(object):

    def __init__(self, interface=None):
        self.interface = interface
        self.zmq_context = zmq.Context.instance()
        if self.interface is None: # create new process locally
            # 1. create temporary socket
            tmp_interface = utils.find_loopback_interface()
            tmp_address = 'tcp://{}:*'.format(tmp_interface)
            tmp_socket = self.zmq_context.socket(zmq.PAIR)
            tmp_socket.setsockopt(zmq.RCVTIMEO, 10000) # ?
            tmp_socket.bind(tmp_address)
            tmp_socket.linger = 1000 # ?
            tmp_address = tmp_socket.getsockopt(zmq.LAST_ENDPOINT)
            tmp_port = utils.extract_port(tmp_address)
            tmp_port = utils.extract_port(tmp_address)
            # 2. spawn manager locally
            command = [sys.executable]
            command += ['-m', 'circusort.cli.manager']
            command += ['-i', tmp_interface]
            command += ['-p', tmp_port]
            self.process = Popen(command, stdin=PIPE)
            # 3. send configuration to the manager process
            conf = {'address': tmp_address.decode()}
            message = json.dumps(conf)
            message = message.encode()
            self.process.stdin.write(message)
            self.process.stdin.close()
            # 4. receive status from the manager process
            message = tmp_socket.recv_json()
            address = message['address'].encode()
            print("status: {}".format(address))
            # TODO create manager client...
            self.client = None
        else: # create new process remotely
            # 1. create temporary socket
            print("Create temporary socket...")
            tmp_interface = utils.find_ethernet_interface()
            tmp_address = 'tcp://{}:*'.format(tmp_interface)
            tmp_socket = self.zmq_context.socket(zmq.PAIR)
            tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
            tmp_socket.bind(tmp_address)
            tmp_socket.linger = 1000
            tmp_address = tmp_socket.getsockopt(zmq.LAST_ENDPOINT)
            tmp_port = utils.extract_port(tmp_address)
            print("  tmp_address: {}".format(tmp_address))
            # 2. spawn manager remotely
            print("Spawn manager remotely...")
            command = ['/usr/bin/python']
            command += ['-m', 'circusort.cli.manager']
            command += ['-i', tmp_interface]
            command += ['-p', tmp_port]
            command = ' '.join(command)
            configuration = load_configuration()
            print("Create SSH connection...")
            ssh_client = paramiko.SSHClient() # basic interface to instantiate server connections and file transfers
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # auto-accept inbound host keys
            ssh_client.connect(self.interface, username=configuration.ssh.username) # connect to the local SSH server
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=10.0) # run command synchronously
            # ssh_transport = client.get_transport()
            # ssh_channel = ssh_transport.open_session()
            # ssh_channel.exec_command(commande) # run command asynchronously
            # 3. receive greetings from the manager process
            print("Receive greetings...")
            message = tmp_socket.recv_json()
            kind = message['kind']
            assert kind == 'greetings', "kind: {k}".format(k=kind)
            self.rpc_interface = message['rpc interface']
            self.rpc_port = message['rpc port']
            print("  rpc interface: {}".format(self.rpc_interface))
            print("  rpc port: {}".format(self.rpc_port))
            # 4. send greetings to the manager process
            print("Send greetings...")
            self.rpc_socket = self.zmq_context.socket(zmq.PAIR)
            self.rpc_socket.connect(self.rpc_address)
            self.rpc_socket.linger = 1000 # ?
            message = {
                'kind': 'greetings',
                'misc': '...',
            }
            self.rpc_socket.send_json(message)
            # TODO close temporary socket
            tmp_socket.close()
            # TODO save RPC socket somewhere...
            # TODO create manager client...
            self.client = None
        self.workers = {}

    def __del__(self):
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

    @property
    def rpc_address(self):
        return "tcp://{i}:{p}".format(i=self.rpc_interface, p=self.rpc_port)

    @property
    def nb_workers(self):
        return len(self.workers)

    def create_reader(self):
        print("Create reader...")
        # TODO remove the two following lines...
        # reader = Reader(self)
        # self.register_worker(reader)
        # TODO create a reader and attach it to this manager...
        message = {
            'kind': 'order',
            'action': 'create_reader',
            'path': '/tmp/spyking-circus-ort/synthetic_data.raw'
        }
        self.rpc_socket.send_json(message)
        message = self.rpc_socket.recv_json()
        kind = message['kind']
        assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        reader = None
        return reader

    def create_computer(self):
        computer = Computer()
        self.register_worker(computer)
        return computer

    def create_writer(self):
        writer = Writer()
        self.register_worker(writer)
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
