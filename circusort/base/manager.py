import json
import paramiko
import re
from subprocess import Popen, PIPE
import sys
from time import sleep
import zmq

from .reader import Reader
from .computer import Computer
from .writer import Writer
from . import utils
from circusort.io import load_configuration



class Manager(object):

    def __init__(self, interface=None):
        self.interface = interface
        if self.interface is None: # create new process locally
            # 1. create temporary socket
            tmp_interface = utils.find_loopback_interface()
            tmp_address = 'tcp://{}:*'.format(tmp_interface)
            tmp_context = zmq.Context.instance()
            tmp_socket = tmp_context.socket(zmq.PAIR)
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
            tmp_context = zmq.Context.instance()
            tmp_socket = tmp_context.socket(zmq.PAIR)
            tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
            tmp_socket.bind(tmp_address)
            tmp_socket.linger = 1000
            tmp_address = tmp_socket.getsockopt(zmq.LAST_ENDPOINT)
            tmp_port = utils.extract_port(tmp_address)
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
            _, stdout, stderr = ssh_client.exec_command(command) # run command
            # print(stdout.readlines())
            # print(stderr.readlines())
            # TODO create manager client
            self.client = None
        self.workers = {}

    @property
    def nb_workers(self):
        return len(self.workers)

    def create_reader(self):
        reader = Reader(self)
        self.register_worker(reader)
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
