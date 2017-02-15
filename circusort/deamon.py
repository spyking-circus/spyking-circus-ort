import os
import paramiko
import subprocess
import sys
import zmq



# hostname = "134.157.180.212"
# username = "baptiste"
# password = ""
# key_filename = os.path.expanduser("~/.ssh/id_rsa.pub")
#
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(hostname, username=username, password=password, key_filename=key_filename)
#
# shell = ssh.invoke_shell()


_password = ""
_key_filename = os.path.expanduser("~/.ssh/id_rsa.pub")

class Deamon(object):
    '''TODO add docstring...'''
    def __init__(self, hostname, username=None, password=_password, key_filename=_key_filename):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.ssh_client = paramiko.SSHClient()
        self.policy = paramiko.AutoAddPolicy()
        self.ssh_client.set_missing_host_key_policy(self.policy)
        self.ssh_client.connect(self.hostname, username=self.username, password=self.password, key_filename=self.key_filename)
        try:
            deamon_command = "python -m circusort.base deamon start --port 4242"
            print(deamon_command)
            stdin, stdout, stderr = self.ssh_client.exec_command(deamon_command)
            print("stdout:\n{}".format(stdout.read()))
            print("stderr:\n{}".format(stderr.read()))
        except Exception as exception:
            raise NotImplementedError(exception)

    def __repr__(self):
        formatter = "Deamon (hostname: {}, username:{})"
        return formatter.format(self.hostname, self.username)


def start_deamon():
    cmd = [sys.executable, __file__]
    print(cmd)
    return

# def start_deamon():
#     protocol = "tcp"
#     interface = "*"
#     port = 4724
#     address = "{}://{}:{}".format(protocol, interface, port)
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.bind(address)
#     while True:
#         # Wait for next request from a client
#         message = socket.recv()
#         print("message: {}".format(message))
