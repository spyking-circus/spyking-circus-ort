import argparse
import json
from logging import DEBUG, getLogger, Handler
# from logging.handlers import SocketHandler
from subprocess import Popen
import sys
import time
import zmq

from circusort.manager import Manager
from circusort.base import utils



# TODO remove...
# def manager_parser():
#     parser = argparse.ArgumentParser(description='Launch a manager.')
#     parser.add_argument('-a', '--address', help='specify the IP address/hostname')
#     parser.add_argument('-p', '--port', help='specify the port number')
#     return parser

class LogHandler(Handler)

    def __init__(self):

        self.host = '134.157.180.205'
        self.port = 9020

        super(LogHandler, self).__init__(self)

        # Set up data connection
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(self.address)

    def __del__(self):
        self.close()

    @property
    def address(self):
        transport = 'tcp'
        endpoint = '{h}:{p}'.format(h=self.host, p=self.port)
        return '{t}://{e}'.format(t=transport, e=endpoint)

    def emit(self, record):
        message = {
            'kind': 'log',
            'record': self.format(record),
        }
        self.socket.send_json(message)
        return

    def handle(self, record):
        super(LogHandler, self).handle(record)
        return

    def close(self):
        message = {
            'kind': 'order',
            'action': 'stop',
        }
        self.socket.send_json(message)
        self.socket.close()
        super(LogHandler, self).close()
        return


def main(arguments):

    handler = LogHandler()

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    # logger.setFormatter(formatter)
    logger.addHandler(handler)

    zmq_context = zmq.Context.instance()
    # Load configuration options
    # TODO remove 3 following lines...
    # message = sys.stdin.read()
    # configuration = json.loads(message)
    # configuration.update(arguments)
    configuration = arguments
    interface = configuration['interface']
    port = configuration['port']
    # Save configuration to file
    logger.debug("interface: {i}".format(i=interface))
    f = open("/tmp/circusort_cli_manager.txt", mode="w")
    f.write("interface: {}\n".format(interface))
    f.write("port: {}\n".format(port))
    f.close()
    # Open a socket to parent process to inform it of the address
    tmp_address = "tcp://{i}:{p}".format(i=interface, p=port)
    tmp_socket = zmq_context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    tmp_socket.linger = 1000 # ?
    # Save configuration to file
    f = open("/tmp/circusort_cli_manager_bis.txt", mode="w")
    f.write("...")
    f.close()

    # Create RPC socket
    rpc_interface = utils.find_ethernet_interface()
    rpc_address = "tcp://{}:*".format(rpc_interface)
    rpc_socket = zmq_context.socket(zmq.PAIR)
    rpc_socket.setsockopt(zmq.RCVTIMEO, 10000)
    rpc_socket.bind(rpc_address)
    rpc_socket.linger = 1000 # ?
    rpc_address = rpc_socket.getsockopt(zmq.LAST_ENDPOINT)
    rpc_port = utils.extract_port(rpc_address)
    # Report status
    message = {
        'kind': 'greetings',
        'rpc interface': rpc_interface,
        'rpc port': rpc_port,
    }
    tmp_socket.send_json(message)
    # Receive greetings
    f = open("/tmp/circusort_cli_manager_ter.txt", mode="w")
    f.write("rpc address: {}\n".format(rpc_address))
    f.close()
    message = rpc_socket.recv_json()
    kind = message['kind']
    misc = message['misc']
    assert(kind == 'greetings')
    f = open("/tmp/circusort_cli_manager_qua.txt", mode="w")
    f.write("misc: {}\n".format(misc))
    f.close()
    # TODO close the temporary socket
    tmp_socket.close()

    while True:
        message = rpc_socket.recv_json()
        kind = message['kind']
        if kind == 'order':
            action = message['action']
            if action == 'stop':
                break
            elif action == 'create_reader':
                # TODO create a temporary socket for the new worker...
                tmp_transport = "ipc"
                tmp_endpoint = "circusort_tmp"
                tmp_address = "{t}://{e}".format(t=tmp_transport, e=tmp_endpoint)
                f = open("/tmp/circusort_cli_manager_qui.txt", mode='w')
                f.write("tmp_endpoint: {e}\n".format(e=tmp_endpoint))
                f.close()
                try:
                    tmp_socket = zmq_context.socket(zmq.PAIR)
                    tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
                    tmp_socket.bind(tmp_address)
                    tmp_socket.linger = 1000 # ?
                except Exception as exception:
                    f = open("/tmp/circusort_cli_manager_sec.txt", mode='w')
                    f.write("exception\n")
                    f.write("{e}\n".format(e=exception))
                    f.close()
                    raise exception
                # TODO spawn the new worker...
                command = ['/usr/bin/python']
                command += ['-m', 'circusort.cli.reader']
                command += ['-e', tmp_endpoint]
                f = open("/tmp/circusort_cli_manager_sec.txt", mode='w')
                f.write("command: {c}\n".format(c=' '.join(command)))
                f.close()
                process = Popen(command)
                # TODO receive greetings from the new worker...
                message = tmp_socket.recv_json()
                f = open("/tmp/circusort_cli_manager_sep.txt", mode='w')
                f.write("recv_json\n")
                f.close()
                kind = message['kind']
                assert kind == 'greetings', "kind: {}".format(kind)
                rpc2_endpoint = message['rpc endpoint']
                f = open("/tmp/circusort_cli_manager_oct.txt", mode='w')
                f.write("rpc2_endpoint: {e}\n".format(e=rpc2_endpoint))
                f.close()
                # TODO connect to the RPC socket of the new worker...
                rpc2_transport = "ipc"
                rpc2_address = "{t}://{e}".format(t=rpc2_transport, e=rpc2_endpoint)
                rpc2_socket = zmq_context.socket(zmq.PAIR)
                rpc2_socket.connect(rpc2_address)
                rpc2_socket.linger = 1000 # ?
                # TODO send greetings to the new worker...
                message = {
                    'kind': 'greetings',
                }
                rpc2_socket.send_json(message)
                f = open("/tmp/circusort_cli_manager_send_greetings.txt", mode='w')
                f.write("rpc2_address: {a}\n".format(a=rpc2_address))
                f.close()
                message = rpc2_socket.recv_json()
                kind = message['kind']
                assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
                # TODO close the temporary socket...
                tmp_socket.close()
            else:
                pass
            message = {
                'kind': 'acknowledgement',
            }
            rpc_socket.send_json(message)
        else:
            pass

    message = {
        'kind': 'acknowledgement',
    }
    rpc_socket.send_json(message)
    rpc_socket.close()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', required=True)
    parser.add_argument('-p', '--port', required=True)

    args = parser.parse_args()
    args = vars(args)

    main(args)
