import argparse
import json
from logging import DEBUG, INFO, WARN, ERROR, CRITICAL, Formatter, getLogger, Handler
# from logging.handlers import SocketHandler
from subprocess import Popen
import sys
import time
import zmq
from zmq import Context, PUB
from zmq.log.handlers import PUBHandler

from circusort.manager import Manager
from circusort.base import utils



# TODO remove...
# def manager_parser():
#     parser = argparse.ArgumentParser(description='Launch a manager.')
#     parser.add_argument('-a', '--address', help='specify the IP address/hostname')
#     parser.add_argument('-p', '--port', help='specify the port number')
#     return parser

class LogHandler(Handler):
    '''A basic logging handler that emits log messages through a PUB socket.'''

    def __init__(self, address):

        Handler.__init__(self)

        self.address = address

        self.hostname = 'host'
        # self.form = "%(message)s" # default formatter
        # self.form = "%(levelname)s %(filename)s %(lineno)d: %(message)s"
        # self.form = "%(levelname)s {h}:%(process)s %(pathname)s %(lineno)d: %(message)s".format(h=self.hostname)
        self.form = "%(levelname)s {h}:%(process)s %(name)s: %(message)s".format(h=self.hostname)
        self.formatters = {
            DEBUG: Formatter(self.form),
            INFO: Formatter(self.form),
            WARN: Formatter(self.form),
            ERROR: Formatter(self.form),
            CRITICAL: Formatter(self.form),
        }

        # Set up data connection
        self.context = Context.instance()
        self.socket = self.context.socket(PUB)
        self.socket.connect(self.address)

    def __del__(self):
        self.close()

    def format(self, record):
        '''Format a record.'''
        return self.formatters[record.levelno].format(record)

    def emit(self, record):
        '''Emit a log message on the PUB socket.'''
        # message = {
        #     'kind': 'log',
        #     'record': self.format(record),
        # }
        # TODO try to send an unformatted pickle instead...
        message = {
            'kind': 'log',
            'record': record.__dict__,
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

    zmq_context = zmq.Context.instance()

    log_address = arguments['log_address']

    handler = LogHandler(log_address)

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    # logger.setFormatter(formatter)
    logger.addHandler(handler)

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
    logger.debug("port: {p}".format(p=port))
    # Open a socket to parent process to inform it of the address
    tmp_address = "tcp://{i}:{p}".format(i=interface, p=port)
    tmp_socket = zmq_context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    tmp_socket.linger = 1000 # ?
    # Save configuration to file
    logger.debug("...")

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
    logger.debug("rpc address: {a}".format(a=rpc_address))
    message = rpc_socket.recv_json()
    kind = message['kind']
    misc = message['misc']
    assert(kind == 'greetings')
    logger.debug("misc: {m}".format(m=misc))
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
                logger.debug("tmp_endpoint: {e}".format(e=tmp_endpoint))
                try:
                    tmp_socket = zmq_context.socket(zmq.PAIR)
                    tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
                    tmp_socket.bind(tmp_address)
                    tmp_socket.linger = 1000 # ?
                except Exception as exception:
                    logger.debug("exception")
                    logger.debug("{e}".format(e=exception))
                    raise exception
                # TODO spawn the new worker...
                command = ['/usr/bin/python']
                command += ['-m', 'circusort.cli.reader']
                command += ['-e', tmp_endpoint]
                logger.debug("command: {c}".format(c=' '.join(command)))
                process = Popen(command)
                # TODO receive greetings from the new worker...
                message = tmp_socket.recv_json()
                logger.debug("recv_json")
                kind = message['kind']
                assert kind == 'greetings', "kind: {}".format(kind)
                rpc2_endpoint = message['rpc endpoint']
                logger.debug("rpc2_endpoint: {e}".format(e=rpc2_endpoint))
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
                logger.debug("rpc2_address: {a}".format(a=rpc2_address))
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
