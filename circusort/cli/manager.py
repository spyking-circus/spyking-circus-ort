import argparse
import subprocess
import zmq

from circusort.manager import Manager
from circusort.base import utils



def main(arguments):

    context = zmq.Context()

    log_address = arguments['log_address']

    # Get logger instance
    logger = utils.get_log(log_address, name=__name__)

    # Load configuration options
    configuration = arguments
    interface = configuration['interface']
    port = configuration['port']
    # Open a socket to parent process to inform it of the address
    tmp_address = "tcp://{i}:{p}".format(i=interface, p=port)
    logger.debug("connect tmp socket to {a}".format(a=tmp_address))
    tmp_socket = context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    # # TODO remove or adapt following line...
    # tmp_socket.linger = 1000 # ?

    # Create RPC socket
    rpc_interface = utils.find_ethernet_interface()
    rpc_address = "tcp://{i}:*".format(i=rpc_interface)
    logger.debug("bind rpc socket at {a}".format(a=rpc_address))
    rpc_socket = context.socket(zmq.PAIR)
    rpc_socket.setsockopt(zmq.RCVTIMEO, 10000)
    rpc_socket.bind(rpc_address)
    # # TODO remove or adapt following line...
    # rpc_socket.linger = 1000 # ?
    rpc_address = rpc_socket.getsockopt(zmq.LAST_ENDPOINT)
    logger.debug("rpc socket binded at {a}".format(a=rpc_address))
    rpc_port = utils.extract_port(rpc_address)
    # Report status
    message = {
        'kind': 'greetings',
        'rpc interface': rpc_interface,
        'rpc port': rpc_port,
    }
    tmp_socket.send_json(message)
    # Receive greetings
    message = rpc_socket.recv_json()
    kind = message['kind']
    assert(kind == 'greetings')
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
                logger.debug("bind tmp socket at {a}".format(a=tmp_address))
                try:
                    tmp_socket = context.socket(zmq.PAIR)
                    tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
                    tmp_socket.bind(tmp_address)
                    # # TODO remove or adapt the following line...
                    # tmp_socket.linger = 1000 # ?
                except Exception as exception:
                    logger.debug("exception")
                    logger.debug("{e}".format(e=exception))
                    raise exception
                # TODO spawn the new worker...
                command = ['/usr/bin/python']
                command += ['-m', 'circusort.cli.reader']
                command += ['-e', tmp_endpoint]
                logger.debug("spawn reader locally with:\n{c}".format(c=' '.join(command)))
                process = subprocess.Popen(command)
                # TODO receive greetings from the new worker...
                message = tmp_socket.recv_json()
                kind = message['kind']
                assert kind == 'greetings', "kind: {}".format(kind)
                rpc2_endpoint = message['rpc endpoint']
                # TODO connect to the RPC socket of the new worker...
                rpc2_transport = "ipc"
                rpc2_address = "{t}://{e}".format(t=rpc2_transport, e=rpc2_endpoint)
                logger.debug("connect rpc socket to {a}".format(a=rpc2_address))
                rpc2_socket = context.socket(zmq.PAIR)
                rpc2_socket.connect(rpc2_address)
                # # TODO remove or adapt the following line...
                # rpc2_socket.linger = 1000 # ?
                # TODO send greetings to the new worker...
                message = {
                    'kind': 'greetings',
                }
                rpc2_socket.send_json(message)
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
