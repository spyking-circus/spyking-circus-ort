from __future__ import print_function
from argparse import ArgumentParser
import json
from logging import basicConfig, getLogger, makeLogRecord
from logging.handlers import DEFAULT_TCP_LOGGING_PORT
from threading import Event, Thread
import zmq

from circusort.base import utils



def receive_log(context, interface):

    basicConfig(format='%(relativeCreated)5d %(name)-15s %(levelname)-8s %(message)s')

    # TODO: connect to the temporary socket...
    log_address = 'inproc://circusort_cli_logger'
    log_socket = context.socket(zmq.PAIR)
    log_socket.connect(log_address)

    # TODO initialize server...
    topic = b'log'
    transport = 'tcp'
    host = interface
    port = '*'
    endpoint = '{h}:{p}'.format(h=host, p=port)
    address = '{t}://{e}'.format(t=transport, e=endpoint)
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic)
    socket.bind(address)
    endpoint = socket.getsockopt(zmq.LAST_ENDPOINT)

    # TODO: send greetings via the temporary socket...
    message = {
        'kind': 'greetings',
        'endpoint': endpoint,
    }
    log_socket.send_json(message)
    # TODO: close the temporary socket...
    log_socket.close()

    # TODO serve until stopped...
    while True:

        topic, data = socket.recv_multipart()
        # data = str(data, 'utf-8') # convert bytes to string
        message = json.loads(data)
        kind = message['kind']
        if kind == 'order':
            action = message['action']
            if action == 'close':
                break
            else:
                pass
        elif kind == 'log':
            record = message['record']
            record = makeLogRecord(record)
            # TODO handle log record
            logger = getLogger(record.name)
            logger.handle(record)
        else:
            pass

    socket.close()

    return


def main(arguments):

    configuration = arguments
    tmp_endpoint = configuration['endpoint']
    interface = configuration['interface']

    context = zmq.Context()

    # I. Initialize logger process
    # 1. create temporary socket
    tmp_transport = 'ipc'
    tmp_address = '{t}://{e}'.format(t=tmp_transport, e=tmp_endpoint)
    tmp_socket = context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    # 2. create RPC socket
    rpc_transport = 'ipc'
    rpc_endpoint = 'circusort_log'
    rpc_address = '{t}://{e}'.format(t=rpc_transport, e=rpc_endpoint)
    rpc_socket = context.socket(zmq.PAIR)
    rpc_socket.bind(rpc_address)
    # 3. send greetings to the director
    message = {
        'kind': 'greetings',
        'rpc endpoint': rpc_endpoint,
    }
    tmp_socket.send_json(message)
    # 4. receive greetings from the director
    message = rpc_socket.recv_json()
    kind = message['kind']
    assert kind == 'greetings', "kind: {k}".format(k=kind)
    # 5. close temporary socket
    tmp_socket.close()

    # TODO create temporary socket for the thread...
    log_address = 'inproc://circusort_cli_logger'
    log_socket = context.socket(zmq.PAIR)
    log_socket.bind(log_address)
    # TODO start thread...
    t = Thread(target=receive_log, args=(context, interface))
    t.setDaemon(True)
    t.start()
    # TODO get log endpoint from temporary socket...
    message = log_socket.recv_json()
    kind = message['kind']
    assert kind == 'greetings', "kind: {k}".format(k=kind)
    log_endpoint = message['endpoint']
    # TODO send log endpoint to the director...

    # 6. send acknowledgement to the director
    message = {
        'kind': 'acknowledgement',
        'endpoint': log_endpoint,
    }
    rpc_socket.send_json(message)

    while True:
        message = rpc_socket.recv_json()
        kind = message['kind']
        if kind == 'order':
            action = message['action']
            if action == 'stop':
                break
            else:
                pass
        else:
            pass

    message = {
        'kind': 'acknowledgement',
    }
    rpc_socket.send_json(message)
    rpc_socket.close()

    return



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-e', '--endpoint', required=True)
    parser.add_argument('-i', '--interface', required=True)

    args = parser.parse_args()
    args = vars(args)

    main(args)
