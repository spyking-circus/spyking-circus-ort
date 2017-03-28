import zmq

from circusort.base import utils



def main(arguments):

    context = zmq.Context()

    log_address = arguments['log_address']

    # Get logger instance
    logger = utils.get_log(log_address, name=__name__)

    # Load configuration options
    configuration = arguments
    endpoint = configuration['endpoint']

    # 1. connect temporary socket to manager
    tmp_transport = "ipc"
    tmp_address = '{t}://{e}'.format(t=tmp_transport, e=endpoint)
    logger.debug("connect tmp socket to {a}".format(a=tmp_address))
    tmp_socket = context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    # # TODO remove or adapt following line...
    # tmp_socket.linger = 1000 # ?
    # 2. bind rpc socket
    rpc_transport = 'ipc'
    rpc_endpoint = 'circusort_rpc'
    rpc_address = '{t}://{e}'.format(t=rpc_transport, e=rpc_endpoint)
    logger.debug("bind rpc socket at {a}".format(a=rpc_address))
    rpc_socket = context.socket(zmq.PAIR)
    rpc_socket.setsockopt(zmq.RCVTIMEO, 10000)
    rpc_socket.bind(rpc_address)
    # 3. send greetings to manager
    logger.debug("send greetings to manager")
    message = {
        'kind': 'greetings',
        'rpc endpoint': rpc_endpoint,
    }
    tmp_socket.send_json(message)
    # 4. receive greetings from manager
    logger.debug("receive greetings from manager")
    message = rpc_socket.recv_json()
    kind = message['kind']
    assert kind == 'greetings', "kind: {k}".format(k=kind)
    # 5. send acknowledgement to manager
    logger.debug("send acknowledgement to manager")
    message = {
        'kind': 'acknowledgement',
    }
    rpc_socket.send_json(message)
    # 6. close temporary socket
    logger.debug("close tmp socket")
    tmp_socket.close()

    # TODO complete...

    return
