from argparse import ArgumentParser
from zmq import Context, PAIR, RCVTIMEO



def main(arguments):

    configuration = arguments
    f = open("/tmp/circusort_cli_reader_arguments.txt", mode='w')
    for key in configuration:
        f.write("{}\n".format(key))
    f.close()
    endpoint = configuration['endpoint']

    f = open("/tmp/circusort_cli_reader.txt", mode='w')
    f.write("endpoint: {e}\n".format(e=endpoint))
    f.close()

    zmq_context = Context.instance()
    # TODO connect to the temporary socket of the manager...
    tmp_transport = "ipc"
    tmp_address = '{t}://{e}'.format(t=tmp_transport, e=endpoint)
    tmp_socket = zmq_context.socket(PAIR)
    tmp_socket.connect(tmp_address)
    tmp_socket.linger = 1000 # ?
    # TODO create rpc socket...
    rpc_transport = 'ipc'
    rpc_endpoint = 'circusort_rpc'
    rpc_address = '{t}://{e}'.format(t=rpc_transport, e=rpc_endpoint)
    rpc_socket = zmq_context.socket(PAIR)
    rpc_socket.setsockopt(RCVTIMEO, 10000)
    rpc_socket.bind(rpc_address)
    # TODO send greetings to the manager...
    message = {
        'kind': 'greetings',
        'rpc endpoint': rpc_endpoint,
    }
    f = open("/tmp/circusort_cli_reader_before_send_json.txt", mode='w')
    f.write("before send json\n")
    f.write("tmp address: {a}\n".format(a=tmp_address))
    f.close()
    tmp_socket.send_json(message)
    f = open("/tmp/circusort_cli_reader_after_send_json.txt", mode='w')
    f.write("after send json\n")
    f.write("rpc address: {a}\n".format(a=rpc_address))
    f.close()
    # TODO receive greetings from the manager...
    message = rpc_socket.recv_json()
    f = open("/tmp/circusort_cli_reader_recv_greetings.txt", mode='w')
    f.write("greetings\n")
    f.close()
    kind = message['kind']
    assert kind == 'greetings', "kind: {k}".format(k=kind)
    message = {
        'kind': 'acknowledgement',
    }
    rpc_socket.send_json(message)
    # TODO close the temporary socket...
    tmp_socket.close()
    return


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-e', '--endpoint', required=True)

    args = parser.parse_args()
    args = vars(args)

    main(args)
