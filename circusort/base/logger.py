from subprocess import Popen
from sys import executable
import zmq



class Logger(object):
    '''Logging server.'''

    def __init__(self):

        zmq_context = zmq.Context.instance()

        # 1. create a temporary socket
        tmp_transport = "ipc"
        tmp_endpoint = "circusort_tmp"
        tmp_address = "{t}://{e}".format(t=tmp_transport, e=tmp_endpoint)
        tmp_socket = zmq_context.socket(zmq.PAIR)
        tmp_socket.bind(tmp_address)

        # 2. spawn logger process
        command = [executable]
        command += ['-m', 'circusort.cli.logger']
        command += ['-e', tmp_endpoint]
        process = Popen(command)

        # 3. receive greetings
        message = tmp_socket.recv_json()
        kind = message['kind']
        assert kind == 'greetings', "kind: {k}".format(k=kind)
        rpc_endpoint = message['rpc endpoint']

        # 4. connect to the RPC socket
        rpc_transport = 'ipc'
        rpc_address = '{t}://{e}'.format(t=rpc_transport, e=rpc_endpoint)
        self.rpc_socket = zmq_context.socket(zmq.PAIR)
        self.rpc_socket.connect(rpc_address)

        # 5. send greetings
        message = {
            'kind': 'greetings',
        }
        self.rpc_socket.send_json(message)

        # 6. receive acknowledgement
        message = self.rpc_socket.recv_json()
        kind = message['kind']
        assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        endpoint = message['endpoint']
        self.address = endpoint

        # 7. close temporary socket
        tmp_socket.close()

    def __del__(self):

        try:
            message = {
                'kind': 'order',
                'action': 'stop',
            }
            self.rpc_socket.send_json(message)
            message = self.rpc_socket.recv_json()
            kind = message['kind']
            assert kind == 'acknowledgement', "kind: {k}".format(k=kind)
        except Exception as exception:
            print(exception)
            self.process.terminate()
            self.process.wait()
