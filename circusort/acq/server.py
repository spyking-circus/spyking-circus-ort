import time
import zmq



def spawn_server(interface, port):
    '''TODO add doctring...'''
    protocol = "tcp"
    address = "{}://{}:{}".format(protocol, interface, port)

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(address)

    print("Server's socket listens on network port:\n  {}".format(address))

    while True:
        message = "Hello world!"
        socket.send(message)
        message = socket.recv()
        print("message: {}".format(message))
        time.sleep(1)

    return
