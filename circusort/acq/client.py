import zmq



def spawn_client(interface, port):
    '''TODO add docstring...'''
    protocol = "tcp"
    address = "{}://{}:{}".format(protocol, interface, port)

    context = zmq.Context()

    socket = context.socket(zmq.PAIR)
    socket.connect("{}://{}:{}".format(protocol, interface, port))

    while True:
        message = socket.recv()
        print("message: {}".format(message))
        socket.send("client message to server1")
        socket.send("client message to server2")
        time.sleep(1)

    return
