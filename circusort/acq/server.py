import zmq



def spawn_server(interface, port=5556):
    '''TODO add doctring...'''
    protocol = "tcp"

    context = zmq.Context()

    socket = context.socket(zmq.PAIR)
    socket.bind("{}://{}:{}".format(protocol, interface, port))

    while True:
        message = "Hello world!"
        socket.send(message)
        message = socket.recv()
        print("message: {}".format(message))
        time.sleep(1)

    return
