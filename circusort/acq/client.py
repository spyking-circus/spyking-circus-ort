import zmq



def spawn_client(port=5556):
    '''TODO add docstring...'''
    context = zmq.Context()

    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:{}".format(port))

    while True:
        messsage = socket.recv()
        print("message: {}".format(message))
        socket.send("client message to server1")
        socket.send("client message to server2")
        time.sleep(1)

    return
