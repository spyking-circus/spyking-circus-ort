import zmq



def spawn_client():
    '''TODO add docstring...'''
    port = "5556"

    context = zmq.Context()

    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % port)

    while True:
        msg = socket.recv()
        print("message: {}".format(msg))
        socket.send("client message to server1")
        socket.send("client message to server2")
        time.sleep(1)

    return
