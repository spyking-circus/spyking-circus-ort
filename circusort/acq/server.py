import zmq



def spawn_server():
    '''TODO add doctring...'''
    port = "5556"

    context = zmq.Context()

    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:{}".format(port))

    while True:
        message = "Hello world!"
        socket.send(message)
        message = socket.recv()
        print("message: {}".format(message))
        time.sleep(1)

    return
