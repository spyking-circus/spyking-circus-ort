import subprocess
import sys
import zmq


def start_deamon():
    cmd = [sys.executable, __file__]
    print(cmd)
    return

# def start_deamon():
#     protocol = "tcp"
#     interface = "*"
#     port = 4724
#     address = "{}://{}:{}".format(protocol, interface, port)
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.bind(address)
#     while True:
#         # Wait for next request from a client
#         message = socket.recv()
#         print("message: {}".format(message))
