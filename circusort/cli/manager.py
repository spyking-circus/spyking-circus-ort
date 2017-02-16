import argparse
import json
import sys
import time
import zmq

from circusort.manager import Manager



# TODO remove...
# def manager_parser():
#     parser = argparse.ArgumentParser(description='Launch a manager.')
#     parser.add_argument('-a', '--address', help='specify the IP address/hostname')
#     parser.add_argument('-p', '--port', help='specify the port number')
#     return parser

def main(arguments):
    # Load configuration options
    message = sys.stdin.read()
    configuration = json.loads(message)
    configuration.update(arguments)
    interface = configuration['interface']
    port = configuration['port']
    # Save configuration to file
    f = open("/tmp/circusort_cli_manager.txt", mode="w")
    f.write("interface: {}\n".format(interface))
    f.write("port: {}\n".format(port))
    f.close()
    # Open a socket to parent process to inform it of the address
    tmp_address = "tcp://{i}:{p}".format(i=interface, p=port)
    tmp_context = zmq.Context.instance()
    tmp_socket = tmp_context.socket(zmq.PAIR)
    tmp_socket.connect(tmp_address)
    tmp_socket.linger = 1000

    status = {
        'address': '...'.decode(),
    }
    # Report status
    start = time.time()
    while time.time() < start + 10.0:
        # send status repeatedly until we receive a reply
        tmp_socket.send_json(status)
        try:
            tmp_socket.recv(zmq.NOBLOCK)
            break
        except zmq.error.Again:
            time.sleep(0.01)
            continue
    tmp_socket.close()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface')
    parser.add_argument('-p', '--port')

    args = parser.parse_args()
    args = vars(args)

    main(args)
