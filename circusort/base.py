import argparse
import os
import subprocess
import sys
import time
import zmq

from .io import load_configuration


PID_FILE_PATH = "~/.spyking-circus-ort/spyking-circus-ort.pid"

ACKNOWLEDGEMENT = b"acknowledgement"
STOP = b"stop"

class Deamon(object):
    '''Deamon class'''
    def __init__(self):
        # Path to the pid lock file
        self.pid_file_path = os.path.expanduser(PID_FILE_PATH)
        self.directory_path = os.path.dirname(self.pid_file_path)
        config = load_configuration()
        self.protocol = config.deamon.protocol
        self.interface = config.deamon.interface
        self.port = config.deamon.port

    @property
    def address(self):
        address = "{}://{}:{}".format(self.protocol, self.interface, self.port)
        return address

    def start(self):
        '''Start the deamon'''
        if os.path.isfile(self.pid_file_path):
            raise Exception("Failed to start, looks like the deamon is already running.")
        else:
            pid = self.fork()
            if pid == 0: # Child process (i.e. deamon process)
                if not os.path.exists(self.directory_path):
                    os.makedirs(self.directory_path)
                with open(self.pid_file_path, 'w+') as f:
                    buf = "{}".format(os.getpid())
                    f.write(buf)
                self.run()
                os.remove(self.pid_file_path)
                sys.exit(0)
            else: # Parent process
                time.sleep(1.0)
                return

    def fork(self):
        '''Forks the process'''
        pid = os.fork()
        return pid

    def run(self):
        '''Run the deamon'''
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.address)
        while True:
            message = socket.recv()
            if message == STOP:
                socket.send(ACKNOWLEDGEMENT)
                break # infinite loop
            else:
                socket.send(ACKNOWLEDGEMENT)
                pass
        return

    def stop(self):
        '''Stop deamon'''
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.address)
        socket.send(STOP)
        message = socket.recv()
        if not message == ACKNOWLEDGEMENT:
            raise Exception("Unknown response: {}".format(message))
        else:
            return


def main_deamon_start(args):
    print("Start spyking-circus-ort deamon...")
    deamon = Deamon()
    deamon.start()
    print("Done.")
    return

def main_deamon_stop(args):
    print("Stop spyking-circus-ort deamon...")
    deamon = Deamon()
    deamon.stop()
    print("Done.")
    return

def main():

    prog = "spyking-circus-ort"
    help = "help"
    help_deamon = "help deamon"
    help_deamon_start = "help deamon start"
    help_deamon_stop = "help deamon stop"

    parser = argparse.ArgumentParser(prog=prog)

    subparsers = parser.add_subparsers()

    parser_deamon = subparsers.add_parser('deamon', help=help_deamon)
    subparsers_deamon = parser_deamon.add_subparsers()
    parser_deamon_start = subparsers_deamon.add_parser('start', help=help_deamon_start)
    parser_deamon_start.set_defaults(main=main_deamon_start)
    parser_deamon_stop = subparsers_deamon.add_parser('stop', help=help_deamon_stop)
    parser_deamon_stop.set_defaults(main=main_deamon_stop)

    args = parser.parse_args()

    args.main(args)

    return
