import argparse
import os
import subprocess
import sys
import time
import zmq

from circusort.io import load_configuration
from circusort.io.configure import CONFIGURATION_PATH
from circusort.io.configure import add, remove_option, remove_section, delete

from .director import Director


PID_FILE_PATH = "~/.spyking-circus-ort/spyking-circus-ort.pid"

ACKNOWLEDGEMENT = b"acknowledgement"
STOP = b"stop"


def create_director():
    '''Create a new director in this process.'''
    # server = RPCServer()
    # server.run() # or server.run_lazy()
    director = Director()
    return director


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
            pid = os.fork()
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
        '''Stop the deamon'''
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

def main_configure_file(args):
    path = os.path.expanduser(CONFIGURATION_PATH)
    print(path)
    return

def main_configure_show(args):
    config = load_configuration()
    if args.section is None:
        print(config)
    else:
        if hasattr(config, args.section):
            section = getattr(config, args.section)
            if args.option is None:
                print(section)
            else:
                if hasattr(section, args.option):
                    option = getattr(section, args.option)
                    print(option)
                else:
                    pass
        else:
            pass
    return

def main_configure_set(args):
    add(args.section, args.option, args.value)
    return

def main_configure_remove(args):
    if args.section is None:
        remove()
    else:
        if args.option is None:
            remove_section(args.section)
        else:
            remove_option(args.section, args.option)
    return


def main():

    prog = "spyking-circus-ort"
    help = "help"
    help_deamon = "help deamon"
    help_deamon_start = "help deamon start"
    help_deamon_stop = "help deamon stop"
    help_configure = "help configure"

    parser = argparse.ArgumentParser(prog=prog)

    subparsers = parser.add_subparsers()

    parser_deamon = subparsers.add_parser('deamon', help=help_deamon)
    subparsers_deamon = parser_deamon.add_subparsers()
    parser_deamon_start = subparsers_deamon.add_parser('start', help=help_deamon_start)
    parser_deamon_start.set_defaults(main=main_deamon_start)
    parser_deamon_stop = subparsers_deamon.add_parser('stop', help=help_deamon_stop)
    parser_deamon_stop.set_defaults(main=main_deamon_stop)

    parser_configure = subparsers.add_parser('configure', help=help_configure)
    subparsers_configure = parser_configure.add_subparsers()
    parser_configure_file = subparsers_configure.add_parser('file')
    parser_configure_file.set_defaults(main=main_configure_file)
    parser_configure_show = subparsers_configure.add_parser('show')
    parser_configure_show.add_argument('section', nargs='?')
    parser_configure_show.add_argument('option', nargs='?')
    parser_configure_show.set_defaults(main=main_configure_show)
    parser_configure_set = subparsers_configure.add_parser('set')
    parser_configure_set.add_argument('section')
    parser_configure_set.add_argument('option')
    parser_configure_set.add_argument('value')
    parser_configure_set.set_defaults(main=main_configure_set)
    parser_configure_remove = subparsers_configure.add_parser('remove')
    parser_configure_remove.add_argument('section', nargs='?')
    parser_configure_remove.add_argument('option', nargs='?')
    parser_configure_remove.set_defaults(main=main_configure_remove)
    # TODO: complete...
    # show [section [option]]
    # set section option value
    # remove [section [option]]

    args = parser.parse_args()

    args.main(args)

    return


if __name__ == '__main__':
    main()
