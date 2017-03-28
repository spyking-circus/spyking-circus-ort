import array
import fcntl
import json
import logging
import random
import re
import socket
import struct
import subprocess
import time
import zmq



class LogHandler(logging.Handler):
    '''Logging handler sending logging output to a ZMQ socket.'''

    def __init__(self, address):

        logging.Handler.__init__(self)

        self.address = address

        self.form = "%(levelname)s %(process)s %(name)s: %(message)s"
        self.formatter = logging.Formatter(self.form)

        # Set up logging connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(self.address)

        # TODO: make sure we are connected...
        #       (see http://stackoverflow.com/questions/23433037/sleep-after-zmq-connect#23437374)
        duration = 0.1 # s
        time.sleep(duration)

    def __del__(self):

        self.socket.close()
        self.context.term()
        super(LogHandler, self).close()

    def format(self, record):
        '''Format a record.'''
        return self.formatter.format(record)

    def emit(self, record):
        '''Emit a log message on the PUB socket.'''
        topic = b'log'
        message = {
            'kind': 'log',
            'record': record.__dict__,
        }
        data = json.dumps(message)
        # data = data.encode('utf-8') # convert string to bytes
        self.socket.send_multipart([topic, data])
        return

    def handle(self, record):
        super(LogHandler, self).handle(record)
        return


def get_log(address, name=None):
    '''Get a logger instance by name.'''

    # Initialize the handler instance
    handler = LogHandler(address)

    # Get a logger with the specified name (or the root logger)
    logger = logging.getLogger(name=name)
    # Set the threshold for this logger
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    # Add the specified handler to this logger
    logger.addHandler(handler)

    return logger


def find_interfaces(format=True):
    '''Enumerate all interfaces available on the system.

    Parameter
    ---------
    format: boolean
        True: interfaces are returned as string
        False: interfaces are returned as byte string

    Code from http://code.activestate.com/recipes/439093/#c1
    '''
    max_possible = 128  # arbitrary. raise if needed.
    bytes = max_possible * 32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    names = array.array('B', '\0' * bytes)
    outbytes = struct.unpack('iL', fcntl.ioctl(
        s.fileno(),
        0x8912,  # SIOCGIFCONF
        struct.pack('iL', bytes, names.buffer_info()[0])
    ))[0]
    namestr = names.tostring()
    lst = []
    for i in range(0, outbytes, 40):
        name = namestr[i:i+16].split('\0', 1)[0]
        ip   = namestr[i+20:i+24]
        lst.append((name, ip))
    if format:
        lst = dict([(i[0], format_ip(i[1])) for i in lst])
    return lst

def format_ip(addr):
    return str(ord(addr[0])) + '.' + \
           str(ord(addr[1])) + '.' + \
           str(ord(addr[2])) + '.' + \
           str(ord(addr[3]))

def find_loopback_interface():
    interfaces = find_interfaces()
    return interfaces['lo']

def find_ethernet_interface():
    interfaces = find_interfaces()
    if 'eth1' in interfaces:
        return interfaces['eth1']
    elif 'enp0s25' in interfaces:
        return interfaces['enp0s25']
    else:
        raise Exception("Can't find ethernet interfaces 'eth0' or 'enp0s25'.")


def extract_port(address):
    port = re.split(":", address)[-1]
    return port

def find_interface_address_towards(host):
    '''TODO add docstring'''

    p = subprocess.Popen(["ip", "route", "get", host],
                         stdout=subprocess.PIPE)
    l = p.stdout.readlines()
    s = l[0]
    k = 'address'
    p = "src (?P<{}>[\d,.]*)".format(k)
    m = re.compile(p)
    r = m.search(s)
    a = r.group(k)

    return a
