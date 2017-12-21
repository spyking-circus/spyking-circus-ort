import array
import fcntl
import json
import logging
# import random
import re
import socket
import struct
import subprocess
import time
import zmq


def append_hdf5(dataset, data):
    """Append 1D-array to a HDF5 dataset.

    Parameters:
        dataset: ?
            HDF5 dataset.
        data: numpy.ndarray
            1D-array.
    """

    old_size = len(dataset)
    new_size = old_size + len(data)
    new_shape = (new_size,)
    dataset.resize(new_shape)
    dataset[old_size:new_size] = data

    return


def find_interfaces(format_=True):
    """Enumerate all interfaces available on the system.

    Parameter
        format_: boolean
            True: interfaces are returned as string
            False: interfaces are returned as byte string

    Code from http://code.activestate.com/recipes/439093/#c1
    """

    max_possible = 128  # arbitrary. raise if needed.
    bytes_ = max_possible * 32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    names = array.array('B', '\0' * bytes_)
    outbytes = struct.unpack('iL', fcntl.ioctl(
        s.fileno(),
        0x8912,  # SIOCGIFCONF
        struct.pack('iL', bytes_, names.buffer_info()[0])
    ))[0]
    namestr = names.tostring()
    lst = []
    for i in range(0, outbytes, 40):
        name = namestr[i:i+16].split('\0', 1)[0]
        ip = namestr[i+20:i+24]
        lst.append((name, ip))
    if format_:
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
