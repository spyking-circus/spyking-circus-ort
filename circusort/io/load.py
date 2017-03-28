import numpy as np
import os.path
import ConfigParser as cp

from .utils import *



def load(path):
    data = raw_binary(path)
    return data


class RawBinary(object):
    '''TODO add doc...'''

    def __init__(self, path, dtype, length, nb_channels, sampling_rate):
        self.path = path
        self.dtype = dtype
        self.length = length
        self.nb_channels = nb_channels
        self.sampling_rate = sampling_rate

    def load(self):
        shape = (self.length, self.nb_channels)
        f = np.memmap(self.path, dtype=self.dtype, mode='r', shape=shape)
        data = f[:, :]
        return data



def raw_binary(path):
    path = os.path.expanduser(path)
    header_path = get_header_path(path)
    header = cp.ConfigParser()
    header.read(header_path)
    dtype = header.get('header', 'dtype')
    length = header.getint('header', 'length')
    nb_channels = header.getint('header', 'nb_channels')
    sampling_rate = header.getfloat('header', 'sampling_rate')
    raw = RawBinary(path, dtype, length, nb_channels, sampling_rate)
    data = raw.load()
    return data
