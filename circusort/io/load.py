import numpy as np
import os.path
import ConfigParser as cp

from .utils import *


def load(path):
    data = raw_binary(path)
    return data


class RawBinary(object):
    """Raw binary"""
    # TODO complete docstring.

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


def load_peaks(path):
    """Load peaks

    Arguments:
        path: string

    """
    # TODO complete docstring.

    return Peaks(path)


class Peaks(object):
    """Peaks"""
    # TODO complete docstring.

    def __init__(self, path):

        self.path = os.path.expanduser(path)
        self.data = np.fromfile(self.path, dtype=np.int32)

    @property
    def electrodes(self):

        return self.data[0::2]

    @property
    def time_steps(self):

        return self.data[1::2]

    def get_time_steps(self, selection=None):

        if selection is None:
            ans = self.time_steps
        elif isinstance(selection, int):
            mask = np.array([e == selection for e in self.electrodes], dtype=np.bool)
            ans = self.time_steps[mask]
        elif isinstance(selection, list):
            mask = np.array([e in selection for e in self.electrodes], dtype=np.bool)
            ans = self.time_steps[mask]
        else:
            raise NotImplementedError()

        return ans
