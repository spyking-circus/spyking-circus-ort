import numpy as np
import os.path



def load(path):
    data = raw_binary(path)
    return data


class RawBinary(object):
    '''TODO add doc...'''

    def __init__(self, path, nb_channels, length, sampling_rate):
        self.path = path
        self.nb_channels = nb_channels
        self.length = length
        self.sampling_rate = sampling_rate

    def load(self):
        shape = (self.length, self.nb_channels)
        f = np.memmap(self.path, dtype='float32', mode='r', shape=shape)
        data = f[:, :]
        return data



def raw_binary(path, nb_channels, length, sampling_rate):
    path = os.path.expanduser(path)
    raw = RawBinary(path, nb_channels, length, sampling_rate)
    data = raw.load()
    return data
