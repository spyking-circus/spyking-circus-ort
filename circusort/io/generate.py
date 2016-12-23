import os.path
import numpy as np

from . import load



class Synthetic(object):
    '''TODO add doc...'''

    def __init__(self, path, nb_channels, duration, sampling_rate):
        self.path = path
        self.nb_channels = nb_channels
        self.length = int(np.ceil(duration * sampling_rate))
        self.sampling_rate = sampling_rate
        self.duration = float(self.length) / self.sampling_rate
        # ...
        self.chunk_length = 40000

    def __repr__(self):
        repr  = "Synthetic dataset:"
        repr += "\n  path: {}".format(self.path)
        repr += "\n  nb channels: {}".format(self.nb_channels)
        repr += "\n  length: {}".format(self.length)
        repr += "\n  sampling rate: {} Hz".format(self.sampling_rate)
        repr += "\n  duration: {} s".format(self.duration)
        return repr

    def save(self):
        shape = (self.length, self.nb_channels)
        f = np.memmap(self.path, dtype='float32', mode='w+', shape=shape)
        # First: generate the gaussian noise chunk by chunk...
        i_start = 0
        i_end = self.chunk_length
        mu = 0.0 # V
        sigma = 1.0e-3 # V
        while i_end < self.length:
            print("Save chunk form {} to {}".format(i_start, i_end))
            shape = (self.chunk_length, self.nb_channels)
            data = np.random.normal(mu, sigma, shape)
            f[i_start:i_end, :] = data[:, :]
            f.flush()
            i_start += self.chunk_length
            i_end += self.chunk_length
        print("Save chunk form {} to {}".format(i_start, self.length))
        shape = (self.length - i_start, self.nb_channels)
        data = np.random.normal(mu, sigma, shape)
        f[i_start:self.length, :] = data[:, :]
        f.flush()
        # Second: add some spikes chunk by chunk...
        # TODO complete...
        return

    def load(self):
        '''TODO add doc...'''
        data = load.raw_binary(self.path, self.nb_channels, self.length, self.sampling_rate)
        return data



def synthetic(path, nb_channels=3, duration=60.0, sampling_rate=20000.0):
    '''TODO add doc...'''
    path = os.path.expanduser(path)
    syn = Synthetic(path, nb_channels, duration, sampling_rate)
    print(syn)
    syn.save()
    data = syn.load()
    return data
