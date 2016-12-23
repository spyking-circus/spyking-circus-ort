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



class Channel(object):
    '''TODO add doc...'''

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        repr = "Channel {} (".format(self.id)
        repr += " x: {}".format(self.x)
        repr += " y: {}".format(self.y)
        repr += ")"
        return repr

class Cell(object):
    '''TODO add doc...'''

    def __init__(self, id, x, y, lam=1.0):
        self.id = id
        self.x = x
        self.y = y
        self.lam = lam # expectation of Poisson interval
        self.times = self.initialize_times()

    def __repr__(self):
        repr = "Cell {} (".format(self.id)
        repr += " x: {}".format(self.x)
        repr += " y: {}".format(self.y)
        repr += ")"
        return repr

    def initialize_times(self):
        times = np.random.poisson(lam=self.lam, size=10)
        times = np.cumsum(times)
        return times


class SyntheticGrid(object):
    '''TODO add doc...'''

    def __init__(self, path, size, duration, sampling_rate):
        self.path = path
        self.size = size
        self.nb_channels = size * size
        self.length = int(np.ceil(duration * sampling_rate))
        self.sampling_rate = sampling_rate
        self.duration = float(self.length) / self.sampling_rate
        self.nb_cells = 1
        # ...
        self.chunk_length = 40000
        self.channels = self.initialize_channels()
        self.cells = self.initilize_cells()

    def initialize_channels(self):
        channels = dict()
        for k in range(0, self.nb_channels):
            x = float(k % self.nb_channels)
            x -= 0.5 * float(self.size - 1)
            x *= 1.0e-4
            y = float(k / self.nb_channels)
            y -= 0.5 * float(self.size - 1)
            y *= 1.0e-4
            channels[k] = Channel(k, x, y)
        return channels

    def initialize_cells(self):
        cells = dict()
        for k in range(0, self.nb_cells):
            # TODO correct the following lines...
            x = np.random.uniform(0.0, float(self.size - 1))
            y = np.random.uniform(0.0, float(self.size - 1))
            cells[k] = Cell(k, x, y)
        return cells

    def save(self):
        return

    def load(self):
        return



def synthetic_grid(path, size=2, duration=60.0, sampling_rate=20000.0):
    '''TODO add doc...'''
    path = os.path.expanduser(path)
    syn_grid = SyntheticGrid(path, size, duration, sampling_rate)
    syn_grid.save()
    data = syn_grid.load()
    return data
