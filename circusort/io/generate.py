import os.path
import matplotlib.pyplot as plt
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

    def __init__(self, id, x, y, t, scale=1.0, refactory_period=0.5):
        self.id = id
        self.x = x # x-position
        self.y = y # y-position
        self.t = t # lifetime
        self.scale = scale # expectation of exponential intervals
        self.refactory_period = refactory_period
        self.times = self.initialize_times()

    def __repr__(self):
        repr = "Cell {} (".format(self.id)
        repr += " x: {}".format(self.x)
        repr += " y: {}".format(self.y)
        repr += ")"
        return repr

    def initialize_times(self):
        ref_time = 0.0 # s
        size = 1 + int(self.t / self.scale)
        times = np.random.exponential(scale=self.scale, size=size)
        times = times[self.refactory_period < times]
        times = ref_time + np.cumsum(times)
        ref_time = times[-1]
        while ref_time < self.t:
            new_times = np.random.exponential(scale=self.scale, size=size)
            new_times = new_times[self.refactory_period < new_times]
            new_times = ref_time + np.cumsum(new_times)
            ref_time = new_times[-1]
            times = np.append(times, new_times)
        times = times[times < self.t]
        return times


class SyntheticGrid(object):
    '''TODO add doc...'''

    def __init__(self, path, size, duration, sampling_rate):
        self.path = path
        self.size = size
        self.nb_channels = size * size
        self.inter_electrode_distance = 5.0e-5
        self.length = int(np.ceil(duration * sampling_rate))
        self.sampling_rate = sampling_rate
        self.duration = float(self.length) / self.sampling_rate
        self.nb_cells = 5
        # ...
        self.chunk_length = 40000
        self.channels = self.initialize_channels()
        self.cells = self.initialize_cells()

    def initialize_channels(self):
        channels = dict()
        for k in range(0, self.nb_channels):
            x = float(k % self.size)
            x -= 0.5 * float(self.size - 1)
            x *= self.inter_electrode_distance
            y = float(k / self.size)
            y -= 0.5 * float(self.size - 1)
            y *= self.inter_electrode_distance
            channels[k] = Channel(k, x, y)
        return channels

    def initialize_cells(self):
        cells = dict()
        for k in range(0, self.nb_cells):
            # TODO correct the following lines...
            x = np.random.uniform(-0.5, float(self.size - 1) + 0.5)
            x -= 0.5 * float(self.size - 1)
            x *= self.inter_electrode_distance
            y = np.random.uniform(-0.5, float(self.size - 1) + 0.5)
            y -= 0.5 * float(self.size - 1)
            y *= self.inter_electrode_distance
            t = self.duration
            cells[k] = Cell(k, x, y, t)
        return cells

    def plot(self):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.plot_spatial_configuration()
        # TODO plot spatial configuration (electrodes + cells)...
        self.plot_temporal_configuration()
        # TODO plot temporal configuration (spike times for each cell)...
        return

    def plot_spatial_configuration(self):
        vmin = - 0.5 * float(self.size) * self.inter_electrode_distance * 1.0e6
        vmax = + 0.5 * float(self.size) * self.inter_electrode_distance * 1.0e6
        plt.figure()
        plt.subplot(1, 1, 1)
        # Plot each channel
        x = [channel.x * 1.0e6 for channel in self.channels.itervalues()]
        y = [channel.y * 1.0e6 for channel in self.channels.itervalues()]
        plt.scatter(x, y, s=10, c='k')
        # Draw horizontal lines
        for k in range(0, self.size):
            k_start = k * self.size
            k_end = (k + 1) * self.size - 1
            plt.plot([x[k_start], x[k_end]], [y[k_start], y[k_end]], color='black', linestyle='dashed')
        # Draw vertical lines
        for k in range(0, self.size):
            k_start = k
            k_end = k + (self.size - 1) * self.size
            plt.plot([x[k_start], x[k_end]], [y[k_start], y[k_end]], color='black', linestyle='dashed')
        # Plot each cell
        x = [cell.x * 1.0e6 for cell in self.cells.itervalues()]
        y = [cell.y * 1.0e6 for cell in self.cells.itervalues()]
        c = [cell.id for cell in self.cells.itervalues()]
        plt.scatter(x, y, s=50, c=c, cmap='viridis')
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        plt.xlabel(r"$x$ $(\mu{}m)$")
        plt.ylabel(r"$y$ $(\mu{}m)$")
        plt.title(r"Spatial configuration")
        plt.axes().set_aspect('equal', adjustable='box')
        return

    def plot_temporal_configuration(self):
        plt.figure()
        plt.subplot(1, 1, 1)
        # Plot spike times of each cell
        for cell in self.cells.itervalues():
            x = cell.times
            y = (float(cell.id) + 0.8) * np.ones_like(x)
            bottom = cell.id
            x = np.insert(x, [0, len(x)], [0.0, self.duration])
            y = np.insert(y, [0, len(y)], [bottom] * 2)
            plt.stem(x, y, linefmt='k-', markerfmt='k ', basefmt='k-', bottom=bottom)
        plt.xlim(0.0, self.duration)
        plt.ylim(-0.1, float(len(self.cells)) - 0.1)
        plt.xlabel(r"time $(s)$")
        plt.ylabel(r"cell")
        plt.title("Temporal configuration")
        return

    def save(self):
        shape = (self.length, self.nb_channels)
        f = np.memmap(self.path, dtype='float32', mode='w+', shape=shape)
        # First: generate the gaussian noise chunk by chunk...
        i_start = 0
        i_end = self.chunk_length
        mu = 0.0 # V
        sigma = 1.0e-3 # V
        ## Save each chunk
        while i_end < self.length:
            if __debug__:
                print("Save chunk form {} to {}".format(i_start, i_end))
            chunk_length = self.chunk_length
            shape = (chunk_length, self.nb_channels)
            data = np.random.normal(mu, sigma, shape)
            f[i_start:i_end, :] = data[:, :]
            f.flush()
            # Update indices
            i_start += self.chunk_length
            i_end += self.chunk_length
        ## Save last chunk (might be shorter than the others)
        if __debug__:
            print("Save final chunk form {} to {}".format(i_start, self.length))
        chunk_length = self.length - i_start
        shape = (chunk_length, self.nb_channels)
        data = np.random.normal(mu, sigma, shape)
        f[i_start:self.length, :] = data[:, :]
        f.flush()
        # Second: add some spikes chunk by chunk...
        i_start = 0
        i_end = self.chunk_length
        while i_end < self.length:
            data = f[i_start:i_end, :]
            for cell in self.cells.itervalues():
                times = cell.times * self.sampling_rate
                times = times[float(i_start) <= times]
                times = times[times < float(i_end)]
                times = times - float(i_start)
                for time in times:
                    data[int(time), :] = 7.5e-3
            f[i_start:i_end] = data
            f.flush()
            # Update indices
            i_start += self.chunk_length
            i_end += self.chunk_length
        i_end = self.chunk_length - i_start
        data = f[i_start:i_end]
        # ...
        f[i_start:i_end] = data
        f.flush()
        return


def synthetic_grid(path, size=2, duration=60.0, sampling_rate=20000.0):
    '''TODO add doc...'''
    path = os.path.expanduser(path)
    syn_grid = SyntheticGrid(path, size, duration, sampling_rate)
    syn_grid.plot()
    syn_grid.save()
    return syn_grid
