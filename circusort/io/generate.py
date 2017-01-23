import os.path
import matplotlib.pyplot as plt
import numpy as np
import ConfigParser as cp

from . import load
from .utils import *



def default(path, visualization=False):
    syn_grid = synthetic_grid(path, visualization=visualization)
    return



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
    syn.save()
    data = syn.load()
    return data



class Channel(object):
    '''TODO add doc...'''

    def __init__(self, id, c, x, y):
        self.id = id
        self.color = c
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

    def __init__(self, id, c, x, y, t, scale=0.5, refactory_period=20.0e-3):
        self.id = id
        self.color = c
        self.x = x # x-position
        self.y = y # y-position
        self.t = t # cell lifetime
        self.scale = scale # expectation of exponential intervals
        self.refactory_period = refactory_period
        self.duration = 5.0e-3 # s
        self.tau = 1.0e-3 # s
        self.alpha = 10.0e-3
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

    def distance_to(self, channel):
        dx = channel.x - self.x
        dy = channel.y - self.y
        d = np.linalg.norm([dx, dy])
        return d

    def sample(self, spike_time, times):
        w = np.zeros_like(times)
        m = spike_time <= times
        t = times - spike_time
        w[m] -= np.sin(2.0 * np.pi * t[m] / t[m][-1])
        w[m] *= (t[m] / self.tau) * np.exp(1.0 - (t[m] / self.tau))
        w[m] *= self.alpha
        return w

class SyntheticGrid(object):
    '''TODO add doc...'''

    def __init__(self, path, size, duration, sampling_rate):
        self.path = path
        self.size = size
        self.nb_channels = size * size
        self.dtype = 'float32'
        self.inter_electrode_distance = 5.0e-5
        self.duration = duration
        self.length = int(duration * sampling_rate)
        self.sampling_rate = sampling_rate
        self.mu = 0.0 # V # noise mean
        self.sigma = 1.0e-3 # V # noise standard deviation
        self.nb_cells = 9
        self.chunk_length = 40000
        self.channels = self.initialize_channels()
        self.cells = self.initialize_cells()

    def initialize_channels(self):
        channels = dict()
        xref = 0.5 * float(self.size - 1)
        yref = 0.5 * float(self.size - 1)
        for k in range(0, self.nb_channels):
            c = plt.cm.viridis(float(k) / float(self.nb_channels - 1))
            x = float(k % self.size)
            x -= xref
            x *= self.inter_electrode_distance
            y = float(k / self.size)
            y -= yref
            y *= self.inter_electrode_distance
            channels[k] = Channel(k, c, x, y)
        return channels

    def initialize_cells(self):
        cells = dict()
        xmin = -0.5
        xref = 0.5 * float(self.size - 1)
        xmax = float(self.size - 1) + 0.5
        ymin = -0.5
        yref = 0.5 * float(self.size - 1)
        ymax = float(self.size - 1) + 0.5
        for k in range(0, self.nb_cells):
            c = plt.cm.viridis(float(k) / float(self.nb_cells - 1))
            x = np.random.uniform(xmin, xmax)
            x -= xref
            x *= self.inter_electrode_distance
            y = np.random.uniform(ymin, ymax)
            y -= yref
            y *= self.inter_electrode_distance
            t = self.duration
            cells[k] = Cell(k, c, x, y, t)
        return cells

    def plot(self):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.plot_spatial_configuration()
        self.plot_temporal_configuration(t_start=0.0, t_end=1.0)
        self.plot_waveforms()
        return

    def plot_spatial_configuration(self):
        vmin = - 0.5 * float(self.size) * self.inter_electrode_distance * 1.0e6
        vmax = + 0.5 * float(self.size) * self.inter_electrode_distance * 1.0e6
        plt.figure()
        plt.subplot(1, 1, 1)
        # Plot grid
        ## Plot each channel
        x = [channel.x * 1.0e6 for channel in self.channels.itervalues()]
        y = [channel.y * 1.0e6 for channel in self.channels.itervalues()]
        c = [channel.color for channel in self.channels.itervalues()]
        plt.scatter(x, y, s=20, c=c, zorder=2, marker='s')
        ## Plot horizontal lines
        for k in range(0, self.size):
            k_start = k * self.size
            k_end = (k + 1) * self.size - 1
            plt.plot([x[k_start], x[k_end]], [y[k_start], y[k_end]], zorder=1, color='black', linestyle='dashed')
        ## Plot vertical lines
        for k in range(0, self.size):
            k_start = k
            k_end = k + (self.size - 1) * self.size
            plt.plot([x[k_start], x[k_end]], [y[k_start], y[k_end]], zorder=1, color='black', linestyle='dashed')
        # Plot cells
        x = [cell.x * 1.0e6 for cell in self.cells.itervalues()]
        y = [cell.y * 1.0e6 for cell in self.cells.itervalues()]
        c = [cell.color for cell in self.cells.itervalues()]
        plt.scatter(x, y, s=50, c=c, zorder=3)
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        plt.xlabel(r"$x$ $(\mu{}m)$")
        plt.ylabel(r"$y$ $(\mu{}m)$")
        plt.title(r"Spatial configuration")
        plt.axes().set_aspect('equal', adjustable='box')
        return

    def plot_temporal_configuration(self, t_start=None, t_end=None):
        if t_start is None:
            t_start = 0.0 # s
        if t_end is None:
            t_end = self.duration # s
        plt.figure()
        plt.subplot(1, 1, 1)
        # Plot spike times of each cell
        for cell in self.cells.itervalues():
            m = np.logical_and(t_start <= cell.times, cell.times < t_end)
            x = cell.times[m]
            y = (float(cell.id) + 0.8) * np.ones_like(x)
            bottom = cell.id
            x = np.insert(x, [0, len(x)], [t_start, t_end])
            y = np.insert(y, [0, len(y)], [bottom] * 2)
            markerline, stemlines, baseline = plt.stem(x, y, bottom=bottom)
            plt.setp(markerline, 'marker', None)
            plt.setp(markerline, 'color', cell.color)
            plt.setp(stemlines, 'color', cell.color)
            plt.setp(baseline, 'color', cell.color)
        yticks = [cell.id for cell in self.cells.itervalues()]
        ylabels = [str(cell.id) for cell in self.cells.itervalues()]
        plt.yticks(yticks, ylabels)
        plt.xlim(t_start, t_end)
        plt.ylim(-0.1, float(len(self.cells)) - 0.1)
        plt.xlabel(r"time $(s)$")
        plt.ylabel(r"cell")
        plt.title(r"Temporal configuration")
        return

    def plot_waveforms(self):
        nb_cells = len(self.cells)
        nb_cols = int(np.sqrt(nb_cells - 1)) + 1
        nb_rows = (nb_cells - 1) / nb_cols + 1
        plt.figure()
        for cell in self.cells.itervalues():
            plt.subplot(nb_rows, nb_cols, cell.id + 1)
            t_min = 0.0
            t_max = float(81) / self.sampling_rate
            t = np.linspace(t_min, t_max, num=81)
            w = cell.sample(0.0, t)
            t = 1.0e3 * t
            plt.plot(t, w, color=cell.color)
            plt.xlim(t[0], t[-1])
        plt.suptitle(r"Waveforms")
        return

    def save(self):
        shape = (self.length, self.nb_channels)
        f = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=shape)
        # First: generate the noise chunk by chunk...
        i_start = 0
        i_end = self.chunk_length
        ## Save each chunk
        while i_end < self.length:
            if __debug__:
                print("Save chunk form {} to {}".format(i_start, i_end))
            chunk_length = self.chunk_length
            shape = (chunk_length, self.nb_channels)
            data = np.random.normal(self.mu, self.sigma, shape)
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
        data = np.random.normal(self.mu, self.sigma, shape)
        f[i_start:self.length, :] = data[:, :]
        f.flush()
        # Second: add some spikes chunk by chunk...
        i_start = 0
        i_end = self.chunk_length
        ## For each chunk
        while i_end < self.length:
            data = f[i_start:i_end, :]
            for cell in self.cells.itervalues():
                # Filter spike times falling into the current chunk
                times = cell.times
                t_ref = float(i_start) / self.sampling_rate
                t_start = t_ref - cell.duration
                t_end = float(i_end - 1) / self.sampling_rate
                times = times[t_start <= times]
                times = times[times <= t_end]
                # Set chunk start time as reference
                times = times - t_ref
                t_start -= t_ref
                t_end -= t_ref
                # For each spike time
                for time in times:
                    # For each channel
                    for channel in self.channels.itervalues():
                        d = cell.distance_to(channel)
                        if time < 0:
                            t_min = 0.0
                        else:
                            t_min = time
                        if time + cell.duration <= t_end:
                            t_max = time + cell.duration
                        else:
                            t_max = t_end
                        j_min = int(np.ceil(t_min * self.sampling_rate))
                        j_max = int(np.floor(t_max * self.sampling_rate)) + 1
                        t_min = float(j_min) / self.sampling_rate
                        t_max = float(j_max - 1) / self.sampling_rate
                        num = j_max - j_min
                        t = np.linspace(t_min, t_max, num=num)
                        w = cell.sample(time, t)
                        v = data[j_min:j_max, channel.id]
                        v += w * np.exp(- d / 50.0e-6)
                        data[j_min:j_max, channel.id] = v
            f[i_start:i_end] = data
            f.flush()
            # Update indices
            i_start += self.chunk_length
            i_end += self.chunk_length
        i_end = self.chunk_length - i_start
        data = f[i_start:i_end]
        for cell in self.cells.itervalues():
            # Filter spike times falling into the current chunk
            times = cell.times
            t_ref = float(i_start) / self.sampling_rate
            t_start = t_ref - cell.duration
            t_end = float(i_end - 1) / self.sampling_rate
            times = times[t_start <= times]
            times = times[times <= t_end]
            # Set chunk start time as reference
            times = times - t_ref
            t_start -= t_ref
            t_end -= t_ref
            # For each spike time
            for time in times:
                # For each channel
                for channel in self.channels.itervalues():
                    d = cell.distance_to(channel)
                    if time < 0:
                        t_min = 0.0
                    else:
                        t_min = time
                    if time + cell.duration <= t_end:
                        t_max = time + cell.duration
                    else:
                        t_max = t_end
                    j_min = int(np.ceil(t_min * self.sampling_rate))
                    j_max = int(np.floor(t_max * self.sampling_rate)) + 1
                    t_min = float(j_min) / self.sampling_rate
                    t_max = float(j_max - 1) / self.sampling_rate
                    num = j_max - j_min
                    t = np.linspace(t_min, t_max, num=num)
                    w = cell.sample(time, t)
                    v = data[j_min:j_max, channel.id]
                    v += w * np.exp(- d / 50.0e-6)
                    data[j_min:j_max, channel.id] = v
        f[i_start:i_end] = data
        f.flush()
        # Save header file
        header_path = get_header_path(self.path)
        header_stream = open(header_path, mode='w')
        header = cp.ConfigParser()
        header.add_section('header')
        header.set('header', 'dtype', self.dtype)
        header.set('header', 'length', self.length)
        header.set('header', 'nb_channels', self.nb_channels)
        header.set('header', 'duration', self.duration)
        header.set('header', 'sampling_rate', self.sampling_rate)
        header.write(header_stream)
        header_stream.close()
        return

    def save_visualization(self):
        '''TODO add docstring...'''
        # Save visualization files
        ## Create visualization directory if necessary
        make_local_directory(self.path)
        ## Save visualization of the spatial configuration
        spatial_configuration_path = get_spatial_configuration_path(self.path)
        self.plot_spatial_configuration()
        plt.savefig(spatial_configuration_path)
        plt.close()
        ## Save visualization of the temporal configuration
        temporal_configuration_path = get_temporal_configuration_path(self.path)
        self.plot_temporal_configuration()
        plt.savefig(temporal_configuration_path)
        plt.close()
        ## Save visualization of the waveforms
        waveforms_path = get_waveforms_path(self.path)
        self.plot_waveforms()
        plt.savefig(waveforms_path)
        plt.close()
        return


def synthetic_grid(path, size=2, duration=60.0, sampling_rate=20000.0, visualization=False):
    '''TODO add doc...'''
    path = os.path.expanduser(path)
    syn_grid = SyntheticGrid(path, size, duration, sampling_rate)
    syn_grid.save()
    if visualization:
        syn_grid.save_visualization()
    else:
        pass
    return syn_grid
