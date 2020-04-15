import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os


from circusort.utils.path import normalize_path


class Peaks(object):

    def __init__(self, path):

        self._path = path
        self._file = h5py.File(self._path, mode='r')

    def __len__(self):

        return self._times.size

    def __getitem__(self, key):

        time = self._times[key]
        channel = self._channels[key]
        polarity = self._polarities[key]

        return time, channel, polarity

    def __str__(self):

        polarities = self._polarities
        channels = self._channels
        nb_times = {
            polarity: {
                channel: 0
                for channel in np.unique(channels)
            }
            for polarity in np.unique(polarities)
        }
        for polarity, channel in zip(polarities, channels):
            nb_times[polarity][channel] += 1
        string = str(nb_times)

        return string

    @property
    def _times(self):

        dataset = self._file['times']
        times = dataset.value

        return times

    @property
    def _nb_times(self):

        return self._times.size

    @property
    def _channels(self):

        dataset = self._file['channels']
        channels = dataset.value

        return channels

    @property
    def _polarities(self):

        dataset = self._file['polarities']
        polarities = dataset.value

        return polarities

    def get_times(self, t_min=None, t_max=None, channels=None, polarities=None):

        flags = np.ones(self._nb_times, dtype=np.bool)

        if t_min is not None:
            flags = np.logical_and(flags, t_min <= self._times)

        if t_max is not None:
            flags = np.logical_and(flags, self._times <= t_max)

        if channels is not None:
            flags_ = np.zeros(self._nb_times, dtype=np.bool)
            for channel in channels:
                flags_ = np.logical_or(flags_, self._channels == channel)
            flags = np.logical_and(flags, flags_)

        if polarities is not None:
            flags_ = np.zeros(self._nb_times, dtype=np.bool)
            for polarity in polarities:
                flags_ = np.logical_or(flags_, self._polarities == polarity)
            flags = np.logical_and(flags, flags_)

        times = self._times[flags]

        return times

    def _plot(self, ax, t_min=0.0, t_max=0.5, **kwargs):
        """Plot peaks from file.

        Arguments:
            ax: none | matplotlib.axes.Axes
            t_min: float (optional)
                The default value is 0.0.
            t_max: float (optional)
                The default value is 0.5.
            kwargs: dict (optional)
                Additional keyword arguments. See matplotlib.axes.Axes.plot for details.
        Return:
            ax: matplotlib.axes.Axes
        """

        for channel in np.unique(self._channels):
            times = self.get_times(t_min=t_min, t_max=t_max, channels=[channel])
            for time in times:
                x = [time, time]
                y = [float(channel) + 0.15, float(channel) + 0.35]
                ax.plot(x, y, **kwargs)

        return ax

    def plot(self, output=None, ax=None, **kwargs):
        """Plot data from file.

        Arguments:
            output: none | string (optional)
                The default value is None.
            ax: none | matplotlib.axes.Axes (optional)
                The default value is None.
            kwargs: dict (optional)
                Additional keyword arguments. See matplotlib.axes.Axes.plot for details.
        Return:
            ax: matplotlib.axes.Axes
        """

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            ax = self._plot(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, **kwargs)

        return ax
