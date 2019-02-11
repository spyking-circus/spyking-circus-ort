import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from circusort.utils.path import normalize_path
from circusort.io.probe import load_probe


class DataFile(object):
    """Data file

    Attributes:
        path: string
        sampling_rate: float
        nb_channels: integer
        dtype: string
        gain: float
    """

    def __init__(self, path, sampling_rate, nb_channels=None, probe=None, dtype='float32', gain=1.0, offset=0, nb_replay=1):
        """Initialize data file.

        Arguments:
            path: string
            sampling_rate: float
            nb_channels: integer
            dtype: string (optional)
                The default value is 'float32'.
            gain: float (optional)
                The default value is 1.0.
            offset: int (optional)
                The offset if the file has a header
        """

        self.path = path
        self.dtype = dtype
        assert (probe is not None) or (nb_channels is not None), "Please provide either a probe file, either a number of channels"
        if probe is None:
            self.total_nb_channels = nb_channels
            self.nb_channels = nb_channels
            self.probe = None
        else:
            self.probe = probe
            self.total_nb_channels = probe.total_nb_channels
            self.nb_channels = probe.nb_channels

        self.sampling_rate = sampling_rate
        self.gain = gain
        self.offset = offset
        self.nb_replay = nb_replay
        self._quantum_offset = float(np.iinfo('int16').min)
        self.data = np.memmap(self.path, dtype=self.dtype, offset=self.offset)

        if self.total_nb_channels > 1:
            self.real_shape = self.data.shape[0] // self.total_nb_channels
            self.nb_samples = self.nb_replay*self.data.shape[0] // self.total_nb_channels
            self.data = self.data.reshape(self.real_shape, self.total_nb_channels)
        elif self.total_nb_channels == 1:
            self.nb_samples = self.nb_replay*self.data.shape[0]
        else:
            raise NotImplementedError()

    def __len__(self):

        return self.nb_samples

    @property
    def duration(self):
        return self.nb_samples
   
    def _get_slice(self, t_min, t_max, samples = True):

        if samples is False:
            b_min = int(np.ceil(t_min * self.sampling_rate))
            b_max = int(np.floor(t_max * self.sampling_rate))

        assert b_min < self.nb_samples, "Please provide valid t_start, t_stop"
        assert b_max < self.nb_samples, "Please provide valid t_start, t_stop"
        b_min = b_min % self.real_shape
        b_max = b_max % self.real_shape
        

        if b_min < b_max:
            return np.arange(b_min, b_max + 1)
        else:
            return list(range(b_max, self.real_shape)) + list(range(b_min + 1))

    def get_snippet(self, t_min, t_max):
        """Get data snippet.

        Arguments:
            t_min: float
            t_max: float
        Return:
            data: numpy.ndarray
        """

        data = self.data[self._get_slice(t_min, t_max, False), :]
        if self.probe is not None:
            data = self.data[:, self.probe.nodes]
        data = data.astype(np.float32)
        data = self.gain * data

        return data

    def take(self, channels=None, ts_min=None, ts_max=None):
        """Take samples from data file.

        Arguments:
            channels: none | iterable (optional)
                The indices of the channels to extract.
                The default value is None.
            ts_min: none | integer (optional)
                The minimum time step to extract.
                The default value is None.
            ts_max: none | integer (optional)
                The maximum time step to extract.
                The default value is None.
        Return:
            data: numpy.ndarray
                The extracted samples.
        """

        if ts_min is None:
            ts_min = 0
        if ts_max is None:
            ts_max = self.data.shape[0] - 1

        time_steps = self._get_slice(ts_min, ts_max)

        if channels is None:
            data = self.data[time_steps, :]
        else:
            data = self.data[np.ix_(time_steps, channels)]

        if self.probe is not None:
            data = self.data[:, self.probe.nodes]

        return data

    def put(self, data, channels=None, ts_min=None, ts_max=None):
        """"Put samples in data file.

        Arguments:
            data: numpy.ndarray
                The data to inject.
            channels: none | iterable (optional)
                The indices of the channels for the injection.
                The default value is None.
            ts_min: none | integer (optional)
                The minimum time step for the injection.
                The default value is None.
            ts_max: none | integer (optional)
                The maximum time step for the injection.
                The default value is None.
        """

        if ts_min is None:
            ts_min = 0
        if ts_max is None:
            ts_max = self.data.shape[0] - 1

        time_steps = self._get_slice(ts_min, ts_max)

        if channels is None:
            self.data[time_steps, :] = data
        else:
            self.data[np.ix_(time_steps, channels)] = data

        return

    def _plot(self, ax, t_min=0.0, t_max=1, colors=None, **kwargs):
        """Plot data from file.

        Arguments:
            ax: none | matplotlib.axes.Axes
            t_min: float (optional)
                The default value is 0.0.
            t_max: float (optional)
                The default value is 1.
            colors: none | iterable (optional)
                The default value is None.
            kwargs: dict (optional)
                Additional keyword arguments. See matplotlib.axes.Axes.plot for details.
        Return:
            ax: matplotlib.axes.Axes
        """

        # b_min = int(np.ceil(t_min * self.sampling_rate))
        # b_max = int(np.floor(t_max * self.sampling_rate))
        # nb_samples = b_max - b_min + 1
        # t_min_ = float(b_min) / self.sampling_rate
        # t_max_ = float(b_max) / self.sampling_rate

        snippet = self.get_snippet(t_min, t_max)
        nb_samples = snippet.shape[0]

        # Define the number of channels to be plotted.
        max_nb_channels = 10
        if self.nb_channels > max_nb_channels:
            string = "Too many channels ({}), sub-selection used ({}) to plot the data."
            message = string.format(self.nb_channels, max_nb_channels)
            warnings.warn(message)
        nb_plotted_channels = min(self.nb_channels, max_nb_channels)

        # Compute the scaling factor for the voltage.
        factor = 0.0
        for channel in range(0, nb_plotted_channels):
            y = snippet[:, channel]
            y = y - np.mean(y)
            factor = max(factor, np.abs(y).max())
        factor = factor if factor > 0.0 else 1.0

        # Plot data.
        x = np.linspace(t_min, t_max, num=nb_samples)
        for count, channel in enumerate(range(0, nb_plotted_channels)):
            y = snippet[:, channel]
            y = y - np.mean(y)
            y = count + 0.5 * y / factor
            if colors is None:
                ax.plot(x, y, **kwargs)
            else:
                kwargs.update(color=colors[channel])
                ax.plot(x, y, **kwargs)
        ax.set_yticks([])
        ax.set_xlabel(u"time (s)")
        ax.set_ylabel(u"channel")
        ax.set_title(u"Data")

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

        if output is not None and ax is None:
            plt.ioff()

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
