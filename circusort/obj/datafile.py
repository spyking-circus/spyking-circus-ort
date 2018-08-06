import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class DataFile(object):
    """Data file

    Attributes:
        path: string
        sampling_rate: float
        nb_channels: integer
        dtype: string
        gain: float
    """

    def __init__(self, path, sampling_rate, nb_channels, dtype='float32', gain=1.0):
        """Initialize data file.

        Arguments:
            path: string
            sampling_rate: float
            nb_channels: integer
            dtype: string (optional)
                The default value is 'float32'.
            gain: float (optional)
                The default value is 1.0.
        """

        self.path = path
        self.dtype = dtype
        self.nb_channels = nb_channels
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.data = np.memmap(self.path, dtype=self.dtype)

        if self.nb_channels > 1:
            self.data = self.data.reshape(self.data.shape[0] // self.nb_channels, self.nb_channels)

    def __len__(self):

        return len(self.data)

    def get_snippet(self, t_min, t_max):
        """Get data snippet.

        Arguments:
            t_min: float
            t_max: float
        """

        b_min = int(np.ceil(t_min * self.sampling_rate))
        b_max = int(np.floor(t_max * self.sampling_rate))

        data = self.data[b_min:b_max + 1, :]
        data = data.astype(np.float32)
        data = self.gain * data

        return data

    def take(self, channels=None, ts_min=None, ts_max=None):

        time_steps = np.arange(ts_min, ts_max + 1)
        data = self.data[np.ix_(time_steps, channels)]

        return data

    def put(self, data, channels=None, ts_min=None, ts_max=None):

        time_steps = np.arange(ts_min, ts_max + 1)
        self.data[np.ix_(time_steps, channels)] = data

        return

    def _plot(self, ax, t_min=0.0, t_max=0.5, **kwargs):
        """Plot data from file.

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

        b_min = int(np.ceil(t_min * self.sampling_rate))
        b_max = int(np.floor(t_max * self.sampling_rate))
        nb_samples = b_max - b_min + 1
        t_min_ = float(b_min) / self.sampling_rate
        t_max_ = float(b_max) / self.sampling_rate

        snippet = self.get_snippet(t_min_, t_max_)

        factor = 0.0
        for channel in range(0, self.nb_channels):
            y = snippet[:, channel]
            y = y - np.mean(y)
            factor = max(factor, np.abs(y).max())
        factor = factor if factor > 0.0 else 1.0

        for count, channel in enumerate(range(0, self.nb_channels)):
            x = np.linspace(t_min_, t_max_, num=nb_samples)
            y = snippet[0:nb_samples, channel]
            y = y - np.mean(y)
            y = count + 0.5 * y / factor
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
