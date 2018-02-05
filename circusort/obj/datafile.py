import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class DataFile(object):
    # TODO add docstring

    def __init__(self, path, sampling_rate, nb_channels, dtype='float32', gain=1.):
        # TODO add docstring.

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

        gmin = int(t_min * self.sampling_rate)
        gmax = int(t_max * self.sampling_rate)

        data = self.data[gmin:gmax, :].astype(np.float32) * self.gain
        return data

    def _plot(self, ax, t_min=0.0, t_max=0.5, **kwargs):

        _ = kwargs  # Discard additional keyword arguments.

        bmin = int(t_min * self.sampling_rate)
        bmax = int(t_max * self.sampling_rate)

        snippet = self.get_snippet(t_min, t_max)

        factor = self.data[bmin:bmax, :].max()

        for count, channel in enumerate(range(self.nb_channels)):
            ax.plot(np.linspace(t_min, t_max, (bmax - bmin)), factor*count + snippet[bmin:bmax, channel], '0.5')

        ax.set_yticks([])
        ax.set_xlabel(u"time (s)")
        ax.set_title(u"Data")

        return

    def plot(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "train.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, **kwargs)

        return
