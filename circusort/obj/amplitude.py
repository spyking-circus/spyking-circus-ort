import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Amplitude(object):
    """The amplitude of a cell through time."""
    # TODO complete docstring.

    def __init__(self, amplitudes, times, path=None):
        """Initialization."""
        # TODO complete docstring.

        self.amplitudes = amplitudes
        self.times = times
        self.path = path

    def save(self, path):
        """Save amplitude to file.

        Parameters:
            path: string
                The path to the file in which to save the amplitude.
        """

        with h5py.File(path, mode='w') as file_:
            kwargs = {
                'shape': self.amplitudes.shape,
                'dtype': self.amplitudes.dtype,
                'data': self.amplitudes,
            }
            file_.create_dataset('amplitudes', **kwargs)
            kwargs = {
                'shape': self.times.shape,
                'dtype': self.times.dtype,
                'data': self.times,
            }
            file_.create_dataset('times', **kwargs)

        self.path = path

        return

    def _plot(self, ax, set_ax=False, t_min=None, t_max=None, **kwargs):

        _ = kwargs  # Discard additional keyword arguments.

        x = self.times
        y = self.amplitudes

        if set_ax:
            x_min = np.min(x) if t_min is None else t_min
            x_max = np.max(x) if t_max is None else t_max
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(u"time (s)")
            ax.set_ylabel(u"amplitude (arb. unit)")
            ax.set_title(u"Amplitude")
        ax.plot(x, y)

        return

    def plot(self, output=None, ax=None, set_ax=False, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = fig.add_subplot(gs[0])
            self._plot(ax_, set_ax=True, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                fig.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "amplitude.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, set_ax=set_ax, **kwargs)

        return
