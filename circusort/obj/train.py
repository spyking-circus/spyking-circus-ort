import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Train(object):
    # TODO add docstring

    def __init__(self, times, t_min=None, t_max=None, path=None):
        # TODO add docstring.

        self.times = times
        self.times = self.times if t_min is None else self.times[t_min <= self.times]
        self.times = self.times if t_max is None else self.times[self.times <= t_max]
        self.t_min = min(0.0, np.min(times)) if t_min is None else t_min
        self.t_max = np.max(times) if t_max is None else t_max
        self.path = path

    def __len__(self):
        return len(self.times)

    def __iter__(self):
        return self.times.__iter__()

    @property
    def nb_times(self):

        return self.times.size

    def reverse(self):
        # TODO add docstring.

        # TODO improve method with two additional attributes: start_time and end_time.
        times = self.t_min + ((self.t_max - self.t_min) - (self.times - self.t_min))
        train = Train(times, t_min=self.t_min, t_max=self.t_max)

        return train

    def slice(self, t_min=None, t_max=None):
        # TODO add docstring.

        times = self.times
        if t_min is None:
            t_min = self.t_min
        elif isinstance(t_min, float):
            times = times[t_min <= times]
        if t_max is None:
            t_max = self.t_max
        elif isinstance(t_max, float):
            times = times[times <= t_max]
        train = Train(times, t_min=t_min, t_max=t_max)

        return train

    def save(self, path):
        """Save train to file.

        Parameters:
            path: string
                The path to the file in which to save the train.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('times', shape=self.times.shape, dtype=self.times.dtype, data=self.times)
        file_.close()

        self.path = path

        return

    def get_parameters(self):
        # TODO add docstring.

        parameters = {
            'mode': "default",
            'path': self.path if self.path is not None else "",
            # TODO complete.
        }

        return parameters

    def _plot(self, ax, t_min=0.0, t_max=10.0, offset=0, **kwargs):

        _ = kwargs  # Discard additional keyword arguments.

        t_min = self.t_min + t_min
        t_max = self.t_min + t_max

        is_selected = np.logical_and(t_min <= self.times, self.times <= t_max)
        x = self.times[is_selected]
        y = offset * np.ones_like(x)
        x_min = t_min
        x_max = t_max

        ax.set_xlim(x_min, x_max)
        ax.scatter(x, y)  # TODO control the radius of the somas of the cells.
        ax.set_yticks([])
        ax.set_xlabel(u"time (s)")
        ax.set_title(u"Train")

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

    # TODO complete.
