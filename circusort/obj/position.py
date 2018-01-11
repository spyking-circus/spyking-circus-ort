# -*- coding: utf-8 -*-

import h5py
import matplotlib.gridspec as gds
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Position(object):
    """The position of a cell through time."""
    # TODO complete docstring.

    def __init__(self, x, y, path=None):
        """Initialization."""
        # TODO complete docstring.

        self.x = x
        self.y = y
        self.path = path

    def save(self, path):
        """Save position to file.

        Parameters:
            path: string
                The path to the file in which to save the position.
        """

        with h5py.File(path, mode='w') as file_:
            file_.create_dataset('x', shape=self.x.shape, dtype=self.x.dtype, data=self.x)
            file_.create_dataset('y', shape=self.y.shape, dtype=self.y.dtype, data=self.y)

        self.path = path

        return

    def get_initial_position(self):

        x = self.x[0]
        y = self.y[0]
        position = (x, y)

        return position

    def _plot(self, ax, probe=None, color='C1', set_ax=False, **kwargs):
        # TODO add docstring.

        _ = kwargs  # Discard additional keyword arguments.

        x = self.x[0:1]
        y = self.y[0:1]
        r = 5.0  # µm

        if probe is None:
            if set_ax:
                x_min = np.amin(x) - 10.0
                x_max = np.amax(x) + 10.0
                y_min = np.amin(y) - 10.0
                y_max = np.amax(y) + 10.0
                ax.set_aspect('equal')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel(u"x (µm)")
                ax.set_ylabel(u"y (µm)")
        else:
            probe.plot(ax=ax)

        patch = ptc.Circle((x, y), radius=r, color=color)
        ax.add_patch(patch)
        ax.set_title(u"Position")

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
                    path = os.path.join(path, "position.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                fig.savefig(path)
        else:
            self._plot(ax, set_ax=set_ax, **kwargs)

        return
