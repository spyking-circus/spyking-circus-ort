import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.utils.path import normalize_path


class Template(object):
    """The template of a cell."""
    # TODO complete docstring

    def __init__(self, channels, waveforms):
        """Initialization.

        Parameters:
            channels: numpy.ndarray
                The channels which define the support of the template. An array of shape (nb_channels,).
            waveforms: numpy.ndarray
                The waveforms of the template. An array of shape: (nb_channels, nb_samples).
        """

        self.channels = channels
        self.waveforms = waveforms

    def save(self, path):
        """Save template to file.

        Parameters:
            path: string
                The path to file in which to save the template.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('channels', shape=self.channels.shape, dtype=self.channels.dtype, data=self.channels)
        file_.create_dataset('waveforms', shape=self.waveforms.shape, dtype=self.waveforms.dtype, data=self.waveforms)
        file_.close()

        return

    def plot(self, output=None, probe=None, **kwargs):
        """Plot template.

        Parameters:
            output: none | string
            probe: none | circusort.obj.Probe
        """
        # TODO complete docstring.

        _ = kwargs  # Discard additional keyword arguments.

        nb_channels, nb_samples = self.waveforms.shape

        if output is not None:
            plt.ioff()

        fig, ax = plt.subplots()
        if probe is None:
            x_min = 0
            x_max = nb_samples
            ax.set_xlim(x_min, x_max)
            x = np.arange(0, nb_samples)
            for k in range(0, nb_channels):
                y = self.waveforms[k, :]
                color = 'C{}'.format(k)
                ax.plot(x, y, color=color)
        else:
            color = 'C0'
            for k in self.channels:
                x_0, y_0 = probe.get_channel_position(k)
                x = 20.0 * np.linspace(-0.5, +0.5, num=nb_samples) + x_0
                y = 0.3 * self.waveforms[k, :] + y_0
                ax.plot(x, y, color=color)
        ax.set_xlabel(u"time (arb. unit)")
        ax.set_ylabel(u"voltage (arb. unit)")
        ax.set_title(u"Template")
        fig.tight_layout()

        if output is None:
            plt.show()
        else:
            path = normalize_path(output)
            if path[-4:] != ".pdf":
                path = os.path.join(path, "template.pdf")
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            fig.savefig(path)

        return

    # TODO complete.
