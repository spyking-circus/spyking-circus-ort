import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from circusort.utils.path import normalize_path
from scipy.sparse import csc_matrix


class TemplateComponent(object):

    def __init__(self, waveforms, indices, nb_channels, amplitudes=None):
        """Initialization.

        Parameters:
            waveforms
            indices
            nb_channels
            amplitudes (optional)
        """

        self.waveforms = waveforms.astype(np.float32)
        self.indices = indices.astype(np.int32)
        self.nb_channels = nb_channels
        self.amplitudes = np.array(amplitudes, dtype=np.float32)

    @property
    def norm(self):

        return np.sqrt(np.sum(self.waveforms**2)/(self.nb_channels * self.temporal_width))

    @property
    def temporal_width(self):

        return self.waveforms.shape[1]

    def to_sparse(self, method='csc', flatten=False):

        data = self.to_dense()
        if method is 'csc':
            if flatten:
                data = data.flatten()[None, :]
            return scipy.sparse.csc_matrix(data, dtype=np.float32)
        elif method is 'csr':
            if flatten:
                data = data.flatten()[:, None]
            return scipy.sparse.csr_matrix(data, dtype=np.float32)

    def to_dense(self):
        result = np.zeros((self.nb_channels, self.temporal_width), dtype=np.float32)
        for count, index in enumerate(self.indices):
            result[index] = self.waveforms[count]
        return result

    def normalize(self):

        self.waveforms /= self.norm

    def similarity(self, component):

        return np.corrcoef(self.to_dense().flatten(), component.to_dense().flatten())[0, 1]


class Template(object):

    def __init__(self, first_component, channel=None, second_component=None, creation_time=0):

        self.first_component = first_component
        assert self.first_component.amplitudes is not None
        self.channel = channel
        self.second_component = second_component
        self.creation_time = creation_time
        self._synthetic_export = None

        if self.channel is None:
            min_voltages = np.min(self.first_component.waveforms, axis=1)
            index = np.argmin(min_voltages)
            self.channel = self.first_component.indices[index]

    @property
    def two_components(self):

        return self.second_component is not None

    @property
    def amplitudes(self):

        return np.array(self.first_component.amplitudes, dtype=np.float32)

    def normalize(self):

        self.first_component.normalize()
        if self.two_components:
            self.second_component.normalize()

    @property
    def temporal_width(self):

        return self.first_component.temporal_width

    @property
    def synthetic_export(self):

        if self._synthetic_export is None:
            channels = self.first_component.indices
            waveforms = self.first_component.waveforms
            nb_channels, nb_timestamps = waveforms.shape

            timestamps = np.arange(0, nb_timestamps) - (nb_timestamps - 1) // 2
            timestamps = timestamps[np.newaxis, :]
            timestamps = np.repeat(timestamps, repeats=nb_channels, axis=0)

            channels = channels[:, np.newaxis]
            channels = np.repeat(channels, repeats=nb_timestamps, axis=1)

            i = timestamps.flatten().astype(np.int32)
            j = channels.flatten().astype(np.int32)
            v = waveforms.flatten().astype(np.float32)

            self._synthetic_export = i, j, v

        return self._synthetic_export

    def similarity(self, template):

        res = [self.first_component.similarity(template.first_component)]
        if template.two_components and self.two_components:
            res += [self.second_component.similarity(template.second_component)]

        return np.mean(res)

    def save(self, path):

        with h5py.File(path, mode='w') as file_:

            file_.create_dataset('waveforms/1', data=self.first_component.waveforms, chunks=True)
            file_.create_dataset('amplitudes', data=self.amplitudes)
            file_.create_dataset('indices', data=self.first_component.indices, chunks=True)
            file_.attrs['channel'] = self.channel
            file_.attrs['nb_channels'] = self.first_component.nb_channels
            file_.attrs['creation_time'] = self.creation_time

            if self.two_components:
                file_.create_dataset('waveforms/2', data=self.second_component.waveforms, chunks=True)

        return

    def plot(self, output=None, probe=None, **kwargs):
        """Plot template.

        Parameters:
            output: none | string
            probe: none | circusort.obj.Probe
        """
        # TODO complete docstring.

        _ = kwargs  # Discard additional keyword arguments.

        nb_channels, nb_samples = self.first_component.waveforms.shape

        if output is not None:
            plt.ioff()

        fig, ax = plt.subplots()
        if probe is None:
            x_min = 0
            x_max = nb_samples
            ax.set_xlim(x_min, x_max)
            x = np.arange(0, nb_samples)
            color = 'C0'
            for k in range(0, nb_channels):
                y = self.first_component.waveforms[k, :]
                ax.plot(x, y, color=color)
        else:
            ax.set_aspect('equal')
            ax.set_xlim(*probe.x_limits)
            ax.set_ylim(*probe.y_limits)
            color = 'C0'
            for k, channel in enumerate(self.first_component.indices):
                x_0, y_0 = probe.get_channel_position(channel)
                x = 20.0 * np.linspace(-0.5, +0.5, num=nb_samples) + x_0
                y = 0.3 * self.first_component.waveforms[k, :] + y_0
                ax.plot(x, y, color=color, solid_capstyle='round')
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
