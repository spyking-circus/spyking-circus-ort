import collections
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

        if len(self.waveforms.shape) != 2:
            self.waveforms = self.waveforms.reshape(len(self.indices), len(self.waveforms) // len(self.indices))

        self.nb_channels = nb_channels
        self.amplitudes = np.array(amplitudes, dtype=np.float32)

        string = "{} different from {}"
        message = string.format(self.waveforms.shape, len(self.indices))
        assert len(self.waveforms) == len(self.indices), message

    @property
    def norm(self):

        norm = np.linalg.norm(self.waveforms)

        return norm

    @property
    def temporal_width(self):

        return self.waveforms.shape[1]

    def __str__(self):

        return 'TemplateComponent for %d channels with amplitudes %s' % (self.nb_channels, self.amplitudes)

    def _compress(self, indices):

        self.waveforms = np.delete(self.waveforms, indices, 0)
        self.indices = np.delete(self.indices, indices, 0)

        return

    def to_sparse(self, method='csr', flatten=False):

        data = self.to_dense()
        if flatten:
            data = data.flatten()[None, :]

        if method is 'csc':
            sparse_data = scipy.sparse.csc_matrix(data, dtype=np.float32)
        elif method is 'csr':
            sparse_data = scipy.sparse.csr_matrix(data, dtype=np.float32)
        else:
            string = "method={}"
            message = string.format(method)
            raise NotImplementedError(message)

        return sparse_data

    def to_dense(self):

        result = np.zeros((self.nb_channels, self.temporal_width), dtype=np.float32)
        result[self.indices] = self.waveforms

        return result

    def normalize(self):

        self.waveforms /= self.norm

        return

    def similarity(self, component):
        """Compute the correlation coefficient between two template components.

        Argument:
            component: circusort.obj.TemplateComponent
                The template component with which the correlation coefficient has to be computed.
        Return:
            coefficient: float
                The correlation coefficient between the to template components. A value between +1.0 and -1.0.
        """

        if self.intersect(component):
            # Format 1st component.
            c1 = self.to_dense()
            c1 = c1.flatten()
            # Format 2nd component.
            c2 = component.to_dense()
            c2 = c2.flatten()
            # Compute the Pearson product-moment correlation coefficients (2 x 2 matrix).
            r = np.corrcoef(c1, c2)
            coefficient = r[0, 1]
        else:
            coefficient = 0.0

        return coefficient

    def intersect(self, component):

        return np.any(np.in1d(self.indices, component.indices))

    def to_dict(self, full=True):

        res = {'wav': self.waveforms, 'amp': self.amplitudes}
        if full:
            res['indices'] = self.indices
            res['nb_channels'] = self.nb_channels

        return res

    def center(self, shift):

        if shift != 0:
            aligned_template = np.zeros(self.waveforms.shape, dtype=np.float32)
            if shift > 0:
                aligned_template[:, shift:] = self.waveforms[:, :-shift]
            elif shift < 0:
                aligned_template[:, :shift] = self.waveforms[:, -shift:]

            self.waveforms = aligned_template

        return


class Template(object):

    def __init__(self, first_component, channel=None, second_component=None, creation_time=0, compress=True, path=None):

        assert isinstance(first_component, TemplateComponent)
        self.first_component = first_component
        assert self.first_component.amplitudes is not None
        self.channel = channel
        if second_component is not None:
            assert isinstance(second_component, TemplateComponent)
            assert np.all(second_component.indices == self.first_component.indices), "Error with indices"
            assert second_component.temporal_width == self.first_component.temporal_width, "Error with time"

        self.second_component = second_component
        self.creation_time = creation_time
        self._synthetic_export = None
        self.compressed = False

        if self.channel is None:
            min_voltages = np.min(self.first_component.waveforms, axis=1)
            index = np.argmin(min_voltages)
            self.channel = self.first_component.indices[index]

        if compress:
            self._auto_compression()

        self.path = path

    def __len__(self):

        if self.second_component is not None:
            return 2
        else:
            return 1

    def __getitem__(self, item):

        if item == 0:
            return self.first_component
        elif len(self) > 1:
            if item == 1:
                return self.second_component
            else:
                print("Only 2 components are available")

    def __str__(self):
        if self.compressed:
            str_comp = 'Compressed'
        else:
            str_comp = 'Non Compressed'

        return '%s template on channel %d (%d indices)\n' % (str_comp, self.channel, len(self.indices))

    def __iter__(self):

        data = [self.first_component]
        if len(self) == 2:
            data += [self.second_component]

        return data.__iter__()

    @property
    def two_components(self):

        return self.second_component is not None

    @property
    def indices(self):

        return self.first_component.indices

    @property
    def amplitudes(self):

        return np.array(self.first_component.amplitudes, dtype=np.float32)

    @property
    def temporal_width(self):

        return self.first_component.temporal_width

    @property
    def nb_channels(self):

        return self.first_component.nb_channels

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

    @property
    def norm(self):

        result = [a.norm for a in self]

        return result

    def normalize(self):

        for component in self:
            component.normalize()

    def intersect(self, template):

        return self.first_component.intersect(template)

    def similarity(self, template):
        """Compute the similarities between templates.

        Argument:
            template: circusort.obj.Template | collections.Iterable
                The templates with which similarities have to be computed.
        Return:
            coefficient: float | numpy.ndarray
                The similarities between templates.
        """

        if isinstance(template, Template):
            coefficient = self.first_component.similarity(template.first_component)
        elif isinstance(template, collections.Iterable):
            coefficient = np.array([
                self.first_component.similarity(t.first_component)
                for t in template
            ])
        else:
            string = "Unexpected type: {}"
            message = string.format(type(template))
            raise TypeError(message)

        return coefficient

    def _auto_compression(self):

        sums = np.sum(self.first_component.waveforms, 1)
        if self.two_components:
            sums += np.sum(self.second_component.waveforms, 1)
        indices = np.where(sums == 0)[0]
        self._compress(indices)

        return

    def _compress(self, indices):

        if len(indices) > 0:
            for component in self:
                component._compress(indices)
            self.compressed = True
            self._synthetic_export = None

        return

    def compress(self, compression_factor=0.5):

        if compression_factor > 0:
            stds = np.std(self.first_component.waveforms, 1)
            threshold = np.percentile(stds, int(compression_factor * 100.0))
            indices = np.where(stds < threshold)[0]
            self._compress(indices)

        return

    def save(self, path):

        with h5py.File(path, mode='w') as file_:

            file_.create_dataset('waveforms/1', data=self.first_component.waveforms, chunks=True)
            file_.create_dataset('amplitudes', data=self.amplitudes)
            file_.create_dataset('indices', data=self.indices, chunks=True)
            file_.attrs['channel'] = self.channel
            file_.attrs['nb_channels'] = self.first_component.nb_channels
            file_.attrs['creation_time'] = self.creation_time
            file_.attrs['compressed'] = self.compressed

            if self.two_components:
                file_.create_dataset('waveforms/2', data=self.second_component.waveforms, chunks=True)

        self.path = path

        return

    def plot(self, ax=None, output=None, probe=None, title=u"Template", with_xaxis=True, with_yaxis=True,
             with_scale_bars=True, **kwargs):
        """Plot template.

        Arguments:
            ax: none | matplotlib.axes.Axes (optional)
                The default value is None.
            output: none | string (optional)
                The default value is None.
            probe: none | circusort.obj.Probe (optional)
                The default value is None.
            title: none | string (optional)
                The default value is u"Template".
            with_xaxis: boolean (optional)
                The default value is True.
            with_yaxis: boolean (optional)
                The default value is True.
            with_scale_bars: boolean (optional)
                The default value is True.
            kwargs: dictionary (optional)
                Additional keyword arguments.
        """

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if 'solid_capstyle' not in kwargs:
            kwargs['solid_capstyle'] = 'round'

        nb_channels, nb_samples = self.first_component.waveforms.shape

        if output is not None:
            plt.ioff()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if probe is None:
            x_min = 0
            x_max = nb_samples
            ax.set_xlim(x_min, x_max)
            x = np.arange(0, nb_samples)
            for k in range(0, nb_channels):
                y = self.first_component.waveforms[k, :]
                ax.plot(x, y, **kwargs)
        else:
            ax.set_aspect('equal')
            x_min, x_max = probe.x_limits
            y_min, y_max = probe.y_limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            for k, channel in enumerate(self.first_component.indices):
                x_0, y_0 = probe.get_channel_position(channel)
                x = 20.0 * np.linspace(-0.5, +0.5, num=nb_samples) + x_0
                y = 0.3 * self.first_component.waveforms[k, :] + y_0
                ax.plot(x, y, **kwargs)
            if with_scale_bars:
                # Add scale bars.
                x_anchor = x_max - 0.1 * (x_max - x_min)
                y_anchor = y_max - 0.1 * (y_max - y_min)
                # # Add time scale bar.
                width = 1  # TODO improve.
                x = [x_anchor, x_anchor - 20.0 * float(width)]
                y = [y_anchor, y_anchor]
                ax.plot(x, y, color='black')
                ax.text(np.mean(x), np.mean(y), u"{} arb. unit".format(width), fontsize=8,
                        horizontalalignment='center', verticalalignment='bottom')
                # # Add voltage scale bar.
                height = 50  # TODO improve.
                x = [x_anchor, x_anchor]
                y = [y_anchor, y_anchor - 0.3 * float(height)]
                ax.plot(x, y, color='black')
                ax.text(np.mean(x), np.mean(y), u"{} µV".format(height), fontsize=8,
                        horizontalalignment='left', verticalalignment='center')

        if with_xaxis:
            ax.set_xlabel(u"x (µm)")
        else:
            ax.set_xticklabels([])
        if with_yaxis:
            ax.set_ylabel(u"y (µm)")
        else:
            ax.set_yticklabels([])
        if title:
            ax.set_title(title)
        fig.tight_layout()

        if output is not None:
            path = normalize_path(output)
            if path[-4:] != ".pdf":
                path = os.path.join(path, "template.pdf")
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            fig.savefig(path)

        return

    def to_dict(self):

        res = {}
        for count, component in enumerate(self):
            key = "{}".format(count)
            res[key] = component.to_dict(False)

        if self.compressed:
            res['compressed'] = self.indices

        res['channel'] = self.channel
        res['time'] = self.creation_time

        return res

    def center(self, peak_type='negative'):

        if peak_type == 'negative':
            tmp_idx = np.divmod(self.first_component.waveforms.argmin(), self.first_component.waveforms.shape[1])
        elif peak_type == 'positive':
            tmp_idx = np.divmod(self.first_component.waveforms.argmax(), self.first_component.waveforms.shape[1])
        else:
            string = "peak_type={}"
            message = string.format(peak_type)
            raise NotImplementedError(message)

        shift = (self.temporal_width - 1) // 2 - tmp_idx[1]

        for component in self:
            component.center(shift)

        return
