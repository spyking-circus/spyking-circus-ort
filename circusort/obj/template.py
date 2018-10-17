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

    def __add__(self, other):

        if not isinstance(other, TemplateComponent):
            string = "unsupported operand type(s) for +: '{}' and '{}'"
            message = string.format(type(self), type(other))
            raise TypeError(message)

        assert self.nb_channels == other.nb_channels
        assert self.temporal_width == other.temporal_width

        indices = np.unique(np.concatenate((self.indices, other.indices)))
        waveforms = np.zeros((self.nb_channels, self.temporal_width), dtype=np.float32)
        for k, index in enumerate(self.indices):
            waveforms[index, :] += self.waveforms[k, :]
        for k, index in enumerate(other.indices):
            waveforms[index, :] += self.waveforms[k, :]
        waveforms = waveforms[indices, :]
        nb_channels = self.nb_channels
        amplitudes = None

        result = TemplateComponent(waveforms, indices, nb_channels, amplitudes=amplitudes)

        return result

    def __mul__(self, other):

        if not isinstance(other, float):
            string = "unsupported operand type(s) for *: '{}' and '{}'"
            message = string.format(type(self), type(other))
            raise TypeError(message)

        waveforms = self.waveforms * other
        indices = self.indices
        nb_channels = self.nb_channels
        amplitudes = None

        result = TemplateComponent(waveforms, indices, nb_channels, amplitudes=amplitudes)

        return result

    @property
    def norm(self):

        norm = np.linalg.norm(self.waveforms)

        return norm

    @property
    def temporal_width(self):

        return self.waveforms.shape[1]

    @property
    def extrema(self):
        index = self.temporal_width//2 + 1
        return (np.min(self.waveforms[:, index]), np.max(self.waveforms[:, index]))
    
    def __str__(self):

        string = "TemplateComponent for {} channels with amplitudes {}"
        message = string.format(self.nb_channels, self.amplitudes)

        return message

    def compress(self, indices):

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

        res = {
            'wav': self.waveforms,
            'amp': self.amplitudes,
        }
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

    def __init__(self, first_component, channel=None, second_component=None,
                 creation_time=0, compressed=False, path=None):

        # Save 1st component.
        assert isinstance(first_component, TemplateComponent)
        assert first_component.amplitudes is not None
        self.first_component = first_component
        # Save central channel.
        if channel is None:
            min_voltages = np.min(self.first_component.waveforms, axis=1)
            index = np.argmin(min_voltages)
            self.channel = self.first_component.indices[index]
        else:
            self.channel = channel
        # Save 2nd component.
        if second_component is not None:
            assert isinstance(second_component, TemplateComponent)
            assert np.all(second_component.indices == self.first_component.indices), "Error with indices..."
            assert second_component.temporal_width == self.first_component.temporal_width, "Error with time..."
        self.second_component = second_component
        # Save creation time.
        self.creation_time = creation_time
        # Save compression arguments.
        self._compressed = compressed
        if not self._compressed:
            self._auto_compression()
        # Save path.
        self.path = path
        # Define internal argument.
        self._synthetic_export = None

    def __len__(self):

        length = 2 if self.has_second_component else 1

        return length

    def __getitem__(self, key):

        if key == 0:
            item = self.first_component
        elif key == 1 and len(self) > 1:
            item = self.second_component
        else:
            string = "component {} is out of bounds for template with {} components"
            message = string.format(key, len(self))
            raise IndexError(message)

        return item

    def __str__(self):

        if self._compressed:
            str_comp = 'Compressed'
        else:
            str_comp = 'Non Compressed'

        return '%s template on channel %d (%d indices)\n' % (str_comp, self.channel, len(self.indices))

    def __iter__(self):

        data = [self.first_component]
        if len(self) == 2:
            data += [self.second_component]

        return data.__iter__()

    def __add__(self, other):
        """Add two templates together.

        Argument:
            other: Template
                Template to add to the current one.
        Return:
            result: Template

        """

        if not isinstance(other, Template):
            string = "unsupported operand type(s) for +: '{}' and '{}'"
            message = string.format(type(self), type(other))
            raise TypeError(message)

        first_component = self.first_component + other.first_component
        channel = None
        if self.has_second_component and other.has_second_component:
            second_component = self.second_component + other.second_component
        else:
            second_component = None
        creation_time = max(self.creation_time, other.creation_time)
        compressed = self.is_compressed and other.is_compressed
        path = None

        result = Template(first_component, channel=channel, second_component=second_component,
                          creation_time=creation_time, compressed=compressed, path=path)

        return result

    def __mul__(self, other):
        """Multiple a template by a factor.

        Argument:
            other: float
                Scale factor.
        Return:
            result: Template
                Scaled template.
        """

        if not isinstance(other, float):
            string = "unsupported operand type(s) for *: '{}' and '{}'"
            message = string.format(type(self), type(other))
            raise TypeError(message)

        first_component = self.first_component * other
        channel = self.channel
        second_component = self.second_component * other if self.has_second_component else None
        creation_time = self.creation_time
        compressed = self.is_compressed
        path = None

        result = Template(first_component, channel=channel, second_component=second_component,
                          creation_time=creation_time, compressed=compressed, path=path)

        return result

    @property
    def two_components(self):

        return self.second_component is not None

    @property
    def has_second_component(self):

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

    @property
    def extrema(self):
        return self.first_component.extrema

    @property
    def is_compressed(self):

        return self._compressed

    def normalize(self):

        for component in self:
            component.normalize()

        return

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
                component.compress(indices)
            self._compressed = True
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
            file_.attrs['compressed'] = self._compressed

            if self.two_components:
                file_.create_dataset('waveforms/2', data=self.second_component.waveforms, chunks=True)

        self.path = path

        return

    def plot(self, ax=None, output=None, probe=None, title=u"Template", with_xaxis=True, with_yaxis=True,
             with_scale_bars=True, mode='superimposed', time_factor=50.0, voltage_factor=0.5, **kwargs):
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
            mode: string (optional)
                The default value is 'superimposed'.
            time_factor: float (optional)
                The default value is 50.0.
            voltage_factor: float (optional)
                The default value is 0.5.
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
            if mode == 'superimposed':
                x_min = 0
                x_max = nb_samples
                ax.set_xlim(x_min, x_max)
                x = np.arange(0, nb_samples)
                for k in range(0, nb_channels):
                    y = self.first_component.waveforms[k, :]
                    label = "waveform {}".format(k)
                    ax.plot(x, y, label=label, **kwargs)
            else:
                x_min = 0
                x_max = nb_samples
                ax.set_xlim(x_min, x_max)
                x = np.arange(0, nb_samples)
                if mode is not None:
                    v_max = mode
                else:
                    v_max = np.max(np.abs(self.first_component.waveforms))
                if v_max > 0.0:
                    waveforms = 0.5 * self.first_component.waveforms / v_max
                else:
                    waveforms = self.first_component.waveforms
                indices = self.first_component.indices
                for k in range(0, nb_channels):
                    y = waveforms[k, :] + float(indices[k])
                    label = "waveform {}".format(k)
                    ax.plot(x, y, label=label, **kwargs)
        else:
            ax.set_aspect('equal')
            x_min, x_max = probe.x_limits
            y_min, y_max = probe.y_limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            for k, channel in enumerate(self.first_component.indices):
                x_0, y_0 = probe.get_channel_position(channel)
                x = time_factor * np.linspace(-0.5, +0.5, num=nb_samples) + x_0
                y = voltage_factor * self.first_component.waveforms[k, :] + y_0
                label = "waveform {}".format(channel)
                ax.plot(x, y, label=label, **kwargs)
            if with_scale_bars:
                # Add scale bars.
                x_anchor = x_max - 0.1 * (x_max - x_min)
                y_anchor = y_max - 0.1 * (y_max - y_min)
                # # Add time scale bar.
                width = 1  # TODO improve.
                x = [x_anchor, x_anchor - time_factor * float(width)]
                y = [y_anchor, y_anchor]
                ax.plot(x, y, color='black', label="time scale bar")
                ax.text(np.mean(x), np.mean(y), u"{} arb. unit".format(width), fontsize=8,
                        horizontalalignment='center', verticalalignment='bottom')
                # # Add voltage scale bar.
                height = 50  # TODO improve.
                x = [x_anchor, x_anchor]
                y = [y_anchor, y_anchor - voltage_factor * float(height)]
                ax.plot(x, y, color='black', label="voltage scale bar")
                ax.text(np.mean(x), np.mean(y), r"{} $\mu$V".format(height), fontsize=8,
                        horizontalalignment='left', verticalalignment='center')

        if with_xaxis:
            ax.set_xlabel(r"x ($\mu$m)")
        else:
            ax.set_xticklabels([])
        if with_yaxis:
            ax.set_ylabel(r"y ($\mu$m)")
        else:
            ax.set_yticklabels([])
        if title is not None:
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

        if self._compressed:
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
