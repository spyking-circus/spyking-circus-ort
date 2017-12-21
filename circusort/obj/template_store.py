import h5py
import numpy as np
import os
import scipy

from scipy.sparse import csc_matrix

from circusort.io.probe import load_probe
from circusort.io.utils import append_hdf5


class TemplateComponent(object):

    def __init__(self, waveforms, indices, nb_channels, amplitudes=None):
        """Initialization.

        Parameters:
            waveforms
            indices
            nb_channels
            amplitudes (optional)
        """

        self.waveforms = waveforms
        self.amplitudes = amplitudes
        self.indices = indices
        self.nb_channels = nb_channels

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


class Template(object):

    def __init__(self, first_component, channel, second_component=None, creation_time=0):

        self.first_component = first_component
        assert self.first_component.amplitudes is not None
        self.channel = channel
        self.second_component = second_component
        self.creation_time = creation_time

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


class TemplateStore(object):

    def __init__(self, file_name, probe_file=None, mode='r+'):

        self.file_name = os.path.abspath(file_name)
        self.probe_file = probe_file
        self.mode = mode
        self._index = -1
        self.mappings = {}
        self._2_components = False
        self._temporal_width = None
        self.h5_file = None
        self._first_creation = None
        self._last_creation = None
        self._channels = None

        self._open(self.mode)

        if self.mode in ['w']:

            assert probe_file is not None
            self.probe = load_probe(self.probe_file)
            for channel, indices in self.probe.edges.items():
                indices = self.probe.edges[channel]
                self.h5_file.create_dataset('mapping/%d' % channel, data=indices, chunks=True, maxshape=(None,))
                self.mappings[channel] = indices
            self.h5_file.create_dataset('indices', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('times', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('channels', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,))
            self.h5_file.attrs['probe_file'] = os.path.abspath(os.path.expanduser(probe_file))

        elif self.mode in ['r', 'r+']:
            indices = self.h5_file['indices'][:]
            if len(indices) > 0:
                self._index = indices.max()

            self.mappings = {}
            for key, value in self.h5_file['mapping'].items():
                self.mappings[int(key)] = value[:]

            if self._index >= 0:
                self._2_components = '2' in self.h5_file['waveforms/%d' % self._index]

            self.probe_file = self.h5_file.attrs['probe_file']
            self.probe = load_probe(self.probe_file)

        self.nb_channels = len(self.mappings)
        self._close()

    def __str__(self):
        string = """
        Template store with {} templates
        two_components : {}
        temporal_width : {}
        probe_file     : {}
        """.format(len(self.indices), self.two_components, self.temporal_width, self.probe_file)
        return string

    def __iter__(self, index):
        for i in self.indices:
            yield self[i]

    def __getitem__(self, index):
        return self.get(self.indices[index])

    def __len__(self):
        return self.nb_templates

    def _add_template_channel(self, channel, index):
        if self._channels is None:
            self._channels = {}

        if channel in self._channels:
            self._channels[channel] += [index]
        else:
            self._channels[channel] = [index]

    @property
    def templates_per_channels(self):
        if self._channels is not None:
            return self._channels
        else:
            for channel, index in zip(self.channels, self.indices):
                self._add_template_channel(channel, index)
            return self._channels

    @property
    def first_creation(self):
        return self.times.min()

    @property
    def last_creation(self):
        return self.times.max()

    @property
    def indices(self):
        self._open(mode='r')
        data = self.h5_file['indices'][:]
        self._close()
        return data

    @property
    def channels(self):
        self._open(mode='r')
        data = self.h5_file['channels'][:]
        self._close()
        return data

    @property
    def times(self):
        self._open(mode='r')
        data = self.h5_file['times'][:]
        self._close()
        return data

    @property
    def nb_templates(self):
        return len(self.indices)

    @property
    def next_index(self):
        self._index += 1
        return self._index

    @property
    def two_components(self):
        if self._2_components is not None:
            return self._2_components
        else:
            indices = self.indices
            if len(indices) > 0:
                self._open('r')
                self._2_components = '2' in self.h5_file['waveforms/%d' % indices[0]]
                self._close()
            return self._2_components

    @property
    def temporal_width(self):
        if self._temporal_width is not None:
            return self._temporal_width
        else:
            assert self.nb_templates > 0
            template = self.__getitem__(0)
            self._temporal_width = template.temporal_width
            return self._temporal_width

    def slice_templates_by_channel(self, channels):
        if not np.iterable(channels):
            channels = [channels]
        result = []
        for t in self.get():
            if t.channel in [channels]:
                result += [t]
        return result

    def slice_templates_by_creation_time(self, start=0, stop=np.inf):
        times = self.times
        result = np.where((times > start) & (times <= stop))[0]
        return self.get(result)

    def is_in_store(self, index):
        if index in self.indices:
            return True
        else:
            return False

    def add(self, templates):

        assert self.mode in ['w', 'r+']
        self._open('r+')

        indices = []
        if not np.iterable(templates):
            templates = [templates]

        for t in templates:

            assert isinstance(t, Template)
            gidx = self.next_index

            self.h5_file.create_dataset('waveforms/%d/1' % gidx, data=t.first_component.waveforms, chunks=True)
            self.h5_file.create_dataset('amplitudes/%d' % gidx, data=t.amplitudes)

            if self._temporal_width is None:
                self._temporal_width = t.temporal_width

            if t.second_component is not None:
                self._2_components = True
                self.h5_file.create_dataset('waveforms/%d/2' % gidx, data=t.second_component.waveforms, chunks=True)

            self._add_template_channel(t.channel, gidx)

            append_hdf5(self.h5_file['times'], np.array([t.creation_time], dtype=np.int32))
            append_hdf5(self.h5_file['indices'], np.array([gidx], dtype=np.int32))
            append_hdf5(self.h5_file['channels'], np.array([t.channel], dtype=np.int32))
            indices += [gidx]

        self._close()
        return indices

    def get(self, elements=None):

        self._open('r')

        if elements is None:
            elements = self.h5_file['indices'][:]

        if not np.iterable(elements):
            singleton = True
            elements = [elements]
        else:
            singleton = False

        result = []
        indices = self.h5_file['indices'][:]
        channels = self.h5_file['channels'][:]
        times = self.h5_file['times'][:]

        for index in elements:

            assert index in indices

            idx_pos = np.where(indices == index)[0]

            waveforms = self.h5_file['waveforms/%d/1' % index][:]
            amplitudes = self.h5_file['amplitudes/%d' % index][:]
            channel = channels[idx_pos][0]
            first_component = TemplateComponent(waveforms, self.mappings[channel], self.nb_channels, amplitudes)
            if self.two_components:
                waveforms2 = self.h5_file['waveforms/%d/2' % index][:]
                second_component = TemplateComponent(waveforms2, self.mappings[channel], self.nb_channels)
            else:
                second_component = None

            result += [Template(first_component, channel, second_component, creation_time=int(times[idx_pos]))]

        self._close()
        if singleton and len(result) == 1:
            result = result[0]

        return result

    def remove(self, indices):

        if not np.iterable(indices):
            indices = [indices]

        self._open('r+')

        for index in indices:
            assert index in indices
            self.h5_file.pop('waveforms/%d' % index)
            self.h5_file.pop('amplitudes/%d' % index)
            channels = self.h5_file.pop('channels')
            times = self.h5_file.pop('times')
            indices = self.h5_file.pop('indices')
            to_remove = np.where(indices == index)[0]
            self.h5_file['channels'] = np.delete(channels, to_remove)
            self.h5_file['indices'] = np.delete(indices, to_remove)
            self.h5_file['times'] = np.delete(times, to_remove)

        self._close()

    def _open(self, mode='r+'):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_name, mode)

    def _close(self):
        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        self._close()
