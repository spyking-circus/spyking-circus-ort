import h5py
import numpy as np
import os
import time

from circusort.io.utils import append_hdf5
from circusort.obj.template import Template, TemplateComponent


class TemplateStore(object):

    def __init__(self, file_name, probe_file=None, mode='r+', compression=None):
        # TODO use compression='gzip' instead.
        # TODO i.e. fix the ValueError: Compression filter "gzip" is unavailable.

        self.file_name = os.path.expanduser(os.path.abspath(file_name))
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
        self.compression = compression
        self._similarities = {}

        self._open(self.mode)
        from circusort.io.probe import load_probe

        if self.mode in ['w']:

            assert probe_file is not None
            self.probe = load_probe(self.probe_file)
            for channel, indices in self.probe.edges.items():
                indices = self.probe.edges[channel]
                self.h5_file.create_dataset('mapping/%d' % channel, data=indices, chunks=True, maxshape=(None,),
                                            compression=self.compression)
                self.mappings[channel] = indices
            self.h5_file.create_dataset('indices', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,),
                                        compression=self.compression)
            self.h5_file.create_dataset('times', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,),
                                        compression=self.compression)
            self.h5_file.create_dataset('channels', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None,),
                                        compression=self.compression)
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

    def __iter__(self):

        for index in self.indices:
            yield self.get(index)

        return

    def __getitem__(self, index):

        template = self.get(index)

        return template

    def __len__(self):

        return self.nb_templates

    def _add_template_channel(self, channel, index):

        if self._channels is None:
            self._channels = {}

        if channel in self._channels:
            self._channels[channel] += [index]
        else:
            self._channels[channel] = [index]

        return

    @property
    def templates_per_channels(self):

        if self._channels is None:
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

    def _add_similarity(self, i, j, value):

        if i not in self._similarities:
            self._similarities[i] = {}
        self._similarities[i][j] = value

        return

    def similarity(self, i, j):

        if i in self._similarities:
            if j in self._similarities[i]:
                return self._similarities[i][j]

        value = self.get(i).first_component.similarity(self.get(j).first_component)
        self._add_similarity(i, j, value)
        self._add_similarity(j, i, value)

        return value

    @property
    def similarities(self):

        res = np.zeros((len(self), len(self)), dtype=np.float32)
        indices = self.indices
        for c1, i in enumerate(indices):
            for c2, j in enumerate(indices):
                res[i, j] = self.similarity(i, j)
                res[j, i] = res[i, j]

        return res

    @property
    def two_components(self):

        if self._2_components is None:
            indices = self.indices
            assert len(indices) > 0
            self._open('r')
            self._2_components = '2' in self.h5_file['waveforms/%d' % indices[0]]
            self._close()

        return self._2_components

    @property
    def is_empty(self):

        return self.nb_templates == 0

    @property
    def temporal_width(self):

        if self._temporal_width is None:

            assert self.nb_templates > 0
            template = self.__getitem__(0)
            self._temporal_width = template.temporal_width

        return self._temporal_width

    def get_putative_merges(self, n_best=None, min_cc=0):

        if n_best is not None:
            n_best = min(n_best + 1, len(self))
        else:
            n_best = len(self)
        ids = []
        ccs = []
        for count, similarity in enumerate(self.similarities):
            idx = np.argsort(similarity)[::-1]
            kidx = np.where(similarity[idx] < min_cc)[0]
            if len(kidx) > 0:
                ids += [self.indices[idx[1:min(kidx[0], n_best)]]]
            else:
                ids += [self.indices[idx[1:n_best]]]
            ccs += [similarity[ids[-1]]]

        return ids, ccs

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
        """Check if a template exists in the store (by index).

        Argument:
            index: integer
                The index for which the existence of a template will be checked.
        Return:
            _: boolean
        """

        return index in self.indices

    def add(self, templates):
        """Add templates to the store.

        Argument:
            templates: list
                A list of templates to add to the store.
        Return:
            indices: list
                A list which contains the indices of the added templates.
        """

        assert self.mode in ['w', 'r+']
        self._open('r+')

        indices = []
        if isinstance(templates, Template):
            templates = [templates]

        for t in templates:

            assert isinstance(t, Template)
            gidx = self.next_index

            self.h5_file.create_dataset('waveforms/%d/1' % gidx, data=t.first_component.waveforms, chunks=True,
                                        compression=self.compression)
            self.h5_file.create_dataset('amplitudes/%d' % gidx, data=t.amplitudes,
                                        compression=self.compression)

            if t.compressed:
                self.h5_file.create_dataset('compressed/%d' % gidx, data=t.indices,
                                            compression=self.compression)

            if self._temporal_width is None:
                self._temporal_width = t.temporal_width

            if t.second_component is not None:
                self._2_components = True
                self.h5_file.create_dataset('waveforms/%d/2' % gidx, data=t.second_component.waveforms, chunks=True,
                                            compression=self.compression)

            self._add_template_channel(t.channel, gidx)

            append_hdf5(self.h5_file['times'], np.array([t.creation_time], dtype=np.int32))
            append_hdf5(self.h5_file['indices'], np.array([gidx], dtype=np.int32))
            append_hdf5(self.h5_file['channels'], np.array([t.channel], dtype=np.int32))
            indices += [gidx]

        self._close()

        return indices

    def get(self, elements=None):
        """Get templates by indices from the store.

        Argument:
            elements: none | integer | iterable (optional)
                The indices of the templates to get from the store. If None then all the templates are selected.
                The default value is None.
        Return:
            result: list
                A list which contains the selected templates.
        """

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

            if 'compressed' in self.h5_file.keys():
                mapping = self.h5_file['compressed/%d' % index][:]
                compressed = True
            else:
                mapping = self.mappings[channel]
                compressed = False

            first_component = TemplateComponent(waveforms, mapping, self.nb_channels, amplitudes)

            if self.two_components:
                waveforms2 = self.h5_file['waveforms/%d/2' % index][:]
                second_component = TemplateComponent(waveforms2, mapping, self.nb_channels)
            else:
                second_component = None

            template = Template(first_component, channel, second_component, creation_time=int(times[idx_pos]))
            template.compressed = compressed
            result += [template]

        self._close()

        if singleton and len(result) == 1:
            result = result[0]

        return result

    def remove(self, indices):
        """Delete templates by indices in the store."""

        if not np.iterable(indices):
            indices = [indices]

        self._open('r+')

        for index in indices:
            assert index in indices
            self.h5_file.pop('waveforms/%d' % index)
            self.h5_file.pop('amplitudes/%d' % index)
            if 'compressed' in self.h5_file.keys():
                self.h5_file.pop('compressed/%d' % index)
            channels = self.h5_file.pop('channels')
            times = self.h5_file.pop('times')
            indices = self.h5_file.pop('indices')
            to_remove = np.where(indices == index)[0]
            self.h5_file['channels'] = np.delete(channels, to_remove)
            self.h5_file['indices'] = np.delete(indices, to_remove)
            self.h5_file['times'] = np.delete(times, to_remove)

        self._close()

        return

    def _open(self, mode='r+'):
        """Open the file which contains the store.

        Argument:
            mode: string
                The opening mode to use to open the file.
                The default value is 'r+'.

        See also:
            h5py.File for a detailed documentation of the mode argument.
        """

        while self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.file_name, mode=mode, swmr=True)
            except IOError:
                duration = 50e-3  # s
                time.sleep(duration)

        return

    def _close(self):
        """Close the file which contains the store."""

        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()
            self.h5_file = None

        return

    def __del__(self):

        self._close()
