# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
import scipy

from scipy.sparse import csc_matrix, csr_matrix, hstack
import numpy as np
from circusort.io.probe import load_probe
from circusort.io.utils import append_hdf5
from circusort.io import generate_probe


def generate_waveform(width=5.0e-3, amplitude=80.0, sampling_rate=20e+3):
    """Generate a waveform.

    Parameters:
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        amplitude: float (optional)
            Voltage amplitude [µV]. The default value is 80.0.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.

    Return:
        waveform: np.array
            Generated waveform.
    """

    i_start = - int(width * sampling_rate / 2.0)
    i_stop = + int(width * sampling_rate / 2.0)
    steps = np.arange(i_start, i_stop + 1)
    times = steps.astype('float32') / sampling_rate
    waveform = - np.cos(times / (width / 2.0) * (1.5 * np.pi))
    gaussian = np.exp(- (times / (width / 4.0)) ** 2.0)
    waveform = np.multiply(waveform, gaussian)
    if np.min(waveform) < - sys.float_info.epsilon:
        waveform /= np.abs(np.min(waveform))
        waveform *= amplitude

    return waveform


def generate_positions(nb_cells, probe):
    """Generate the positions of the cells.

    Parameters:
        nb_cells: integer
            The number of cells.
        probe: circusort.obj.Probe
            The probe.
    """

    fov = probe.field_of_view
    x_min = fov['x_min']
    x_max = fov['x_max']
    y_min = fov['y_min']
    y_max = fov['y_max']

    positions = []
    for _ in range(0, nb_cells):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        position = (x, y)
        positions.append(position)

    return positions


def generate_amplitudes(nb_cells):
    """Generate the amplitudes of the cells."""

    amplitudes = np.random.normal(80.0, scale=2.5, size=nb_cells)

    return amplitudes


def generate_templates(nb_templates=3, probe=None,
                       positions=None, max_amps=None,
                       radius=None, width=5.0e-3, sampling_rate=20e+3):
    """Generate templates.

    Parameters:
        nb_templates: none | integer (optional)
            Number of templates to generate. The default value is 3.
        probe: none | circusort.io.Probe
            Description of the probe (e.g. spatial layout). The default value is None.
        positions: none | list (optional)
            Coordinates of position of the centers (spatially) of the templates [µm]. The default value is None.
        max_amps: none | float (optional)
            Maximum amplitudes of the templates [µV]. The default value is None.
        radius: none | float (optional)
            Radius of the signal horizon [µm]. The default value is None.
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.

    Return:
        templates: dictionary
            Generated dictionary of templates.
    """

    if probe is None:
        probe = generate_probe()

    if positions is None:
        positions = generate_positions(nb_templates, probe)

    if max_amps is None:
        max_amps = generate_amplitudes(nb_templates)

    if radius is None:
        radius = probe.radius

    nb_samples = 1 + 2 * int(width * sampling_rate / 2.0)

    templates = {}
    for k in range(0, nb_templates):
        # Get distance to the nearest electrode.
        position = positions[k]
        nearest_electrode_distance = probe.get_nearest_electrode_distance(position)
        # Get channels before signal horizon.
        x, y = position
        channels, distances = probe.get_channels_around(x, y, radius + nearest_electrode_distance)
        # Declare waveforms.
        nb_electrodes = len(channels)
        shape = (nb_electrodes, nb_samples)
        waveforms = np.zeros(shape, dtype=np.float)
        # Initialize waveforms.
        amplitude = max_amps[k]
        waveform = generate_waveform(width=width, amplitude=amplitude, sampling_rate=sampling_rate)
        for i, distance in enumerate(distances):
            gain = (1.0 + distance / 40.0) ** -2.0
            waveforms[i, :] = gain * waveform
        # Store template.
        template = (channels, waveforms)
        templates[k] = template

    return templates


def save_templates(directory, templates):
    """Save templates.

    Parameters:
        directory: string
            Directory in which to save the templates.
        templates: dictionary
            Dictionary of templates.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for k, template in templates.iteritems():
        channels, waveforms = template
        filename = "{}.h5".format(k)
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='w')
        f.create_dataset('channels', shape=channels.shape, dtype=channels.dtype, data=channels)
        f.create_dataset('waveforms', shape=waveforms.shape, dtype=waveforms.dtype, data=waveforms)
        f.close()

    return


def list_templates(directory):
    """List template paths contained in the specified directory.

    Parameter:
        directory: string
            Directory from which to list the templates.

    Return:
        paths: list
            List of template paths found in the specified directory.
    """

    if not os.path.isdir(directory):
        message = "No such template directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()
    paths = [os.path.join(directory, filename) for filename in filenames]

    return paths


def load_template(path):
    """Load template.

    Parameter:
        path: string
            Path from which to load the template.

    Return:
        template: tuple
            Template. The first element of the tuple contains the support of the template (i.e. channels). The second
            element contains the corresponding waveforms.
    """

    f = h5py.File(path, mode='r')
    channels = f.get('channels').value
    waveforms = f.get('waveforms').value
    f.close()
    template = (channels, waveforms)

    return template


def load_templates(directory):
    """Load templates.

    Parameter:
        directory: string
            Directory from which to load the templates.

    Return:
        templates: dictionary
            Dictionary of templates.
    """

    paths = list_templates(directory)

    templates = {
        k: load_template(path)
        for k, path in enumerate(paths)
    }

    return templates


class TemplateComponent(object):

    def __init__(self, waveforms, indices, nb_channels, amplitudes=None):
        self.waveforms   = waveforms
        self.amplitudes  = amplitudes
        self.indices     = indices
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

    def __init__(self, first_component, channel, second_component=False, creation_time=0):

        self.first_component  = first_component
        assert self.first_component.amplitudes is not None
        self.channel          = channel
        self.second_component = second_component
        self.creation_time    = creation_time

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

        self.file_name       = os.path.abspath(file_name)
        self.mode            = mode
        self._index          = -1
        self.mappings        = {}
        self._2_components   = False
        self._temporal_width = None
        self.h5_file         = None
        
        self._open(self.mode)

        if self.mode in ['w']:
            
            assert probe_file is not None
            self.probe        = load_probe(probe_file)
            for channel, indices in self.probe.edges.items():
                indices = self.probe.edges[channel]
                self.h5_file.create_dataset('mapping/%d' %channel, data=indices, chunks=True, maxshape=(None, ))
                self.mappings[channel] = indices
            self.h5_file.create_dataset('indices', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None, ))
            self.h5_file.create_dataset('times', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None, ))
            self.h5_file.create_dataset('channels', data=np.zeros(0, dtype=np.int32), chunks=True, maxshape=(None, ))

        elif self.mode in ['r', 'r+']:
            indices       = self.h5_file['indices'][:]
            if len(indices) > 0:
                self._index = indices.max()
                
            self.mappings = {}
            for key, value in self.h5_file['mapping'].items():
                self.mappings[int(key)] = value[:]
            
            if self._index >= 0:
                self._2_components = '2' in self.h5_file['waveforms/%d' %self._index]
        
        self.nb_channels = len(self.mappings)
        self._close()
        
    def __iter__(self, index):
        for i in self.indices:
            yield self[i]

    def __getitem__(self, index):
        return self.get(self.indices[index])

    def __len__(self):
        return self.nb_templates

    @property
    def indices(self):
        self._open(mode='r')
        data = self.h5_file['indices'][:]
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
                self._2_components = '2' in self.h5_file['waveforms/%d' %indices[0]]
                self._close()
            return self._2_components

    @property
    def temporal_width(self):
        if self._temporal_width is not None:
            return self._temporal_width
        else:
            assert self.nb_templates > 0
            template = self.__getitem__(0)
            self._temporal_width = template[template.keys()[0]].temporal_width
            return self._temporal_width

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

            self.h5_file.create_dataset('waveforms/%d/1' %gidx, data=t.first_component.waveforms, chunks=True)
            self.h5_file.create_dataset('amplitudes/%d' %gidx, data=t.amplitudes)

            if self._temporal_width is None:
                self._temporal_width = t.temporal_width

            if t.second_component is not None:
                self._2_components = True
                self.h5_file.create_dataset('waveforms/%d/2' %gidx, data=t.second_component.waveforms, chunks=True)

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
            elements = [elements]

        result   = {}
        indices  = self.h5_file['indices'][:]
        channels = self.h5_file['channels'][:]
        times    = self.h5_file['times'][:]

        for index in elements:
            
            assert index in indices

            idx_pos       = np.where(indices == index)[0]
            result[index] = {}

            waveforms = self.h5_file['waveforms/%d/1' %index][:]
            if self.two_components:
                waveforms2 = self.h5_file['waveforms/%d/2' %index][:]

            amplitudes = self.h5_file['amplitudes/%d' %index][:]
            second_component = None

            channel         = channels[idx_pos][0]
            first_component = TemplateComponent(waveforms, self.mappings[channel], self.nb_channels, amplitudes)
            if self.two_components:
                second_component = TemplateComponent(waveforms2, self.mappings[channel], self.nb_channels)

            result[index] = Template(first_component, channel, second_component, creation_time=int(times[idx_pos]))

        self._close()

        return result
        
    def remove(self, indices):

        if not np.iterable(indices):
            indices = [indices]

        self._open('r+')

        for index in indices:

            assert index in indices
            self.h5_file.pop('waveforms/%d' %index)
            self.h5_file.pop('amplitudes/%d' %index)
            channels = self.h5_file.pop('channels')
            times    = self.h5_file.pop('times')
            indices  = self.h5_file.pop('indices')
            to_remove = np.where(indices == index)[0]
            self.h5_file['channels'] = np.delete(channels, to_remove)
            self.h5_file['indices']  = np.delete(indices, to_remove)
            self.h5_file['times']    = np.delete(times, to_remove)

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

