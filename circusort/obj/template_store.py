import h5py
import numpy as np
import os

from scipy.sparse import csc_matrix, hstack

from circusort.io.utils import append_hdf5


class TemplateStore(object):
    variables = [
        'norms',
        'templates',
        'amplitudes',
        'channels',
        'times',
    ]

    def __init__(self, file_name, initialized=False, mode='w',
                 two_components=False, N_t=None):

        self.file_name = os.path.abspath(file_name)
        self.initialized = initialized
        self.mode = mode
        self.two_components = two_components
        self._spike_width = N_t

        if self.two_components:
            self.variables += ['norms2', 'templates2']
        self.h5_file = None

    @property
    def width(self):

        self.h5_file = h5py.File(self.file_name, 'r')
        res = self.h5_file['N_t'][0]
        self.h5_file.close()

        return res

    def add(self, data):

        norms = data['norms']
        templates = data['templates']
        amplitudes = data['amplitudes']
        channels = data['channels']
        times = data['times']

        if self.two_components:
            norms2 = data['norms2']
            templates2 = data['templates2']

        if not self.initialized:
            self.h5_file = h5py.File(self.file_name, self.mode)

            self.h5_file.create_dataset('norms', data=norms, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('channels', data=channels, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('amplitudes', data=amplitudes, chunks=True, maxshape=(None, 2))
            self.h5_file.create_dataset('times', data=times, chunks=True, maxshape=(None,))

            if self.two_components:
                self.h5_file.create_dataset('norms2', data=norms2, chunks=True, maxshape=(None,))
                self.h5_file.create_dataset('data2', data=templates2.data, chunks=True, maxshape=(None,))

            self.h5_file.create_dataset('data', data=templates.data, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('indptr', data=templates.indptr, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('indices', data=templates.indices, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('shape', data=templates.shape, chunks=True)

            if self._spike_width is not None:
                self.h5_file.create_dataset('N_t', data=np.array([self._spike_width]))

            self.initialized = True
            self.h5_file.close()

        else:

            self.h5_file = h5py.File(self.file_name, 'r+')
            append_hdf5(self.h5_file['norms'], norms)
            append_hdf5(self.h5_file['amplitudes'], amplitudes)
            append_hdf5(self.h5_file['channels'], channels)
            append_hdf5(self.h5_file['times'], times)
            self.h5_file['shape'][1] = len(self.h5_file['norms'])
            if self.two_components:
                append_hdf5(self.h5_file['norms2'], norms2)
                append_hdf5(self.h5_file['data2'], templates2.data)
            append_hdf5(self.h5_file['data'], templates.data)
            append_hdf5(self.h5_file['indices'], templates.indices)
            to_write = templates.indptr[1:] + self.h5_file['indptr'][-1]
            append_hdf5(self.h5_file['indptr'], to_write)
            self.h5_file.close()

    @property
    def nb_templates(self):
        self.h5_file = h5py.File(self.file_name, 'r')
        res = len(self.h5_file['amplitudes'])
        self.h5_file.close()
        return res

    @property
    def nnz(self):
        self.h5_file = h5py.File(self.file_name, 'r')
        res = len(self.h5_file['data'])
        self.h5_file.close()
        return res

    def get(self, indices=None, variables=None):

        result = {}

        if variables is None:
            variables = self.variables
        elif not isinstance(variables, list):
            variables = [variables]

        self.h5_file = h5py.File(self.file_name, 'r')

        if indices is None:

            for var in ['norms', 'amplitudes', 'channels', 'times', 'norms2']:
                if var in variables:
                    result[var] = self.h5_file[var][:]

            if 'templates' in variables:
                result['templates'] = csc_matrix((self.h5_file['data'][:],
                                                  self.h5_file['indices'][:],
                                                  self.h5_file['indptr'][:]),
                                                 shape=self.h5_file['shape'][:])

            if 'templates2' in variables:
                result['templates2'] = csc_matrix((self.h5_file['data'][:],
                                                   self.h5_file['indices'][:],
                                                   self.h5_file['indptr'][:]),
                                                  shape=self.h5_file['shape'][:])
        else:

            if not np.iterable(indices):
                indices = [indices]

            for var in ['norms', 'channels', 'times', 'norms2']:
                if var in variables:
                    result[var] = self.h5_file[var][indices]

            if 'amplitudes' in variables:
                result['amplitudes'] = self.h5_file['amplitudes'][indices, :]

            load_t1 = 'templates' in variables
            load_t2 = 'templates2' in variables
            are_templates_to_load = load_t1 or load_t2

            if are_templates_to_load:
                myshape = self.h5_file['shape'][0]
                indptr = self.h5_file['indptr'][:]

            if load_t1:
                result['templates'] = csc_matrix((myshape, 0), dtype=np.float32)

            if load_t2:
                result['templates2'] = csc_matrix((myshape, 0), dtype=np.float32)

            if are_templates_to_load:
                for item in indices:
                    mask = np.zeros(len(self.h5_file['data']), dtype=np.bool)
                    mask[indptr[item]:indptr[item + 1]] = 1
                    n_data = indptr[item + 1] - indptr[item]
                    if load_t1:
                        temp = csc_matrix((self.h5_file['data'][mask],
                                           (self.h5_file['indices'][mask], np.zeros(n_data))),
                                          shape=(myshape, 1))
                        result['templates'] = hstack((result['templates'], temp), 'csc')

                    if load_t2:
                        temp = csc_matrix((self.h5_file['data2'][mask],
                                           (self.h5_file['indices'][mask], np.zeros(n_data))),
                                          shape=(myshape, 1))
                        result['templates2'] = hstack((result['templates2'], temp), 'csc')

        self.h5_file.close()

        return result

    def remove(self, index):

        raise NotImplementedError()

    def close(self):

        try:
            self.h5_file.close()
        except Exception:
            pass

        return
