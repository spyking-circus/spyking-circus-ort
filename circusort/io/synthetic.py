import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import os
import numpy
from circusort.io.utils import append_hdf5

class SyntheticStore(object):

    variables = ['x', 'y', 'z', 'r', 'e', 'spike_times', 'waveform/x', 'waveform/y']

    def __init__(self, file_name, mode='w'):

        self.file_nam= os.path.abspath(file_name)
        self.initialized = {}
        self.mode = mode
        self.is_created = False

    def add(self, data):

        cell_id = data.pop('cell_id')
        if not self.initialized.has_key(cell_id):
            self.initialized[cell_id] = []

        if not self.is_created:
            self.h5_file = h5py.File(self.file_name, self.mode)
        else:
            self.h5_file = h5py.File(self.file_name, 'r+')

        if not str(cell_id) in self.h5_file.keys():
            self.h5_file.create_group('{}'.format(cell_id))

        for key in data.keys():
            if key in self.initialized[cell_id]:
                append_hdf5(self.h5_file['{c}/{d}'.format(c=cell_id, d=key)], data[key])
            else:
                self.h5_file.create_dataset('{c}/{d}'.format(c=cell_id, d=key), data=data[key], chunks=True, maxshape=(None,))
                self.initialized[cell_id] += [key]
                self.is_created = True

        self.h5_file.close()


    @property
    def nb_cells(self):
        self.h5_file = h5py.File(self.file_name, 'r')
        res = len(self.h5_file.keys())
        self.h5_file.close()
        return res

    def get(self, indices=None, variables=None):

        result = {}

        self.h5_file = h5py.File(self.file_name, 'r')

        if indices is None:
            indices = self.h5_file.keys()
        elif not numpy.iterable(indices):
            indices = [indices]

        if variables is None:
            variables = self.variables
        elif not isinstance(variables, list):
            variables = [variables]

        for cell_id in indices:
            result[cell_id] = {}
            for key in variables:
                result[cell_id][key] = self.h5_file['{c}/{d}'.format(c=cell_id, d=key)][:]

        self.h5_file.close()

        return result

    def close(self):
        try:
            self.h5_file.close()
        except Exception:
            pass
