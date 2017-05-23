import h5py
import os

class TemplateStore(object):

    def __init__(self, file_name):

        self.file_name = os.path.abspath(file_name)
        self.h5_file   = h5py.File(self.file_name, 'w')
        self.templates = None

    def add(templates, norms, amplitudes, templates2=None, norms2=None):

        self.h5_file.create_dataset('norms', data=norms, chunks=True)
        self.h5_file.create_dataset('amplitudes', data=amplitudes, chunks=True)

        if norms2 is not None:
            self.h5_file.create_dataset('norms2', data=norms2, chunks=True)
            self.h5_file.create_dataset('data2', data=templates2.data)

        self.h5_file.create_dataset('data', data=templates.data)
        self.h5_file.create_dataset('indptr', data=templates.indptr)
        self.h5_file.create_dataset('indices', data=templates.indices)
        self.h5_file.create_dataset('shape', data=templates.shape)

    def get(self, two_component=True):

        templates = scipy.sparse.csc_matrix((self.h5_file['data'][:], self.h5_file['indices'][:], self.h5_file['indptr'][:]),
                          shape=self.h5_file['shape'][:])

        if two_component:
            templates2 = scipy.sparse.csc_matrix((self.h5_file['data2'][:], self.h5_file['indices'][:], self.h5_file['indptr'][:]),
                          shape=self.h5_file['shape'][:])

            return templates, self.h5_file['norms'][:], self.h5_file['amplitudes'][:], templates2, self.h5_file['norms2'][:]

        else:

            return templates, self.h5_file['norms'][:], self.h5_file['amplitudes'][:]
    
    def remove(self, index):
        pass

    def get_overlaps(self):
        pass

    def __del__(self):
        self.h5_file.close()