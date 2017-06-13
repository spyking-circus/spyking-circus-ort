import h5py
import os
import scipy.sparse
import numpy
from circusort.io.utils import append_hdf5

class TemplateStore(object):

    def __init__(self, file_name, mode='w', two_components=False):

        self.file_name      = os.path.abspath(file_name)
        self.initialized    = False
        self.mode           = mode
        self.two_components = two_components

    def add(self, data):

        norms      = data['norms']
        templates  = data['templates']
        amplitudes = data['amplitudes']
        channels   = data['channels']
        if self.two_components:
            norms2      = data['norms2']
            templates2  = data['templates2']

        if not self.initialized:
            self.h5_file = h5py.File(self.file_name, self.mode)

            self.h5_file.create_dataset('norms', data=norms, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('channels', data=channels, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('amplitudes', data=amplitudes, chunks=True, maxshape=(None, 2))

            if self.two_components:
                self.h5_file.create_dataset('norms2', data=norms2, chunks=True, maxshape=(None,))
                self.h5_file.create_dataset('data2', data=templates2.data, chunks=True, maxshape=(None, ))

            self.h5_file.create_dataset('data', data=templates.data, chunks=True, maxshape=(None,))
            self.h5_file.create_dataset('indptr', data=templates.indptr, chunks=True, maxshape=(None, ))
            self.h5_file.create_dataset('indices', data=templates.indices, chunks=True, maxshape=(None, ))
            self.h5_file.create_dataset('shape', data=templates.shape, chunks=True)
            self.initialized = True
            self.h5_file.close()

        else:
    
            self.h5_file = h5py.File(self.file_name, 'r+')
            append_hdf5(self.h5_file['norms'], norms)
            append_hdf5(self.h5_file['amplitudes'], amplitudes)
            append_hdf5(self.h5_file['channels'], channels)
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

    def get(self, indices=None):

        result = {}

        self.h5_file = h5py.File(self.file_name, 'r')

        if indices is None:
            result['norms']      = self.h5_file['norms'][:]
            result['amplitudes'] = self.h5_file['amplitudes'][:]
            result['channels']   = self.h5_file['channels'][:]
            result['templates']  = scipy.sparse.csc_matrix((self.h5_file['data'][:], self.h5_file['indices'][:], self.h5_file['indptr'][:]),
                        shape=self.h5_file['shape'][:])
            if self.two_components:
                result['norms2']     = self.h5_file['norms2'][:]
                result['templates2'] = scipy.sparse.csc_matrix((self.h5_file['data'][:], self.h5_file['indices'][:], self.h5_file['indptr'][:]),
                        shape=self.h5_file['shape'][:])
        else:
            result['norms']      = self.h5_file['norms'][indices]
            result['amplitudes'] = self.h5_file['amplitudes'][indices, :]
            result['channels']   = self.h5_file['channels'][indices]

            myshape = self.h5_file['shape'][0]
            indptr  = self.h5_file['indptr'][:]

            result['templates'] = scipy.sparse.csc_matrix((myshape, 0), dtype=numpy.float32)

            if self.two_components:
                result['norms2']     = self.h5_file['norms2'][indices]
                result['templates2'] = scipy.sparse.csc_matrix((myshape, 0), dtype=numpy.float32)

            for item in indices:
                mask    = numpy.zeros(len(self.h5_file['data']), dtype=numpy.bool)
                mask[indptr[item]:indptr[item+1]] = 1
                n_data  = indptr[item+1] - indptr[item]
                temp    = scipy.sparse.csc_matrix((self.h5_file['data'][mask], (self.h5_file['indices'][mask], numpy.zeros(n_data))), shape=(myshape, 1))    
                result['templates']  = scipy.sparse.hstack((result['templates'], temp), 'csc')
 
                if self.two_components:
                    temp    = scipy.sparse.csc_matrix((self.h5_file['data2'][mask], (self.h5_file['indices'][mask], numpy.zeros(n_data))), shape=(myshape, 1))    
                    result['templates2'] = scipy.sparse.hstack((result['templates2'], temp), 'csc')

        self.h5_file.close()

        return result
        
    def remove(self, index):
        pass

    def close(self):
        try:
            self.h5_file.close()
        except Exception:
            pass