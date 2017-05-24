import h5py
import os
import scipy.sparse
import numpy

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
            nb_template  = len(norms)
            nb_data      = len(templates.data)
            nb_indptr    = len(templates.indptr)

            self.h5_file = h5py.File(self.file_name, self.mode)


            self.h5_file.create_dataset('norms', data=norms.reshape(nb_template, 1), chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset('channels', data=channels.reshape(nb_template, 1), chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset('amplitudes', data=amplitudes, chunks=True, maxshape=(None, 2))

            if self.two_components:
                self.h5_file.create_dataset('norms2', data=norms2.reshape(nb_template, 1), chunks=True, maxshape=(None, 1))
                self.h5_file.create_dataset('data2', data=templates2.data.reshape(nb_data, 1), chunks=True, maxshape=(None, 1))

            self.h5_file.create_dataset('data', data=templates.data.reshape(nb_data, 1), chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset('indptr', data=templates.indptr.reshape(nb_indptr, 1), chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset('indices', data=templates.indices.reshape(nb_data, 1), chunks=True, maxshape=(None, 1))
            self.h5_file.create_dataset('shape', data=templates.shape, chunks=True)
            self.initialized = True
            self.h5_file.close()

        else:
            
            self.h5_file = h5py.File(self.file_name, 'r+')

            nb_templates = self.nb_templates
            nb_new       = len(amplitudes)
            new_shape_1  = nb_templates + nb_new

            nb_data      = self._nb_nnz
            nb_new       = len(templates.data)
            new_shape_2  = nb_data + nb_new

            nb_indptr    = self._nb_indptr
            nb_new       = len(templates.indptr)
            new_shape_3  = nb_indptr + nb_new - 1

            self.h5_file['norms'].resize((new_shape_1, 1))
            self.h5_file['channels'].resize((new_shape_1, 1))
            self.h5_file['amplitudes'].resize((new_shape_1, 2))
            self.h5_file['amplitudes'][nb_templates:, :] = amplitudes
            self.h5_file['norms'][nb_templates:, 0]      = norms
            self.h5_file['channels'][nb_templates:, 0]   = channels
            self.h5_file['shape'][1]                     = new_shape_1

            if self.two_components:
                self.h5_file['norms2'].resize((new_shape_1, 1))
                self.h5_file['norms2'][nb_templates:, 0] = norms2
                self.h5_file['data2'].resize((new_shape_2, 1))
                self.h5_file['data2'][nb_data:, 0] = templates2.data

            self.h5_file['data'].resize((new_shape_2, 1))
            self.h5_file['data'][nb_data:, 0] = templates.data

            self.h5_file['indices'].resize((new_shape_2, 1))
            self.h5_file['indices'][nb_data:, 0] = templates.indices

            to_write = templates.indptr[1:] + self.h5_file['indptr'][-1, 0]
            self.h5_file['indptr'].resize((new_shape_3, 1))
            self.h5_file['indptr'][nb_indptr:, 0] = to_write
            self.h5_file.close()

    @property
    def nb_templates(self):
        return len(self.h5_file['amplitudes'])

    @property
    def _nb_nnz(self):
        return len(self.h5_file['data'])

    @property
    def _nb_indptr(self):
        return len(self.h5_file['indptr'])

    @property
    def info(self):
        return self._nb_nnz, self._nb_indptr

    def get(self, indices=None):

        result = {}

        self.h5_file = h5py.File(self.file_name, 'r')

        if indices is None:
            result['norms']      = self.h5_file['norms'][:, 0]
            result['amplitudes'] = self.h5_file['amplitudes'][:, :]
            result['channels']   = self.h5_file['channels'][:, 0]
            result['templates']  = scipy.sparse.csc_matrix((self.h5_file['data'][:, 0], self.h5_file['indices'][:, 0], self.h5_file['indptr'][:, 0]),
                        shape=self.h5_file['shape'][:])
            if self.two_components:
                result['norms2']     = self.h5_file['norms2'][:]
                result['templates2'] = scipy.sparse.csc_matrix((self.h5_file['data'][:, 0], self.h5_file['indices'][:, 0], self.h5_file['indptr'][:, 0]),
                        shape=self.h5_file['shape'][:])
        else:
            result['norms']      = self.h5_file['norms'][indices, 0]
            result['amplitudes'] = self.h5_file['amplitudes'][indices, :]
            result['channels']   = self.h5_file['channels'][indices, 0]

            myshape = self.h5_file['shape'][0]
            indptr  = self.h5_file['indptr'][:, 0]

            result['templates'] = scipy.sparse.csc_matrix((myshape, 0), dtype=numpy.float32)

            for item in indices:
                myslice = numpy.arange(indptr[item], indptr[item+1])
                n_data  = len(myslice)
                temp    = scipy.sparse.csc_matrix((self.h5_file['data'][myslice, 0], (self.h5_file['indices'][myslice, 0], numpy.zeros(n_data))), shape=(myshape, 1))    
                result['templates']  = scipy.sparse.hstack((result['templates'], temp), 'csc')
 
            if self.two_components:
                result['norms2']     = self.h5_file['norms2'][indices]
                result['templates2'] = scipy.sparse.csc_matrix((myshape, 0), dtype=numpy.float32)

                for item in indices:
                    myslice = numpy.arange(indptr[item], indptr[item+1])
                    n_data  = len(myslice)
                    temp    = scipy.sparse.csc_matrix((self.h5_file['data2'][myslice, 0], (self.h5_file['indices'][myslice, 0], numpy.zeros(n_data))), shape=(myshape, 1))    
                    result['templates2'] = scipy.sparse.hstack((result['templates2'], temp), 'csc')

        self.h5_file.close()

        return result
        
    def remove(self, index):
        pass

    def get_overlaps(self):
        pass

    def close(self):
        try:
            self.h5_file.close()
        except Exception:
            pass