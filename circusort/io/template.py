import h5py
import os

class TemplateStore(object):

    def __init__(self, file_name, mode='w', two_components=False):

        self.file_name      = os.path.abspath(file_name)
        self.h5_file        = h5py.File(self.file_name, mode)
        self.initialized    = False
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
            nb_template = len(norms)
            nb_data     = len(templates.data)
            nb_indptr   = len(templates.indptr)
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
        else:

            nb_templates = self.nb_templates
            nb_new       = len(amplitudes)
            new_shape_1  = nb_templates + nb_new

            nb_data      = self._nb_nnz
            nb_new       = len(templates.data)
            new_shape_2  = nb_data + nb_new

            nb_indptr    = self._nb_indptr
            nb_new       = len(templates.indptr)
            new_shape_3  = nb_indptr + nb_new

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

            self.h5_file['indptr'].resize((new_shape_3, 1))
            self.h5_file['indptr'][nb_data:, 0] = templates.indptr + self.h5_file['indptr'][-1, 0]

        print self.h5_file['shape'][:]

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

    def get(self):

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