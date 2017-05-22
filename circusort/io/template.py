import h5py
import os

class TemplateInterface(object):

    def __init__(self, file_name):

        self.file_name = os.path.abspath(file_name)

    def save_all(templates, norms, amplitudes, templates2=None, norms2=None):

        self.h5_file = h5py.File(self.file_name, 'w')
        self.h5_file.create_dataset('norms', data=norms, chunks=True)
        self.h5_file.create_dataset('amplitudes', data=amplitudes, chunks=True)

        if norms2 is not None:
            self.h5_file.create_dataset('norms2', data=norms2)

        for count in xrange(len(templates)):
            pass

    def get_all(self, two_component=True):

        template = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

        if two_component:
            template2 = scipy.sparse.csr_matrix((loader['data2'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

        template, loader['norms'], loader['amplitudes'], template2, loader['norms2']
