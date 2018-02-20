# -*- coding: utf-8 -*-
import numpy as np
import h5py
import scipy.sparse
from circusort.obj.overlaps import Overlaps


class OverlapsStore(object):

    def __init__(self, template_store=None, compress=False):

        self.template_store = template_store
        self.two_components = self.template_store.two_components
        self.nb_channels = self.template_store.nb_channels
        self._temporal_width = None
        self.compress = compress
        self._indices = []
        self._masks = {0: {}}

        self.overlaps = {
            '1': {}
        }

        self.nb_elements = self.nb_channels * self.temporal_width
        self.size = 2 * self.temporal_width - 1

        self.first_component = scipy.sparse.csr_matrix((0, self.nb_elements), dtype=np.float32)
        self.norms = {
            '1': np.zeros(0, dtype=np.float32)
        }
        self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        self.electrodes = np.zeros(0, dtype=np.int32)

        if self.two_components:
            self.second_component = scipy.sparse.csr_matrix((0, self.nb_elements), dtype=np.float32)
            self.norms['2'] = np.zeros(0, dtype=np.float32)
            self.overlaps['2'] = {}

        self._cols = np.arange(self.nb_channels * self._temporal_width).astype(np.int32)
        self._scols = {
            'delays': np.arange(1, self.temporal_width + 1),
            'left': {},
            'right': {}
        }

        for idelay in self._scols['delays']:
            self._scols['left'][idelay] = np.where(self._cols % self.temporal_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self.temporal_width >= (self.temporal_width - idelay))[0]

        self.update(self.template_store.indices)
        self._all_components = None

    def __len__(self):

        return self.first_component.shape[0]

    @property
    def temporal_width(self):

        if self._temporal_width is None:
            self._temporal_width = self.template_store.temporal_width

        return self._temporal_width

    @property
    def all_components(self):

        if self._all_components is None:
            if not self.two_components:
                self._all_components = self.first_component.tocsc()
            else:
                self._all_components = scipy.sparse.vstack((self.first_component, self.second_component), format='csr').tocsc()

        return self._all_components

    @property
    def nb_templates(self):

        return self.first_component.shape[0]

    def get_non_zeros(self, index):
        if self.compress:
            pass
        else:
            return np.arange(self.nb_templates)

    def _update_masks(self, index, new_indices):

        if not np.any(np.in1d(self._indices[index], new_indices)):
            self._masks[index][self.nb_templates] = False
        else:
            self._masks[index][self.nb_templates] = True

        self._masks[self.nb_templates] = {}

    def get_overlaps(self, index, component='1'):

        if index not in self.overlaps[component]:
            target = self.all_components
            template = self.all_components[index]
            self.overlaps[component][index] = Overlaps(template, target, self._scols, self.size, self.temporal_width)

        else:
            if self.overlaps[component][index].do_update:
                target = self.all_components
                template = self.all_components[index]
                self.overlaps[component][index].update(template, target)

        return self.overlaps[component][index].overlaps

    def clear_overlaps(self):

        for key in self.overlaps.keys():
            self.overlaps[key] = {}

        return

    def dot(self, waveforms):

        scalar_products = self.all_components.dot(waveforms)

        return scalar_products

    def add_template(self, template):

        # Add new and updated templates to the dictionary.
        self.norms['1'] = np.concatenate((self.norms['1'], [template.first_component.norm]))
        self.amplitudes = np.vstack((self.amplitudes, template.amplitudes))
        if self.two_components:
            self.norms['2'] = np.concatenate((self.norms['2'], [template.second_component.norm]))

        template.normalize()

        if self.compress:
            for index in range(self.nb_templates):
                self._update_masks(index, template.indices)

            self._indices += [template.indices]

        csr_template = template.first_component.to_sparse('csr', flatten=True)
        self.first_component = scipy.sparse.vstack((self.first_component, csr_template), format='csr')

        if self.two_components:
            csr_template = template.second_component.to_sparse('csr', flatten=True)
            self.second_component = scipy.sparse.vstack((self.second_component, csr_template), format='csr')

        self.electrodes = np.concatenate((self.electrodes, [template.channel]))

        self._all_components = None
        for key, value in self.overlaps.items():
            for index in value.keys():
                self.overlaps[key][index].new_indices += [len(self) - 1]

    def update(self, indices):

        templates = self.template_store.get(indices)

        for template in templates:
            self.add_template(template)

    def precompute_overlaps(self):

        import time
        t_start = time.time()
        for index in range(self.nb_templates):
            self.get_overlaps(index, component='1')
            if self.two_components:
                self.get_overlaps(index, component='2')

        print time.time() - t_start

    def _open(self, mode='r+'):

        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_name, mode=mode, swmr=True)

        return

    def _close(self):

        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()
            self.h5_file = None

        return

    def __del__(self):

        self._close()