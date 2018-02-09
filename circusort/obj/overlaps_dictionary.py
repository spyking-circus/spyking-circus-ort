# -*- coding: utf-8 -*-
from scipy.sparse import vstack, csr_matrix
import numpy as np
import scipy.sparse


class OverlapsDictionary(object):

    def __init__(self, template_store=None):

        self.template_store = template_store
        self.two_components = self.template_store.two_components
        self.nb_channels = self.template_store.nb_channels
        self._temporal_width = None
        self.overlaps = {
            '1': {}
        }
        self._spike_width = self.template_store.temporal_width
        self._nb_elements = self.nb_channels * self.temporal_width
        self._delays = np.arange(1, self.temporal_width + 1)
        self._cols = np.arange(self.nb_channels * self._temporal_width).astype(np.int32)
        self._overlap_size = 2 * self._spike_width - 1
        self.first_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
        self.norms = {
            '1': np.zeros(0, dtype=np.float32)
        }
        self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        self._scols = {
            'left': {},
            'right': {}
        }

        if self.two_components:
            self.second_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
            self.norms['2'] = np.zeros(0, dtype=np.float32)
            self.overlaps['2'] = {}

        for idelay in self._delays:
            self._scols['left'][idelay] = np.where(self._cols % self._spike_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]

        self.update(self.template_store.indices)

    @property
    def temporal_width(self):

        if self._temporal_width is None:
            self._temporal_width = self.template_store.temporal_width

        return self._temporal_width

    @property
    def all_components(self):

        if not self.two_components:
            all_components = self.first_component
        else:
            all_components = vstack((self.first_component, self.second_component), format='csc')

        return all_components

    @property
    def nb_templates(self):

        return self.first_component.shape[0]        

    def get_overlaps(self, index, component='1'):

        if index not in self.overlaps[component]:
            target = self.all_components
            overlaps = self._get_overlaps(self.first_component[index], target)
            self.overlaps['1'][index] = overlaps
            if self.two_components:
                overlaps = self._get_overlaps(self.second_component[index], target)
                self.overlaps['2'][index] = overlaps

        return self.overlaps[component][index]

    def _get_overlaps(self, template, target):
        
        all_x = np.zeros(0, dtype=np.int32)
        all_y = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        for idelay in self._delays:
            tmp_1 = template[:, self._scols['left'][idelay]]
            tmp_2 = target[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            dx, dy = data.nonzero()
            ones = np.ones(len(dx), dtype=np.int32)
            all_x = np.concatenate((all_x, dx * target.shape[0] + dy))
            all_y = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self._spike_width:
                tmp_1 = template[:, self._scols['right'][idelay]]
                tmp_2 = target[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                dx, dy = data.nonzero()
                ones = np.ones(len(dx), dtype=np.int32)
                all_x = np.concatenate((all_x, dx * target.shape[0] + dy))
                all_y = np.concatenate((all_y, (self._overlap_size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        shape = (target.shape[0] * template.shape[0], self._overlap_size)

        return csr_matrix((all_data, (all_x, all_y)), shape=shape)

    def clear_overlaps(self):

        for key in self.overlaps.keys():
            self.overlaps[key] = {}

        return

    def dot(self, waveforms):

        scalar_products = self.all_components.dot(waveforms)

        return scalar_products

    def update(self, indices):

        templates = self.template_store.get(indices)

        amplitudes = [t.amplitudes for t in templates]
        norms_1 = [t.first_component.norm for t in templates]

        self.amplitudes = np.vstack((self.amplitudes, amplitudes))
        self.norms['1'] = np.concatenate((self.norms['1'], norms_1))

        if self.two_components:
            norms_2 = [t.second_component.norm for t in templates]
            self.norms['2'] = np.concatenate((self.norms['2'], norms_2))

        for t in templates:
            t.normalize()

        csc_templates = [t.first_component.to_sparse('csc', flatten=True) for t in templates]
        self.first_component = vstack([self.first_component] + csc_templates, format='csc')

        if self.two_components:
            csc_templates = [t.second_component.to_sparse('csc  ', flatten=True) for t in templates]
            self.second_component = vstack([self.second_component] + csc_templates, format='csc')
