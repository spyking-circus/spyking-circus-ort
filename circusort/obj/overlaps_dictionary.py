# -*- coding: utf-8 -*-
from scipy.sparse import vstack, csr_matrix
import numpy as np
import scipy.sparse


class Overlaps(object):

    def __init__(self, template, target, _delays, _scols, overlap_size, spike_width):

        self._delays = _delays
        self._scols = _scols
        self._overlap_size = overlap_size
        self._spike_width = spike_width
        self.new_indices = []
        self.overlaps = self._get_overlaps(template, target)

    @property
    def do_update(self):

        return len(self.new_indices) > 0

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

    def update(self, template, target):

        new_overlaps = self._get_overlaps(template, target[self.new_indices])
        self.overlaps = scipy.sparse.vstack((self.overlaps, new_overlaps))
        self.new_indices = []


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
        self.first_component = scipy.sparse.csr_matrix((0, self._nb_elements), dtype=np.float32)
        self.norms = {
            '1': np.zeros(0, dtype=np.float32)
        }
        self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        self._scols = {
            'left': {},
            'right': {}
        }

        if self.two_components:
            self.second_component = scipy.sparse.csr_matrix((0, self._nb_elements), dtype=np.float32)
            self.norms['2'] = np.zeros(0, dtype=np.float32)
            self.overlaps['2'] = {}

        for idelay in self._delays:
            self._scols['left'][idelay] = np.where(self._cols % self._spike_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]

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
                self._all_components = self.first_component
            else:
                self._all_components = vstack((self.first_component, self.second_component), format='csr')

        return self._all_components

    @property
    def nb_templates(self):

        return self.first_component.shape[0]        

    def get_overlaps(self, index, component='1'):

        if index not in self.overlaps[component]:
            target = self.all_components
            template = self.all_components[index]
            self.overlaps[component][index] = Overlaps(template, target, self._delays,
                                                       self._scols, self._overlap_size, self._spike_width)

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

        csr_template = template.first_component.to_sparse('csr', flatten=True)
        self.first_component = vstack((self.first_component, csr_template), format='csr')

        if self.two_components:
            csr_template = template.second_component.to_sparse('csr', flatten=True)
            self.second_component = vstack((self.second_component, csr_template), format='csr')

        self._all_components = None
        for key, value in self.overlaps.items():
            for index in value.keys():
                self.overlaps[key][index].new_indices += [len(self) - 1]

    def update(self, indices):

        templates = self.template_store.get(indices)

        for template in templates:
            self.add_template(template)