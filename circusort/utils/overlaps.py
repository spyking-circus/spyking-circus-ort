# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix, vstack, hstack, csr_matrix
import numpy as np
import scipy.sparse

class TemplateDictionary(object):

    def __init__(self, template_store=None, cc_merge=None, cc_mixture=None):
        self.template_store  = template_store
        self.mappings        = self.template_store.mappings
        self.nb_channels     = self.template_store.nb_channels
        self.first_component = None
        self.cc_merge        = cc_merge
        self.cc_mixture      = cc_mixture
        self._duplicates     = None
        self._mixtures       = None

    def _init_from_template(self, template):
        self._nb_elements    = self.nb_channels * template.temporal_width
        self._delays         = np.arange(1, template.temporal_width + 1)
        self._spike_width    = template.temporal_width
        self._overlap_size   = 2 * self._spike_width - 1
        self._cols           = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.first_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
        self._scols          = {'left' : {}, 'right' : {}}

        for idelay in self._delays:
            self._scols['left'][idelay]  = np.where(self._cols % self._spike_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]

        if self.cc_merge is not None:
            self.cc_merge *= self._nb_elements

        if self.cc_mixture is not None:
            self.cc_mixture *= self._nb_elements


    def __len__(self):
        return self.first_component.shape[0]

    def __str__(self):
        string = """
        Template dictionary with {m} templates
        rejected as duplicates: {l}
        rejected as mixtures  : {k}
        """.format(m=self.nb_templates, l=self.nb_duplicates, k=self.nb_mixtures)
        return string

    def __iter__(self, index):
        for i in self.first_component:
            yield self[i]

    def __getitem__(self, index):
        return self.first_component[index]

    def __len__(self):
        return self.nb_templates

    @property
    def nb_templates(self):
        return self.first_component.shape[0]

    @property
    def nb_mixtures(self):
        if self._mixtures is None:
            return 0
        else:
            return np.sum([len(value) for value in self._mixtures.items()])

    @property
    def nb_duplicates(self):
        if self._duplicates is None:
            return 0
        else:
            return np.sum([len(value) for value in self._duplicates.items()])

    def _add_duplicates(self, template):
        if self._duplicates is None:
            self._duplicates = {}

        if self._duplicates.has_key(template.channel):
            self._duplicates[template.channel] += [template.creation_time]
        else:
            self._duplicates[template.channel] = [template.creation_time]

    def _add_mixtures(self, template):
        if self._mixtures is None:
            self._mixtures = {}

        if self._mixtures.has_key(template.channel):
            self._mixtures[template.channel] += [template.creation_time]
        else:
            self._mixtures[template.channel] = [template.creation_time]

    def _add_template(self, template, csc_template):
        self.first_component = scipy.sparse.vstack((self.first_component, csc_template), 'csc')
        return self.template_store.add(template)

    def _remove_template(self, template, index):
        pass


    def add(self, templates):

        nb_duplicates = 0
        nb_mixtures   = 0
        accepted      = []

        for t in templates:

            if self.first_component is None:
                self._init_from_template(t)

            csc_template  = t.first_component.to_sparse('csc', flatten=True)
            norm          = t.first_component.norm
            csc_template /= norm
            is_present    = self._is_present(csc_template)
            is_mixture    = self._is_mixture(csc_template)

            if is_present:
                nb_duplicates += 1
                self._add_duplicates(t)

            if is_mixture:
                nb_mixtures += 1
                self._add_mixture(t)

            if not is_present and not is_mixture:
                accepted += self._add_template(t, csc_template)

        return accepted, nb_duplicates, nb_mixtures


    def _is_present(self, csc_template):

        if (self.cc_merge is None) or (self.nb_templates == 0):
            return False

        for idelay in self._delays:
            tmp_1 = csc_template[:, self._scols['left'][idelay]]
            tmp_2 = self.first_component[:, self._scols['right'][idelay]]
            data  = tmp_1.dot(tmp_2.T)
            if np.any(data.data >= self.cc_merge):
                return True

            if idelay < self._spike_width:
                tmp_1 = csc_template[:, self._scols['right'][idelay]]
                tmp_2 = self.first_component[:, self._scols['left'][idelay]]
                data  = tmp_1.dot(tmp_2.T)
                if np.any(data.data >= self.cc_merge):
                    return True
        return False

    def _is_mixture(self, csc_template):
        
        if (self.cc_mixture is None) or (self.nb_templates == 0):
            return False

        all_x    = np.zeros(0, dtype=np.int32)
        all_y    = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        for idelay in self._delays:
            tmp_1    = csc_template[:, self._scols['left'][idelay]]
            tmp_2    = self.first_component[:, self._scols['right'][idelay]]
            data     = tmp_1.dot(tmp_2.T)
            dx, dy   = data.nonzero()
            ones     = np.ones(len(dx), dtype=np.int32)
            all_x    = np.concatenate((all_x, dx * self.nb_templates + dy))
            all_y    = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self._spike_width:
                tmp_1    = csc_template[:, self._scols['right'][idelay]]
                tmp_2    = self.first_component[:, self._scols['left'][idelay]]
                data     = tmp_1.dot(tmp_2.T)
                dx, dy   = data.nonzero()
                ones     = np.ones(len(dx), dtype=np.int32)
                all_x    = np.concatenate((all_x, dx * self.nb_templates + dy))
                all_y    = np.concatenate((all_y, (self._overlap_size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        shape     = (self.nb_templates, self._overlap_size)
        overlap   = csr_matrix((all_data, (all_x, all_y)), shape=shape)
        distances = np.argmax(overlap.toarray(), 1)

        # for i in xrange(self.nb_templates):
        #     M[0, 0] = overlap[i, i]
        #     V[0, 0] = overlap_k[0, distances[0, i]]
        #     for j in xrange(self.nb_templates):
        #         M[1, 1]  = overlap[j, j]
        #         M[1, 0]  = overlap_i[j, distances[k, i] - distances[k, j]]
        #         M[0, 1]  = M[1, 0]
        #         V[1, 0]  = overlap_k[j, distances[k, j]]

        return False



class OverlapsDictionary(object):

    def __init__(self, template_store=None):
        self.template_store  = template_store
        self.two_components  = self.template_store.two_components
        self.nb_channels     = self.template_store.nb_channels
        self._temporal_width = None
        self.overlaps        = {'first_component' : {}}
        self._spike_width    = self.template_store.temporal_width
        self._nb_elements    = self.nb_channels * self.temporal_width
        self._delays         = np.arange(1, self.temporal_width + 1)
        self._cols           = np.arange(self.nb_channels * self._temporal_width).astype(np.int32)
        self._overlap_size   = 2 * self._spike_width - 1
        self.first_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
        self.norms           = {'1' : np.zeros(0, dtype=np.float32)}
        self.amplitudes      = np.zeros((0, 2), dtype=np.float32)
        self._scols          = {'left' : {}, 'right' : {}}

        if self.two_components:
            self.second_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
            self.norms['2']       = np.zeros(0, dtype=np.float32)
            self.overlaps['second_component'] = {}

        for idelay in self._delays:
            self._scols['left'][idelay]  = np.where(self._cols % self._spike_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]

        self.update(self.template_store.indices)

    @property
    def temporal_width(self):
        if self._temporal_width is not None:
            return self._temporal_width
        else:
            self._temporal_width = self.template_store.temporal_width
            return self._temporal_width

    @property
    def all_components(self):
        if not self.two_components:
            return self.first_component
        else:
            return vstack((self.first_component, self.second_component), 'csc')

    @property
    def nb_templates(self):
        return self.first_component.shape[0]        

    def get_overlaps(self, index, component='first_component'):
        if self.overlaps[component].has_key(index):
            return self.overlaps[component][index]
        else:
            target   = self.all_components
            overlaps = self._get_overlaps(self.first_component[index], target)
            self.overlaps['first_component'][index] = overlaps
            if self.two_components:
                overlaps = self._get_overlaps(self.second_component[index], target)
                self.overlaps['second_component'][index] = overlaps

            return self.overlaps[component][index]

    def _get_overlaps(self, template, target):
        
        all_x    = np.zeros(0, dtype=np.int32)
        all_y    = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        for idelay in self._delays:
            tmp_1    = template[:, self._scols['left'][idelay]]
            tmp_2    = target[:, self._scols['right'][idelay]]
            data     = tmp_1.dot(tmp_2.T)
            dx, dy   = data.nonzero()
            ones     = np.ones(len(dx), dtype=np.int32)
            all_x    = np.concatenate((all_x, dx * target.shape[0] + dy))
            all_y    = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self._spike_width:
                tmp_1    = template[:, self._scols['right'][idelay]]
                tmp_2    = target[:, self._scols['left'][idelay]]
                data     = tmp_1.dot(tmp_2.T)
                dx, dy   = data.nonzero()
                ones     = np.ones(len(dx), dtype=np.int32)
                all_x    = np.concatenate((all_x, dx * target.shape[0] + dy))
                all_y    = np.concatenate((all_y, (self._overlap_size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        shape = (target.shape[0] * template.shape[0], self._overlap_size)
        return csr_matrix((all_data, (all_x, all_y)), shape=shape)

    def clear_overlaps(self):
        for key in self.overlaps.keys():
            self.overlaps[key] = {}

    def dot(self, waveforms):
        if not self.two_components:
            return self.first_component.dot(waveforms)
        else:
            tmp1 = self.first_component.dot(waveforms)
            tmp2 = self.second_component.dot(waveforms)
            return np.vstack((tmp1, tmp2))

    def update(self, indices):

        templates = self.template_store.get(indices)

        for t in templates:

            # Add new and updated templates to the dictionary.
            self.norms['1'] = np.concatenate((self.norms['1'], [t.first_component.norm]))
            self.amplitudes = np.vstack((self.amplitudes, t.amplitudes))
            if self.two_components:
                self.norms['2'] = np.concatenate((self.norms['2'], [t.second_component.norm]))

            t.normalize()

            self.first_component = vstack((self.first_component, t.first_component.to_sparse('csc', flatten=True)), 'csc')
            if self.two_components:
                self.second_component = vstack((self.second_component, t.second_component.to_sparse('csc', flatten=True)), 'csc')