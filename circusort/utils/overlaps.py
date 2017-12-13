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

    def _init_from_template(self, template):
        self._nb_elements    = self.nb_channels * template.temporal_width
        self._delays         = np.arange(1, template.temporal_width + 1)
        self._spike_width    = template.temporal_width
        self._cols           = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.first_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)

        if self.cc_merge is not None:
            self.cc_merge *= self._nb_elements

        if self.cc_mixture is not None:
            self.cc_mixture *= self._nb_elements

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

            if is_mixture:
                nb_mixtures   += 1

            if not is_present and not is_mixture:
                accepted += self._add_template(t, csc_template)

        return accepted, nb_duplicates, nb_mixtures


    def _is_present(self, csc_template):

        if self.cc_merge is None:
            return False

        tmp_loc_c1 = csc_template
        tmp_loc_c2 = self.first_component

        for idelay in self._delays:
            scols    = np.where(self._cols % self._spike_width < idelay)[0]
            tmp_1    = tmp_loc_c1[:, scols]
            scols    = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]
            tmp_2    = tmp_loc_c2[:, scols]
            data     = tmp_1.dot(tmp_2.T)
            if np.any(data.data >= self.cc_merge):
                return True

            if idelay < self._spike_width:
                scols    = np.where(self._cols % self._spike_width < idelay)[0]
                tmp_1    = tmp_loc_c2[:, scols]
                scols    = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]
                tmp_2    = tmp_loc_c1[:, scols]
                data     = tmp_1.dot(tmp_2.T)
                if np.any(data.data >= self.cc_merge):
                    return True
        return False

    def _is_mixture(self, csc_template):
        
        if self.cc_mixture is None:
            return False

    def _add_template(self, template, csc_template):
        
        self.first_component = scipy.sparse.vstack((self.first_component, csc_template), 'csc')
        return self.template_store.add(template)



class OverlapsDictionary(object):

    def __init__(self, template_store=None):
        self.template_store  = template_store
        self.two_components  = self.template_store.two_components
        self.nb_channels     = self.template_store.nb_channels
        self._temporal_width = None
        self.overlaps        = {'first_component' : {}}
        self._init_from_template_store()

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
            return vstack((self.first_component, self.second_component))

    @property
    def nb_templates(self):
        return self.first_component.shape[0]

    def _init_from_template_store(self):
        self._spike_width    = self.template_store.temporal_width
        self._nb_elements    = self.nb_channels * self.temporal_width
        self._delays         = np.arange(1, self.temporal_width + 1)
        self._cols           = np.arange(self.nb_channels * self._temporal_width).astype(np.int32)
        self._overlap_size   = 2 * self._spike_width + 1
        self.first_component = scipy.sparse.csr_matrix((0, self._nb_elements), dtype=np.float32)
        self.norms           = {'1' : np.zeros(0, dtype=np.float32)}
        self.amplitudes      = {'1' : np.zeros(0, dtype=np.float32)}
        if self.two_components:
            self.second_component = scipy.sparse.csr_matrix((0, self._nb_elements), dtype=np.float32)
            self.norms['2']       = np.zeros(0, dtype=np.float32)
            self.amplitudes['2']  = np.zeros(0, dtype=np.float32)
            self.overlaps['second_component'] = {}

        self.update(self.template_store.indices)


    def update_overlaps(self, sources):

        sources = np.array(sources, dtype=np.int32)

        # if self.two_components:
        #     sources = np.concatenate((sources, sources + self.nb_templates))

        selection = list(set(sources).difference(self.overlaps['first_component'].keys()))

        if len(selection) > 0:

            tmp_loc_c1 = self.first_component[selection]
            tmp_loc_c2 = self.all_components

            all_x = np.zeros(0, dtype=np.int32)
            all_y = np.zeros(0, dtype=np.int32)
            all_data = np.zeros(0, dtype=np.float32)

            for idelay in self._delays:
                scols = np.where(self._cols % self._spike_width < idelay)[0]
                tmp_1 = tmp_loc_c1[:, scols]
                scols = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]
                tmp_2 = tmp_loc_c2[:, scols]
                data = tmp_1.dot(tmp_2.T).toarray()

                dx, dy = data.nonzero()
                data = data[data.nonzero()].ravel()

                all_x = np.concatenate((all_x, dx * self.nb_templates + dy))
                all_y = np.concatenate((all_y, (idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                all_data = np.concatenate((all_data, data))

                if idelay < self._spike_width:
                    all_x = np.concatenate((all_x, dy * len(selection) + dx))
                    all_y = np.concatenate((all_y, (2 * self._spike_width - idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                    all_data = np.concatenate((all_data, data))

            shape    = (self.nb_templates * len(selection), self._overlap_size)

            print shape, all_x.max(), all_y.max()

            overlaps = csr_matrix((all_data, (all_x, all_y)), shape=shape)

            for count, c in enumerate(selection):
                self.overlaps['first_component'][c] = overlaps[count * self.nb_templates:(count + 1) * self.nb_templates]

        return


    def update(self, indices):

        templates = self.template_store.get(indices)

        for key, t in templates.items():

            # Add new and updated templates to the dictionary.
            self.norms['1']      = np.concatenate((self.norms['1'], [t.first_component.norm]))
            self.amplitudes['1'] = np.concatenate((self.amplitudes['1'], [t.first_component.amplitude]))
            if self.two_components:
                self.norms['2']      = np.concatenate((self.norms['2'], [t.second_component.norm]))
                self.amplitudes['2'] = np.concatenate((self.amplitudes['2'], [t.second_component.amplitude]))

            t.normalize()

            self.first_component = vstack((self.first_component, t.first_component.to_sparse('csr', flatten=True)))
            if self.two_components:
                self.second_component = vstack((self.second_component, t.second_component.to_sparse('csr', flatten=True)))

        self.update_overlaps(indices)