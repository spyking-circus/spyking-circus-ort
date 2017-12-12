# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix, hstack
import numpy as np
import scipy.sparse

class TemplateDictionary(object):

    def __init__(self, template_store=None, cc_merge=None, cc_mixture=None):
        self.template_store = template_store
        self.mappings       = self.template_store.mappings
        self.nb_channels    = self.template_store.nb_channels
        self.templates      = None
        self.cc_merge       = cc_merge
        self.cc_mixture     = cc_mixture

    def _init_from_template(self, template):
        self._nb_elements  = self.nb_channels * template.temporal_width
        self._delays       = np.arange(1, template.temporal_width + 1)
        self._spike_width  = template.temporal_width
        self._rows         = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.templates     = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)

        if self.cc_merge is not None:
            self.cc_merge *= self._nb_elements

        if self.cc_mixture is not None:
            self.cc_mixture *= self._nb_elements

    def add(self, templates):

        nb_duplicates = 0
        nb_mixtures   = 0
        accepted      = []

        for t in templates:

            if self.templates is None:
                self._init_from_template(t)

            csc_template = t.first_component.to_sparse('csc', flatten=True)
            norm         = t.first_component.norm
            is_present   = self._is_present(csc_template/norm)
            is_mixture   = self._is_mixture(csc_template)

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
        tmp_loc_c2 = self.templates

        for idelay in self._delays:
            srows    = np.where(self._rows % self._spike_width < idelay)[0]
            tmp_1    = tmp_loc_c1[srows]
            srows    = np.where(self._rows % self._spike_width >= (self._spike_width - idelay))[0]
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2)
            if np.any(data.data >= self.cc_merge):
                return True

            if idelay < self._spike_width:
                srows    = np.where(self._rows % self._spike_width < idelay)[0]
                tmp_1    = tmp_loc_c2[srows]
                srows    = np.where(self._rows % self._spike_width >= (self._spike_width - idelay))[0]
                tmp_2    = tmp_loc_c1[srows]
                data     = tmp_1.T.dot(tmp_2)
                if np.any(data.data >= self.cc_merge):
                    return True
        return False

    def _is_mixture(self, csc_template):
        
        if self.cc_mixture is None:
            return False

    def _add_template(self, template, csc_template):
        self.templates = scipy.sparse.hstack((self.templates, csc_template))
        return self.template_store.add(template)



class OverlapsDictionary(object):

    def __init__(self, template_store=None):
        self.template_store = template_store
        self.templates      = None
        self.two_components = self.template_store.two_components
        self.nb_channels    = self.template_store.nb_channels
        # self.norms = np.zeros(0, dtype=np.float32)
        # self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        # self.templates = None
        # self.variables = ['norms', 'templates', 'amplitudes']
        # if self.two_components:
        #     self.norms2 = np.zeros(0, dtype=np.float32)
        #     self.variables += ['norms2', 'templates2']


    def _init_from_template(self, template):
        self._nb_elements  = self.nb_channels * template.temporal_width
        self._delays       = np.arange(1, template.temporal_width + 1)
        self._spike_width  = template.temporal_width
        self._cols         = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.first_components = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)
        if self.two_components:
            self.second_component = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)

    def _update_overlaps(self, sources):

        sources = np.array(sources, dtype=np.int32)

        if self.two_components:
            sources = np.concatenate((sources, sources + self.nb_templates))

        selection = list(set(sources).difference(self.overlaps.keys()))

        if len(selection) > 0:

            tmp_loc_c1 = self.templates[selection]
            tmp_loc_c2 = self.templates

            all_x = np.zeros(0, dtype=np.int32)
            all_y = np.zeros(0, dtype=np.int32)
            all_data = np.zeros(0, dtype=np.float32)

            for idelay in self.all_delays:
                scols = np.where(self.all_cols % self._spike_width_ < idelay)[0]
                tmp_1 = tmp_loc_c1[:, scols]
                scols = np.where(self.all_cols % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2 = tmp_loc_c2[:, scols]
                data = tmp_1.dot(tmp_2.T).toarray()

                dx, dy = data.nonzero()
                data = data[data.nonzero()].ravel()

                all_x = np.concatenate((all_x, dx * self.templates.shape[0] + dy))
                all_y = np.concatenate((all_y, (idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                all_data = np.concatenate((all_data, data))

                if idelay < self._spike_width_:
                    all_x = np.concatenate((all_x, dy * len(selection) + dx))
                    all_y = np.concatenate((all_y, (2 * self._spike_width_ - idelay - 1) * np.ones(len(dx), dtype=np.int32)))
                    all_data = np.concatenate((all_data, data))

            shape = (self.templates.shape[0] * len(selection), self._overlap_size)
            overlaps = csr_matrix((all_data, (all_x, all_y)), shape=shape)

            for count, c in enumerate(selection):
                self.overlaps[c] = overlaps[count * self.templates.shape[0]:(count + 1) * self.templates.shape[0]]

        return


    # # Retrieve data associated to new and updated templates.
    #         templates = self.template_store.get(updater['indices'], variables=self.variables)

    #         for t in templates:
    #             # Add new and updated templates to the dictionary.
    #             self.norms      = np.concatenate((self.norms, t.first_component.norm))
    #             t.normalize()
    #             self.amplitudes = np.vstack((self.amplitudes, t.amplitudes))
    #             self.templates  = vstack((self.templates, t.first_component.to_sparse('csr', flatten=True)))
    #             if t.second_component is not None:
    #                 self.norms2    = np.concatenate((self.norms2, t.second_component.norm))
    #                 self.templates = vstack((self.templates, data.pop('templates2').T), 'csr')

    #         # Reinitialize overlapping matrix for recomputation.
    #         self.overlaps = {}