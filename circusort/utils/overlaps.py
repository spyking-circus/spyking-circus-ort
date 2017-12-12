# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix, hstack
import numpy as np
import scipy.sparse

class TemplateDictionary(object):

    def __init__(self, template_store=None, similarity_threshold=0.975, similarity_mixture=None):
        self.template_store = template_store
        self.mappings       = self.template_store.mappings
        self.nb_channels    = self.template_store.nb_channels
        self._nb_elements   = self.nb_channels * self.template_store.temporal_width
        self.templates      = scipy.sparse.csr_matrix((self._nb_elements, 0), dtype=np.float32)
        self.all_delays     = np.arange(1, self.template_store.temporal_width + 1)

        self.cc_similarity  = similarity_threshold * self._nb_elements
        self.cc_mixture     = similarity_mixture * self._nb_elements


    def add(self, templates):

        nb_duplicates = 0
        nb_mixtures   = 0
        accepted      = []

        for t in templates:

            csc_template = new_template.first_component.to_sparse('csc', flatten=True)
            norm         = new_template.first_component.norm
            is_present   = self._is_duplicated(csc_template/norm)
            is_mixture   = False #self._is_mxture(csc_template)

            if is_present:
                nb_duplicates += 1
            if is_mixture:
                nb_mixtures   += 1

            if not is_present and not is_mixture:
                accepted += self._add_template(template, csc_template)
            else:
                pass

            if nb_duplicates > 0:
                self.log.debug('{n} rejected {s} duplicated templates'.format(n=self.name, s=nb_duplicates))
            if nb_mixtures > 0:
                self.log.debug('{n} rejected {s} composite templates'.format(n=self.name, s=nb_mixtures))
            if nb_accepted > 0:
                self.log.debug('{n} accepted {t} templates'.format(n=self.name, t=nb_accepted))


        return new_templates


    def _is_present(self, csc_template):

        tmp_loc_c1 = template
        tmp_loc_c2 = self.templates

        for idelay in self.all_delays:
            srows    = np.where(self.all_rows % self._spike_width_ < idelay)[0]
            tmp_1    = tmp_loc_c1[srows]
            srows    = np.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2)
            if np.any(data.data >= self.cc_similarity):
                return True

            if idelay < self._spike_width_:
                srows    = np.where(self.all_rows % self._spike_width_ < idelay)[0]
                tmp_1    = tmp_loc_c2[srows]
                srows    = np.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2    = tmp_loc_c1[srows]
                data     = tmp_1.T.dot(tmp_2)
                if np.any(data.data >= self.cc_merging):
                    return True
        return False

    def _is_mixture(self, template):
        pass

    def _add_template(self, template, csc_template):
        self.templates = scipy.sparse.hstack((self.templates, csc_template))
        return self.template_store.add(template)