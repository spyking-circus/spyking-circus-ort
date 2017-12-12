# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix, hstack
import numpy as np
import scipy.sparse

class TemplateDictionary(object):

    def __init__(self, template_store=None, thr_merging=None, thr_mixture=None, logger=None):
        self.template_store = template_store
        self.mappings       = self.template_store.mappings
        self.nb_channels    = self.template_store.nb_channels
        self.templates      = None
        self.thr_merging    = thr_merging
        self.thr_mixture    = thr_mixture
        self.log            = logger

    def _init_from_template(self, template):
        self._nb_elements  = self.nb_channels * template.temporal_width
        self._delays       = np.arange(1, template.temporal_width + 1)
        self._spike_width  = template.temporal_width
        self._rows         = np.arange(self.nb_channels * self._spike_width)
        self.templates     = scipy.sparse.csr_matrix((self._nb_elements, 0), dtype=np.float32)

        if self.thr_merging is not None:
            self.thr_merging *= self._nb_elements

        if self.thr_mixture is not None:
            self.thr_mixture *= self._nb_elements

    def add(self, templates):

        nb_duplicates = 0
        nb_mixtures   = 0
        accepted      = []

        for t in templates:

            if self.templates is None:
                self._init_from_template(t)

            csr_template = t.first_component.to_sparse('csr', flatten=True)
            norm         = t.first_component.norm
            is_present   = self._is_present(csr_template/norm)
            is_mixture   = self._is_mixture(csr_template)

            if is_present:
                nb_duplicates += 1

            if is_mixture:
                nb_mixtures   += 1

            if not is_present and not is_mixture:
                accepted += self._add_template(t, csr_template)

        return accepted, nb_duplicates, nb_mixtures


    def _is_present(self, csr_template):

        if self.thr_merging is None:
            return False

        tmp_loc_c1 = csr_template
        tmp_loc_c2 = self.templates

        self.log.debug('{}'.format(tmp_loc_c1.shape))
        self.log.debug('{}'.format(tmp_loc_c2.shape))
        self.log.debug('{}'.format(self._rows.shape))
        self.log.debug('{}'.format(self._spike_width))

        for idelay in self._delays:
            srows    = np.where(self._rows % self._spike_width < idelay)[0]
            self.log.debug('{}'.format(srows))
            tmp_1    = tmp_loc_c1[srows]
            srows    = np.where(self._rows % self._spike_width >= (self._spike_width - idelay))[0]
            self.log.debug('{}'.format(srows))
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2)
            if np.any(data.data >= self.thr_merging):
                return True

            if idelay < self._spike_width:
                srows    = np.where(self._rows % self._spike_width < idelay)[0]
                tmp_1    = tmp_loc_c2[srows]
                srows    = np.where(self._rows % self._spike_width >= (self._spike_width - idelay))[0]
                tmp_2    = tmp_loc_c1[srows]
                data     = tmp_1.T.dot(tmp_2)
                if np.any(data.data >= self.thr_merging):
                    return True
        return False

    def _is_mixture(self, csr_template):
        
        if self.thr_mixture is None:
            return False

    def _add_template(self, template, csr_template):
        self.templates = scipy.sparse.hstack((self.templates, csr_template))
        return self.template_store.add(template)