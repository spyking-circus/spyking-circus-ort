# -*- coding: utf-8 -*-
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse


class TemplateDictionary(object):

    def __init__(self, template_store=None, cc_merge=None, cc_mixture=None):

        self.template_store = template_store
        self.mappings = self.template_store.mappings
        self.nb_channels = self.template_store.nb_channels
        self.first_component = None
        self.cc_merge = cc_merge
        self.cc_mixture = cc_mixture
        self._duplicates = None
        self._mixtures = None

    def _init_from_template(self, template):

        self._nb_elements = self.nb_channels * template.temporal_width
        self._delays = np.arange(1, template.temporal_width + 1)
        self._spike_width = template.temporal_width
        self._overlap_size = 2 * self._spike_width - 1
        self._cols = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.first_component = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=np.float32)
        self._scols = {
            'left': {},
            'right': {}
        }

        for idelay in self._delays:
            self._scols['left'][idelay] = np.where(self._cols % self._spike_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self._spike_width >= (self._spike_width - idelay))[0]

        if self.cc_merge is not None:
            self.cc_merge *= self._nb_elements

        if self.cc_mixture is not None:
            self.cc_mixture *= self._nb_elements

    def __str__(self):

        string = """
        Template dictionary with {} templates
        rejected as duplicates: {}
        rejected as mixtures  : {}
        """.format(self.nb_templates, self.nb_duplicates, self.nb_mixtures)

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
            nb_mixtures = 0
        else:
            nb_mixtures = np.sum([len(value) for value in self._mixtures.items()])

        return nb_mixtures

    @property
    def nb_duplicates(self):

        if self._duplicates is None:
            nb_duplicates = 0
        else:
            nb_duplicates = np.sum([len(value) for value in self._duplicates.items()])

        return nb_duplicates

    def _add_duplicates(self, template):

        if self._duplicates is None:
            self._duplicates = {}

        if template.channel in self._duplicates:
            self._duplicates[template.channel] += [template.creation_time]
        else:
            self._duplicates[template.channel] = [template.creation_time]

        return

    def _add_mixtures(self, template):

        if self._mixtures is None:
            self._mixtures = {}

        if template.channel in self._mixtures:
            self._mixtures[template.channel] += [template.creation_time]
        else:
            self._mixtures[template.channel] = [template.creation_time]

        return

    def _add_template(self, template, csc_template):

        self.first_component = scipy.sparse.vstack((self.first_component, csc_template), format='csc')
        indices = self.template_store.add(template)

        return indices

    def _add_templates(self, templates, csc_templates):

        self.first_component = scipy.sparse.vstack([self.first_component] + csc_templates, format='csc')
        indices = [
            index
            for template in templates
            for index in self.template_store.add(template)
        ]

        return indices

    def _remove_template(self, template, index):

        pass  # TODO implement or remove this method?

    def initialize(self, templates):

        accepted, _, _ = self.add(templates, force=True)

        return accepted

    def add(self, templates, force=False):

        nb_duplicates = 0
        nb_mixtures = 0
        accepted = []

        if force:

            if self.first_component is None and len(templates) > 0:
                self._init_from_template(templates[0])

            def get_csc_template(template_):
                csc_template_ = template_.first_component.to_sparse('csc', flatten=True)
                csc_template_ /= template_.first_component.norm
                return csc_template_

            csc_templates = [
                get_csc_template(template)
                for template in templates
            ]

            accepted = self._add_templates(templates, csc_templates)

        else:

            for k, t in enumerate(templates):

                if self.first_component is None:
                    self._init_from_template(t)

                csc_template = t.first_component.to_sparse('csc', flatten=True)
                norm = t.first_component.norm
                csc_template /= norm

                is_present = self._is_present(csc_template)
                is_mixture = self._is_mixture(csc_template)
                if is_present:
                    nb_duplicates += 1
                    self._add_duplicates(t)
                if is_mixture:
                    nb_mixtures += 1
                    self._add_mixtures(t)
                if not is_present and not is_mixture:
                    accepted += self._add_template(t, csc_template)
                    # TODO add templates with self._add_templates instead of self._add_template.
                    # TODO this will be more efficient (i.e. nb_templates times faster).

        return accepted, nb_duplicates, nb_mixtures

    def _is_present(self, csc_template):

        if (self.cc_merge is None) or (self.nb_templates == 0):
            return False

        for idelay in self._delays:
            tmp_1 = csc_template[:, self._scols['left'][idelay]]
            tmp_2 = self.first_component[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            if np.any(data.data >= self.cc_merge):
                return True

            if idelay < self._spike_width:
                tmp_1 = csc_template[:, self._scols['right'][idelay]]
                tmp_2 = self.first_component[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                if np.any(data.data >= self.cc_merge):
                    return True
        return False

    def _is_mixture(self, csc_template):
        
        if (self.cc_mixture is None) or (self.nb_templates == 0):
            return False

        all_x = np.zeros(0, dtype=np.int32)
        all_y = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        for idelay in self._delays:
            tmp_1 = csc_template[:, self._scols['left'][idelay]]
            tmp_2 = self.first_component[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            dx, dy = data.nonzero()
            ones = np.ones(len(dx), dtype=np.int32)
            all_x = np.concatenate((all_x, dx * self.nb_templates + dy))
            all_y = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self._spike_width:
                tmp_1 = csc_template[:, self._scols['right'][idelay]]
                tmp_2 = self.first_component[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                dx, dy = data.nonzero()
                ones = np.ones(len(dx), dtype=np.int32)
                all_x = np.concatenate((all_x, dx * self.nb_templates + dy))
                all_y = np.concatenate((all_y, (self._overlap_size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        shape = (self.nb_templates, self._overlap_size)
        overlap = csr_matrix((all_data, (all_x, all_y)), shape=shape)
        distances = np.argmax(overlap.toarray(), 1)
        _ = distances

        # for i in xrange(self.nb_templates):
        #     M[0, 0] = overlap[i, i]
        #     V[0, 0] = overlap_k[0, distances[0, i]]
        #     for j in xrange(self.nb_templates):
        #         M[1, 1]  = overlap[j, j]
        #         M[1, 0]  = overlap_i[j, distances[k, i] - distances[k, j]]
        #         M[0, 1]  = M[1, 0]
        #         V[1, 0]  = overlap_k[j, distances[k, j]]

        return False