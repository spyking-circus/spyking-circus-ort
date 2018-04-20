# -*- coding: utf-8 -*-
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse


class TemplateDictionary(object):

    def __init__(self, template_store=None, cc_merge=None, cc_mixture=None, optimize=True):

        self.template_store = template_store
        self.mappings = self.template_store.mappings
        self.nb_channels = self.template_store.nb_channels
        self.first_component = None
        self.cc_merge = cc_merge
        self.cc_mixture = cc_mixture
        self._duplicates = None
        self._mixtures = None
        self.optimize = optimize
        self._indices = []

    def _init_from_template(self, template):

        self._nb_elements = self.nb_channels * template.temporal_width
        self._delays = np.arange(1, template.temporal_width + 1)
        self._spike_width = template.temporal_width
        self._overlap_size = 2 * self._spike_width - 1
        self._cols = np.arange(self.nb_channels * self._spike_width).astype(np.int32)
        self.first_component = scipy.sparse.csr_matrix((0, self._nb_elements), dtype=np.float32)
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

    def _add_template(self, template, csr_template):
        """Add a template to the template dictionary.

        Arguments:
            template
            csr_template

        Return:
            indices: list
                A list which contains the indices of templates successfully added to the underlying template store.
        """
        # TODO complete docstring.

        self.first_component = scipy.sparse.vstack((self.first_component, csr_template), format='csr')
        indices = self.template_store.add(template)

        return indices

    def _remove_template(self, template, index):

        pass  # TODO implement or remove this method?

    @staticmethod
    def _get_csr_template(template):

        csr_template = template.first_component.to_sparse('csr', flatten=True)
        norm = template.first_component.norm
        csr_template /= norm

        return csr_template

    def non_zeros(self, indices):

        res = np.zeros(0, dtype=np.int32)
        for count, i in enumerate(self._indices):
            if np.any(np.in1d(i, indices)):
                res = np.concatenate((res, [count]))

        return res

    def initialize(self, templates):
        """Initialize the template dictionary with templates.

        Argument:
            templates
        Return:
            accepted: list
                A list which contains the indices of templates successfully added to the underlying template store.
        """
        # TODO complete docstring.

        accepted, _, _ = self.add(templates, force=True)

        return accepted

    def add(self, templates, force=False):
        """Add templates to the template dictionary.

        Arguments:
            templates
            force: boolean (optional)
                The default value is False.
        Returns:
            accepted: list
                A list which contains the indices of templates successfully added to the underlying template store.
            nb_duplicates: integer
                The number of duplicates.
            nb_mixtures: integer
                The number of mixtures.
        """
        # TODO complete docstring.

        nb_duplicates = 0
        nb_mixtures = 0
        accepted = []

        if force:
            for t in templates:

                if self.first_component is None:
                    self._init_from_template(t)

                csr_template = self._get_csr_template(t)
                accepted += self._add_template(t, csr_template)

        else:
            for t in templates:

                if self.first_component is None:
                    self._init_from_template(t)

                csr_template = self._get_csr_template(t)

                if self.optimize:
                    non_zeros = self.non_zeros(t.indices)
                else:
                    non_zeros = None

                is_present = self._is_present(csr_template, non_zeros)
                is_mixture = self._is_mixture(csr_template, non_zeros)

                if is_present:
                    nb_duplicates += 1
                    self._add_duplicates(t)

                if is_mixture:
                    nb_mixtures += 1
                    self._add_mixtures(t)

                if not is_present and not is_mixture:
                    accepted += self._add_template(t, csr_template)

                    if self.optimize:
                        self._indices += [t.indices]

        return accepted, nb_duplicates, nb_mixtures

    def _is_present(self, csr_template, non_zeros=None):

        if (self.cc_merge is None) or (self.nb_templates == 0):
            return False

        if non_zeros is not None:
            sub_target = self.first_component[non_zeros]
        else:
            sub_target = self.first_component

        for idelay in self._delays:
            tmp_1 = csr_template[:, self._scols['left'][idelay]]
            tmp_2 = sub_target[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            if np.any(data.data >= self.cc_merge):
                return True

            if idelay < self._spike_width:
                tmp_1 = csr_template[:, self._scols['right'][idelay]]
                tmp_2 = sub_target[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                if np.any(data.data >= self.cc_merge):
                    return True
        return False

    def _is_mixture(self, csr_template, non_zeros=None):
        
        if (self.cc_mixture is None) or (self.nb_templates == 0):
            return False

        return False
