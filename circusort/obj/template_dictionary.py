# -*- coding: utf-8 -*-

import numpy as np

from circusort.obj.overlaps_store import OverlapsStore


class TemplateDictionary(object):

    def __init__(self, template_store=None, cc_merge=None, cc_mixture=None, optimize=True, overlap_path=None):

        self.template_store = template_store
        self.nb_channels = self.template_store.nb_channels
        self.cc_merge = cc_merge
        self.cc_mixture = cc_mixture
        self._duplicates = None
        self._mixtures = None
        self.optimize = optimize
        self._indices = []
        self.overlaps_store = OverlapsStore(self.template_store, optimize=self.optimize, path=overlap_path)

    @property
    def is_empty(self):

        return len(self.template_store) == 0

    @property
    def first_component(self):

        return self.overlaps_store.first_component

    def _init_from_template(self, template):
        """Initialize template dictionary based on a sampled template.

        Argument:
            template: circusort.obj.Template
                The sampled template used to initialize the dictionary. This template won't be added to the dictionary.
        """

        self.nb_elements = self.nb_channels * template.temporal_width

        return

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

        return

    def __getitem__(self, index):

        return self.first_component[index]

    def __len__(self):

        return self.nb_templates

    @property
    def to_json(self):

        result = {
            'template_store': self.template_store.file_name,
            'overlaps': self.overlaps_store.to_json
        }

        return result

    @property
    def nb_templates(self):

        return self.overlaps_store.nb_templates

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

    def _add_template(self, template):
        """Add a template to the template dictionary.

        Arguments:
            template: circusort.obj.Template
        Return:
            indices: list
                A list which contains the indices of templates successfully added to the underlying template store.
        """

        indices = self.template_store.add(template)
        self.overlaps_store.add_template(template)

        return indices

    def compute_overlaps(self):

        self.overlaps_store.compute_overlaps()

    def save_overlaps(self):

        self.overlaps_store.save_overlaps()

    def non_zeros(self, channel_indices):
        """Get indices of templates whose spatial supports include at least one of the given channel indices.

        Argument:
            channel_indices: iterable
                The channel indices.
        Return:
            template_indices: numpy.ndarray
                The template indices.
        """

        template_indices = np.array([
            k
            for k, channel_indices_bis in enumerate(self._indices)
            if np.any(np.in1d(channel_indices_bis, channel_indices))
        ], dtype=np.int32)

        return template_indices

    def initialize(self, templates):
        """Initialize the template dictionary with templates.

        Argument:
            templates
        Return:
            accepted: list
                A list which contains the indices of templates successfully added to the underlying template store.
        """

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

        nb_duplicates = 0
        nb_mixtures = 0
        accepted = []

        # Initialize the dictionary (if necessary).
        if self.is_empty:
            template = next(iter(templates))
            self._init_from_template(template)

        if force:
            # Add all the given templates without checking duplicates and mixtures.

            for t in templates:

                accepted += self._add_template(t)

                if self.optimize:
                    self._indices += [t.indices]

        else:

            for t in templates:

                csr_template = t.first_component.to_sparse('csr', flatten=True)
                norm = t.first_component.norm
                csr_template /= norm

                if self.optimize:
                    non_zeros = self.non_zeros(t.indices)
                else:
                    non_zeros = None

                # Check if the template is already in this dictionary.
                is_present = self._is_present(csr_template, non_zeros)
                if is_present:
                    nb_duplicates += 1
                    self._add_duplicates(t)

                # TODO Removing mixture online is difficult.
                # TODO Some mixture might have been added before the templates which compose this mixture.
                # TODO For now, let's say that there is no mixture in the signal.

                # # Check if the template is a mixture of templates already present in the dictionary.
                # is_mixture = self._is_mixture(csr_template, non_zeros)
                # if is_mixture:
                #     nb_mixtures += 1
                #     self._add_mixtures(t)

                is_mixture = False

                if not is_present and not is_mixture:
                    accepted += self._add_template(t)

                    if self.optimize:
                        self._indices += [t.indices]

        return accepted, nb_duplicates, nb_mixtures

    def _is_present(self, csr_template, non_zeros=None):

        if (self.cc_merge is None) or (self.nb_templates == 0):
            return False

        return self.overlaps_store.is_present(csr_template, self.cc_merge, non_zeros)

    def _is_mixture(self, csr_template, non_zeros=None):

        if (self.cc_mixture is None) or (self.nb_templates == 0):
            return False

        # TODO complete/clean.
        _ = csr_template  # discard this argument
        _ = non_zeros  # discard this argument
        # return self.overlaps_store.is_mixture(csr_template, self.cc_mixture, non_zeros)
        return False
