# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle  # TODO check if we should use cPickle instead.
import scipy.sparse

from circusort.obj.overlaps import Overlaps


class OverlapsStore(object):
    """Overlap store.

    Attributes:
        templates_store: none | circusort.obj.TemplateStore
        optimize
        path: none | string
        two_components
        nb_channels
        overlaps
        nb_elements
        size
        first_component
        norms
        amplitudes
        electrodes
        second_component
        temporal_width
        all_components
        nb_templates
    """
    # TODO complete docstring.

    def __init__(self, template_store=None, optimize=True, path=None):
        """Initialize overlap store.

        Arguments:
            template_store: none | circusort.obj.TemplateStore (optional)
                The default value is None.
            optimize: boolean (optional)
                The default value is True.
            path: none | string (optional)
                The default value is True.
        """
        # TODO complete docstring.

        self.template_store = template_store
        self.optimize = optimize
        self.path = path

        self.two_components = self.template_store.two_components
        self.nb_channels = self.template_store.nb_channels

        self._temporal_width = None
        self._indices = []
        self._masks = {}
        self._first_component = None
        self._is_initialized = False

        if not self.template_store.is_empty:
            self.update(self.template_store.indices, laziness=False)

        self._all_components = None

        return

    def __len__(self):
        if self._first_component is None:
            return 0
        else:
            return self.first_component.shape[0]

    def _init_from_template(self, template):
        # TODO add docstring.

        self.overlaps = {
            '1': {}
        }

        self.norms = {}

        self._temporal_width = template.temporal_width
        self.nb_elements = self.nb_channels * self.temporal_width
        self.size = 2 * self.temporal_width - 1

        self.first_component = scipy.sparse.csr_matrix((0, self.nb_elements), dtype=np.float32)
        self.norms['1'] = np.zeros(0, dtype=np.float32)

        self.amplitudes = np.zeros((0, 2), dtype=np.float32)
        self.electrodes = np.zeros(0, dtype=np.int32)

        if self.two_components:
            self.second_component = scipy.sparse.csr_matrix((0, self.nb_elements), dtype=np.float32)
            self.overlaps['2'] = {}
            self.norms['2'] = np.zeros(0, dtype=np.float32)

        self._cols = np.arange(self.nb_channels * self._temporal_width).astype(np.int32)
        self._scols = {
            'delays': np.arange(1, self.temporal_width + 1),
            'left': {},
            'right': {}
        }

        for idelay in self._scols['delays']:
            self._scols['left'][idelay] = np.where(self._cols % self.temporal_width < idelay)[0]
            self._scols['right'][idelay] = np.where(self._cols % self.temporal_width >=
                                                    (self.temporal_width - idelay))[0]

        self._is_initialized = True

        return

    @property
    def temporal_width(self):

        if self._temporal_width is None:
            self._temporal_width = self.template_store.temporal_width

        return self._temporal_width

    @property
    def all_components(self):

        if self._all_components is None:
            if not self.two_components:
                self._all_components = self.first_component.tocsc()
            else:
                self._all_components = scipy.sparse.vstack((self.first_component,
                                                            self.second_component), format='csr').tocsc()

        return self._all_components

    @property
    def nb_templates(self):

        return self.first_component.shape[0]

    @property
    def to_json(self):
        return {'path': self.path}

    def non_zeros(self, index):
        if self.optimize:
            res = np.zeros(0, dtype=np.bool)
            for i in range(index + 1):
                res = np.concatenate((res, [self._masks[i, index]]))
            for i in range(index + 1, self.nb_templates):
                res = np.concatenate((res, [self._masks[index, i]]))
            return np.where(res == True)[0]
        else:
            return None

    def _update_masks(self, index, new_indices):

        if np.any(np.in1d(self._indices[index], new_indices)):
            self._masks[index, self.nb_templates] = True
        else:
            self._masks[index, self.nb_templates] = False

    def get_overlaps(self, index, component='1'):
        # TODO add docstring.

        if index not in self.overlaps[component]:
            target = self.all_components
            template = self.all_components[index]
            self.overlaps[component][index] = Overlaps(self._scols, self.size, self.temporal_width)
            self.overlaps[component][index].initialize(template, target, self.non_zeros(index))
        else:
            if self.overlaps[component][index].do_update:
                target = self.all_components
                template = self.all_components[index]
                self.overlaps[component][index].update(template, target, self.non_zeros(index))

        return self.overlaps[component][index].overlaps

    def clear_overlaps(self):
        # TODO add docstring.

        for key in self.overlaps.keys():
            self.overlaps[key] = {}

        return

    def dot(self, waveforms):
        # TODO add docstring.

        scalar_products = self.all_components.dot(waveforms)

        return scalar_products

    def add_template(self, template):
        # TODO add docstring.

        if not self._is_initialized:
            self._init_from_template(template)

        # Add new and updated templates to the dictionary.
        self.norms['1'] = np.concatenate((self.norms['1'], [template.first_component.norm]))
        self.amplitudes = np.vstack((self.amplitudes, template.amplitudes))
        if self.two_components:
            self.norms['2'] = np.concatenate((self.norms['2'], [template.second_component.norm]))

        template.normalize()

        if self.optimize:

            self._masks[self.nb_templates, self.nb_templates] = True

            for index in range(self.nb_templates):
                self._update_masks(index, template.indices)

            self._indices += [template.indices]

        csr_template = template.first_component.to_sparse('csr', flatten=True)
        self.first_component = scipy.sparse.vstack((self.first_component, csr_template), format='csr')

        if self.two_components:
            csr_template = template.second_component.to_sparse('csr', flatten=True)
            self.second_component = scipy.sparse.vstack((self.second_component, csr_template), format='csr')

        self.electrodes = np.concatenate((self.electrodes, [template.channel]))

        self._all_components = None
        for key, value in self.overlaps.items():
            for index in value.keys():
                self.overlaps[key][index].indices_ += [len(self) - 1]

    def update(self, indices, laziness=True):
        # TODO add docstring.

        templates = self.template_store.get(indices)

        for template in templates:
            self.add_template(template)

        if not laziness:
            if self.path is not None and os.path.isfile(self.path):
                # Load precomputed overlaps.
                self.load_overlaps(self.path)
            else:
                # Precompute overlaps.
                self.compute_overlaps()

        return

    def compute_overlaps(self):
        # TODO add docstring.

        for index in range(0, self.nb_templates):
            self.get_overlaps(index, component='1')
            if self.two_components:
                self.get_overlaps(index, component='2')

        return

    def save_overlaps(self, path=None):
        """Save the internal dictionary of the overlaps store to file.

        Argument:
            path: none | string (optional)
        """
        # TODO complete docstring.

        # Check argument.
        if path is None:
            path = self.path
        assert path is not None, "Missing argument: path"

        # Normalize path.
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        # Dump overlaps.
        print "Saving overlaps..."
        with open(path, mode='wb') as file_:
            pickle.dump(self.overlaps, file_)

        return

    def load_overlaps(self, path=None):
        """Load the internal dictionary of the overlaps store from file.

        Argument:
            path: none | string
        """
        # TODO complete docstring.

        # Check argument.
        if path is None:
            path = self.path

        assert path is not None, "Missing argument: path"

        # Normalize path.
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        print "Loading overlaps..."
        # Load overlaps.
        with open(path, mode='rb') as file_:
            self.overlaps = pickle.load(file_)

        return

    def is_present(self, csr_template, cc_merge, non_zeros=None):

        if non_zeros is not None:
            sub_target = self.first_component[non_zeros]
            norms = self.norms['1'][non_zeros]
        else:
            sub_target = self.first_component
            norms = self.norms['1']

        for idelay in self._scols['delays']:
            tmp_1 = csr_template[:, self._scols['left'][idelay]]
            tmp_2 = sub_target[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            if np.any(data.data >= cc_merge):
                return True

            if idelay < self.temporal_width:
                tmp_1 = csr_template[:, self._scols['right'][idelay]]
                tmp_2 = sub_target[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                if np.any(data.data >= cc_merge):
                    return True

        return False

    def _is_mixture(self, csr_template, non_zeros=None):

        if (self.cc_mixture is None) or (self.nb_templates == 0):
            return False

        return False
