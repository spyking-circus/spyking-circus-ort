from .block import Block
import numpy as np
import os
import tempfile
from circusort.io.probe import Probe
import scipy.sparse

# from circusort.io.utils import save_pickle
from circusort.io.template import TemplateStore


class Template_updater(Block):
    """Template updater"""
    # TODO complete docstring.

    name = "Template updater"

    params = {
        'spike_width': 5.,
        'probe': None,
        'radius': None,
        'sampling_rate': 20000,
        'cc_merge': 0.95,
        'data_path': None,
        'nb_channels': 10,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('templates')
        self.add_output('updater', 'dict')

    def _initialize(self):
        self.spikes         = {}
        self.global_id      = 0
        self.two_components = False
        self.temp_indices   = {}
        self._spike_width_  = int(self.sampling_rate*self.spike_width*1e-3)
        if np.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width        = (self._spike_width_-1)//2
        self._overlap_size = 2*self._spike_width_ - 1
        self.all_delays    = np.arange(1, self._spike_width_ + 1)

        if self.data_path is None:
            self.data_path = self._get_tmp_path()

        self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.log.info('{n} records templates into {k}'.format(k=self.data_path, n=self.name))
        self.template_store = TemplateStore(os.path.join(self.data_path, 'template_store.h5'), N_t=self._spike_width_)
        return

    @property
    def nb_templates(self):
        return self.templates.shape[1]

    def _get_tmp_path(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name))
        tmp_file.close()
        return data_path

    def _guess_output_endpoints(self):
        self._nb_elements = self.nb_channels*self._spike_width_
        self.cc_merging   = self.cc_merge * self._nb_elements
        self.all_rows     = np.arange(self.nb_channels*self._spike_width_)
        self.templates    = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)
        self.norms        = np.zeros(0, dtype=np.float32)
        self.amplitudes   = np.zeros((0, 2), dtype=np.float32)
        self.channels     = np.zeros(0, dtype=np.int32)
        self.overlaps     = {}
        self.max_nn_chan  = 0

        for channel in xrange(self.nb_channels):
            indices = self.probe.edges[channel]

            if len(indices) > self.max_nn_chan:
                self.max_nn_chan = len(indices)

            self.temp_indices[channel] = np.zeros(0, dtype=np.int32)
            for i in indices:
                tmp = np.arange(i*self._spike_width_, (i+1)*self._spike_width_)
                self.temp_indices[channel] = np.concatenate((self.temp_indices[channel], tmp))

        if self.data_path is not None:
            self.overlaps_file  = os.path.join(self.data_path, 'overlaps')


    def _is_duplicated(self, template):

        tmp_loc_c1 = template
        tmp_loc_c2 = self.templates

        for idelay in self.all_delays:
            srows    = np.where(self.all_rows % self._spike_width_ < idelay)[0]
            tmp_1    = tmp_loc_c1[srows]
            srows    = np.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2)
            if np.any(data.data >= self.cc_merging):
                self.nb_duplicates += 1
                return True

            if idelay < self._spike_width_:
                srows    = np.where(self.all_rows % self._spike_width_ < idelay)[0]
                tmp_1    = tmp_loc_c2[srows]
                srows    = np.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2    = tmp_loc_c1[srows]
                data     = tmp_1.T.dot(tmp_2)
                if np.any(data.data >= self.cc_merging):
                    self.nb_duplicates += 1
                    return True

        return False

    def _add_template(self, template, amplitude):
        self.amplitudes = np.vstack((self.amplitudes, amplitude))
        template_norm   = np.sqrt(np.sum(template.data**2)/self._nb_elements)
        self.norms      = np.concatenate((self.norms, [template_norm]))
        to_write        = template/template_norm
        to_write.data   = to_write.data.astype(np.float32)
        self.templates  = scipy.sparse.hstack((self.templates, to_write), format='csc')

    def _add_second_template(self, template):
        template_norm   = np.sqrt(np.sum(template.data**2)/self._nb_elements)
        self.norms2     = np.concatenate((self.norms2, [template_norm]))
        to_write        = template/template_norm
        to_write.data   = to_write.data.astype(np.float32)
        self.templates2 = scipy.sparse.hstack((self.templates2, to_write), format='csc')

    def _construct_templates(self, templates_data):

        new_templates = []
        self.nb_duplicates = 0

        for key in templates_data['dat'].keys():
            for channel in templates_data['dat'][key].keys():
                templates  = np.array(templates_data['dat'][key][channel]).astype(np.float32)
                amplitudes = np.array(templates_data['amp'][key][channel]).astype(np.float32)

                if self.two_components:
                    templates2 = np.array(templates_data['two'][key][channel]).astype(np.float32)

                if len(templates) > 0:
                    tmp_pos = self.temp_indices[int(channel)]
                    n_data  = len(tmp_pos)

                    for count in xrange(len(templates)):
                        template = scipy.sparse.csc_matrix((templates[count].ravel(), (tmp_pos, np.zeros(n_data))), shape=(self._nb_elements, 1))
                        template_norm = np.sqrt(np.sum(template.data**2)/self._nb_elements)
                        is_duplicated = self._is_duplicated(template/template_norm)
                        if not is_duplicated:
                            self._add_template(template, amplitudes[count])
                            if self.two_components:
                                template2 = scipy.sparse.csc_matrix((templates2[count].ravel(), (tmp_pos, np.zeros(n_data))), shape=(self._nb_elements, 1))
                                self._add_second_template(template2)

                            self.channels = np.concatenate((self.channels, [int(channel)]))
                            self.log.debug('{n} has now a dictionary with {k} templates'.format(n=self.name, k=self.nb_templates))
                            new_templates  += [self.global_id]
                            self.global_id += 1

        self.log.debug('{n} rejected {s} duplicated templates'.format(n=self.name, s=self.nb_duplicates))

        return new_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)
        if data is not None:

            if not self.is_active:
                self._set_active_mode()
                if data.has_key('two'):
                    self.two_components = True
                    self.template_store.two_components = True
                    self.templates2     = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)
                    self.norms2         = np.zeros(0, dtype=np.float32)

            self.log.debug("{n} updates the dictionary of templates".format(n=self.name))

            if self.two_components:
                self.templates2 = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=np.float32)

            nb_before     = self.nb_templates
            new_templates = self._construct_templates(data)
            offset        = data.pop('offset')

            if len(new_templates) > 0:

                params = {'templates'  : self.templates[:, nb_before:],
                          'norms'      : self.norms[nb_before:],
                          'amplitudes' : self.amplitudes[nb_before:],
                          'channels'   : self.channels[nb_before:], 
                          'times'      : [offset] * len(new_templates)}

                if self.two_components:
                    params['templates2'] = self.templates2
                    params['norms2']     = self.norms2[nb_before:]

                self.template_store.add(params)
                self.output.send({'templates_file' : self.template_store.file_name, 'indices' : new_templates})
        return
