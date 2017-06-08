from .block import Block
import numpy, os, tempfile
from circusort.io.probe import Probe
import scipy.sparse
from circusort.io.utils import save_pickle
from circusort.io.template import TemplateStore
from circusort.io.overlap import OverlapStore


class Template_updater(Block):
    '''TODO add docstring'''

    name   = "Template updater"

    params = {'spike_width'   : 5.,
              'probe'         : None,
              'radius'        : None,
              'sampling_rate' : 20000,
              'cc_merge'      : 0.9,
              'data_path'     : None,
              'nb_channels'   : 10}

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
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width        = (self._spike_width_-1)//2
        self._overlap_size = 2*self._spike_width_ - 1
        self.all_delays    = numpy.arange(1, self._spike_width_ + 1)

        if self.data_path is None:
            self.data_path = self._get_tmp_path()
        
        self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.log.info('{n} records templates into {k}'.format(k=self.data_path, n=self.name))        
        self.template_store = TemplateStore(os.path.join(self.data_path, 'template_store.h5'))
        self.overlap_store  = OverlapStore(os.path.join(self.data_path, 'overlap_store.h5'))
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
        self.all_rows     = numpy.arange(self.nb_channels*self._spike_width_)
        self.templates    = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=numpy.float32)
        self.norms        = numpy.zeros(0, dtype=numpy.float32)
        self.amplitudes   = numpy.zeros((0, 2), dtype=numpy.float32)
        self.channels     = numpy.zeros(0, dtype=numpy.int32)
        self.overlaps     = {}
        self.max_nn_chan  = 0

        for channel in xrange(self.nb_channels):
            indices = self.probe.edges[channel]
            
            if len(indices) > self.max_nn_chan:
                self.max_nn_chan = len(indices)

            self.temp_indices[channel] = numpy.zeros(0, dtype=numpy.int32)
            for i in indices:
                tmp = numpy.arange(i*self._spike_width_, (i+1)*self._spike_width_)
                self.temp_indices[channel] = numpy.concatenate((self.temp_indices[channel], tmp))

        if self.data_path is not None:
            self.overlaps_file  = os.path.join(self.data_path, 'overlaps')


    def _is_duplicated(self, template):

        tmp_loc_c1 = template
        tmp_loc_c2 = self.templates
        all_data   = numpy.zeros(0, dtype=numpy.float32)

        for idelay in self.all_delays:
            srows    = numpy.where(self.all_rows % self._spike_width_ < idelay)[0]
            tmp_1    = tmp_loc_c1[srows]
            srows    = numpy.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2)
            all_data = numpy.concatenate((all_data, data.data))

            if idelay < self._spike_width_:
                srows    = numpy.where(self.all_rows % self._spike_width_ < idelay)[0]
                tmp_1    = tmp_loc_c2[srows]
                srows    = numpy.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2    = tmp_loc_c1[srows]
                data     = tmp_1.T.dot(tmp_2)
                all_data = numpy.concatenate((all_data, data.data))

        if numpy.any(all_data >= self.cc_merge*self._nb_elements):
            self.nb_duplicates += 1
            return True
        return False

    def _add_template(self, template, amplitude):
        self.amplitudes = numpy.vstack((self.amplitudes, amplitude))
        template_norm   = numpy.sqrt(numpy.sum(template.data**2)/self._nb_elements)
        self.norms      = numpy.concatenate((self.norms, [template_norm]))
        to_write        = template/template_norm
        to_write.data   = to_write.data.astype(numpy.float32)
        self.templates  = scipy.sparse.hstack((self.templates, to_write), format='csc')

    def _add_second_template(self, template):
        template_norm   = numpy.sqrt(numpy.sum(template.data**2)/self._nb_elements)
        self.norms2     = numpy.concatenate((self.norms2, [template_norm]))
        to_write        = template/template_norm
        to_write.data   = to_write.data.astype(numpy.float32)
        self.templates2 = scipy.sparse.hstack((self.templates2, to_write), format='csc')

    def _update_overlaps(self, indices):

        #### First pass ##############
        tmp_loc_c1 = self.templates[:, indices]
        tmp_loc_c2 = self.templates

        all_x      = numpy.zeros(0, dtype=numpy.int32)
        all_y      = numpy.zeros(0, dtype=numpy.int32)
        all_data   = numpy.zeros(0, dtype=numpy.float32)
        
        for idelay in self.all_delays:
            srows    = numpy.where(self.all_rows % self._spike_width_ < idelay)[0]
            tmp_1    = tmp_loc_c1[srows]
            srows    = numpy.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
            tmp_2    = tmp_loc_c2[srows]
            data     = tmp_1.T.dot(tmp_2).toarray()

            dx, dy   = data.nonzero()
            data     = data[data.nonzero()].ravel()
            
            all_x    = numpy.concatenate((all_x, dx*self.nb_templates + dy))
            all_y    = numpy.concatenate((all_y, (idelay - 1)*numpy.ones(len(dx), dtype=numpy.int32)))
            all_data = numpy.concatenate((all_data, data))

            if idelay < self._spike_width_:
                all_x    = numpy.concatenate((all_x, dy*len(indices) + dx))
                all_y    = numpy.concatenate((all_y, (2*self._spike_width_ - idelay - 1)*numpy.ones(len(dx), dtype=numpy.int32)))
                all_data = numpy.concatenate((all_data, data))

        overlaps  = scipy.sparse.csr_matrix((all_data, (all_x, all_y)), shape=(self.nb_templates*len(indices), self._overlap_size))

        del all_x, all_y, all_data

        selection = list(set(range(self.nb_templates)).difference(indices))

        for count, c1 in enumerate(indices):
            self.overlaps[c1] = overlaps[count*self.nb_templates:(count+1)*self.nb_templates]
            for t in selection:
                overlap          = self.overlaps[c1][t]
                overlap.data     = overlap.data[::-1]
                self.overlaps[t] = scipy.sparse.vstack((self.overlaps[t], overlap), format='csr')


    def _construct_templates(self, templates_data):

        new_templates = []
        self.nb_duplicates = 0

        for key in templates_data['dat'].keys():
            for channel in templates_data['dat'][key].keys():
                templates  = numpy.array(templates_data['dat'][key][channel]).astype(numpy.float32)
                amplitudes = numpy.array(templates_data['amp'][key][channel]).astype(numpy.float32)

                if self.two_components:
                    templates2 = numpy.array(templates_data['two'][key][channel]).astype(numpy.float32)

                if len(templates) > 0:
                    tmp_pos = self.temp_indices[int(channel)]
                    n_data  = len(tmp_pos)

                    for count in xrange(len(templates)):
                        template = scipy.sparse.csc_matrix((templates[count].ravel(), (tmp_pos, numpy.zeros(n_data))), shape=(self._nb_elements, 1))
                        template_norm = numpy.sqrt(numpy.sum(template.data**2)/self._nb_elements)
                        is_duplicated = self._is_duplicated(template/template_norm)
                        if not is_duplicated:
                            self._add_template(template, amplitudes[count])
                            if self.two_components:
                                template2 = scipy.sparse.csc_matrix((templates2[count].ravel(), (tmp_pos, numpy.zeros(n_data))), shape=(self._nb_elements, 1))
                                self._add_second_template(template2)

                            self.channels = numpy.concatenate((self.channels, [int(channel)]))
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
                    self.templates2     = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=numpy.float32)
                    self.norms2         = numpy.zeros(0, dtype=numpy.float32)

            self.log.debug("{n} updates the dictionary of templates".format(n=self.name))

            if self.two_components:
                self.templates2 = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=numpy.float32)

            nb_before     = self.nb_templates
            new_templates = self._construct_templates(data)

            if len(new_templates) > 0:

                params = {'templates'  : self.templates[:, nb_before:], 
                          'norms'      : self.norms[nb_before:], 
                          'amplitudes' : self.amplitudes[nb_before:], 
                          'channels'   : self.channels[nb_before:]}

                if self.two_components:
                    params['templates2'] = self.templates2
                    params['norms2']     = self.norms2[nb_before:]
                
                self.template_store.add(params)

                # #self.overlap_store.add(new_templates, params)
                # self._update_overlaps(new_templates)
                # save_pickle(self.overlaps_file, self.overlaps)
                # self.output.send({'templates_file' : self.template_store.file_name, 'indices' : new_templates, 'overlaps' : self.overlaps_file})

                self.output.send({'templates_file' : self.template_store.file_name, 'indices' : new_templates})
        return
