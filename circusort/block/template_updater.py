from .block import Block
import numpy, os, tempfile
from circusort.config.probe import Probe
import scipy.sparse
from circusort.io.utils import save_sparse, save_pickle

class Template_updater(Block):
    '''TODO add docstring'''

    name   = "Template updater"

    params = {'spike_width'   : 5.,
              'probe'         : None,
              'radius'        : None,
              'sampling_rate' : 20000,
              'cc_merge'      : 0.25, 
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
        self.spikes        = {}
        self.global_id     = 0
        self.temp_indices  = {}
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.all_delays    = numpy.arange(1, self.spike_width + 1)
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2
        self._overlap_size = 2*self._spike_width_ - 1

        if self.data_path is None:
            tmp_file  = tempfile.NamedTemporaryFile()
            data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name))
            tmp_file.close()
            self.data_path = data_path
        
        self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.log.info('Templates data are saved in {k}'.format(k=self.data_path))
        return

    @property
    def nb_templates(self):
        return self.templates.shape[1]

    def _guess_output_endpoints(self):
        self._nb_elements = self.nb_channels*self._spike_width_
        self.all_rows     = numpy.arange(self.nb_channels*self._spike_width_)
        self.templates    = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=numpy.float32)
        self.norms        = numpy.zeros(0, dtype=numpy.float32)
        self.best_elec    = numpy.zeros(0, dtype=numpy.int32)
        self.amplitudes   = numpy.zeros((0, 2), dtype=numpy.float32)
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
            mapping_sparse  = -1 * numpy.ones((self.nb_channels, self.max_nn_chan), dtype=numpy.int32)
            self.writers               = {}
            self.writers['mapping']    = os.path.join(self.data_path, 'mapping')
            self.writers['amplitudes'] = open(os.path.join(self.data_path, 'amplitudes.dat'), 'wb')
            self.writers['channels']   = open(os.path.join(self.data_path, 'channels.dat'), 'wb')
            self.writers['templates']  = open(os.path.join(self.data_path, 'templates.dat'), 'wb')
            for channel in xrange(self.nb_channels):
                indices = self.probe.edges[channel]
                mapping_sparse[channel, :len(indices)] = indices
            numpy.save(self.writers['mapping'], mapping_sparse)

        # ## We normalize the templates
        # for idx in xrange(self.nb_templates):
        #     self.norms[idx] = numpy.sqrt(self.templates[:, idx].sum()**2)/self._nb_elements 
        #     myslice = numpy.arange(self.templates.indptr[idx], self.templates.indptr[idx+1])
        #     self.templates.data[myslice] /= self.norms[idx]


    def _get_temp_indices(self, channel):
        if not self.temp_indices.has_key(channel):
            indices = self.probe.edges[channel]
            self.temp_indices[channel] = numpy.zeros(0, dtype=numpy.int32)
            for i in indices:
                tmp = numpy.arange(i*self._spike_width_, (i+1)*self._spike_width_)
                self.temp_indices[channel] = numpy.concatenate((self.temp_indices[channel], tmp))
        return self.temp_indices[channel]


    def _write_template_data(self, template, amplitudes, channel):
        self.writers['channels'].write(numpy.array([channel], dtype=numpy.int32))
        self.writers['amplitudes'].write(amplitudes.flatten())

        indices  = self.probe.edges[channel]
        to_write = numpy.zeros((self._spike_width_, self.max_nn_chan), dtype=numpy.float32)
        template = template.toarray().reshape(self.nb_channels, self._spike_width_).T
        to_write[:, :len(indices)] = template[:, indices]
        self.writers['templates'].write(to_write.flatten())

    def _cross_corr(self, t1, t2):
        t1 = t1.toarray().reshape(self.nb_channels, self._spike_width_)
        t2 = t2.toarray().reshape(self.nb_channels, self._spike_width_)
        return numpy.corrcoef(t1.flatten(), t2.flatten())[0, 1]

    def _is_duplicate(self, template):
        for idx in xrange(self.nb_templates):
            if self._cross_corr(template, self.templates[:, idx]) >= self.cc_merge:
                self.log.debug('A duplicate template is found, thus rejected')
                return True
        return False

    def _add_template(self, template, amplitude):
        self.amplitudes = numpy.vstack((self.amplitudes, amplitude))
        template_norm   = numpy.sqrt(template.sum()**2)/self._nb_elements
        self.norms      = numpy.concatenate((self.norms, [template_norm]))
        self.templates  = scipy.sparse.hstack((self.templates, template), format='csc')


    def _update_overlaps(self, indices):

        tmp_loc_c2 = self.templates.tocsr()

        for c1 in indices:
            tmp_loc_c1 = self.templates[:, c1].tocsr()
            all_x      = numpy.zeros(0, dtype=numpy.int32)
            all_y      = numpy.zeros(0, dtype=numpy.int32)
            all_data   = numpy.zeros(0, dtype=numpy.float32)
            
            for idelay in self.all_delays:
                srows      = numpy.where(self.all_rows % self._spike_width_ < idelay)[0]
                tmp_1      = tmp_loc_c1[srows]

                srows      = numpy.where(self.all_rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                tmp_2      = tmp_loc_c2[srows]

                data       = tmp_1.T.dot(tmp_2)
                data       = data.toarray()

                dx, dy     = data.nonzero()
                data       = data[data.nonzero()].ravel()
                all_x      = numpy.concatenate((all_x, dx))
                all_y      = numpy.concatenate((all_y, dy))
                all_data   = numpy.concatenate((all_data, data))

            self.overlaps[c1] = scipy.sparse.csr_matrix((all_data, (all_x, all_y)), shape=(self.nb_templates, self._overlap_size))

            # Now we need to add those overlaps to the old ones
            for t in range(min(indices)):
                overlap          = self.overlaps[c1][t]
                overlap.data     = overlap.data[::-1]
                self.overlaps[t] = scipy.sparse.vstack((self.overlaps[t], overlap), format='csr')

    def _construct_templates(self, templates_data):

        new_templates = []

        for key in templates_data['dat'].keys():
            for channel in templates_data['dat'][key].keys():
                templates  = numpy.array(templates_data['dat'][key][channel]).astype(numpy.float32)
                amplitudes = numpy.array(templates_data['amp'][key][channel]).astype(numpy.float32)
                if len(templates) > 0:
                    tmp_pos = self._get_temp_indices(int(channel))
                    n_data  = len(tmp_pos)
                    for count, t in enumerate(templates):
                        template = scipy.sparse.csc_matrix((t.ravel(), (tmp_pos, numpy.zeros(n_data))), shape=(self._nb_elements, 1))

                        is_duplicated = self._is_duplicate(template)
                        if not is_duplicated:
                            self._add_template(template, amplitudes[count])
                            self._write_template_data(template, amplitudes[count], int(channel))
                            self.log.debug('The dictionary has now {k} templates'.format(k=self.nb_templates))
                            new_templates  += [self.global_id]
                            self.global_id += 1

        return new_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)
        if data is not None:
            self.log.debug("{n} updates the dictionary of templates".format(n=self.name_and_counter))
            new_templates = self._construct_templates(data)
            save_sparse('templates', self.templates)
            save_pickle('amplitudes', self.amplitudes)
            save_pickle('norms', self.norms)

            if len(new_templates) > 0:
                self._update_overlaps(new_templates)
                save_pickle('overlaps', self.overlaps)
                self.output.send({'templates' : 'templates', 'overlaps' : 'overlaps', 'amplitudes' : 'amplitudes', 'norms' : 'norms'})

        return
