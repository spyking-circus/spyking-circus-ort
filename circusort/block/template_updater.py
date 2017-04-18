from .block import Block
import numpy
from circusort.config.probe import Probe
import scipy.sparse

class Template_updater(Block):
    '''TODO add docstring'''

    name   = "Template updater"

    params = {'spike_width'   : 5.,
              'probe'         : None,
              'radius'        : None,
              'sampling_rate' : 20000,
              'cc_merge'      : 0.25    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('templates')
        self.add_input('data')
        self.add_output('overlaps')

    def _initialize(self):
        self.spikes        = {}
        self.global_id     = 0
        self.temp_indices  = {}
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.all_delays    = numpy.arange(1, self.spike_width + 1)
        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_templates(self):
        return self.templates.shape[1]

    def _guess_output_endpoints(self):
        self._nb_elements = self.nb_channels*self._spike_width_
        self.templates    = scipy.sparse.csc_matrix((self._nb_elements, 0), dtype=numpy.float32)
        self.norms        = numpy.zeros(0, dtype=numpy.float32)
        self.best_elec    = numpy.zeros(0, dtype=numpy.int32)
        self.amplitudes   = numpy.zeros((0, 2), dtype=numpy.float32)
        self.overlaps     = {}

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


    def _dump_template(self, key, channel):
        pass


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
        self.global_id += 1


    def _update_overlaps(self, indices):

        for c1 in xrange(indices):
            for c2 in xrange(self.nb_templates):



    def _construct_overlaps(self):
        over_x    = numpy.zeros(0, dtype=numpy.int32)
        over_y    = numpy.zeros(0, dtype=numpy.int32)
        over_data = numpy.zeros(0, dtype=numpy.float32)
        rows      = numpy.arange(self.nb_channels*self._spike_width_)

        to_explore = numpy.arange(self.nb_channels)

        #local_templates = numpy.zeros(0, dtype=numpy.int32)
        #for ielec in range(comm.rank, N_e, comm.size):
        #    local_templates = numpy.concatenate((local_templates, numpy.where(best_elec == ielec)[0]))
        local_templates = numpy.arange(self.nb_templates)

        #if half:
        nb_total     = len(local_templates)
        upper_bounds = self.templates.shape[1]
        #else:
        #    nb_total     = 2*len(local_templates)
        #    upper_bounds = N_tm//2

        for count, ielec in enumerate(to_explore):

            local_idx = numpy.where(self.best_elec == ielec)[0]
            len_local = len(local_idx)

            # if not half:
            #     local_idx = numpy.concatenate((local_idx, local_idx + upper_bounds))

            if len_local > 0:

                to_consider   = numpy.arange(upper_bounds)
                #if not half:
                #    to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))

                loc_templates  = self.templates[:, local_idx].tocsr()
                loc_templates2 = self.templates[:, to_consider].tocsr()

                for idelay in self.all_delays:

                    srows = numpy.where(rows % self._spike_width_ < idelay)[0]
                    tmp_1 = loc_templates[srows]

                    srows = numpy.where(rows % self._spike_width_ >= (self._spike_width_ - idelay))[0]
                    tmp_2 = loc_templates2[srows]

                    data  = tmp_1.T.dot(tmp_2)
                    data  = data.toarray()

                    dx, dy     = data.nonzero()
                    ddx        = numpy.take(local_idx, dx).astype(numpy.int32)
                    ddy        = numpy.take(to_consider, dy).astype(numpy.int32)
                    data       = data.ravel()
                    dd         = data.nonzero()[0].astype(numpy.int32)
                    over_x     = numpy.concatenate((over_x, ddx*self.nb_templates + ddy))
                    over_y     = numpy.concatenate((over_y, (idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                    over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))
                    if idelay < self._spike_width_:
                        over_x     = numpy.concatenate((over_x, ddy*self.nb_templates + ddx))
                        over_y     = numpy.concatenate((over_y, (2*self._spike_width_-idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                        over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))

        overlaps = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(self.nb_templates**2, 2*self._spike_width_ - 1))
        # To be faster, we rearrange the overlaps into a dictionnary. This has a cost: twice the memory usage for 
        # a short period of time
        self.overlaps = {}

        for i in xrange(self.nb_templates):
            self.overlaps[i] = overlaps[i*self.nb_templates:(i+1)*self.nb_templates]
        del overlaps

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

                        is_duplicate = self._is_duplicate(template)
                        if not is_duplicate:
                            self._add_template(template, amplitudes[count])
                            self.log.debug('The dictionary has now {k} templates'.format(k=self.nb_templates))
                            new_templates += [self.global_id]

        return new_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)
        if data is not None:
            self.log.debug("{n} updates the dictionary of templates".format(n=self.name_and_counter))
            new_templates = self._construct_templates(data)
            
            if len(new_templates) > 0:
                self._update_overlaps(new_templates)

        #if self.nb_templates > 0:
        #    self.output.send(self.result)

        return
