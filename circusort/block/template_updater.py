from .block import Block
import numpy
from circusort.config.probe import Probe
import scipy.sparse

class Template_matcher(Block):
    '''TODO add docstring'''

    name   = "Template updater"

    params = {'spike_width'   : 5.,
              'probe'         : None,
              'radius'        : None,
              'sampling_rate' : 20000,
              'decay_time'    : 3600}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = Probe(self.probe, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('templates')
        self.add_input('data')
        self.add_input('peaks')
        self.add_output('spikes', 'dict')

    def _initialize(self):
        self.spikes        = {}
        self.nb_templates  = 0
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

    def _guess_output_endpoints(self):
        self._nb_elements  = self.nb_channels*self._spike_width_

    def _is_valid(self, peak):
        return (peak >= self._width) & (peak + self._width < self.nb_samples)

    def _get_temp_indices(self, channel):

        if not self.temp_indices.has_key(channel):
            indices = self.probe.edges[channel]
            self.temp_indices[channel] = numpy.zeros(0, dtype=numpy.int32)
            for i in indices:
                tmp = numpy.arange(i*self._spike_width_, (i+1)*self._spike_width_)
                self.temp_indices[channel] = numpy.concatenate((self.temp_indices[channel], tmp))
        return self.temp_indices[channel]

    def _get_all_valid_peaks(self, peaks):
        all_peaks = set([])
        for key in peaks.keys():
            for channel in peaks[key].keys():
                all_peaks = all_peaks.union(peaks[key][channel])

        all_peaks = numpy.array(list(all_peaks), dtype=numpy.int32)
        mask = self._is_valid(all_peaks)
        return all_peaks[mask]


    def _dump_template(self, key, channel):
        pass

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

    def _construct_templates(self, templates):

        data      = numpy.zeros(0, dtype=numpy.float32)
        positions = numpy.zeros((2, 0), dtype=numpy.int32)
        self.best_elec = numpy.zeros(0, dtype=numpy.int32)
        self.amplitudes = numpy.zeros((0, 2), dtype=numpy.float32)

        self.nb_templates = 0

        for key in templates['dat'].keys():
            for channel in templates['dat'][key].keys():
                template   = numpy.array(templates['dat'][key][channel]).astype(numpy.float32)
                if len(template) > 0:
                    tmp_pos    = numpy.zeros((2, len(self.probe.edges[int(channel)])*self._spike_width_), dtype=numpy.int32)
                    tmp_pos[0] = self._get_temp_indices(int(channel))
                    for t in template:
                        self.best_elec = numpy.concatenate((self.best_elec, [int(channel)]))
                        data           = numpy.concatenate((data, t.ravel()))
                        tmp_pos[1]     = self.nb_templates
                        positions      = numpy.hstack((positions, tmp_pos))
                        self.nb_templates += 1

                    amplitudes = numpy.array(templates['amp'][key][channel]).astype(numpy.float32)
                    self.amplitudes = numpy.vstack((self.amplitudes, amplitudes))
    
        self.templates  = scipy.sparse.csc_matrix((data, (positions[0], positions[1])), shape=(self._nb_elements, self.nb_templates))
        self.norms      = numpy.zeros(self.nb_templates, dtype=numpy.float32)

        ## We normalize the templates
        for idx in xrange(self.nb_templates):
            self.norms[idx] = numpy.sqrt(self.templates[:, idx].sum()**2)/self._nb_elements 
            myslice = numpy.arange(self.templates.indptr[idx], self.templates.indptr[idx+1])
            self.templates.data[myslice] /= self.norms[idx]

    def _process(self):
        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)

        if peaks is not None:
            self.offset = peaks.pop('offset')
            data = self.inputs['templates'].receive(blocking=False)
            if data is not None:
                self.log.debug("{n} is receiving some templates: needs to update the dictionary".format(n=self.name_and_counter))
                self._construct_templates(data)
                if self.nb_templates > 0:
                    self._construct_overlaps()

            if self.nb_templates > 0:
                self.output.send(self.result)

        return
