from .block import Block
import numpy
from circusort.config.probe import Probe
import scipy.sparse

class Template_matcher(Block):
    '''TODO add docstring'''

    name   = "Template matcher"

    params = {'spike_width' : 5.,
              'probe'         : None,
              'radius'        : None,
              'sampling_rate' : 20000}

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
        self.space_explo   = 0.5
        self.nb_chances    = 3
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
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
        return (peak >= 2*self._width) & (peak + 2*self._width < self.nb_samples)

    def _get_all_valid_peaks(self, peaks):
        all_peaks = set([])
        for key in peaks.keys():
            for channel in peaks[key].keys():
                all_peaks = all_peaks.union(peaks[key][channel])

        all_peaks = numpy.array(list(all_peaks), dtype=numpy.int32)
        mask = self._is_valid(all_peaks)
        return all_peaks[mask]


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
                if not half:
                    to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))

                loc_templates  = self.templates[:, local_idx]
                loc_templates2 = self.templates[:, to_consider]

                for idelay in all_delays:

                    srows = numpy.where(rows % N_t < idelay)[0]
                    tmp_1 = loc_templates[srows]

                    srows = numpy.where(rows % N_t >= (N_t - idelay))[0]
                    tmp_2 = loc_templates2[srows]

                    data  = tmp_1.T.dot(tmp_2)
                    data  = data.toarray()

                    dx, dy     = data.nonzero()
                    ddx        = numpy.take(local_idx, dx).astype(numpy.int32)
                    ddy        = numpy.take(to_consider, dy).astype(numpy.int32)
                    data       = data.ravel()
                    dd         = data.nonzero()[0].astype(numpy.int32)
                    over_x     = numpy.concatenate((over_x, ddx*N_tm + ddy))
                    over_y     = numpy.concatenate((over_y, (idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                    over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))
                    if idelay < self._spike_width_:
                        over_x     = numpy.concatenate((over_x, ddy*N_tm + ddx))
                        over_y     = numpy.concatenate((over_y, (2*N_t-idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                        over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))

            self.overlaps = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(self.nb_templates**2, 2*self._spike_width_ - 1))

    def _construct_templates(self, templates):

        data      = numpy.zeros(0, dtype=numpy.float32)
        positions = numpy.zeros((2, 0), dtype=numpy.int32)
        self.best_elec = numpy.zeros(0, dtype=numpy.int32)

        self.nb_templates = 0

        for key in templates.keys():
            for channel in templates[key].keys():
                template = numpy.array(templates[key][channel]).astype(numpy.float32)
                for t in template:
                    self.best_elec = numpy.concatenate((self.best_elec, [channel]))
                    t             = t.ravel()
                    data          = numpy.concatenate((data, t))
                    tmp_pos       = numpy.zeros((2, len(t)), dtype=numpy.int32)
                    #### TO DO ####
                    indices       = self.probe.edges[int(channel)]
                    tmp_pos[1]    = self.nb_templates
                    ###############
                    positions     = numpy.hstack((positions, tmp_pos))
                    self.nb_templates += 1

        self.templates = scipy.sparse.csr_matrix((data, (positions[0], positions[1])), shape=(self._nb_elements, self.nb_templates))
        self.norms     = numpy.zeros(self.nb_templates, dtype=numpy.float32)

        for idx in xrange(self.nb_templates):
            self.norms[idx] = numpy.sqrt(self.templates[:, idx].sum()**2)/self._nb_elements 
            #myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
            #templates.data[myslice] /= norm_templates[idx]

    def _fit_chunk(self, batch, peaks):

        peaks   = self._get_all_valid_peaks(peaks)
        n_peaks = len(peaks)

        if len(peaks) > 0:

            sub_batch = numpy.zeros((self.nb_channels, (2*self._width + 1), n_peaks), dtype=numpy.float32)

            for count, peak in enumerate(peaks):
                sub_batch[:, :, count] = batch[:, peak - self._width:peak + self._width + 1]

            sub_batch    = sub_batch.reshape(sub_batch.shape[0]*sub_batch.shape[1], sub_batch.shape[2])
            b            = self.templates.T.dot(sub_batch)                


            #local_offset = padding[0] + t_offset
            #local_bounds = (2*self._width, len_chunk - 2*self._width)
            
            #all_spikes   = peaks + self.offset

            # Because for GPU, slicing by columns is more efficient, we need to transpose b
            #b           = b.transpose()

            failure     = numpy.zeros(n_peaks, dtype=numpy.int32)
            mask        = numpy.ones((self.nb_templates, n_peaks), dtype=numpy.float32)
            sub_b       = b[:self.nb_templates, :]

            min_time    = peaks.min()
            max_time    = peaks.max()
            local_len   = max_time - min_time + 1
            min_times   = numpy.maximum(peaks - min_time - 2*self._width, 0)
            max_times   = numpy.minimum(peaks - min_time + 2*self._width + 1, max_time - min_time)
            max_n_peaks = int(self.space_explo*(max_time-min_time+1)//(2*2*self._width + 1))
                    
            while (numpy.mean(failure) < self.nb_chances):

                data        = sub_b * mask
                argmax_bi   = numpy.argsort(numpy.max(data, 0))[::-1]
                print argmax_bi

                while (len(argmax_bi) > 0):

                    subset          = []
                    indices         = []
                    all_times       = numpy.zeros(local_len, dtype=numpy.bool)

                    for count, idx in enumerate(argmax_bi):
                        myslice = all_times[min_times[idx]:max_times[idx]]
                        if not myslice.any():
                            subset  += [idx]
                            indices += [count]
                            all_times[min_times[idx]:max_times[idx]] = True
                        if len(subset) > max_n_peaks:
                            break

                    subset    = numpy.array(subset, dtype=numpy.int32)
                    argmax_bi = numpy.delete(argmax_bi, indices)

                    inds_t, inds_temp = subset, numpy.argmax(sub_b[:, subset], 0)

                    best_amp  = sub_b[inds_temp, inds_t]/self._nb_elements
                    best_amp2 = b[inds_temp + self.nb_templates, inds_t]/self._nb_elements

                    mask[inds_temp, inds_t] = 0

                    best_amp_n   = best_amp/numpy.take(norm_templates, inds_temp)
                    best_amp2_n  = best_amp2/numpy.take(norm_templates, inds_temp + self.nb_templates)

                    ######## To EDIT
                    all_idx      = ((best_amp_n >= 0) & (best_amp_n <= 1000))
                    
                    to_keep      = numpy.where(all_idx == True)[0]
                    to_reject    = numpy.where(all_idx == False)[0]
                    ts           = numpy.take(self.n_peaks, inds_t[to_keep])
                    good         = (ts >= local_bounds[0]) & (ts < local_bounds[1])
                    
                    if len(ts) > 0:
                        
                        tmp      = numpy.dot(numpy.ones((len(ts), 1), dtype=numpy.int32), peaks.reshape((1, n_t)))
                        tmp     -= ts.reshape((len(ts), 1))
                        condition = numpy.abs(tmp) <= 2*self._width

                        for count, keep in enumerate(to_keep):
                            
                            idx_b    = numpy.compress(condition[count, :], all_indices)
                            ytmp     = tmp[count, condition[count, :]] + 2*self._width
                            
                            indices  = numpy.zeros((S_over, len(ytmp)), dtype=numpy.float32)
                            indices[ytmp, numpy.arange(len(ytmp))] = 1

                            tmp1   = c_overs[inds_temp[keep]].multiply(-best_amp[keep]).dot(indices)
                            tmp2   = c_overs[inds_temp[keep] + self.nb_templates].multiply(-best_amp2[keep]).dot(indices)
                            b[:, idx_b] += tmp1 + tmp2

                            if good[count]:

                                t_spike               = ts[count] + self.offset
                                result['spiketimes'] += [t_spike]
                                result['amplitudes'] += [(best_amp_n[keep], best_amp2_n[keep])]
                                result['templates']  += [inds_temp[keep]]

                    myslice           = numpy.take(inds_t, to_reject)
                    failure[myslice] += 1
                    sub_idx           = (numpy.take(failure, myslice) >= self.nb_chances)
                    
                    mask[:, numpy.compress(sub_idx, myslice)] = 0


    def _process(self):
        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)

        if peaks is not None:
            self.offset = peaks.pop('offset')
            templates = self.inputs['templates'].receive(blocking=False)
            if templates is not None:
                self._construct_templates(templates)
                self._construct_overlaps()

            print self.nb_templates
            if self.nb_templates > 0:
                self._fit_chunk(batch, peaks)

        return
