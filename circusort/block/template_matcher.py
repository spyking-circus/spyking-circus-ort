from .block import Block
import numpy
import scipy

class Template_matcher(Block):
    '''TODO add docstring'''

    name   = "Template matcher"

    params = {'spike_width' : 5.,
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('templates', 'dict')
        self.add_input('data')
        self.add_input('peaks', 'dict')
        self.add_output('spikes', 'dict')

    def _initialize(self):
        self.spikes        = {}
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
        pass


    def _is_valid(self, peak):
        return (peak >= 2*self._width) and (peak + 2*self._width < self.nb_samples)

    def _get_all_peaks(self, peaks):
        all_peaks = set([])
        for key in peaks.values():
            for channel in peaks[key].values():
                all_peaks = all_peaks.union(peaks[key][channel])

        all_peaks = numpy.array(all_peaks, dtype=numpy.int32)
        mask = self._is_valid(all_peaks)
        return all_peaks[all_peaks]


    def _computes_overlaps(self, templates):
        pass


    def _fit_chunk(self, batch, peaks):

        peaks   = self._get_all_peaks(peaks)
        n_peaks = len(peaks)

        if len(peaks) > 0:

            batch     = batch.ravel()
            sub_batch = numpy.zeros((self.nb_channels*(2*self._width + 1), n_peaks), dtype=numpy.float32)

            if len_chunk != last_chunk_size:
                slice_indices = numpy.zeros(0, dtype=numpy.int32)
                for idx in xrange(N_e):
                    slice_indices = numpy.concatenate((slice_indices, len_chunk*idx + temp_window))
                last_chunk_size = len_chunk

            for count, idx in enumerate(peaks):
                sub_mat[:, count] = numpy.take(batch, slice_indices + idx)

            del local_chunk

            b       = templates.dot(sub_mat)                

            del sub_mat

            local_offset = padding[0] + t_offset
            local_bounds = (temp_2_shift, len_chunk - temp_2_shift)
            all_spikes   = local_peaktimes + local_offset

            # Because for GPU, slicing by columns is more efficient, we need to transpose b
            #b           = b.transpose()

            failure     = numpy.zeros(n_t, dtype=numpy.int32)
            mask        = numpy.ones((n_tm, n_t), dtype=numpy.float32)
            sub_b       = b[:n_tm, :]

            min_time     = local_peaktimes.min()
            max_time     = local_peaktimes.max()
            local_len    = max_time - min_time + 1
            min_times    = numpy.maximum(local_peaktimes - min_time - temp_2_shift, 0)
            max_times    = numpy.minimum(local_peaktimes - min_time + temp_2_shift + 1, max_time - min_time)
            max_n_t      = int(space_explo*(max_time-min_time+1)//(2*temp_2_shift + 1))
                    
            while (numpy.mean(failure) < nb_chances):

                data        = sub_b * mask
                argmax_bi   = numpy.argsort(numpy.max(data, 0))[::-1]

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
                        if len(subset) > max_n_t:
                            break

                    subset    = numpy.array(subset, dtype=numpy.int32)
                    argmax_bi = numpy.delete(argmax_bi, indices)

                    inds_t, inds_temp = subset, numpy.argmax(numpy.take(sub_b, subset, axis=1), 0)

                    best_amp  = sub_b[inds_temp, inds_t]/n_scalar
                    best_amp2 = b[inds_temp + n_tm, inds_t]/n_scalar

                    mask[inds_temp, inds_t] = 0

                    best_amp_n   = best_amp/numpy.take(norm_templates, inds_temp)
                    best_amp2_n  = best_amp2/numpy.take(norm_templates, inds_temp + n_tm)

                    all_idx      = ((best_amp_n >= amp_limits[inds_temp, 0]) & (best_amp_n <= amp_limits[inds_temp, 1]))
                    to_keep      = numpy.where(all_idx == True)[0]
                    to_reject    = numpy.where(all_idx == False)[0]
                    ts           = numpy.take(local_peaktimes, inds_t[to_keep])
                    good         = (ts >= local_bounds[0]) & (ts < local_bounds[1])

                    # We reduce to only the good times that will be kept
                    #to_keep      = to_keep[good]
                    #ts           = ts[good]
                    
                    if len(ts) > 0:
                        
                        tmp      = numpy.dot(numpy.ones((len(ts), 1), dtype=numpy.int32), local_peaktimes.reshape((1, n_t)))
                        tmp     -= ts.reshape((len(ts), 1))
                        condition = numpy.abs(tmp) <= temp_2_shift

                        for count, keep in enumerate(to_keep):
                            
                            idx_b    = numpy.compress(condition[count, :], all_indices)
                            ytmp     = tmp[count, condition[count, :]] + temp_2_shift
                            
                            indices  = numpy.zeros((S_over, len(ytmp)), dtype=numpy.float32)
                            indices[ytmp, numpy.arange(len(ytmp))] = 1

                            tmp1   = c_overs[inds_temp[keep]].multiply(-best_amp[keep]).dot(indices)
                            tmp2   = c_overs[inds_temp[keep] + n_tm].multiply(-best_amp2[keep]).dot(indices)
                            b[:, idx_b] += tmp1 + tmp2

                            if good[count]:

                                t_spike               = ts[count] + local_offset
                                result['spiketimes'] += [t_spike]
                                result['amplitudes'] += [(best_amp_n[keep], best_amp2_n[keep])]
                                result['templates']  += [inds_temp[keep]]

                    myslice           = numpy.take(inds_t, to_reject)
                    failure[myslice] += 1
                    sub_idx           = (numpy.take(failure, myslice) >= nb_chances)
                    
                    mask[:, numpy.compress(sub_idx, myslice)] = 0


    def _process(self):
        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)
        templates = self.inputs['templates'].receive(blocking=False)
        if templates is not None:
            self._computes_overlaps(templates)
        # print self.counter
        # for key in templates.keys():
        #     for channel in templates[key].values():
        #         pass
        


        return
