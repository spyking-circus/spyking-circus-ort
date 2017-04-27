from .block import Block
import numpy
import sys
import scipy.sparse
from circusort.io.utils import load_pickle, save_pickle

def load_data(filename, format='csr'):
    loader = numpy.load(filename + '.npz')
    if format == 'csr':
        template = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    elif format == 'csc':
        template = scipy.sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    return template, loader['norms'], loader['amplitudes']



class Template_fitter(Block):
    '''TODO add docstring'''

    name   = "Template fitter"

    params = {'spike_width'   : 5.,
              'sampling_rate' : 20000}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('updater')
        self.add_input('data')
        self.add_input('peaks')
        self.add_output('spikes', 'dict')

    def _initialize(self):
        self.space_explo   = 0.5
        self.nb_chances    = 3
        self._spike_width_ = int(self.sampling_rate*self.spike_width*1e-3)
        self.two_components = False

        if numpy.mod(self._spike_width_, 2) == 0:
            self._spike_width_ += 1
        self._width = (self._spike_width_-1)//2
        self._overlap_size = 2*self._spike_width_ - 1
        return

    @property
    def nb_channels(self):
        return self.inputs['data'].shape[0]

    @property
    def nb_samples(self):
        return self.inputs['data'].shape[1]

    @property
    def nb_templates(self):
        return self.templates.shape[0]

    def _guess_output_endpoints(self):
        self._nb_elements  = self.nb_channels*self._spike_width_
        self.templates     = scipy.sparse.csc_matrix((0, self._nb_elements), dtype=numpy.float32)
        self.slice_indices = numpy.zeros(0, dtype=numpy.int32)
        temp_window        = numpy.arange(-self._width, self._width + 1)
        for idx in xrange(self.nb_channels):
            self.slice_indices = numpy.concatenate((self.slice_indices, self.nb_samples*idx + temp_window))

    def _is_valid(self, peak):
        return (peak >= self._width) & (peak + self._width < self.nb_samples)

    def _get_all_valid_peaks(self, peaks):
        all_peaks = set([])
        for key in peaks.keys():
            for channel in peaks[key].keys():
                all_peaks = all_peaks.union(peaks[key][channel])

        all_peaks = numpy.array(list(all_peaks), dtype=numpy.int32)
        mask = self._is_valid(all_peaks)
        return all_peaks[mask]

    def _reset(self):
        self.result = {'spike_times' : numpy.zeros(0, dtype=numpy.int32),
                       'amplitudes'  : numpy.zeros(0, dtype=numpy.float32),
                       'templates'   : numpy.zeros(0, dtype=numpy.int32),
                       'offset'      : self.offset}


    def _fit_chunk(self, batch, peaks):

        self._reset()
        peaks       = self._get_all_valid_peaks(peaks)
        n_peaks     = len(peaks)
        all_indices = numpy.arange(n_peaks)

        if n_peaks > 0:

            batch     = batch.flatten()
            sub_batch = numpy.zeros((self.nb_channels*self._spike_width_, n_peaks), dtype=numpy.float32)

            for count, peak in enumerate(peaks):
                sub_batch[:, count] = batch[self.slice_indices + peak]

            b           = self.templates.dot(sub_batch)                
            failure     = numpy.zeros(n_peaks, dtype=numpy.int32)
            mask        = numpy.ones((self.nb_templates, n_peaks), dtype=numpy.int32)
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

                while (len(argmax_bi) > 0):
                    subset          = numpy.zeros(0, dtype=numpy.int32)
                    indices         = numpy.zeros(0, dtype=numpy.int32)
                    all_times       = numpy.zeros(local_len, dtype=numpy.bool)

                    for count, idx in enumerate(argmax_bi):
                        myslice = all_times[min_times[idx]:max_times[idx]]
                        if not myslice.any():
                            subset  = numpy.concatenate((subset, [idx]))
                            indices = numpy.concatenate((indices, [count]))
                            all_times[min_times[idx]:max_times[idx]] = True
                        if len(subset) > max_n_peaks:
                            break

                    argmax_bi = numpy.delete(argmax_bi, indices)

                    inds_t, inds_temp = subset, numpy.argmax(numpy.take(sub_b, subset, axis=1), 0)

                    best_amp  = sub_b[inds_temp, inds_t]/self._nb_elements
                    if self.two_components:
                        best_amp2 = b[inds_temp + self.nb_templates, inds_t]/self._nb_elements

                    mask[inds_temp, inds_t] = 0

                    best_amp_n   = best_amp/numpy.take(self.norms, inds_temp)
                    if self.two_components:
                        best_amp2_n  = best_amp2/numpy.take(norm_templates, inds_temp + self.nb_templates)

                    all_idx      = ((best_amp_n >= self.amplitudes[inds_temp, 0]) & (best_amp_n <= self.amplitudes[inds_temp, 1]))
                    to_keep      = numpy.where(all_idx == True)[0]
                    to_reject    = numpy.where(all_idx == False)[0]
                    ts           = numpy.take(peaks, inds_t[to_keep])
                    good         = (ts >= 2*self._width) & (ts + 2*self._width < self.nb_samples)

                    if len(ts) > 0:
                        
                        tmp      = numpy.dot(numpy.ones((len(ts), 1), dtype=numpy.int32), peaks.reshape((1, n_peaks)))
                        tmp     -= ts.reshape((len(ts), 1))
                        condition = numpy.abs(tmp) <= 2*self._width

                        for count, keep in enumerate(to_keep):
                            
                            idx_b    = numpy.compress(condition[count, :], all_indices)
                            ytmp     = tmp[count, condition[count, :]] + 2*self._width
                            
                            indices  = numpy.zeros((self._overlap_size, len(ytmp)), dtype=numpy.int32)
                            indices[ytmp, numpy.arange(len(ytmp))] = 1

                            tmp1   = self.overlaps[inds_temp[keep]].multiply(-best_amp[keep]).dot(indices)
                            b[:, idx_b] += tmp1

                            if self.two_components:
                                tmp2   = c_overs[inds_temp[keep] + self.nb_templates].multiply(-best_amp2[keep]).dot(indices)
                                b[:, idx_b] += tmp2

                            if good[count]:
                                self.result['spike_times']  = numpy.concatenate((self.result['spike_times'], [ts[count]]))
                                self.result['amplitudes']   = numpy.concatenate((self.result['amplitudes'], [best_amp_n[keep]]))
                                self.result['templates']    = numpy.concatenate((self.result['templates'], [inds_temp[keep]]))

                    myslice           = numpy.take(inds_t, to_reject)
                    failure[myslice] += 1
                    sub_idx           = (numpy.take(failure, myslice) >= self.nb_chances)
                    mask[:, numpy.compress(sub_idx, myslice)] = 0

            if len(self.result['spike_times']) > 0:
                self.log.debug('{n} fitted {k} spikes from {m} templates'.format(n=self.name_and_counter, k=len(self.result['spike_times']), m=self.nb_templates))
            else:
                self.log.debug('{n} fitted no spikes from {s} peaks'.format(n=self.name_and_counter, s=n_peaks))

    def _process(self):
        batch = self.inputs['data'].receive()
        peaks = self.inputs['peaks'].receive(blocking=False)

        if peaks is not None:

            if not self.is_active:
                self._set_active_mode()

            while peaks.pop('offset')/self.nb_samples < self.counter:
                    peaks = self.inputs['peaks'].receive()

            self.offset = self.counter * self.nb_samples

            updater  = self.inputs['updater'].receive(blocking=False)
            
            if updater is not None:
                self.templates, self.norms, self.amplitudes  = load_data(updater['templates'], format='csc')
                self.templates = self.templates.T
                self.overlaps  = load_pickle(updater['overlaps'])
                
            if self.nb_templates > 0:
                self._fit_chunk(batch, peaks)
                self.output.send(self.result)

        return
