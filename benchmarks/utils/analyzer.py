import numpy
import time
import os
import matplotlib
from circusort import io
from circusort.io.template import TemplateStore

class Analyzer(object):

    def __init__(self, spk_writer_params, probe, template_store, synthetic_file=None, filtered_data=None):

        self.probe    = io.Probe(probe)

        self.spikes   = numpy.fromfile(spk_writer_params['spike_times'], dtype=numpy.int32)
        self.temp_ids = numpy.fromfile(spk_writer_params['templates'], dtype=numpy.int32)
        self.amps     = numpy.fromfile(spk_writer_params['amplitudes'], dtype=numpy.float32)

        if filtered_data is not None:
            self.filtered_data = numpy.fromfile(filtered_data, dtype=numpy.float32)
            self.filtered_data = self.filtered_data.reshape(self.filtered_data.size/self.nb_channels, self.nb_channels)
        else:
            self.filtered_data = None

        self.template_store = TemplateStore(os.path.join(os.path.abspath(template_store), 'template_store.h5'), 'r')

        if synthetic_file is not None:
            self.synthetic_file = synthetic_file

    @property
    def nb_channels(self):
        return self.probe.nb_channels

    def 

    # def view_time_slice(self, t_min=None, t_max=None):


    #     nb_buffers = 10
    #     nb_samples = 1024

    #     t_max    = spikes.max() + nb_samples

    #     t_min    = t_max - nb_buffers * nb_samples

    #     N_t       = updater._spike_width_

        

    #     data          = template_store.get()
    #     all_templates = data.pop('templates').T
    #     norms         = data.pop('norms')

    #     curve = numpy.zeros((nb_channels, t_max-t_min), dtype=numpy.float32)

    #     idx    = numpy.where(spikes > t_min)[0]

    #     for spike, temp_id, amp in zip(spikes[idx], temp_ids[idx], amps[idx]):
    #         if spike > t_min + N_t/2:
    #             spike -= t_min
    #             tmp1   = all_templates[temp_id].toarray().reshape(nb_channels, N_t)
    #             curve[:, spike-N_t/2:spike+N_t/2+1] += amp*tmp1*norms[temp_id]
            
    #     neg_peaks = numpy.fromfile('/tmp/peaks.dat', dtype=numpy.int32)
    #     neg_peaks = neg_peaks.reshape(neg_peaks.size/2, 2)

    #     spacing  = 10
    #     pylab.figure()
    #     for i in xrange(nb_channels):
    #         pylab.plot(numpy.arange(t_min, t_max), raw_data[t_min:t_max, i] + i*spacing, '0.5')
    #         pylab.plot(numpy.arange(t_min, t_max), curve[i, :] + i*spacing, 'r')
    #         idx = numpy.where((neg_peaks[:,1] < t_max) & (neg_peaks[:,1] >= t_min) & (neg_peaks[:,0] == i))
    #         sub_peaks = neg_peaks[idx]
    #         pylab.scatter(sub_peaks[:, 1], spacing*sub_peaks[:, 0], c='k')



    #     pylab.show()