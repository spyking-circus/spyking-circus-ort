# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import scipy
from circusort.io.utils import generate_fake_probe
from circusort.io.template import TemplateStore


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
sampling_rate = 20000
two_components= False
nb_fitters    = 4
probe_file    = generate_fake_probe(nb_channels, radius=2, prb_file='test.prb')

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=500, log_level=logging.DEBUG, two_components=two_components)
updater       = manager2.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels, log_level=logging.DEBUG)

dispatcher    = manager.create_block('chunk_dispatcher', nb_groups=nb_fitters)
fitters       = [manager2.create_block('template_fitter', log_level=logging.INFO, two_components=two_components) for i in xrange(nb_fitters)]

peak_dispatch = manager.create_block('peak_dispatcher', nb_groups=nb_fitters)

writer        = manager.create_block('writer', data_path='/tmp/output.dat')
writer_2      = manager2.create_block('spike_writer')
writer_3      = manager2.create_block('peak_writer', neg_peaks='/tmp/peaks.dat')

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data'), writer.input])
director.connect(whitening.output, dispatcher.input)
for i in xrange(nb_fitters):
    director.connect(dispatcher.get_output('data_%d' %i), fitters[i].get_input('data'))

director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks'), writer_3.input, peak_dispatch.input])

for i in xrange(nb_fitters):
    director.connect(peak_dispatch.get_output('peaks_%d' %i), fitters[i].get_input('peaks'))

director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))

for i in xrange(nb_fitters):
    director.connect(updater.get_output('updater'), fitters[i].get_input('updater'))
    director.connect(fitters[i].output, writer_2.input)

director.start()
director.sleep(duration=60.0)
director.stop()

import numpy, pylab

spikes   = numpy.fromfile(writer_2.recorded_data['spike_times'], dtype=numpy.int32)
temp_ids = numpy.fromfile(writer_2.recorded_data['templates'], dtype=numpy.int32)
amps     = numpy.fromfile(writer_2.recorded_data['amplitudes'], dtype=numpy.float32)

raw_data = numpy.fromfile('/tmp/output.dat', dtype=numpy.float32)
raw_data = raw_data.reshape(raw_data.size/nb_channels, nb_channels)

nb_buffers = 10
nb_samples = 1024

t_max    = spikes.max() + nb_samples

t_min    = t_max - nb_buffers * nb_samples

template_store = TemplateStore('templates/template_store.h5', 'r')
N_t            = template_store.width

data          = template_store.get()
all_templates = data.pop('templates').T
norms         = data.pop('norms')

curve = numpy.zeros((nb_channels, t_max-t_min), dtype=numpy.float32)

idx    = numpy.where(spikes > t_min)[0]

for spike, temp_id, amp in zip(spikes[idx], temp_ids[idx], amps[idx]):
    if spike > t_min + N_t/2:
        spike -= t_min
        tmp1   = all_templates[temp_id].toarray().reshape(nb_channels, N_t)
        curve[:, spike-N_t/2:spike+N_t/2+1] += amp*tmp1*norms[temp_id]
    
neg_peaks = numpy.fromfile('/tmp/peaks.dat', dtype=numpy.int32)
neg_peaks = neg_peaks.reshape(neg_peaks.size/2, 2)

spacing  = 10
pylab.figure()
for i in xrange(nb_channels):
    pylab.plot(numpy.arange(t_min, t_max), raw_data[t_min:t_max, i] + i*spacing, '0.5')
    pylab.plot(numpy.arange(t_min, t_max), curve[i, :] + i*spacing, 'r')
    idx = numpy.where((neg_peaks[:,1] < t_max) & (neg_peaks[:,1] >= t_min) & (neg_peaks[:,0] == i))
    sub_peaks = neg_peaks[idx]
    pylab.scatter(sub_peaks[:, 1], spacing*sub_peaks[:, 0], c='k')



pylab.show()
