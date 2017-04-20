# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging
from circusort.io.utils import generate_fake_probe


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
sampling_rate = 20000
probe_file    = generate_fake_probe(nb_channels)

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=2000)
updater       = manager2.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels)
fitter        = manager2.create_block('template_fitter')
writer        = manager.create_block('writer', data_path='/tmp/output.dat')
writer_2      = manager2.create_block('spike_writer')

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks'), fitter.get_input('peaks')])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))
director.connect(updater.get_output('updater'), fitter.get_input('updater'))
director.connect(fitter.output, writer_2.input)

director.start()
director.sleep(duration=60.0)
director.stop()


import numpy, pylab

spikes   = numpy.fromfile(writer_2.recorded_data['spike_times'], dtype=numpy.int32)
temp_ids = numpy.fromfile(writer_2.recorded_data['templates'], dtype=numpy.int32)
amps     = numpy.fromfile(writer_2.recorded_data['amplitudes'], dtype=numpy.float32)

raw_data = numpy.fromfile('/tmp/output.dat', dtype=numpy.float32)
raw_data = raw_data.reshape(raw_data.size/nb_channels, nb_channels)

t_min    = spikes[0] - int(0.1*sampling_rate/1000)
t_max    = spikes[100] + int(0.1*sampling_rate/1000)

spacing  = 10

for i in xrange(nb_channels):
    pylab.plot(numpy.arange(t_min, t_max), raw_data[t_min:t_max, i]+ i*spacing, '0.5')

pylab.scatter(spikes[:100], spacing*temp_ids[:100])


N_t       = updater._spike_width_
templates = numpy.fromfile('templates/templates.dat', dtype=numpy.float32)
elecs     = numpy.fromfile('templates/channels.dat', dtype=numpy.int32)
mapping   = numpy.load('templates/mapping.npy')

templates = templates.reshape(len(elecs), mapping.shape[1]*N_t)

import scipy.sparse
all_templates = scipy.sparse.csr_matrix((0, nb_channels*N_t))
basis         = numpy.arange(N_t)

for idx in xrange(len(templates)):
    indices = mapping[elecs[idx]]
    indices = indices[indices > -1]
    pos_y   = numpy.zeros(0, dtype=numpy.int32)
    
    for i in indices:
        pos_y = numpy.concatenate((pos_y, i*N_t + basis))

    t = scipy.sparse.csr_matrix((templates[idx, :len(indices)*N_t], (numpy.zeros(len(indices)*N_t), pos_y)), shape=(1, nb_channels*N_t))
    all_templates = scipy.sparse.vstack((all_templates, t))

def get_template(id, all_templates):
    return all_templates[id].toarray().reshape(N_t, nb_channels).T