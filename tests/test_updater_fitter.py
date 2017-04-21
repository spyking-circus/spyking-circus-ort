# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging
import scipy
from circusort.io.utils import generate_fake_probe


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
sampling_rate = 20000
probe_file    = generate_fake_probe(nb_channels, radius=1.1)

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=2000, log_level=logging.DEBUG)
updater       = manager2.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels, log_level=logging.DEBUG)
fitter        = manager2.create_block('template_fitter', log_level=logging.DEBUG)
writer        = manager.create_block('writer', data_path='/tmp/output.dat')
writer_2      = manager2.create_block('spike_writer')
writer_3      = manager2.create_block('peak_writer', neg_peaks='/tmp/peaks.dat')

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks'), fitter.get_input('peaks'), writer_3.input])
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

t_min    = spikes[0] - int(10*sampling_rate/1000)
t_max    = spikes[100] + int(10*sampling_rate/1000)

N_t       = updater._spike_width_
# templates = numpy.fromfile('templates/templates.dat', dtype=numpy.float32)
elecs     = numpy.fromfile('templates/channels.dat', dtype=numpy.int32)
# mapping   = numpy.load('templates/mapping.npy')

# templates = templates.reshape(len(elecs), mapping.shape[1]*N_t)

# import scipy.sparse
# all_templates = scipy.sparse.csr_matrix((0, nb_channels*N_t))
# basis         = numpy.arange(N_t)

# for idx in xrange(len(templates)):
#     indices = mapping[elecs[idx]]
#     indices = indices[indices > -1]
#     pos_y   = numpy.zeros(0, dtype=numpy.int32)
    
#     for i in indices:
#         pos_y = numpy.concatenate((pos_y, i*N_t + basis))

#     t = scipy.sparse.csr_matrix((templates[idx, :len(indices)*N_t], (numpy.zeros(len(indices)*N_t), pos_y)), shape=(1, nb_channels*N_t))
#     all_templates = scipy.sparse.vstack((all_templates, t))

def get_template(id, all_templates):
    return all_templates[id].toarray().reshape(nb_channels, N_t)


def load_data(filename, format='csr'):
    loader = numpy.load(filename + '.npz')
    if format == 'csr':
        template = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'], dtype=numpy.float32)
    elif format == 'csc':
        template = scipy.sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'], dtype=numpy.float32)
    return template, loader['norms'], loader['amplitudes']

all_templates, norms, amplitudes = load_data('templates/templates', 'csc')
all_templates = all_templates.T


curve = numpy.zeros((nb_channels, t_max-t_min), dtype=numpy.float32)

for spike, temp_id, amp in zip(spikes[:100], temp_ids[:100], amps[:100]):
    spike -= t_min
    tmp1   = get_template(temp_id, all_templates)
    try:
        curve[:, spike-tmp1.shape[1]/2:spike+tmp1.shape[1]/2+1] += amp*tmp1*norms[temp_id]
    except Exception:
        pass


neg_peaks = numpy.fromfile('/tmp/peaks.dat', dtype=numpy.int32)
neg_peaks = neg_peaks.reshape(neg_peaks.size/2, 2)

spacing  = 10
for i in xrange(nb_channels):
    pylab.plot(numpy.arange(t_min, t_max), raw_data[t_min:t_max, i]+ i*spacing, '0.5')
    pylab.plot(numpy.arange(t_min, t_max), curve[i, :]+ i*spacing, 'r')
    idx = numpy.where((neg_peaks[:,1] < t_max) & (neg_peaks[:,1] >= t_min) & (neg_peaks[:,0] == i))
    sub_peaks = neg_peaks[idx]
    pylab.scatter(sub_peaks[:, 1], spacing*sub_peaks[:, 0], c='k')



pylab.show()
