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
probe_file    = generate_fake_probe(nb_channels, radius=1.1)

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=1000)
updater       = manager2.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels, log_level=logging.DEBUG)

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data')])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks')])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))

director.start()
director.sleep(duration=30.0)
director.stop()

import numpy, pylab

N_t       = updater._spike_width_
# templates = numpy.fromfile('templates/templates.dat', dtype=numpy.float32)
elecs     = numpy.fromfile('templates/channels.dat', dtype=numpy.int32)
# mapping   = numpy.load('templates/mapping.npy')

# templates = templates.reshape(len(elecs), mapping.shape[1]*N_t)

# import scipy.sparse
# all_templates = scipy.sparse.csr_matrix((0, nb_channels*N_t), dtype=numpy.float32)
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

labels = numpy.unique(elecs)
for l in labels:
    idx = numpy.where(elecs == l)[0]
    pylab.figure()
    nb_cols  = 3
    nb_lines = int(len(idx)/nb_cols) + 1 
    count =  1
    for i in idx:
        data = get_template(i, all_templates)*norms[i]
        pylab.subplot(nb_lines, nb_cols, count)
        pylab.imshow(data, aspect='auto')
        pylab.colorbar()
        pylab.clim(-5, 5)
        count += 1

pylab.show()