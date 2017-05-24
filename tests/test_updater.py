# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import scipy
import cPickle
from circusort.io.utils import generate_fake_probe, load_pickle
from circusort.io.template import TemplateStore


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
sampling_rate = 20000
two_components= True
probe_file    = generate_fake_probe(nb_channels, radius=5)


noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=500, two_components=two_components, log_level=logging.DEBUG)
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

template_store = TemplateStore('templates/template_store.h5', 'r')
overlaps = load_pickle('templates/overlaps')

data          = template_store.get()
all_templates = data.pop('templates').T
elecs         = data.pop('channels')
norms         = data.pop('norms')

labels = numpy.unique(elecs)
for l in labels:
    idx = numpy.where(elecs == l)[0]
    pylab.figure()
    nb_cols  = 3
    nb_lines = int(len(idx)/nb_cols) + 1 
    count =  1
    for i in idx:
        data = all_templates[i].toarray().reshape(nb_channels, N_t)*norms[i]
        pylab.subplot(nb_lines, nb_cols, count)
        pylab.imshow(data, aspect='auto')
        pylab.colorbar()
        #pylab.clim(-5, 5)
        count += 1

pylab.show()
