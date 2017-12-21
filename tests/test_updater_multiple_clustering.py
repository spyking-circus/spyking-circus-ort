# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import scipy
import cPickle
from circusort.io.utils import generate_fake_probe
from circusort.obj import TemplateStore


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
sampling_rate = 20000
two_components= True
nb_clustering = 2
probe_file    = generate_fake_probe(nb_channels, radius=5)


noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=5000)

clusters      = []

for i in xrange(nb_clustering):
    clusters += [manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=500, log_level=logging.DEBUG, two_components=two_components, channels=range(i, nb_channels, nb_clustering))]

updater       = manager2.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels, log_level=logging.DEBUG)

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)

to_connect = [mad_estimator.input, peak_detector.get_input('data'), pca.get_input('data')]
for i in xrange(nb_clustering):
    to_connect += [clusters[i].get_input('data')]

director.connect(whitening.output, to_connect)

to_connect = [peak_detector.get_input('mads')]
for i in xrange(nb_clustering):
    to_connect += [clusters[i].get_input('mads')]

director.connect(mad_estimator.output, to_connect)

to_connect = [pca.get_input('peaks')]
for i in xrange(nb_clustering):
    to_connect += [clusters[i].get_input('peaks')]

director.connect(peak_detector.get_output('peaks'), to_connect)


director.connect(pca.get_output('pcs'), [clusters[i].get_input('pcs') for i in xrange(nb_clustering)])

for i in xrange(nb_clustering):
    director.connect(clusters[i].get_output('templates'), updater.get_input('templates'))

director.start()
director.sleep(duration=30.0)
director.stop()


import numpy, pylab


template_store = TemplateStore('templates/template_store.h5', 'r')
N_t            = template_store.width

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
        template = all_templates[i].toarray().reshape(nb_channels, N_t)*norms[i]
        pylab.subplot(nb_lines, nb_cols, count)
        pylab.imshow(template, aspect='auto')
        pylab.colorbar()
        #pylab.clim(-5, 5)
        count += 1

pylab.show()
