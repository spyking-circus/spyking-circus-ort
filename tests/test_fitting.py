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

nb_channels   = 50
probe_file    = generate_fake_probe(nb_channels)

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=5)
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=100)
fitter        = manager2.create_block('template_matcher', probe=probe_file)
writer_1      = manager2.create_block('spike_writer')
writer_2      = manager2.create_block('writer')

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer_2.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks'), fitter.get_input('peaks')])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), fitter.get_input('templates'))
director.connect(fitter.output, writer_1.input)

director.start()
director.sleep(duration=60.0)
director.stop()