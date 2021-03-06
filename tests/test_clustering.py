# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
from circusort.io.utils import generate_fake_probe

host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host)
manager2      = director.create_manager(host=host)

nb_channels   = 10
probe_file    = generate_fake_probe(nb_channels)

noise         = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6, sign_peaks='both')
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager2.create_block('density_clustering', probe=probe_file, nb_waveforms=1000, two_components=False, log_level=logging.DEBUG)

director.initialize()

director.connect(noise.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), pca.get_input('data'), cluster.get_input('data')])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks')])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))

director.start()
director.sleep(duration=60.0)

director.stop()
