# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director      = circusort.create_director(host=host)
manager       = director.create_manager(host=host, log_level=logging.INFO)

# def generate_mapping()



noise         = manager.create_block('noise_generator')
filter        = manager.create_block('filter')
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=5, sign_peaks='both')
pca           = manager.create_block('pca', nb_waveforms=5000)
cluster       = manager.create_block('density_clustering', probe='test.prb')

manager.initialize()

manager.connect(noise.output, filter.input)
manager.connect(filter.output, whitening.input)
manager.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), pca.get_input('data')])
manager.connect(mad_estimator.output, peak_detector.get_input('mads'))
manager.connect(peak_detector.get_output('peaks'), pca.get_input('peaks'))
manager.connect(pca.get_output('peaks'), cluster.get_input('peaks'))
manager.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
manager.connect(pca.get_output('data'), cluster.get_input('data'))

manager.start()
director.sleep(duration=5.0)

director.stop()
