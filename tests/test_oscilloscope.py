# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
# import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)

nb_channels = 10
nb_samples  = 1024

noise    = manager.create_block('fake_spike_generator', nb_channels=nb_channels, nb_samples=nb_samples)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter', cut_off=100)
oscillo  = manager.create_block('oscilloscope', data_path='oscillo', spacing=0.1)
mad_estimator = manager.create_block('mad_estimator', threshold=6, epsilon=0.2)
peak_detector = manager.create_block('peak_detector', sign_peaks='both')

manager.initialize()

manager.connect(noise.output, selector.input)
manager.connect(selector.output, filter.input)
manager.connect(filter.output, [oscillo.get_input('data'), mad_estimator.input, peak_detector.get_input('data')])
manager.connect(mad_estimator.get_output('mads'), [oscillo.get_input('mads'), peak_detector.get_input('mads')])
manager.connect(peak_detector.get_output('peaks'), [oscillo.get_input('peaks')])

manager.start()
director.sleep(duration=10.0)

director.stop()

delay = 100

# import os
# os.system('convert -delay %d oscillo/oscillo_*.png oscilloscope.mov' %delay)