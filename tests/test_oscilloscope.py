# Test the oscilloscope to visualize the filtered signal, the median absolute deviations and the detected peaks
# continuously.

import circusort
# import logging


host = '127.0.0.1'  # to run the test locally

director = circusort.create_director(host=host)
manager = director.create_manager(host=host)

nb_channels = 10
nb_samples = 1024

noise = manager.create_block('fake_spike_generator', nb_channels=nb_channels, nb_samples=nb_samples)
filter_ = manager.create_block('filter', cut_off=100)
oscilloscope = manager.create_block('oscilloscope', spacing=0.1)
mad_estimator = manager.create_block('mad_estimator', threshold=6, epsilon=0.2)
peak_detector = manager.create_block('peak_detector', sign_peaks='both')

manager.initialize()

manager.connect(noise.output, filter_.input)
manager.connect(filter_.output, [oscilloscope.get_input('data'), mad_estimator.input, peak_detector.get_input('data')])
manager.connect(mad_estimator.get_output('mads'), [oscilloscope.get_input('mads'), peak_detector.get_input('mads')])
manager.connect(peak_detector.get_output('peaks'), [oscilloscope.get_input('peaks')])

manager.start()
director.sleep(duration=10.0)

director.stop()
