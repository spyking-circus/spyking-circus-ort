# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)


noise    = manager.create_block('noise_generator',)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter')
writer_1   = manager.create_block('writer')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector')

manager.initialize()

manager.connect(noise.output, selector.input)
manager.connect(selector.output, filter.input)
manager.connect(filter.output, [mad_estimator.input, writer_1.input, peak_detector.get_input('data')])
manager.connect(mad_estimator.get_output('mads'), peak_detector.get_input('mads'))


manager.start()
director.sleep(duration=2.0)

director.stop()
