# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host, log_level=logging.INFO)

noise    = manager.create_block('noise_generator',)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter')
writer_1   = manager.create_block('writer')
writer_2   = manager.create_block('writer', data_path='/tmp/means.dat')
mad_estimator = manager.create_block('mad_estimator')

manager.initialize()

manager.connect(noise.output, selector.input, protocol='ipc')
manager.connect(selector.output, filter.input, protocol='ipc')
manager.connect(filter.output, [writer_1.input, mad_estimator.input], protocol='ipc')
manager.connect(mad_estimator.get_output('thresholds'), writer_2.input, protocol='ipc')

manager.start()
director.sleep(duration=2.0)

director.stop()
