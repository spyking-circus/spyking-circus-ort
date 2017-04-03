# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host, log_level=logging.INFO)


noise    = manager.create_block('noise_generator',)
writer_1 = manager.create_block('writer', data_path='/tmp/output_1.dat')
writer_2 = manager.create_block('writer', data_path='/tmp/output_2.dat')
manager.initialize()

manager.connect(noise.output, [writer_1.input, writer_2.input])


manager.start()

# TODO save computational times to file
director.sleep(duration=2.0)

director.stop()
