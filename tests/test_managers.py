# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging


host = '127.0.0.1' # to run the test locally

size = 2 # square root of number of electrodes (for buffer)
nb_samples = 100 # number of time samples (for buffer)
data_type = 'float32' # data type (for buffer)
nb_buffer = 1000 # number of buffers to process


director = circusort.create_director(host=host, log_level=logging.INFO)

manager = director.create_manager(host=host, log_level=logging.INFO)
reader = manager.create_block('reader', log_level=logging.INFO  )

manager2 = director.create_manager(host=host, log_level=logging.INFO)
writer = manager2.create_block('writer')

reader.size = size
reader.nb_samples = nb_samples
reader.dtype = data_type
reader.force = True

director.initialize()
director.connect(reader.output, writer.input, protocol='tcp')
director.start(nb_samples)
director.sleep(duration=1.0)