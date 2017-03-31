# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally
# host = settings.host # to run the test remotely

size       = 2 # square root of number of electrodes (for buffer)
nb_samples = 100 # number of time samples (for buffer)
data_type  = 'float32' # data type (for buffer)
nb_buffer  = 1000 # number of buffers to process


interface = circusort.utils.find_interface_address_towards(host)
director = circusort.create_director(interface=interface, log_level=logging.INFO)
manager = director.create_manager(host=host, log_level=logging.INFO)

reader = manager.create_block('reader', log_level=logging.INFO  )
writer = manager.create_block('writer')

reader.size = size
reader.nb_samples = nb_samples
reader.dtype = data_type
reader.force = True

manager.initialize()

director.connect(reader.output, writer.input)

manager.start(nb_steps=nb_samples)  
director.sleep(duration=2.0)
