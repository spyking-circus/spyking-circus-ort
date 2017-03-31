# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally
# host = settings.host # to run the test remotely

interface = circusort.utils.find_interface_address_towards(host)
director  = circusort.create_director(interface=interface)
manager   = director.create_manager(host=host, log_level=logging.INFO)

noise    = manager.create_block('noise_generator')
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter')
writer   = manager.create_block('writer')

manager.initialize()

manager.connect(noise.output, selector.input)
manager.connect(selector.output, filter.input)
manager.connect(filter.output, writer.input)

manager.start()

# TODO save computational times to file
director.sleep(duration=2.0)
director.stop()