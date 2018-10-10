# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging


host = '127.0.0.1'  # to run the test locally
nb_groups = 4

director = circusort.create_director(host=host)
manager = director.create_manager(host=host, log_level=logging.INFO)


noise = manager.create_block('noise_generator')
dispatcher = manager.create_block('channel_dispatcher', nb_groups=nb_groups)
filters = [
    manager.create_block('filter', name='Filter %d' % i)
    for i in range(nb_groups)
]
regrouper = manager.create_block('channel_grouper', nb_groups=nb_groups)

manager.initialize()

manager.connect(noise.output, dispatcher.input)


for i in range(nb_groups):
    manager.connect(dispatcher.get_output('data_%d' % i), filters[i].input)
    manager.connect(filters[i].output, regrouper.get_input('data_%d' % i))


manager.start()

# TODO save computational times to file
director.sleep(duration=2.0)

director.stop()
