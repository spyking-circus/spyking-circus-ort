# Test to check that the parallelization in space (i.e. along the channels) works correctly. It allows to compare the
# computational efficiency of one block which filters all the channels versus four blocks which filter the channels, a
# quarter for each block.

import tempfile

import circusort
import logging


host = '127.0.0.1'  # to run the test locally
nb_groups = 4
duration = 10.0  # s

with tempfile.NamedTemporaryFile(suffix='.h5') as data_file:
    data_path = data_file.name

director = circusort.create_director(host=host)
manager = director.create_manager(host=host, log_level=logging.INFO)
noise = manager.create_block('noise_generator')
dispatcher = manager.create_block('channel_dispatcher', nb_groups=nb_groups)
filters = [
    manager.create_block('filter', name='Filter {}'.format(k))
    for k in range(0, nb_groups)
]
regrouper = manager.create_block('channel_grouper', nb_groups=nb_groups)
writer = manager.create_block('writer', data_path=data_path)

manager.initialize()

manager.connect(noise.get_output('data'), dispatcher.get_input('data'))
for k in range(0, nb_groups):
    manager.connect(dispatcher.get_output('data_{}'.format(k)), filters[k].get_input('data'))
for k in range(0, nb_groups):
    manager.connect(filters[k].get_output('data'), regrouper.get_input('data_{}'.format(k)))
manager.connect(regrouper.get_output('data'), writer.get_input('data'))

manager.start()

director.sleep(duration=duration)
director.stop()
director.join()
director.destroy()
