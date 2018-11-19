# Same as 'test/test_parallel_channels.py' except that we use the subnetwork provided by SpyKING CIRCUS ORT to
# parallelize the filtering instead of constructing it by hand.

import tempfile

import circusort
import logging


host = '127.0.0.1'  # to run the test locally
nb_channels = 512
nb_groups = 4
with tempfile.NamedTemporaryFile(suffix='.h5') as data_file:
    data_path = data_file.name
duration = 1.0 * 60.0  # s

director = circusort.create_director(host=host)
manager = director.create_manager(host=host, log_level=logging.INFO)
noise = manager.create_block('noise_generator', nb_channels=nb_channels)
filter_ = manager.create_network('filter', degree=nb_groups)
writer = manager.create_block('writer', data_path=data_path)

manager.initialize()

manager.connect(noise.get_output('data'), filter_.get_input('data'))
manager.connect_network(filter_)
manager.connect(filter_.get_output('data'), writer.get_input('data'))

manager.start()

director.sleep(duration=duration)
director.stop()
director.join()
director.destroy()
