# Same as 'test/test_parallel_peak_detection.py' except that we use the subnetwork provided by SpyKING CIRCUS ORT to
# parallelize the peak detection instead of constructing it by hand.

import logging
import tempfile

import circusort


host = '127.0.0.1'  # to run the test locally
nb_groups = 4  # i.e. number of peak detectors in parallel
with tempfile.NamedTemporaryFile(suffix='.h5') as data_file:
    data_path = data_file.name
duration = 10.0  # s

director = circusort.create_director(host=host)
manager = director.create_manager(host=host, log_level=logging.INFO)

noise = manager.create_block('noise_generator')  # TODO replace by a synthetic generator?
filter_ = manager.create_block('filter')
mad = manager.create_block('mad_estimator')
detector = manager.create_network('peak_detector', degree=nb_groups)
writer = manager.create_block('peak_writer', data_path=data_path)


manager.initialize()

manager.connect(noise.get_output('data'), [
    filter_.get_input('data'),
])
manager.connect(filter_.get_output('data'), [
    mad.get_input('data'),
    detector.get_input('data'),
])
manager.connect(mad.get_output('mads'), [
    detector.get_input('data'),
])
manager.connect_network(detector)
manager.connect(detector.get_output('peaks'), [
    writer.get_input('peaks'),
])

manager.start()

director.sleep(duration=duration)
director.stop()
director.join()
director.destroy()
