# Test to check that the parallelization in space (i.e. along the channels) for the peak detection works correctly. It
# allows to compare the computational efficiency of one block which detect the peaks among all the channels versus four
# blocks which detect the peaks among a quarter of all the channels.

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
data_dispatcher = manager.create_block('channel_dispatcher', name='Data dispatcher', nb_groups=nb_groups)
mad_dispatcher = manager.create_block('channel_dispatcher', name='MAD dispatcher', nb_groups=nb_groups)
detectors = [
    manager.create_block('peak_detector', name='Peak detector {}'.format(k))
    for k in range(0, nb_groups)
]
peak_grouper = manager.create_block('peak_grouper', nb_groups=nb_groups)
writer = manager.create_block('peak_writer', data_path=data_path)


manager.initialize()

manager.connect(noise.get_output('data'), [
    filter_.get_input('data'),
])
manager.connect(filter_.get_output('data'), [
    mad.get_input('data'),
    data_dispatcher.get_input('data'),
])
manager.connect(mad.get_output('mads'), [
    mad_dispatcher.get_input('data'),
])
for k in range(0, nb_groups):
    manager.connect(data_dispatcher.get_output('data_{}'.format(k)), [
        detectors[k].get_input('data')
    ])
    manager.connect(mad_dispatcher.get_output('data_{}'.format(k)), [
        detectors[k].get_input('mads')
    ])
for k in range(0, nb_groups):
    manager.connect(detectors[k].get_output('peaks'), [
        peak_grouper.get_input('peaks_{}'.format(k))
    ])
manager.connect(peak_grouper.get_output('peaks'), [
    writer.get_input('peaks'),
])

manager.start()

director.sleep(duration=duration)
director.stop()
director.join()
director.destroy()
