from logging import DEBUG

import circusort


host = '127.0.0.1'

director = circusort.create_director(host=host)
manager = director.create_manager(host=host)

sampling_rate = 20.0e+3  # Hz
nb_samples = 1024  # number of samples per buffer
simulation_duration = 10.0  # s

cells_args = [
    {
        'r': 'r_ref',
        's': 0.99 * float(nb_samples) / sampling_rate,
        't': 'periodic',
    }
]
cells_params = {
    # 'r_ref': 100.0,  # Hz
    'r_ref': sampling_rate / float(nb_samples),  # Hz
}
hdf5_path = None
probe_path = "mea_4_copy.prb"

generator = manager.create_block('synthetic_generator', cells_args=cells_args, cells_params=cells_params,
                                 hdf5_path=hdf5_path, probe=probe_path, log_level=DEBUG)
filter_ = manager.create_block('filter', cut_off=100, log_level=DEBUG)
whitening = manager.create_block('whitening', log_level=DEBUG)
mad_estimator = manager.create_block('mad_estimator', log_level=DEBUG)
peak_detector = manager.create_block('peak_detector', threshold=5, log_level=DEBUG)
mua_viewer = manager.create_block('mua_viewer', probe=probe_path, nb_samples=nb_samples, log_level=DEBUG)

director.initialize()


director.connect(generator.output, [filter_.input])
director.connect(filter_.output, [whitening.input])
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data')])
director.connect(mad_estimator.output, [peak_detector.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [mua_viewer.get_input('peaks')])


director.start()
director.sleep(duration=simulation_duration)
director.stop()


director.destroy()
