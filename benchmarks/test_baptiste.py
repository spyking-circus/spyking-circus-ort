import os
import logging

import circusort


host = '127.0.0.1'

director = circusort.create_director(host=host)
manager = director.create_manager(host=host)

sampling_rate = 20.0e+3  # Hz
# two_components = True
nb_samples = 1024
# nb_channels = 16
simulation_duration = 10.0  # s

cells_args = [{'r': 'r_ref'}]
cells_params = {'r_ref': 100.0}  # Hz

hdf5_path = None
# probe_path = "mea_16_copy.prb"
probe_path = "mea_4_copy.prb"
tmp_dirname = circusort.io.get_tmp_dirname()
generator_path = os.path.join(tmp_dirname, "generator.dat")
filter_path = os.path.join(tmp_dirname, "filter.dat")
whitening_path = os.path.join(tmp_dirname, "whitening.dat")
peak_detector_path = os.path.join(tmp_dirname, "peak_detector.dat")
peak_fitter_path = os.path.join(tmp_dirname, "peak_fitter.dat")
mad_estimator_path = os.path.join(tmp_dirname, "mad_estimator.dat")

generator = manager.create_block('synthetic_generator', cells_args=cells_args, cells_params=cells_params,
                                 hdf5_path=hdf5_path, probe=probe_path, log_level=logging.DEBUG)
filter_1 = manager.create_block('filter', cut_off=100, log_level=logging.DEBUG)
whitening = manager.create_block('whitening', log_level=logging.DEBUG)
mad_estimator = manager.create_block('mad_estimator', log_level=logging.DEBUG)
peak_detector = manager.create_block('peak_detector', threshold=5, log_level=logging.DEBUG)
peak_fitter = manager.create_block('peak_detector', threshold=5, safety_time=0)
writer = manager.create_block('writer', data_path=generator_path)
writer_2 = manager.create_block('peak_writer', neg_peaks=peak_detector_path)
writer_3 = manager.create_block('peak_writer', neg_peaks=peak_fitter_path)
writer_4 = manager.create_block('writer', data_path=mad_estimator_path)
writer_5 = manager.create_block('writer', data_path=filter_path)
writer_6 = manager.create_block('writer', data_path=whitening_path)


director.initialize()


director.connect(generator.output, [filter_1.input, writer.input])
director.connect(filter_1.output, [whitening.input, writer_5.input])
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'),
                                    peak_fitter.get_input('data'), writer_6.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), peak_fitter.get_input('mads'), writer_4.input])
director.connect(peak_detector.get_output('peaks'), [writer_2.input])
director.connect(peak_fitter.get_output('peaks'), [writer_3.input])


director.start()
director.sleep(duration=30.0)
director.stop()


# director.destroy()
