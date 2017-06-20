# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import numpy
import os
## In this example, we have 20 fixed neurons. 10 are active during the first 30s of the experiment, and then 10 new ones
## are appearing after 30s. The goal here is to study how the clsutering can handle such an discountinuous change


host = '127.0.0.1' # to run the test locally
data_path  = '/tmp/output.dat'
hdf5_path  = '/tmp/synthetic.h5'
peak_path  = '/tmp/peaks.dat'
thres_path = '/tmp/thresholds.dat'
temp_path  = '/tmp/templates'
probe_file = 'mea_4.prb'

for file in [data_path, hdf5_path, peak_path, thres_path]:
    if os.path.exists(file):
        os.remove(file)

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)

sampling_rate  = 20000
two_components = True
nb_samples     = 1024
nb_channels    = 4
sim_time       = 200

nb_cells   = 5
cell_obj_1 = {'r': 'r_ref*(t < tc)'}
cell_obj_2 = {'r': '2*r_ref*(t >= tc)'}

cells_params = {'r_ref': 10.0, # reference firing rate (i.e. mean firing rate)
                'tc'   : (sim_time/2.)*sampling_rate/nb_samples,
            }

cells_args = []

for i in xrange(2*nb_cells):
    if i < nb_cells:
        cells_args += [cell_obj_1]
    else:
        cells_args += [cell_obj_2]

generator     = manager.create_block('synthetic_generator', cells_args=cells_args, cells_params=cells_params, hdf5_path=hdf5_path, probe=probe_file)
filter        = manager.create_block('filter', cut_off=100)
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=5)
peak_fitter   = manager.create_block('peak_detector', threshold=5, safety_time=0)
pca           = manager.create_block('pca', nb_waveforms=1000)
cluster       = manager.create_block('density_clustering', probe=probe_file, nb_waveforms=500, two_components=two_components, log_level=logging.DEBUG)
updater       = manager.create_block('template_updater', probe=probe_file, data_path=temp_path, nb_channels=nb_channels)
fitter        = manager.create_block('template_fitter', log_level=logging.INFO, two_components=two_components)
writer        = manager.create_block('writer', data_path=data_path)
writer_2      = manager.create_block('spike_writer')
writer_3      = manager.create_block('peak_writer', neg_peaks=peak_path)
writer_4      = manager.create_block('writer', data_path=thres_path)

director.initialize()

director.connect(generator.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), peak_fitter.get_input('data'), cluster.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), peak_fitter.get_input('mads'), cluster.get_input('mads'), writer_4.input])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks')])
director.connect(peak_fitter.get_output('peaks'), [fitter.get_input('peaks'), writer_3.input])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))
director.connect(updater.get_output('updater'), fitter.get_input('updater'))
director.connect(fitter.output, writer_2.input)

director.start()
director.sleep(duration=sim_time)
director.stop()

start_time = mad_estimator.start_step
stop_time  = fitter.counter

from utils.analyzer import Analyzer
r = Analyzer(writer_2.recorded_data, probe_file, temp_path, synthetic_store=hdf5_path, filtered_data=data_path, threshold_data=thres_path, start_time=start_time, stop_time=stop_time)
