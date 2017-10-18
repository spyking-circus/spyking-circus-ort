# TODO add short description.

from logging import DEBUG, INFO
import os

import circusort
import utils


# Parameters

host = '127.0.0.1'  # i.e. run the test locally

cell_obj = {'r': 'r_ref'}
cells_args = [cell_obj]
cells_params = {'r_ref': 1.0}  # firing rate [Hz]

tmp_dir = os.path.join('/', 'tmp', 'spyking_circus_ort', 'one_neuron')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
hdf5_path = os.path.join(tmp_dir, 'synthetic.h5')
probe_path = os.path.join('..', 'mea_16.prb')
temp_path = os.path.join(tmp_dir, 'templates')


# Define the elements of the Circus network.

director = circusort.create_director(host=host)

manager = director.create_manager(host=host)

generator = manager.create_block('synthetic_generator',
                                 cells_args=cells_args,
                                 cells_params=cells_params,
                                 hdf5_path=hdf5_path,
                                 probe=probe_path,
                                 log_level=DEBUG)
filtering = manager.create_block('filter',
                                 cut_off=100.0,
                                 log_level=DEBUG)
whitening = manager.create_block('whitening',
                                 log_level=DEBUG)
mad_estimator = manager.create_block('mad_estimator',
                                     log_level=DEBUG)
peak_detector = manager.create_block('peak_detector',
                                     threshold=5.0,
                                     log_level=DEBUG)
pca = manager.create_block('pca',
                           nb_waveforms=1000,
                           log_level=DEBUG)
cluster = manager.create_block('density_clustering',
                               probe=probe_path,
                               nb_waveforms=500,
                               two_components=True,
                               log_level=DEBUG)
updater = manager.create_block('template_updater',
                               probe=probe_path,
                               data_path=temp_path,
                               nb_channels=16,
                               log_level=INFO)
fitter = manager.create_block('template_fitter',
                              two_components=True,
                              log_level=INFO)
spike_writer = manager.create_block('spike_writer',
                                    directory=tmp_dir,
                                    log_level=INFO)


# Initialize the elements of the Circus network.

director.initialize()


# Connect the elements of the Circus network.

director.connect(generator.output, filtering.input)
director.connect(filtering.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input,
                                    peak_detector.get_input('data'),
                                    cluster.get_input('data'),
                                    pca.get_input('data'),
                                    fitter.get_input('data')])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'),
                                        cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'),
                                                     cluster.get_input('peaks'),
                                                     fitter.get_input('peaks')])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))
director.connect(updater.get_output('updater'), fitter.get_input('updater'))
director.connect(fitter.output, spike_writer.input)


# Launch the Circus network.

director.start()
director.sleep(duration=30.0)
director.stop()
director.join()


# Analyze the results.

ans = utils.Results(generator, spike_writer, probe_path, temp_path)
