# TODO add short description.

import argparse
from logging import DEBUG, INFO
import os

import circusort
import utils


# Parse command line.

parser = argparse.ArgumentParser()
parser.add_argument('--no-generation', dest='skip_generation', action='store_true')
args = parser.parse_args()


# Parameters

host = '127.0.0.1'  # i.e. run the test locally

cell_obj = {'r': 'r_ref'}
cells_args = [cell_obj]
cells_params = {'r_ref': 50.0}  # firing rate [Hz]

tmp_dir = os.path.join('/', 'tmp', 'spyking_circus_ort', 'one_neuron')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
# hdf5_path = os.path.join(tmp_dir, 'synthetic.h5')
probe_path = os.path.join('..', 'mea_16.prb')
# signal_path = os.path.join(tmp_dir, 'signal.raw')
# mad_path = os.path.join(tmp_dir, 'mad.raw')
# peak_path = os.path.join(tmp_dir, 'peaks.raw')
# temp_path = os.path.join(tmp_dir, 'templates')

generator_kwargs = {
    'hdf5_path': os.path.join(tmp_dir, 'synthetic.h5'),
    'log_path': os.path.join(tmp_dir, 'synthetic.json'),
    'probe': probe_path,
}
signal_writer_kwargs = {
    'data_path': os.path.join(tmp_dir, 'signal.raw'),
}
mad_writer_kwargs = {
    'data_path': os.path.join(tmp_dir, 'mad.raw')
}
peak_writer_kwargs = {
    'neg_peaks': os.path.join(tmp_dir, 'peaks.raw')
}
updater_kwargs = {
    'data_path': os.path.join(tmp_dir, 'templates')
}
spike_writer_kwargs = {
    'spike_times': os.path.join(tmp_dir, 'spike_times.raw'),
    'templates': os.path.join(tmp_dir, 'templates.raw'),
    'amplitudes': os.path.join(tmp_dir, 'amplitudes.raw'),
}

if args.skip_generation:

    print("Warning: generation skipped.")

else:

    # Define the elements of the Circus network.

    director = circusort.create_director(host=host)

    manager = director.create_manager(host=host)

    generator = manager.create_block('synthetic_generator',
                                     cells_args=cells_args,
                                     cells_params=cells_params,
                                     log_level=DEBUG,
                                     **generator_kwargs)
    filtering = manager.create_block('filter',
                                     cut_off=100.0,
                                     log_level=DEBUG)
#     whitening = manager.create_block('whitening',
#                                      log_level=DEBUG)
    signal_writer = manager.create_block('writer',
                                         log_level=DEBUG,
                                         **signal_writer_kwargs)
    mad_estimator = manager.create_block('mad_estimator',
                                         log_level=DEBUG)
    mad_writer = manager.create_block('writer',
                                      log_level=DEBUG,
                                      **mad_writer_kwargs)
    peak_detector = manager.create_block('peak_detector',
                                         threshold_factor=7.0,
                                         log_level=DEBUG)
    peak_writer = manager.create_block('peak_writer',
                                       log_level=INFO,
                                       **peak_writer_kwargs)
    pca = manager.create_block('pca',
                               nb_waveforms=100,  # 1000
                               log_level=DEBUG)
    cluster = manager.create_block('density_clustering',
                                   threshold_factor=7.0,
                                   probe=probe_path,
                                   nb_waveforms=100,  # 500
                                   two_components=False,
                                   log_level=DEBUG)
    updater = manager.create_block('template_updater',
                                   probe=probe_path,
                                   nb_channels=16,
                                   log_level=INFO,
                                   **updater_kwargs)
    fitter = manager.create_block('template_fitter',
                                  two_components=False,
                                  log_level=DEBUG)
    spike_writer = manager.create_block('spike_writer',
                                        log_level=DEBUG,
                                        **spike_writer_kwargs)


    # Initialize the elements of the Circus network.

    director.initialize()


    # Connect the elements of the Circus network.

    director.connect(generator.output, filtering.input)
    # director.connect(filtering.output, whitening.input)
    # director.connect(whitening.output, [mad_estimator.input,
    #                                     peak_detector.get_input('data'),
    #                                     cluster.get_input('data'),
    #                                     pca.get_input('data'),
    #                                     fitter.get_input('data')])
    director.connect(filtering.output, [mad_estimator.input,
                                        peak_detector.get_input('data'),
                                        cluster.get_input('data'),
                                        pca.get_input('data'),
                                        fitter.get_input('data'),
                                        signal_writer.input])
    director.connect(mad_estimator.output, [peak_detector.get_input('mads'),
                                            cluster.get_input('mads'),
                                            mad_writer.input])
    director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'),
                                                         cluster.get_input('peaks'),
                                                         fitter.get_input('peaks'),
                                                         peak_writer.input])
    director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
    director.connect(cluster.get_output('templates'), updater.get_input('templates'))
    director.connect(updater.get_output('updater'), fitter.get_input('updater'))
    director.connect(fitter.output, spike_writer.input)


    # Launch the Circus network.

    director.start()
    director.sleep(duration=60.0)
    director.stop()
    # director.join()


# Analyze the results.

ans = utils.Results(generator_kwargs, signal_writer_kwargs, mad_writer_kwargs,
                    peak_writer_kwargs, updater_kwargs, spike_writer_kwargs)
