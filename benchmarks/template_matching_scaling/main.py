# TODO add short description.

import argparse
from logging import DEBUG, INFO
import os
import circusort
import utils


# Parse command line.

parser = argparse.ArgumentParser()
parser.add_argument('--no-generation', dest='skip_generation', action='store_true')
parser.add_argument('--init-temp-dict', dest='init_temp_dict', action='store_true')
args = parser.parse_args()


# Parameters

host = '127.0.0.1'  # i.e. run the test locally

sampling_rate = 20e+3
nb_samples = 1024
r = {
    0: "r_ref*(t<t_a)+0.25*r_ref*(t_a<=t)*(t<t_b)+0.8*r_ref*(t_b<=t)*(t<t_d)+0.6*r_ref*(t_d<=t)",
    1: "r_ref*(t<t_a)+0.50*r_ref*(t_a<=t)*(t<t_b)+0.3*r_ref*(t_b<=t)*(t<t_c)+0.4*r_ref*(t_c<=t)",
    2: "r_ref*(t<t_a)+0.75*r_ref*(t_a<=t)*(t<t_b)+0.4*r_ref*(t_b<=t)*(t<t_d)+0.7*r_ref*(t_d<=t)",
}
def define_cell_obj(k):
    cell_obj = {
        'x': 'x_{}'.format(k),
        'y': 'y_{}'.format(k),
        'r': r[k],
        't': 'default',
    }
    return cell_obj
cells_args = [define_cell_obj(k) for k in range(0, 3)]
cells_params = {
    'x_0': +50.0, 'y_0': +50.0,
    'x_1': -50.0, 'y_1': +50.0,
    'x_2': -50.0, 'y_2': -50.0,
    'r_ref': sampling_rate / float(nb_samples),  # firing rate  # Hz
    't_a': 2 * 60 * 20,
    't_b': 5 * 60 * 20,
    't_c': 7 * 60 * 20,
    't_d': 8 * 60 * 20,
}

tmp_dir = os.path.join('/', 'tmp', 'spyking_circus_ort', 'template_matching_scaling')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

probe_path = os.path.join('..', 'mea_16.prb')

generator_kwargs = {
    'hdf5_path': os.path.join(tmp_dir, 'synthetic.h5'),
    'log_path': os.path.join(tmp_dir, 'synthetic.json'),
    'probe': probe_path,
}
raw_signal_writer_kwargs = {
    'data_path': os.path.join(tmp_dir, 'raw_signal.raw'),
}
signal_writer_kwargs = {
    'data_path': os.path.join(tmp_dir, 'signal.raw'),
}
mad_writer_kwargs = {
    'data_path': os.path.join(tmp_dir, 'mad.raw'),
}
peak_writer_kwargs = {
    'neg_peaks': os.path.join(tmp_dir, 'peaks.raw'),
}
if args.init_temp_dict:
    cluster_kwargs = {
        'nb_waveforms': 100000,  # i.e. delay clustering (template already exists)
    }
else:
    cluster_kwargs = {
        'nb_waveforms': 200,  # i.e. precipitate clustering (template does not exist)
    }
updater_kwargs = {
    'data_path': os.path.join(tmp_dir, 'templates.h5'),
}
if args.init_temp_dict:
    fitter_kwargs = {
        'init_path': os.path.join(tmp_dir, 'initial_templates.h5'),
        'with_rejected_times': True,
    }
else:
    fitter_kwargs = {}
spike_writer_kwargs = {
    'spike_times': os.path.join(tmp_dir, 'spike_times.raw'),
    'templates': os.path.join(tmp_dir, 'templates.raw'),
    'amplitudes': os.path.join(tmp_dir, 'amplitudes.raw'),
    'rejected_times': os.path.join(tmp_dir, 'rejected_times.raw'),
    'rejected_amplitudes': os.path.join(tmp_dir, 'rejected_amplitudes.raw'),
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
                                     seed=43,
                                     log_level=DEBUG,
                                     **generator_kwargs)
    raw_signal_writer = manager.create_block('writer',
                                             log_level=DEBUG,
                                             **raw_signal_writer_kwargs)
    filtering = manager.create_block('filter',
                                     cut_off=100.0,
                                     log_level=DEBUG)
    signal_writer = manager.create_block('writer',
                                         log_level=DEBUG,
                                         **signal_writer_kwargs)
    mad_estimator = manager.create_block('mad_estimator',
                                         log_level=DEBUG, 
                                         time_constant=10)
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
                               nb_waveforms=2000,
                               log_level=DEBUG)
    cluster = manager.create_block('density_clustering',
                                   threshold_factor=7.0,
                                   probe=probe_path,
                                   two_components=True,
                                   log_level=DEBUG,
                                   **cluster_kwargs)
    updater = manager.create_block('template_updater',
                                   probe=probe_path,
                                   nb_channels=16,
                                   log_level=INFO,
                                   **updater_kwargs)
    fitter = manager.create_block('template_fitter',
                                  two_components=True,
                                  log_level=INFO,
                                  **fitter_kwargs)
    spike_writer = manager.create_block('spike_writer',
                                        log_level=DEBUG,
                                        **spike_writer_kwargs)

    # Initialize the elements of the Circus network.

    director.initialize()

    # Connect the elements of the Circus network.

    director.connect(generator.output, [filtering.input,
                                        raw_signal_writer.input])
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
    director.sleep(duration=10.0+5.0*60.0)
    director.stop()
    # director.join()

# Analyze the results.

ans = utils.Results(generator_kwargs, raw_signal_writer_kwargs, signal_writer_kwargs,
                    mad_writer_kwargs, peak_writer_kwargs, updater_kwargs, spike_writer_kwargs)
