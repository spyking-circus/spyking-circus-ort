# TODO add short description.

import argparse
from logging import DEBUG, INFO
import os
import circusort
# import utils


# Parse command line.
# TODO check the following lines.
parser = argparse.ArgumentParser()
parser.add_argument('--no-realism', dest='is_realistic', action='store_false', default=True)
parser.add_argument('--no-sorting', dest='skip_sorting', action='store_true', default=False)
parser.add_argument('--init-temp-dict', dest='init_temp_dict', action='store_true')
args = parser.parse_args()


# Define parameters
host = '127.0.0.1'  # i.e. run the test locally
nb_channels = 16
nb_samples = 1024
sampling_rate = 20e+3  # Hz

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling")
directory = os.path.expanduser(directory)
if not os.path.isdir(directory):
    message = "Directory does not exist: {}".format(directory)
    raise OSError(message)
generation_directory = os.path.join(directory, "generation")
probe_path = os.path.join(generation_directory, "probe.prb")
if not os.path.isfile(probe_path):
    message = "File does not exist: {}".format(directory)
    raise OSError(message)
sorting_directory = os.path.join(directory, "sorting")
if not os.path.isdir(sorting_directory):
    os.makedirs(sorting_directory)


# Define keyword arguments.
reader_kwargs = {
    'data_path': os.path.join(directory, "data.raw"),
    'dtype': 'int16',
    'nb_channels': nb_channels,
    'nb_samples': nb_samples,
    'sampling_rate': sampling_rate,
    'is_realistic': args.is_realistic,
}
signal_writer_kwargs = {
    'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
}
mad_writer_kwargs = {
    'data_path': os.path.join(sorting_directory, "mads.h5"),
    'name': 'mads',
}
peak_writer_kwargs = {
    'data_path': os.path.join(sorting_directory, "peaks.h5"),
    'sampling_rate': sampling_rate,
}
if args.init_temp_dict:
    cluster_kwargs = {
        'nb_waveforms': 100000,  # i.e. delay clustering (template already exists)
    }
else:
    cluster_kwargs = {
        'nb_waveforms': 100,  # i.e. precipitate clustering (template does not exist)
    }
updater_kwargs = {
    'data_path': os.path.join(sorting_directory, "templates.h5"),
}
if args.init_temp_dict:
    fitter_kwargs = {
        # TODO correct the following line.
        'init_path': os.path.join(directory, "initial_templates.h5"),
        'with_rejected_times': True,
    }
else:
    fitter_kwargs = {}
spike_writer_kwargs = {
    'data_path': os.path.join(sorting_directory, "spikes.h5"),
    'sampling_rate': sampling_rate,
}


if args.skip_sorting:

    print("Warning: sorting skipped.")

else:

    # Define the elements of the Circus network.
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader',
                                  **reader_kwargs)
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
                                   two_components=False,
                                   log_level=DEBUG,
                                   **cluster_kwargs)
    updater = manager.create_block('template_updater',
                                   probe=probe_path,
                                   nb_channels=16,
                                   log_level=DEBUG,
                                   **updater_kwargs)
    fitter = manager.create_block('template_fitter',
                                  two_components=False,
                                  log_level=DEBUG,
                                  **fitter_kwargs)
    spike_writer = manager.create_block('spike_writer',
                                        log_level=DEBUG,
                                        **spike_writer_kwargs)

    # Initialize the elements of the Circus network.
    director.initialize()

    # Connect the elements of the Circus network.
    director.connect(reader.output, [filtering.input])
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
    director.join()
    director.destroy()

# TODO load data_filtered.raw?
# TODO load mads.h5?
# TODO load templates.h5?
templates = circusort.io.load_templates(updater_kwargs['data_path'])
# TODO save initial_template.h5?
# TODO load initial_template.h5?
spikes = circusort.io.load_spikes(spike_writer_kwargs['data_path'])
sorted_cells = spikes.to_units()
sorted_cells.save(sorting_directory)

generated_cells = circusort.io.load_cells(generation_directory)

matching = circusort.utils.find_matching(sorted_cells, generated_cells, t_min=220.0, t_max=300.0)

# # Analyze the results.
#
# ans = utils.Results(generator_kwargs, raw_signal_writer_kwargs, signal_writer_kwargs,
#                     mad_writer_kwargs, peak_writer_kwargs, updater_kwargs, spike_writer_kwargs)
