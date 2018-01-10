import os

import circusort

from logging import DEBUG, INFO


name = "network"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", "electrodes", name)
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "filter",
    "writer",
    # TODO complete.
]


def sorting(configuration_name):
    """Create the sorting network.

    Parameter:
        configuration_name: string
            The name of the configuration (i.e. context).
    """

    # Define directories.
    if not os.path.isdir(directory):
        message = "Directory does not exist: {}".format(directory)
        raise OSError(message)
    generation_directory = os.path.join(directory, "generation", configuration_name)
    sorting_directory = os.path.join(directory, "sorting", configuration_name)
    introspection_directory = os.path.join(directory, "introspection", configuration_name)

    # Load generation parameters.
    parameters = circusort.io.get_data_parameters(generation_directory)

    # Define parameters.
    host = '127.0.0.1'  # i.e. run the test locally
    dtype = parameters['general']['dtype']
    nb_channels = parameters['probe']['nb_channels']
    nb_samples = parameters['general']['buffer_width']
    sampling_rate = parameters['general']['sampling_rate']
    is_template_dictionary_initialized = False
    probe_path = os.path.join(generation_directory, "probe.prb")

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    if not os.path.isdir(introspection_directory):
        os.makedirs(introspection_directory)

    # Define keyword arguments.
    reader_kwargs = {
        'name': "reader",
        'data_path': os.path.join(directory, "data.raw"),
        'dtype': dtype,
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'introspection_path': introspection_directory,
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 100.0,  # Hz
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    signal_writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    mad_writer_kwargs = {
        'data_path': os.path.join(sorting_directory, "mads.h5"),
        'dataset_name': 'mads',
    }
    peak_writer_kwargs = {
        'data_path': os.path.join(sorting_directory, "peaks.h5"),
        'sampling_rate': sampling_rate,
    }
    if is_template_dictionary_initialized:
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
    if is_template_dictionary_initialized:
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

    # Define the elements of the network.
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    filtering = manager.create_block('filter', **filter_kwargs)
    signal_writer = manager.create_block('writer', **signal_writer_kwargs)
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
                                   probe_path=probe_path,
                                   two_components=False,
                                   log_level=DEBUG,
                                   **cluster_kwargs)
    updater = manager.create_block('template_updater',
                                   probe_path=probe_path,
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
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
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
    # Launch the network.
    director.start()
    director.join()
    director.destroy()

    # TODO remove the following lines (i.e. analysis)?
    # # TODO load data_filtered.raw?
    # # TODO load mads.h5?
    # templates = circusort.io.load_templates(updater_kwargs['data_path'])
    # nb_templates = len(templates)
    # # TODO save initial_template.h5?
    # # TODO load initial_template.h5?
    # spikes = circusort.io.load_spikes(spike_writer_kwargs['data_path'], nb_units=nb_templates)
    # sorted_cells = spikes.to_units()
    # sorted_cells.save(sorting_directory)
    #
    # generated_cells = circusort.io.load_cells(generation_directory)
    #
    # matching = circusort.utils.find_matching(sorted_cells, generated_cells, t_min=220.0, t_max=300.0)

    return
