import os

import circusort

from logging import DEBUG, INFO


name = "network"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "local")
directory = os.path.expanduser(directory)

nb_fitters = 4
nb_replay = 4

def sorting(configuration_name):
    """Create the 1st sorting subnetwork.

    Parameter:
        configuration_name: string
            The name of the configuration (i.e. context).
    """

    # Define directories.
    generation_directory = os.path.join(directory, "generation", configuration_name)
    sorting_directory = os.path.join(directory, "sorting", configuration_name)
    introspection_directory = os.path.join(directory, "introspection", configuration_name)
    log_directory = os.path.join(directory, "log", configuration_name)

    # Load generation parameters.
    parameters = circusort.io.get_data_parameters(generation_directory)

    # Define parameters.
    hosts = {
        'master': '127.0.0.1',
    }
    hosts_keys = [  # ordered
        'master',
    ]
    dtype = parameters['general']['dtype']
    nb_channels = parameters['probe']['nb_channels']
    nb_samples = parameters['general']['buffer_width']
    sampling_rate = parameters['general']['sampling_rate']
    threshold_factor = 7.0
    probe_path = os.path.join(generation_directory, "probe.prb")
    precomputed_template_paths = [
        cell.template.path
        for cell in circusort.io.load_cells(generation_directory)
    ]

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    if not os.path.isdir(introspection_directory):
        os.makedirs(introspection_directory)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)

    # Define keyword arguments.
    director_kwargs = {
        'log_path': os.path.join(log_directory, "log.txt"),
    }
    reader_kwargs = {
        'name': "reader",
        'data_path': os.path.join(generation_directory, "data.raw"),
        'dtype': dtype,
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
        'nb_replay': nb_replay
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 1.0,  # Hz
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    mad_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    detector_kwargs = {
        'name': "detector",
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    peak_writer_kwargs = {
        'name': "peak_writer",
        'data_path': os.path.join(sorting_directory, "peaks.h5"),
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    pca_kwargs = {
        'name': "pca",
        'nb_waveforms': 200,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_kwargs = {
        'name': "cluster",
        'sampling_rate': sampling_rate,
        'nb_waveforms': 200,
        'probe_path': probe_path,
        'two_components': False,
        'introspection_path': introspection_directory,
        'log_level': INFO,
    }
    updater_kwargs = {
        'name': "updater",
        'probe_path': probe_path,
        'templates_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_path': os.path.join(sorting_directory, "overlaps.p"),
        #'precomputed_template_paths': precomputed_template_paths,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    fitter_kwargs = {
        'name': "fitter",
        'degree': nb_fitters,
        # 'templates_init_path': os.path.join(sorting_directory, "templates.h5"),
        # 'overlaps_init_path': os.path.join(sorting_directory, "overlaps.p"),
        'sampling_rate': sampling_rate,
        'discarding_eoc_from_updater': True,
        'introspection_path': introspection_directory,
        'log_level': INFO,
    }
    writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "spikes.h5"),
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': INFO,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=hosts['master'], **director_kwargs)
    managers = {
        key: director.create_manager(host=hosts[key])
        for key in hosts_keys
    }
    reader = managers['master'].create_block('reader', **reader_kwargs)
    filter_ = managers['master'].create_block('filter', **filter_kwargs)
    mad = managers['master'].create_block('mad_estimator', **mad_kwargs)
    detector = managers['master'].create_block('peak_detector', **detector_kwargs)
    peak_writer = managers['master'].create_block('peak_writer', **peak_writer_kwargs)
    pca = managers['master'].create_block('pca', **pca_kwargs)
    cluster = managers['master'].create_block('density_clustering', **cluster_kwargs)
    updater = managers['master'].create_block('template_updater', **updater_kwargs)
    fitter = managers['master'].create_network('fitter', **fitter_kwargs)
    writer = managers['master'].create_block('spike_writer', **writer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.output, [
        filter_.input
    ])
    director.connect(filter_.output, [
        mad.input,
        detector.get_input('data'),
        pca.get_input('data'),
        cluster.get_input('data'),
        fitter.get_input('data'),
    ])
    director.connect(mad.output, [
        detector.get_input('mads')
    ])
    director.connect(detector.get_output('peaks'), [
        peak_writer.get_input('peaks'),
        pca.get_input('peaks'),
        cluster.get_input('peaks'),
        fitter.get_input('peaks'),
    ])
    director.connect(pca.get_output('pcs'), [
        cluster.get_input('pcs'),
    ])
    director.connect(cluster.get_output('templates'), [
        updater.get_input('templates'),
    ])
    director.connect(updater.get_output('updater'), [
        fitter.get_input('updater'),
    ])
    director.connect_network(fitter)
    director.connect(fitter.get_output('spikes'), [
        writer.get_input('spikes'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
