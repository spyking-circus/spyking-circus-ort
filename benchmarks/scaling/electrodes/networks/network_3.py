import os

import circusort

from logging import DEBUG, INFO


name = "network_3"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", "electrodes", name)
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "filter",
    "mad",
    "detector",
    "pca",
    # "cluster",
    # "updater",
    "fitter",
    "writer",
]


def sorting(configuration_name):
    """Create the 1st sorting subnetwork.

    Parameter:
        configuration_name: string
            The name of the configuration (i.e. context).
    """

    # Define directories.
    generation_directory = os.path.join(directory, "generation", configuration_name)
    sorting_directory = os.path.join(directory, "sorting", configuration_name)
    log_directory = os.path.join(directory, "log", configuration_name)
    introspection_directory = os.path.join(directory, "introspection", configuration_name)

    # Load generation parameters.
    parameters = circusort.io.get_data_parameters(generation_directory)

    # Define parameters.
    host = '127.0.0.1'  # i.e. run the test locally
    dtype = parameters['general']['dtype']
    nb_channels = parameters['probe']['nb_channels']
    nb_samples = parameters['general']['buffer_width']
    sampling_rate = parameters['general']['sampling_rate']
    threshold_factor = 7.0
    alignment = True
    spike_width = 5.0  # ms
    spike_jitter = 1.0  # ms
    spike_sigma = 2.75  # µV
    probe_path = os.path.join(generation_directory, "probe.prb")
    precomputed_template_paths = [
        cell.template.path
        for cell in circusort.io.load_cells(generation_directory)
    ]

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    if not os.path.isdir(introspection_directory):
        os.makedirs(introspection_directory)

    # Define keyword arguments.
    director_kwargs = {
        'log_path': os.path.join(log_directory, "log.txt"),
        'log_level': INFO,
    }
    reader_kwargs = {
        'name': "reader",
        'data_path': os.path.join(generation_directory, "data.raw"),
        'dtype': dtype,
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'speed_factor': 1.0,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 1.0,  # Hz
        'order': 1,
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
    pca_kwargs = {
        'name': "pca",
        'spike_width': spike_width,
        'spike_jitter': spike_jitter,
        'spike_sigma': spike_sigma,
        'nb_waveforms': 2000,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_kwargs = {
        'name': "cluster",
        'threshold_factor': threshold_factor,
        'alignment': alignment,
        'spike_width': spike_width,
        'spike_jitter': spike_jitter,
        'spike_sigma': spike_sigma,
        'sampling_rate': sampling_rate,
        'nb_waveforms': 100000,
        'probe_path': probe_path,
        'two_components': False,
        'local_merges': 3,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    updater_kwargs = {
        'name': "updater",
        'probe_path': probe_path,
        'templates_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_path': os.path.join(sorting_directory, "overlaps.p"),
        'precomputed_template_paths': precomputed_template_paths,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    fitter_kwargs = {
        'name': "fitter",
        'sampling_rate': sampling_rate,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "spikes.h5"),
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=host, **director_kwargs)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    filter_ = manager.create_block('filter', **filter_kwargs)
    mad = manager.create_block('mad_estimator', **mad_kwargs)
    detector = manager.create_block('peak_detector', **detector_kwargs)
    pca = manager.create_block('pca', **pca_kwargs)
    cluster = manager.create_block('density_clustering', **cluster_kwargs)
    updater = manager.create_block('template_updater', **updater_kwargs)
    fitter = manager.create_block('template_fitter', **fitter_kwargs)
    writer = manager.create_block('spike_writer', **writer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.get_output('data'), [
        filter_.get_input('data'),
    ])
    director.connect(filter_.get_output('data'), [
        mad.get_input('data'),
        detector.get_input('data'),
        pca.get_input('data'),
        cluster.get_input('data'),
        fitter.get_input('data'),
    ])
    director.connect(mad.get_output('mads'), [
        detector.get_input('mads'),
        cluster.get_input('mads'),
    ])
    director.connect(detector.get_output('peaks'), [
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
    director.connect(fitter.get_output('spikes'), [
        writer.get_input('spikes'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
