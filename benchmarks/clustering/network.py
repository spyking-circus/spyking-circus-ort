import os

import circusort

from logging import DEBUG


name = "network"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "clustering")
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "filter",
    "mad",
    "detector",
    "pca",
    "cluster",
    "cluster_writer",
]
block_nb_buffers = {}
block_labels = {}


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
        'nb_waveforms': 10000,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_kwargs = {
        'name': "cluster",
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        'nb_waveforms': 100,
        'probe_path': probe_path,
        'two_components': False,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_writer_kwargs = {
        'name': "cluster_writer",
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    updater_bis_kwargs = {
        'name': "updater_bis",
        'probe_path': probe_path,
        'templates_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_path': os.path.join(sorting_directory, "overlaps.p"),
        'precomputed_template_paths': precomputed_template_paths,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    updater_writer_kwargs = {
        'name': "updater_writer",
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
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
    cluster_writer = managers['master'].create_block('cluster_writer', **cluster_writer_kwargs)
    updater = managers['master'].create_block('template_updater_bis', **updater_bis_kwargs)
    updater_writer = managers['master'].creatr_block('updater_writer', **updater_writer_kwargs)
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
    ])
    director.connect(mad.output, [
        detector.get_input('mads'),
        cluster.get_input('mads'),
    ])
    director.connect(detector.get_output('peaks'), [
        peak_writer.get_input('peaks'),
        pca.get_input('peaks'),
        cluster.get_input('peaks'),
    ])
    director.connect(pca.get_output('pcs'), [
        cluster.get_input('pcs'),
    ])
    director.connect(cluster.get_output('templates'), [
        cluster_writer.get_input('templates'),
        updater.get_input('templates'),
    ])
    director.connect(updater.get_output('updater'), [
        updater_writer.get_input('updater'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
