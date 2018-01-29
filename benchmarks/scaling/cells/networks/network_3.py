import os

import circusort

from logging import DEBUG


name = "network_3"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", "cells", name)
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "filter",
    "mad",
    "detector",
    "pca",
    "cluster",
    "updater",
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

    # Define keyword arguments.
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
        'cut_off': 0.0,  # Hz
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
        'nb_waveforms': 100000,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_kwargs = {
        'name': "cluster",
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        'nb_waveforms': 100000,
        'probe_path': probe_path,
        'two_components': False,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    updater_kwargs = {
        'name': "updater",
        'probe_path': probe_path,
        'data_path': os.path.join(sorting_directory, "templates.h5"),
        'precomputed_template_paths': precomputed_template_paths,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    fitter_kwargs = {
        'name': "fitter",
        'sampling_rate': sampling_rate,
        'discarding_eoc_from_updater': True,
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
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    filter_ = manager.create_block('filter', **filter_kwargs)
    mad = manager.create_block('mad_estimator', **mad_kwargs)
    detector = manager.create_block('peak_detector', **detector_kwargs)
    # TODO implement `_introspect` for the following blocks.
    pca = manager.create_block('pca', **pca_kwargs)
    cluster = manager.create_block('density_clustering', **cluster_kwargs)
    updater = manager.create_block('template_updater', **updater_kwargs)
    fitter = manager.create_block('template_fitter', **fitter_kwargs)
    writer = manager.create_block('spike_writer', **writer_kwargs)
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
    director.connect(fitter.output, [
        writer.input,
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
