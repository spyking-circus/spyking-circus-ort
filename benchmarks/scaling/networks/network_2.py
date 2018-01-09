import os

import circusort

from logging import DEBUG, INFO


name = "network_2"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", name)
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "filter",
    "mad",
    "detector",
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
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 100.0,  # Hz
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    mad_estimator_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    peak_detector_kwargs = {
        'name': "detector",
        'threshold_factor': 7.0,
        'sampling_rate': sampling_rate,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    peak_writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "peaks.h5"),
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': INFO,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    filter_ = manager.create_block('filter', **filter_kwargs)
    mad_estimator = manager.create_block('mad_estimator', **mad_estimator_kwargs)
    peak_detector = manager.create_block('peak_detector', **peak_detector_kwargs)
    peak_writer = manager.create_block('peak_writer', **peak_writer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.output, [
        filter_.input
    ])
    director.connect(filter_.output, [
        mad_estimator.input,
        peak_detector.get_input('data'),
    ])
    director.connect(mad_estimator.output, [
        peak_detector.get_input('mads')
    ])
    director.connect(peak_detector.get_output('peaks'), [
        peak_writer.input,
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
