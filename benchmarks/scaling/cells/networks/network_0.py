import os

import circusort

from logging import DEBUG


name = "network_0"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", "cells", name)
directory = os.path.expanduser(directory)

block_names = [
    "reader",
    "writer",
]


def sorting(configuration_name):
    """Create the 2nd sorting subnetwork.

    Parameter:
        configuration_name: string
            The name of the configuration (i.e. context).
    """

    # Create directories.
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
    signal_writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }

    # Define the elements of the Circus network.
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    writer = manager.create_block('writer', **signal_writer_kwargs)
    # Initialize the elements of the Circus network.
    director.initialize()
    # Connect the elements of the Circus network.
    director.connect(reader.output, writer.input)
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
