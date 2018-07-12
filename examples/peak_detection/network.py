import os

import circusort

from logging import DEBUG


def detection(directory):

    # Define directories.
    generation_directory = os.path.join(directory, "generation")
    detection_directory = os.path.join(directory, "detection")
    log_directory = os.path.join(directory, "log")

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

    # Create directories (if necessary).
    if not os.path.isdir(detection_directory):
        os.makedirs(detection_directory)
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
        'log_level': DEBUG,
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 1.0,  # Hz
        'log_level': DEBUG,
    }
    mad_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        'log_level': DEBUG,
    }
    detector_kwargs = {
        'name': "detector",
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        'log_level': DEBUG,
    }
    peak_writer_kwargs = {
        'name': "peak_writer",
        'data_path': os.path.join(detection_directory, "peaks.h5"),
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
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.output, [
        filter_.input
    ])
    director.connect(filter_.output, [
        mad.input,
        detector.get_input('data'),
    ])
    director.connect(mad.output, [
        detector.get_input('mads'),
    ])
    director.connect(detector.get_output('peaks'), [
        peak_writer.get_input('peaks'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()

    return
