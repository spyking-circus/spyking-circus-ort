# coding=utf-8
import argparse
import os
import shutil

import circusort

from logging import DEBUG


directory = os.path.join("~", ".spyking-circus-ort", "examples", "read_n_display_peak_detection")
directory = os.path.expanduser(directory)


def main():

    # Define the working directory.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--probe-path', dest='probe_path', required=True)
    args = parser.parse_args()

    # Define log directory.
    log_directory = os.path.join(directory, "log")

    # Clean log directory (if necessary).
    if os.path.isdir(log_directory):
        shutil.rmtree(log_directory)
    os.makedirs(log_directory)

    # TODO remove the 2 following lines.
    # # Load generation parameters.
    # params = circusort.io.get_data_parameters(generation_directory)

    # Define parameters.
    host = '127.0.0.1'

    # Define keyword arguments.
    director_kwargs = {
        'log_path': os.path.join(log_directory, "log.txt")
    }
    reader_kwargs = {
        'name': "reader",
        'data_path': args.data_path,
        'dtype': 'int16',
        # 'nb_channels': 30,  # TODO remove?
        'nb_channels': 256,  # TODO auto?
        'nb_samples': 2000,
        'sampling_rate': 20e+3,
        'is_realistic': True,
        'log_level': DEBUG,
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 20.0,  # Hz
        'log_level': DEBUG,
    }
    mad_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        'log_level': DEBUG,
    }
    detector_kwargs = {
        'name': "detector",
        'threshold_factor': 7.0,
        'sampling_rate': 20e+3,
        'log_level': DEBUG,
    }
    peak_displayer_kwargs = {
        'name': "displayer",
        'probe_path': args.probe_path,
        'log_level': DEBUG,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=host, **director_kwargs)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    filter_ = manager.create_block('filter', **filter_kwargs)
    mad = manager.create_block('mad_estimator', **mad_kwargs)
    detector = manager.create_block('peak_detector', **detector_kwargs)
    peak_displayer = manager.create_block('peak_displayer', **peak_displayer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.get_output('data'), [
        filter_.get_input('data'),
    ])
    director.connect(filter_.get_output('data'), [
        mad.get_input('data'),
        detector.get_input('data'),
        peak_displayer.get_input('data'),
    ])
    director.connect(mad.get_output('mads'), [
        detector.get_input('mads'),
        peak_displayer.get_input('mads'),
    ])
    director.connect(detector.get_output('peaks'), [
        peak_displayer.get_input('peaks'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()

    return


if __name__ == '__main__':

    main()
