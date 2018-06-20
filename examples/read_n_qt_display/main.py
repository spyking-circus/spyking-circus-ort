# coding=utf-8
import argparse
import os
import shutil

import circusort

from logging import DEBUG


directory = os.path.join("~", ".spyking-circus-ort", "examples", "read_n_qt_display")
directory = os.path.expanduser(directory)


def main():

    # Define the working directory.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--display', dest='pending_display', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_configuration is None and args.pending_generation is None and args.pending_display is None:
        args.pending_configuration = True
        args.pending_generation = True
        args.pending_display = True
    else:
        args.pending_configuration = args.pending_configuration is True
        args.pending_generation = args.pending_generation is True
        args.pending_display = args.pending_display is True

    configuration_directory = os.path.join(directory, "configuration")

    if args.pending_configuration:

        # Clean configuration directory (if necessary).
        if os.path.isdir(configuration_directory):
            shutil.rmtree(configuration_directory)
        os.makedirs(configuration_directory)

        # Generate configuration.
        kwargs = {
            'general': {
                'duration': 5.0,  # s
                'name': "read_n_display",
            },
            'probe': {
                'mode': 'mea',
                'nb_rows': 3,
                'nb_columns': 3,
                'radius': 100.0,  # µm
            },
            'cells': {
                'nb_cells': 27,
            }
        }
        configuration = circusort.io.generate_configuration(**kwargs)
        configuration.save(configuration_directory)

    generation_directory = os.path.join(directory, "generation")

    if args.pending_generation:

        # Clean generation directory (if necessary).
        if os.path.isdir(generation_directory):
            shutil.rmtree(generation_directory)
        os.makedirs(generation_directory)

        circusort.net.pregenerator(
            configuration_directory=configuration_directory,
            generation_directory=generation_directory,
        )

    if args.pending_display:

        # Define log directory.
        log_directory = os.path.join(directory, "log")

        # Clean log directory (if necessary).
        if os.path.isdir(log_directory):
            shutil.rmtree(log_directory)
        os.makedirs(log_directory)

        # Load generation parameters.
        params = circusort.io.get_data_parameters(generation_directory)

        # Define parameters.
        host = '127.0.0.1'

        # Define keyword arguments.
        director_kwargs = {
            'log_path': os.path.join(log_directory, "log.txt")
        }
        reader_kwargs = {
            'name': "reader",
            'data_path': os.path.join(generation_directory, "data.raw"),
            'dtype': params['general']['dtype'],
            'nb_channels': params['probe']['nb_channels'],
            'nb_samples': params['general']['buffer_width'],
            'sampling_rate': params['general']['sampling_rate'],
            'is_realistic': True,
            'log_level': DEBUG,
        }
        qt_displayer_kwargs = {
            'name': "displayer",
            'log_level': DEBUG,
        }

        # Define the elements of the network.
        director = circusort.create_director(host=host, **director_kwargs)
        manager = director.create_manager(host=host)
        reader = manager.create_block('reader', **reader_kwargs)
        qt_displayer = manager.create_block('qt_displayer', **qt_displayer_kwargs)
        # Initialize the elements of the network.
        director.initialize()
        # Connect the elements of the network.
        director.connect(reader.get_output('data'), [
            qt_displayer.get_input('data'),
        ])
        # Launch the network.
        director.start()
        director.join()
        director.destroy()

    return


if __name__ == '__main__':

    main()
