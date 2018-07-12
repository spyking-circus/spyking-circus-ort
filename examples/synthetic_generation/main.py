# -*- coding=utf-8 -*-
import argparse
import os
import shutil

import circusort

from circusort.io.datafile import load_datafile


directory = os.path.join("~", ".spyking-circus-ort", "examples", "synthetic_generation")
directory = os.path.expanduser(directory)

nb_rows = 8
nb_columns = 8
radius = 100.0  # µm
nb_cells = 128
duration = 5.0 * 60.0  # s


def main():

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

    # Create the working directory (if necessary).
    if not os.path.isdir(directory):
        os.makedirs(directory)
    configuration_directory = os.path.join(directory, "configuration")

    if args.pending_configuration:

        # Clean the configuration directory (if necessary).
        if os.path.isdir(configuration_directory):
            shutil.rmtree(configuration_directory)
        os.makedirs(configuration_directory)

        # Generate configuration.
        kwargs = {
            'general': {
                'duration': duration,
                'name': 'synthetic_generation',
            },
            'probe': {
                'mode': 'mea',
                'nb_rows': nb_rows,
                'nb_columns': nb_columns,
                'radius': radius,
            },
            'cells': {
                'nb_cells': nb_cells,
            }
        }
        configuration = circusort.io.generate_configuration(**kwargs)
        configuration.save(configuration_directory)

    configuration_directory = os.path.join(directory, "configuration")
    generation_directory = os.path.join(directory, "generation")

    # Generate data (if necessary).
    if args.pending_generation:

        circusort.net.pregenerator(configuration_directory=configuration_directory,
                                   generation_directory=generation_directory)

    # Display data (if necessary).
    if args.pending_display:

        # Load generation parameters.
        parameters = circusort.io.get_data_parameters(generation_directory)

        # Define parameters.
        sampling_rate = parameters['general']['sampling_rate']
        nb_channels = parameters['probe']['nb_channels']

        # Load the data.
        data_path = os.path.join(generation_directory, "data.raw")
        data = load_datafile(data_path, sampling_rate, nb_channels, 'int16', 0.1042)

        # Create the output directory (if necessary).
        output_directory = os.path.join(directory, "output")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # Plot the data.
        path = os.path.join(output_directory, "data.pdf")
        t_min = duration - 1.0  # s
        t_max = duration  # s
        data.plot(output=path, t_min=t_min, t_max=t_max, linewidth=0.25)

        # Print message.
        string = "output file: {}"
        message = string.format(path)
        print(message)

    return


if __name__ == '__main__':

    main()
