# -*- coding=utf-8 -*-
import argparse
import os
import shutil

import circusort

import network

from circusort.io.datafile import load_datafile
from circusort.io.peaks import load_peaks


directory = os.path.join("~", ".spyking-circus-ort", "examples", "peak_detection")
directory = os.path.expanduser(directory)

nb_rows = 4
nb_columns = 4
radius = 100.0  # µm
nb_cells = 16
duration = 5.0 * 60.0  # s


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--detection', dest='pending_detection', action='store_true', default=None)
    parser.add_argument('--display', dest='pending_display', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_configuration is None and args.pending_generation is None and args.pending_detection is None \
            and args.pending_display is None:
        args.pending_configuration = True
        args.pending_generation = True
        args.pending_detection = True
        args.pending_display = True
    else:
        args.pending_configuration = args.pending_configuration is True
        args.pending_generation = args.pending_generation is True
        args.pending_detection = args.pending_detection is True
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
    detection_directory = os.path.join(directory, "detection")
    output_directory = os.path.join(directory, "output")

    # Generate data (if necessary).
    if args.pending_generation:

        circusort.net.pregenerator(configuration_directory=configuration_directory,
                                   generation_directory=generation_directory)

    # Detect peaks (if necessary).
    if args.pending_detection:

        network.detection(directory)

    # Display result (if necessary).
    if args.pending_display:

        # Create the output directory (if necessary).
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # Load generation parameters.
        parameters = circusort.io.get_data_parameters(generation_directory)

        # Define parameters.
        sampling_rate = parameters['general']['sampling_rate']
        nb_channels = parameters['probe']['nb_channels']

        # Load the data.
        data_path = os.path.join(generation_directory, "data.raw")
        data = load_datafile(data_path, sampling_rate, nb_channels, 'int16', 0.1042)

        # Load the peaks.
        peaks_path = os.path.join(detection_directory, "peaks.h5")
        peaks = load_peaks(peaks_path)

        t_min = duration - 1.5  # s
        t_max = duration - 0.5 # s

        # Plot the data.
        ax = data.plot(t_min=t_min, t_max=t_max, linewidth=0.25, color='C0')

        # Plot the peaks.
        ax = peaks.plot(ax=ax, t_min=t_min, t_max=t_max, linewidth=0.25, color='C1', zorder=0)

        # Save figure.
        path = os.path.join(output_directory, "data.pdf")
        fig = ax.get_figure()
        fig.savefig(path)

        # Print message.
        string = "output file: {}"
        message = string.format(path)
        print(message)

    return


if __name__ == '__main__':

    main()
