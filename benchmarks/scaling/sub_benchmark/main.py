import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import circusort

from collections import OrderedDict

from networks import network_1 as network


nb_rows_range = [4, 8, 16, 32]
nb_columns_range = [4, 8, 16, 32]
duration = 5.0 * 60.0  # s


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    parser.add_argument('--introspection', dest='pending_introspection', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_generation is None and args.pending_sorting is None and args.pending_introspection is None:
        args.pending_generation = True
        args.pending_sorting = True
        args.pending_introspection = True
    else:
        args.pending_generation = args.pending_generation is True
        args.pending_sorting = args.pending_sorting is True
        args.pending_introspection = args.pending_introspection is True

    # Define the working directory.
    directory = network.directory
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    configuration_directory = os.path.join(directory, "configuration")
    if not os.path.isdir(configuration_directory):
        os.makedirs(configuration_directory)
        # Generate configurations.
        for nb_rows, nb_columns in zip(nb_rows_range, nb_columns_range):
            name = str(nb_rows * nb_columns)
            kwargs = {
                'general': {
                    'duration': duration,
                    'name': name,
                },
                'probe': {
                    'mode': 'mea',
                    'nb_rows': nb_rows,
                    'nb_columns': nb_columns,
                }
            }
            configuration = circusort.io.generate_configuration(**kwargs)
            configuration_directory_ = os.path.join(configuration_directory, name)
            configuration.save(configuration_directory_)

    # Load configurations.
    configurations = circusort.io.get_configurations(configuration_directory)

    # Configure Matplotlib.
    plt.ioff()
    plt.style.use('seaborn-paper')

    # Process each configuration.
    for configuration in configurations:

        name = configuration['general']['name']

        configuration_directory = os.path.join(directory, "configuration", name)
        generation_directory = os.path.join(directory, "generation", name)
        # sorting_directory = os.path.join(directory, "sorting", name)
        # introspection_directory = os.path.join(directory, "introspection", name)

        # Generate data (if necessary).
        if args.pending_generation:

            circusort.net.pregenerator(configuration_directory=configuration_directory,
                                       generation_directory=generation_directory)

        # Sort data (if necessary).
        if args.pending_sorting:

            network.sorting(name)

            # # TODO remove the following lines.
            # # Load generation parameters.
            # parameters = circusort.io.get_data_parameters(generation_directory)
            #
            # # Define parameters.
            # host = '127.0.0.1'  # i.e. run the test locally
            # dtype = parameters['general']['dtype']
            # nb_channels = parameters['probe']['nb_channels']
            # nb_samples = parameters['general']['buffer_width']
            # sampling_rate = parameters['general']['sampling_rate']
            #
            # # Create directories (if necessary).
            # if not os.path.isdir(sorting_directory):
            #     os.makedirs(sorting_directory)
            # if not os.path.isdir(introspection_directory):
            #     os.makedirs(introspection_directory)
            #
            # # Define keyword arguments.
            # reader_kwargs = {
            #     'data_path': os.path.join(generation_directory, "data.raw"),
            #     'dtype': dtype,
            #     'nb_channels': nb_channels,
            #     'nb_samples': nb_samples,
            #     'sampling_rate': sampling_rate,
            #     'is_realistic': True,
            #     'introspection_path': introspection_directory,
            # }
            # signal_writer_kwargs = {
            #     'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
            #     'introspection_path': introspection_directory,
            # }
            #
            # # Define the elements of the Circus network.
            # director = circusort.create_director(host=host)
            # manager = director.create_manager(host=host)
            # reader = manager.create_block('reader', **reader_kwargs)
            # writer = manager.create_block('writer', log_level=DEBUG, **signal_writer_kwargs)
            # # Initialize the elements of the Circus network.
            # director.initialize()
            # # Connect the elements of the Circus network.
            # director.connect(reader.output, writer.input)
            # # or  # director.connect(reader.output, [reader.input])  # if the previous line does not work.
            # # Launch the network.
            # director.start()
            # director.join()
            # director.destroy()

    # Introspect sorting (if necessary).
    if args.pending_introspection:

        block_names = network.block_names
        speed_factors = OrderedDict()
        output_directory = os.path.join(directory, "output")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        image_format = 'png'

        configuration_names = [
            configuration['general']['name']
            for configuration in configurations
        ]

        # Load data from each configuration.
        for configuration_name in configuration_names:

            generation_directory = os.path.join(directory, "generation", configuration_name)
            introspection_directory = os.path.join(directory, "introspection", configuration_name)

            # Load generation parameters.
            parameters = circusort.io.get_data_parameters(generation_directory)

            # Define parameters.
            nb_samples = parameters['general']['buffer_width']
            sampling_rate = parameters['general']['sampling_rate']
            duration_buffer = float(nb_samples) / sampling_rate

            # Load time measurements from disk.
            speed_factors[configuration_name] = OrderedDict()
            for block_name in block_names:
                measurements = circusort.io.load_time_measurements(introspection_directory, name=block_name)
                durations = measurements['end'] - measurements['start']
                speed_factors[configuration_name][block_name] = duration_buffer / durations

        # Plot real-time performances of blocks for each condition.
        for configuration_name in configuration_names:

            data = [
                speed_factors[configuration_name][block_name]
                for block_name in block_names
            ]

            flierprops = {
                'marker': 's',
                'markersize': 1,
                'markerfacecolor': 'k',
                'markeredgecolor': 'k',
            }
            output_filename = "real_time_performances_{}.{}".format(configuration_name, image_format)
            output_path = os.path.join(output_directory, output_filename)

            fig, ax = plt.subplots(1, 1, num=0, clear=True)
            ax.boxplot(data, notch=True, whis=1.5, labels=block_names, flierprops=flierprops)
            ax.set_ylabel("speed factor")
            ax.set_title("Real-time performances ({} channels)".format(configuration_name))
            fig.tight_layout()
            fig.savefig(output_path)

        # Plot real-time performances of conditions for each block.
        for block_name in block_names:

            data = [
                speed_factors[configuration_name][block_name]
                for configuration_name in configuration_names
            ]

            flierprops = {
                'marker': 's',
                'markersize': 1,
                'markerfacecolor': 'k',
                'markeredgecolor': 'k',
            }
            output_filename = "real_time_performances_{}.{}".format(block_name, image_format)
            output_path = os.path.join(output_directory, output_filename)

            fig, ax = plt.subplots(1, 1, num=0, clear=True)
            ax.boxplot(data, notch=True, whis=1.5, labels=configuration_names, flierprops=flierprops)
            ax.set_xlabel("number of channels")
            ax.set_ylabel("speed factor")
            ax.set_title("Real-time performances ({})".format(block_name))
            fig.tight_layout()
            fig.savefig(output_path)

        # Plot median real-time performances.
        output_filename = "median_real_time_performances.{}".format(image_format)
        output_path = os.path.join(output_directory, output_filename)

        fig, ax = plt.subplots(1, 1, num=0, clear=True)
        x = [
            k
            for k, _ in enumerate(configuration_names)
        ]
        for block_name in block_names:
            y = [
                np.median(speed_factors[configuration_name][block_name])
                for configuration_name in configuration_names
            ]
            plt.plot(x, y, marker='o', label=block_name)
        ax.set_xticks(x)
        ax.set_xticklabels(configuration_names)
        ax.set_xlabel("number of channels")
        ax.set_ylabel("median speed factor")
        ax.set_title("Median real-time performances")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)


if __name__ == '__main__':

    main()
