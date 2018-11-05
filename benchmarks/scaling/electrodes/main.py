# coding=utf-8
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

from collections import OrderedDict

from networks import network_4 as network

nb_rows_range = [2, 4, 8, 16, 32]
nb_columns_range = [2, 4, 8, 16, 32]
radius = 100.0  # µm
cell_density = 0.25  # cells / electrode
duration = 5.0 * 60.0  # s


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    parser.add_argument('--introspection', dest='pending_introspection', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_configuration is None and args.pending_generation is None \
            and args.pending_sorting is None and args.pending_introspection is None:
        args.pending_configuration = True
        args.pending_generation = True
        args.pending_sorting = True
        args.pending_introspection = True
    else:
        args.pending_configuration = args.pending_configuration is True
        args.pending_generation = args.pending_generation is True
        args.pending_sorting = args.pending_sorting is True
        args.pending_introspection = args.pending_introspection is True

    # Define the working directory.
    directory = network.directory
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    configuration_directory = os.path.join(directory, "configuration")

    if args.pending_configuration:

        # Clean the configuration directory (if necessary).
        if os.path.isdir(configuration_directory):
            shutil.rmtree(configuration_directory)
        os.makedirs(configuration_directory)

        # Generate configurations.
        for nb_rows, nb_columns in zip(nb_rows_range, nb_columns_range):
            nb_electrodes = nb_rows * nb_columns
            name = str(nb_electrodes)
            kwargs = {
                'general': {
                    'duration': duration,
                    'name': name,
                },
                'probe': {
                    'mode': 'mea',
                    'nb_rows': nb_rows,
                    'nb_columns': nb_columns,
                    'radius': radius,
                },
                'cells': {
                    'nb_cells': int(cell_density * float(nb_electrodes)),
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

        # Generate data (if necessary).
        if args.pending_generation:

            circusort.net.pregenerator(configuration_directory=configuration_directory,
                                       generation_directory=generation_directory)

        # Sort data (if necessary).
        if args.pending_sorting:

            network.sorting(name)

    # Introspect sorting (if necessary).
    if args.pending_introspection:

        block_names = network.block_names
        block_groups = network.block_groups
        block_nb_buffers = network.block_nb_buffers
        showfliers = False
        durations = OrderedDict()
        duration_factors = OrderedDict()
        output_directory = os.path.join(directory, "output")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        image_format = 'pdf'

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

            # Load time measurements from disk.
            durations[configuration_name] = OrderedDict()
            duration_factors[configuration_name] = OrderedDict()
            for block_name in block_names:
                measurements = circusort.io.load_time_measurements(introspection_directory, name=block_name)
                end_times = measurements.get('end', np.empty(shape=0))
                start_times = measurements.get('start', np.empty(shape=0))
                durations_ = end_times - start_times
                nb_buffers = block_nb_buffers[block_name]
                duration_buffer = float(nb_buffers * nb_samples) / sampling_rate
                duration_factors_ = np.log10(durations_ / duration_buffer)
                durations[configuration_name][block_name] = durations_
                duration_factors[configuration_name][block_name] = duration_factors_

        # Plot real-time performances of blocks for each condition.
        for configuration_name in configuration_names:

            data = [
                duration_factors[configuration_name][block_name]
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
            ax.set(yscale='log')
            ax_ = ax.twinx()
            ax_.boxplot(data, notch=True, whis=1.5, labels=block_names,
                        flierprops=flierprops, showfliers=showfliers)
            ax_.set_yticks([])
            ax_.set_yticklabels([])
            ax_.set_ylabel("")
            ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
            ax.set_ylabel("duration factor")
            ax.set_title("Real-time performances ({} channels)".format(configuration_name))
            fig.tight_layout()
            fig.savefig(output_path)

        # Plot real-time performances of conditions for each block.
        for block_name in block_names:

            data = [
                duration_factors[configuration_name][block_name]
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
            ax.set(yscale='log')
            ax_ = ax.twinx()
            ax_.boxplot(data, notch=True, whis=1.5, labels=configuration_names,
                        flierprops=flierprops, showfliers=showfliers)
            ax_.set_yticks([])
            ax_.set_yticklabels([])
            ax_.set_ylabel("")
            ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
            ax.set_xlabel("number of channels")
            ax.set_ylabel("duration factor")
            ax.set_title("Real-time performances ({})".format(block_name))
            fig.tight_layout()
            fig.savefig(output_path)

        # Plot median real-time performances.
        output_filename = "median_real_time_performances.{}".format(image_format)
        output_path = os.path.join(output_directory, output_filename)

        fig, ax = plt.subplots(1, 1, num=0, clear=True)
        ax.set(yscale='log')
        ax_ = ax.twinx()
        x = [
            k
            for k, _ in enumerate(configuration_names)
        ]
        for block_name in block_names:
            y = [
                np.median(duration_factors[configuration_name][block_name])
                for configuration_name in configuration_names
            ]
            ax_.plot(x, y, marker='o', label=block_name)
        ax_.set_yticks([])
        ax_.set_yticklabels([])
        ax_.set_ylabel("")
        ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
        ax.set_xticks(x)
        ax.set_xticklabels(configuration_names)
        ax.set_xlabel("number of channels")
        ax.set_ylabel("median duration factor")
        ax.set_title("Median real-time performances")
        ax_.legend()
        fig.tight_layout()
        fig.savefig(output_path)

        # Plot real-time performances.
        output_filename = "real_time_performances.{}".format(image_format)
        output_path = os.path.join(output_directory, output_filename)

        mode = 'mean_and_standard_deviation'
        # mode = 'median_and_median_absolute_deviation'

        fig, ax = plt.subplots(1, 1, num=0, clear=True)
        ax.set(yscale='log')
        ax_ = ax.twinx()
        x = [
            k
            for k, _ in enumerate(configuration_names)
        ]
        # Compute y_min.
        y_min = float('inf')
        for block_group, block_names in block_groups.items():
            d = {
                configuration_name: np.concatenate([
                    durations[configuration_name][block_name] / float(block_nb_buffers[block_name])
                    for block_name in block_names
                ])
                for configuration_name in configuration_names
            }
            if mode == 'mean_and_standard_deviation':
                y = np.array([
                    np.mean(d[configuration_name])
                    for configuration_name in configuration_names
                ])
                y_error = np.array([
                    np.std(d[configuration_name])
                    for configuration_name in configuration_names
                ])
            elif mode == 'median_and_median_absolute_deviation':
                y = np.array([
                    np.median(d[configuration_name])
                    for configuration_name in configuration_names
                ])
                y_error = np.array([
                    1.4826 * np.median(np.abs(d[configuration_name] - y[k]))
                    for k, configuration_name in enumerate(configuration_names)
                ])
            else:
                raise ValueError("unexpected mode value: {}".format(mode))
            y1 = y - y_error
            if np.any(y1 > 0.0):
                y_min = min(y_min, np.min(y1[y1 > 0.0]))
        # Plot everything.
        colors = {
            block_group: "C{}".format(k % 10)
            for k, block_group in enumerate(block_groups.keys())
        }
        for block_group, block_names in block_groups.items():
            d = {
                configuration_name: np.concatenate([
                    durations[configuration_name][block_name] / float(block_nb_buffers[block_name])
                    for block_name in block_names
                ])
                for configuration_name in configuration_names
            }
            if mode == 'mean_and_standard_deviation':
                y = np.array([
                    np.mean(d[configuration_name])
                    for configuration_name in configuration_names
                ])
                y_error = np.array([
                    np.std(d[configuration_name])
                    for configuration_name in configuration_names
                ])
            elif mode == 'median_and_median_absolute_deviation':
                # Median and median absolute deviation.
                y = np.array([
                    np.median(d[configuration_name])
                    for configuration_name in configuration_names
                ])
                y_error = np.array([
                    1.4826 * np.median(np.abs(d[configuration_name] - y[k]))
                    for k, configuration_name in enumerate(configuration_names)
                ])
            else:
                raise ValueError("unexpected mode value: {}".format(mode))
            color = colors[block_group]
            y1 = y - y_error
            y1[y1 <= 0.0] = y_min  # i.e. replace negative values
            y1 = np.log10(y1)
            y2 = np.log10(y + y_error)
            ax_.fill_between(x, y1, y2, alpha=0.5, facecolor=color, edgecolor=None)
            ax_.plot(x, np.log10(y), color=color, marker='o', label=block_group)
        ax_.set_yticks([])
        ax_.set_yticklabels([])
        ax_.set_ylabel("")
        ax_.set_ylim(bottom=np.log10(y_min))
        ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
        ax.set_xticks(x)
        ax.set_xticklabels(configuration_names)
        ax.set_xlabel("number of channels")
        ax.set_ylabel("duration (s)")
        ax.set_title("Real-time performances")
        ax_.legend()
        fig.tight_layout()
        fig.savefig(output_path)


if __name__ == '__main__':

    main()
