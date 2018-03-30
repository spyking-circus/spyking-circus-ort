import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

from collections import OrderedDict

from networks import network_4 as network
# from networks import network_3 as network


nb_rows = 16
nb_columns = 16
# nb_cells_range = [3, 12, 48, 192]
nb_cells_range = [192]
duration = 5.0 * 60.0  # s


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    parser.add_argument('--introspection', dest='pending_introspection', action='store_true', default=None)
    parser.add_argument('--validation', dest='pending_validation', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_configuration is None and args.pending_generation is None \
            and args.pending_sorting is None and args.pending_introspection is None \
            and args.pending_validation is None:
        args.pending_configuration = True
        args.pending_generation = True
        args.pending_sorting = True
        args.pending_introspection = True
        args.pending_validation = False
    else:
        args.pending_configuration = args.pending_configuration is True
        args.pending_generation = args.pending_generation is True
        args.pending_sorting = args.pending_sorting is True
        args.pending_introspection = args.pending_introspection is True
        args.pending_validation = args.pending_validation is True

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
        for nb_cells in nb_cells_range:
            name = str(nb_cells)
            kwargs = {
                'general': {
                    'duration': duration,
                    'name': name,
                },
                'probe': {
                    'mode': 'mea',
                    'nb_rows': nb_rows,
                    'nb_columns': nb_columns,
                },
                'cells': {
                    'nb_cells': nb_cells,
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
        try:
            block_nb_buffers = network.block_nb_buffers
        except AttributeError:
            block_nb_buffers = {}
        showfliers = False
        duration_factors = OrderedDict()
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

            # Load time measurements from disk.
            duration_factors[configuration_name] = OrderedDict()
            for block_name in block_names:
                measurements = circusort.io.load_time_measurements(introspection_directory, name=block_name)
                end_times = measurements.get('end', np.empty(shape=0))
                start_times = measurements.get('start', np.empty(shape=0))
                durations = end_times - start_times
                nb_buffers = block_nb_buffers.get(block_name, 1)
                duration_buffer = float(nb_buffers * nb_samples) / sampling_rate
                duration_factors_ = np.log10(durations / duration_buffer)
                duration_factors[configuration_name][block_name] = duration_factors_

        # Plot real-time performances of blocks for each condition (i.e. number of cells).
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
            ax.set_title("Real-time performances ({} cells)".format(configuration_name))
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
            ax.set_xlabel("number of cells")
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
        ax.set_xlabel("number of cells")
        ax.set_ylabel("median duration factor")
        ax.set_title("Median real-time performances")
        ax_.legend()
        fig.tight_layout()
        fig.savefig(output_path)

    # Validate sorting (if necessary).
    if args.pending_validation:

        configuration_names = [
            configuration['general']['name']
            for configuration in configurations
        ]

        # Load data from each configuration.
        for configuration_name in configuration_names:

            generation_directory = os.path.join(directory, "generation", configuration_name)
            sorting_directory = os.path.join(directory, "sorting", configuration_name)

            # Load generation parameters.
            parameters = circusort.io.get_data_parameters(generation_directory)

            # Define parameters.
            nb_channels = nb_rows * nb_columns
            sampling_rate = parameters['general']['sampling_rate']

            print("# Loading data...")

            injected_cells = circusort.io.load_cells(generation_directory)

            from circusort.io.spikes import spikes2cells
            from circusort.io.template_store import load_template_store

            detected_spikes_path = os.path.join(sorting_directory, "spikes.h5")
            detected_spikes = circusort.io.load_spikes(detected_spikes_path)

            detected_templates_path = os.path.join(sorting_directory, "templates.h5")
            detected_templates = load_template_store(detected_templates_path)

            detected_cells = spikes2cells(detected_spikes, detected_templates)

            # Load the data.
            data_path = os.path.join(generation_directory, "data.raw")
            from circusort.io.datafile import load_datafile
            data = load_datafile(data_path, sampling_rate, nb_channels, 'int16', 0.1042)

            # Load the MADs (if possible).
            mads_path = os.path.join(sorting_directory, "mad.raw")
            if os.path.isfile(mads_path):
                from circusort.io.madfile import load_madfile
                mads = load_madfile(mads_path, 'float32', nb_channels, 1024, sampling_rate)
            else:
                mads = None

            # Load the peaks (if possible).
            peaks_path = os.path.join(sorting_directory, "peaks.h5")
            if os.path.isfile(peaks_path):
                from circusort.io.peaks import load_peaks
                peaks = load_peaks(peaks_path)
            else:
                peaks = None

            # Load the filtered data (if possible).
            filtered_data_path = os.path.join(sorting_directory, "data.raw")
            if os.path.isfile(filtered_data_path):
                from circusort.io.datafile import load_datafile
                filtered_data = load_datafile(filtered_data_path, sampling_rate, nb_channels, 'float32', 1.0)
            else:
                filtered_data = None

            # Compute the similarities between detected and injected cells.
            print("# Computing similarities...")
            similarities = injected_cells.compute_similarities(detected_cells)
            print(similarities)

            # Compute the matches between detected and injected cells.
            print("# Computing matches...")
            t_min = 1.0 * 60.0  # s  # discard the 1st minute
            t_max = None
            matches = injected_cells.compute_matches(detected_cells, t_min=t_min, t_max=t_max)
            print(matches)

            from circusort.plt.cells import plot_reconstruction

            t = 3.0 * 60.0  # s  # start time of the reconstruction plot
            d = 1.0  # s  # duration of the reconstruction plot
            plot_reconstruction(detected_cells, t, t + d, sampling_rate, data,
                                mads=mads, peaks=peaks, filtered_data=filtered_data)
            plt.show()


if __name__ == '__main__':

    main()
