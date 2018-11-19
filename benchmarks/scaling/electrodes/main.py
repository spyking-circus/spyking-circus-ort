# coding=utf-8
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

from collections import OrderedDict

from networks import network_4 as network

nb_rows_range = [4, 4, 8, 8, 16, 16]
nb_columns_range = [4, 8, 8, 16, 16, 32]
radius = 100.0  # µm
cell_density = 0.25  # cells / electrode
duration = 10.0 * 60.0  # s


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
            and args.pending_sorting is None and args.pending_introspection is None and args.pending_validation is None:
        args.pending_configuration = True
        args.pending_generation = True
        args.pending_sorting = True
        args.pending_introspection = True
        args.pending_validation = True
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
        # fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4), num=1, clear=True)
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
        # ax_.set_ylim(bottom=np.log10(y_min), top=0.0)
        ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
        ax.set_xticks(x)
        ax.set_xticklabels(configuration_names)
        # ax.set_xticklabels(["$2^{" + "{}".format(2 * i) + "}$" for i in [1, 2, 3, 4, 5]])
        ax.set_xlabel("number of channels")
        ax.set_ylabel("duration (s)")
        ax.set_title("Real-time performances")
        ax_.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
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

            figure_format = 'png'

            generation_directory = os.path.join(directory, "generation", configuration_name)
            sorting_directory = os.path.join(directory, "sorting", configuration_name)
            output_directory = os.path.join(directory, "output", configuration_name)

            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

            # Compute the interspike interval histograms of the generated spike trains (if necessary).
            output_directory_ = os.path.join(output_directory, "generated_interspike_interval_histograms")
            if not os.path.isdir(output_directory_):
                os.makedirs(output_directory_)
                cells = circusort.io.load_cells(generation_directory)
                # Compute the refractory period violation coefficients.
                nb_rpv = {}
                rpv = {}
                for cell_id in cells.ids:
                    cell = cells[cell_id]
                    train = cell.train
                    nb_rpv[cell_id] = train.nb_refractory_period_violations()
                    rpv[cell_id] = train.refractory_period_violation_coefficient()
                # Compute the interspike interval histograms.
                for cell_id in cells.ids:
                    cell = cells[cell_id]
                    train = cell.train
                    nb_spikes = len(train)
                    bin_counts, bin_edges = train.interspike_interval_histogram(bin_width=0.5, width=25.0)
                    bar_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    bar_heights = bin_counts
                    bar_widths = bin_edges[1:] - bin_edges[:-1]
                    fig, ax = plt.subplots()
                    ax.bar(bar_centers, bar_heights, width=bar_widths)
                    ax.axvline(x=2.0, color='black', linestyle='--')
                    ax.set_xlim(bin_edges[0], bin_edges[-1])
                    ax.set_xlabel("interspike interval (ms)")
                    ax.set_ylabel("number of intervals")
                    if 100.0 * rpv[cell_id] > 1e-3:
                        title_string = "ISIH of template {} (2 ms RPV: {:.3f}%, {}/{})"
                        title = title_string.format(cell_id, 100.0 * rpv[cell_id], nb_rpv[cell_id], nb_spikes)
                    else:
                        title_string = "ISIH of template {} (2 ms RPV: <1e-3%, {}/{})"
                        title = title_string.format(cell_id, nb_rpv[cell_id], nb_spikes)
                    ax.set_title(title)
                    fig.tight_layout()
                    filename = "generated_interspike_interval_histogram_{}.{}".format(cell_id, figure_format)
                    path = os.path.join(output_directory, filename)
                    fig.savefig(path)
                    plt.close(fig)

            # Compute the auto-correlograms.
            output_directory_ = os.path.join(output_directory, "autocorrelograms")
            if not os.path.isdir(output_directory_):
                os.makedirs(output_directory_)
                spikes_path = os.path.join(sorting_directory, "spikes.h5")
                spikes = circusort.io.load_spikes(spikes_path)
                for cell_id in range(0, len(spikes)):
                    cell = spikes.get_cell(cell_id)
                    train = cell.train
                    nb_spikes = len(train)
                    bin_counts, bin_edges = train.auto_correlogram()
                    bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    bar_heights = bin_counts
                    bar_widths = bin_edges[1:] - bin_edges[:-1]
                    fig, ax = plt.subplots()
                    ax.bar(bar_centers, bar_heights, width=bar_widths)
                    ax.set_xlabel("lag (ms)")
                    ax.set_ylabel("number of spikes")
                    ax.set_title("Auto-correlogram of template {} ({} spikes)".format(cell_id, nb_spikes))
                    fig.tight_layout()
                    filename = "autocorrelogram_{}.{}".format(cell_id, figure_format)
                    path = os.path.join(output_directory, filename)
                    fig.savefig(path)
                    plt.close(fig)

            # Compute the interspike interval histograms of the sorted spike trains (if necessary).
            output_directory_ = os.path.join(output_directory, "interspike_interval_histograms")
            if not os.path.isdir(output_directory_):
                os.makedirs(output_directory_)
                spikes_path = os.path.join(sorting_directory, "spikes.h5")
                spikes = circusort.io.load_spikes(spikes_path)
                # Compute the refractory period violation coefficients.
                nb_rpv = {}
                rpv = {}
                for cell_id in range(0, len(spikes)):
                    cell = spikes.get_cell(cell_id)
                    train = cell.train
                    nb_rpv[cell_id] = train.nb_refractory_period_violations()
                    rpv[cell_id] = train.refractory_period_violation_coefficient()
                # Compute the interspike interval histograms.
                for cell_id in range(0, len(spikes)):
                    cell = spikes.get_cell(cell_id)
                    train = cell.train
                    nb_spikes = len(train)
                    bin_counts, bin_edges = train.interspike_interval_histogram(bin_width=0.5, width=25.0)
                    bar_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    bar_heights = bin_counts
                    bar_widths = bin_edges[1:] - bin_edges[:-1]
                    fig, ax = plt.subplots()
                    ax.bar(bar_centers, bar_heights, width=bar_widths)
                    ax.axvline(x=2.0, color='black', linestyle='--')
                    ax.set_xlim(bin_edges[0], bin_edges[-1])
                    ax.set_xlabel("interspike interval (ms)")
                    ax.set_ylabel("number of intervals")
                    if 100.0 * rpv[cell_id] > 1e-3:
                        title_string = "ISIH of template {} (2 ms RPV: {:.3f}%, {}/{})"
                        title = title_string.format(cell_id, 100.0 * rpv[cell_id], nb_rpv[cell_id], nb_spikes)
                    else:
                        title_string = "ISIH of template {} (2 ms RPV: <1e-3%, {}/{})"
                        title = title_string.format(cell_id, nb_rpv[cell_id], nb_spikes)
                    ax.set_title(title)
                    fig.tight_layout()
                    filename = "interspike_interval_histogram_{}.{}".format(cell_id, figure_format)
                    path = os.path.join(output_directory, filename)
                    fig.savefig(path)
                    plt.close(fig)

            # TODO compare the sorted spike trains with the generated spike trains.

            # TODO find the matching between sorted templates and detected templates based on:
            # TODO - template waveforms only?
            # TODO - spike trains only?
            # TODO - template waveforms and spike trains?

            injected_cells = circusort.io.load_cells(generation_directory)

            from circusort.io import load_spikes
            detected_spikes_path = os.path.join(sorting_directory, "spikes.h5")
            t_max = 10.0 * 60.0  # s
            detected_spikes = load_spikes(detected_spikes_path, t_max=t_max)

            from circusort.io.template_store import load_template_store
            detected_templates_path = os.path.join(sorting_directory, "templates.h5")
            detected_templates = load_template_store(detected_templates_path)

            from circusort.io.spikes import spikes2cells
            detected_cells = spikes2cells(detected_spikes, detected_templates)

            # Compute the similarities between detected and injected cells.
            print("# Computing similarities...")
            similarities = detected_cells.compute_similarities(injected_cells)
            ordering = True
            similarities_filename = "similarities.pdf"
            similarities_path = os.path.join(output_directory, similarities_filename)
            similarities.plot(ordering=ordering, path=similarities_path)

            # Compute the matches between detected and injected cells.
            print("# Computing matches...")
            t_min = 5.0 * 60.0  # s  # discard the 5 first minutes
            t_max = 10.0 * 60.0  # s
            for cell in detected_cells:
                print("{}, {}, {}".format(cell.train.t_min, cell.train.t_max, cell.train.nb_times))
            matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max)
            ordering = True
            matches_filename = "matches.pdf"
            matches_path = os.path.join(output_directory, matches_filename)
            matches.plot(ordering=ordering, path=matches_path)

            # cells = circusort.io.load_cells(generation_directory)
            # for cell_id in cells.ids:
            #     cell = cells[cell_id]
            #     train = cell.train

            # spikes_path = os.path.join(sorting_directory, "spikes.h5")
            # spikes = circusort.io.load_spikes(spikes_path)
            # for cell_id in range(0, len(spikes)):
            #     cell = spikes.get_cell(cell_id)
            #     train = cell.train


if __name__ == '__main__':

    main()
