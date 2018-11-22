# -*- coding=utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

from collections import OrderedDict

from circusort.plt.cells import plot_reconstruction

from networks import network_4 as network


nb_rows = 16
nb_columns = 16
radius = 100.0  # µm
# nb_cells_range = [3, 12, 48, 192]
cell_densities = [0.25, 1.0, 4.0]
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
            and args.pending_sorting is None and args.pending_introspection is None \
            and args.pending_validation is None:
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
        for cell_density in cell_densities:
            nb_electrodes = nb_rows * nb_columns
            nb_cells = max(1, int(cell_density * float(nb_electrodes)))
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
                    'radius': radius,
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

        if len(configuration_names) == 1:
            configuration_name = configuration_names[0]

            # TODO clean the following copied lines.
            # Load data from each configuration.
            generation_directory = os.path.join(directory, "generation", configuration_name)
            introspection_directory = os.path.join(directory, "introspection", configuration_name)
            # # Load generation parameters.
            parameters = circusort.io.get_data_parameters(generation_directory)
            # # Define parameters.
            nb_samples = parameters['general']['buffer_width']
            sampling_rate = parameters['general']['sampling_rate']
            # # Load time measurements from disk.
            duration_factors_bis = OrderedDict()
            for block_name in block_names:
                measurements = circusort.io.load_time_measurements(introspection_directory, name=block_name)
                keys = [k for k in measurements.keys()]
                start_keys = [k for k in keys if k.endswith(u"_start")]
                end_keys = [k for k in keys if k.endswith(u"_end")]
                start_keys = [k[0:-len(u"_start")] for k in start_keys]
                end_keys = [k[0:-len(u"_end")] for k in end_keys]
                keys = [k for k in start_keys if k in end_keys]
                if keys:
                    keys = [u""] + keys
                    duration_factors_bis[block_name] = OrderedDict()
                    for key in keys:
                        start_key = u"start" if key == u"" else "{}_start".format(key)
                        end_key = u"end" if key == u"" else "{}_end".format(key)
                        start_times = measurements.get(start_key, np.empty(shape=0))
                        end_times = measurements.get(end_key, np.empty(shape=0))
                        durations = end_times - start_times
                        nb_buffers = block_nb_buffers.get(block_name, 1)
                        duration_buffer = float(nb_buffers * nb_samples) / sampling_rate
                        duration_factors_bis_ = np.log10(durations / duration_buffer)
                        duration_factors_bis[block_name][key] = duration_factors_bis_

            # Plot additional real-time performances of conditions for each block.
            for block_name in block_names:
                if block_name in duration_factors_bis:
                    key_names = duration_factors_bis[block_name].keys()
                    data = [
                        duration_factors_bis[block_name][k]
                        for k in key_names
                    ]
                    flierprops = {
                        'marker': 's',
                        'markersize': 1,
                        'markerfacecolor': 'k',
                        'markeredgecolor': 'k',
                    }
                    output_filename = "real_time_performances_{}_bis.{}".format(block_name, image_format)
                    output_path = os.path.join(output_directory, output_filename)
                    fig, ax = plt.subplots(1, 1, num=0, clear=True)
                    ax.set(yscale='log')
                    ax_ = ax.twinx()
                    ax_.boxplot(data, notch=True, whis=1.5, labels=key_names,
                                flierprops=flierprops, showfliers=showfliers)
                    ax_.set_yticks([])
                    ax_.set_yticklabels([])
                    ax_.set_ylabel("")
                    ax.set_ylim(10.0 ** np.array(ax_.get_ylim()))
                    xticklabels = [t[2] for t in ax.xaxis.iter_ticks()]
                    ax.set_xticklabels(xticklabels, rotation=45, horizontalalignment='right')
                    ax.set_xlabel("measurement")
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
        ax.set_xlabel("number of cells")
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

            generation_directory = os.path.join(directory, "generation", configuration_name)
            sorting_directory = os.path.join(directory, "sorting", configuration_name)
            output_directory = os.path.join(directory, "output")

            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

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
            t_min = 0.0  # s
            t_max = duration
            detected_spikes = circusort.io.load_spikes(detected_spikes_path, t_min=t_min, t_max=t_max)

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

            ordering = True
            image_format = 'pdf'

            # Compute the similarities between detected and injected cells.
            similarities_filename = "similarities_{}.{}".format(configuration_name, image_format)
            similarities_path = os.path.join(output_directory, similarities_filename)
            if not os.path.isfile(similarities_path):
                print("# Computing similarities...")
                similarities_cache_filename = "similarities_{}.npz".format(configuration_name)
                similarities_cache_path = os.path.join(output_directory, similarities_cache_filename)
                similarities = detected_cells.compute_similarities(injected_cells, path=similarities_cache_path)
                similarities.plot(ordering=ordering, path=similarities_path)

            # Compute the matches between detected and injected cells.
            matches_filename = "matches_{}.{}".format(configuration_name, image_format)
            matches_path = os.path.join(output_directory, matches_filename)
            matches_curve_filename = "matches_curve_{}.{}".format(configuration_name, image_format)
            matches_curve_path = os.path.join(output_directory, matches_curve_filename)
            if not os.path.isfile(matches_path) or not os.path.isfile(matches_curve_path):
                print("# Computing matches...")
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                if not os.path.isfile(matches_path):
                    matches.plot(ordering=ordering, path=matches_path)
                if not os.path.isfile(matches_curve_path):
                    matches.plot_curve(path=matches_curve_path)

            # # Consider the match with the worst error.
            # sorted_indices = np.argsort(matches.errors)
            # sorted_index = sorted_indices[-1]
            # match = matches[sorted_index]
            # # Determine if false positives or false negatives are dominant.
            # # r_fp = match.compute_false_positive_rate()
            # # r_fn = match.compute_false_negative_rate()
            # # Collect the spike times associated to the false positives / negatives.
            # from circusort.plt.base import plot_times_of_interest
            # train_fn = match.collect_false_negatives()
            # # Plot the reconstruction around these spike times (if necessary).
            # if len(train_fn) > 0:
            #     times_of_interest = train_fn.sample(size=10)
            #     times_of_interest_filename = "times_of_interest_{}.{}".format(configuration_name, image_format)
            #     times_of_interest_path = os.path.join(output_directory, times_of_interest_filename)
            #     plot_times_of_interest(data, times_of_interest,
            #                            window=10e-3, cells=detected_cells, sampling_rate=sampling_rate,
            #                            mads=mads, peaks=peaks, filtered_data=filtered_data,
            #                            path=times_of_interest_path)

            # Plot the reconstruction.
            reconstruction_filename = "reconstruction_{}.{}".format(configuration_name, image_format)
            reconstruction_path = os.path.join(output_directory, reconstruction_filename)
            if not os.path.isfile(reconstruction_path):
                print("# Computing reconstruction...")
                t = 3.0 * 60.0  # s  # start time of the reconstruction plot
                d = 1.0  # s  # duration of the reconstruction plot
                if nb_channels > 32:
                    channels = np.random.choice(nb_channels, size=32, replace=False)
                else:
                    channels = None
                buffer_width = 1024
                plot_reconstruction(detected_cells, t, t + d, sampling_rate, data, channels=channels,
                                    mads=mads, peaks=peaks, filtered_data=filtered_data, buffer_width=buffer_width,
                                    output=reconstruction_path)

            # TODO plot the highest similarity with another template against the false negative rate.
            sim_vs_fnr_filename = "sim_vs_fnr_{}.{}".format(configuration_name, image_format)
            sim_vs_fnr_path = os.path.join(output_directory, sim_vs_fnr_filename)
            if not os.path.isfile(sim_vs_fnr_path):
                print("# Computing sim. v.s. FNR...")
                # Retrieve the template similarities.
                similarities_cache_filename = "similarities_{}.npz".format(configuration_name)
                similarities_cache_path = os.path.join(output_directory, similarities_cache_filename)
                similarities = detected_cells.compute_similarities(injected_cells, path=similarities_cache_path)
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Plot similarity v.s. FNR.
                x = 100.0 * matches.false_negative_rates
                y = similarities.highest_similarities(num=2)
                y = y[:, 1]
                # y = np.mean(y, axis=1)
                fig, ax = plt.subplots()
                ax.scatter(x, y, s=5, color='black')
                ax.set_xlabel("false negative rate (%)")
                ax.set_ylabel("2nd highest similarity (arb. unit)")
                fig.tight_layout()
                fig.savefig(sim_vs_fnr_path)
                plt.close(fig)

            # TODO plot the templates with worst FNR.
            worst_fnr_filename = "worst_fnr_{}.{}".format(configuration_name, image_format)
            worst_fnr_path = os.path.join(output_directory, worst_fnr_filename)
            if not os.path.isfile(worst_fnr_path):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Find the worst FNR.
                fnrs = matches.false_negative_rates
                index = np.argmax(fnrs)
                match = matches[index]
                sorted_cell, injected_cell = match.get_cells()
                sorted_template = sorted_cell.template
                injected_template = injected_cell.template
                probe_filename = "probe.prb"
                probe_path = os.path.join(generation_directory, probe_filename)
                probe = circusort.io.load_probe(probe_path)
                # Plot templates with worst FNR.
                fig, ax = plt.subplots(ncols=2)
                sorted_template.plot(ax=ax[0], output=worst_fnr_path, probe=probe, time_factor=25.0)
                injected_template.plot(ax=ax[1], output=worst_fnr_path, probe=probe, time_factor=25.0)
                plt.close(fig)

            # TODO plot template norm against the false negative rates.
            norm_vs_fnr_filename = "norm_vs_fnr_{}.{}".format(configuration_name, image_format)
            norm_vs_fnr_path = os.path.join(output_directory, norm_vs_fnr_filename)
            if not os.path.isfile(norm_vs_fnr_path):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Plot template norm against FNR.
                x = 100.0 * matches.false_negative_rates
                y = np.array([cell.template.norm for cell in detected_cells])
                fig, ax = plt.subplots()
                ax.scatter(x, y, s=5, color='black')
                ax.set_xlabel("false negative rate (%)")
                ax.set_ylabel("template norm (arb. unit)")
                ax.set_title("Template norm against false negative rate")
                fig.tight_layout()
                fig.savefig(norm_vs_fnr_path)
                plt.close(fig)

            # TODO plot template amplitude against the false negative rates.
            ampl_vs_fnr_filename = "ampl_vs_fnr_{}.{}".format(configuration_name, image_format)
            ampl_vs_fnr_path = os.path.join(output_directory, ampl_vs_fnr_filename)
            if not os.path.isfile(ampl_vs_fnr_path):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Plot template norm against FNR.
                x = 100.0 * matches.false_negative_rates
                y = np.array([cell.template.peak_amplitude() for cell in detected_cells])
                z = detected_cells.ids
                fig, ax = plt.subplots()
                ax.scatter(x, y, s=5, color='black')
                for x_, y_, z_ in zip(x, y, z):
                    ax.annotate(z_, (x_, y_), fontsize=1, color='grey',
                                verticalalignment='center', horizontalalignment='center')
                ax.set_xlabel("false negative rate (%)")
                ax.set_ylabel("template amplitude (µV)")
                ax.set_title("Template amplitude against false negative rate")
                fig.tight_layout()
                fig.savefig(ampl_vs_fnr_path)
                plt.close(fig)

            # TODO plot false negative count through time.
            fnr_filename = "fnr_{}.{}".format(configuration_name, image_format)
            fnr_path = os.path.join(output_directory, fnr_filename)
            if not os.path.isfile(norm_vs_fnr_path):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Plot FNR through time.
                mode = 'sorted'
                x, y, z = matches.false_negative_counts(nb_bins=100)
                if mode == 'sorted':
                    i = np.argsort(matches.false_negative_rates)
                    z = z[i, :]
                fig, ax = plt.subplots()
                c = ax.pcolor(x, y, z)
                ax.set_xlabel("time (s)")
                ax.set_ylabel("detected unit")
                ax.set_title("False negative count through time")
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label("false negative count")
                fig.tight_layout()
                fig.savefig(fnr_path)
                plt.close(fig)

            # TODO plot the templates with worst FNR.
            nb_worst_templates = 10
            worst_fnr_paths = {}
            for k in range(0, nb_worst_templates):
                worst_fnr_filename = "worst_fnr_{}_{}.{}".format(configuration_name, k, image_format)
                worst_fnr_paths[k] = os.path.join(output_directory, worst_fnr_filename)
            if np.any([not os.path.isfile(path) for path in worst_fnr_paths.values()]):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Find the worst FNRs.
                fnrs = matches.false_negative_rates
                indices = np.argsort(fnrs)
                for k in range(0, nb_worst_templates):
                    if not os.path.isfile(worst_fnr_paths[k]):
                        index = indices[-k-1]
                        match = matches[index]
                        sorted_cell, injected_cell = match.get_cells()
                        sorted_template = sorted_cell.template
                        injected_template = injected_cell.template
                        probe_filename = "probe.prb"
                        probe_path = os.path.join(generation_directory, probe_filename)
                        probe = circusort.io.load_probe(probe_path)
                        # Plot templates with worst FNR.
                        fig, ax = plt.subplots(ncols=2)
                        sorted_template.plot(ax=ax[0], output=worst_fnr_paths[k], probe=probe,
                                             time_factor=25.0, voltage_factor=0.5)
                        injected_template.plot(ax=ax[1], output=worst_fnr_paths[k], probe=probe,
                                               time_factor=25.0, voltage_factor=0.5)
                        plt.close(fig)

            # TODO plot the templates with best FNR.
            nb_best_templates = 10
            best_fnr_paths = {}
            for k in range(0, nb_best_templates):
                best_fnr_filename = "best_fnr_{}_{}.{}".format(configuration_name, k, image_format)
                best_fnr_paths[k] = os.path.join(output_directory, best_fnr_filename)
            if np.any([not os.path.isfile(path) for path in best_fnr_paths.values()]):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Find the best FNRs.
                fnrs = matches.false_negative_rates
                indices = np.argsort(fnrs)
                for k in range(0, nb_best_templates):
                    if not os.path.isfile(best_fnr_paths[k]) or force:
                        index = indices[k]
                        match = matches[index]
                        sorted_cell, injected_cell = match.get_cells()
                        sorted_template = sorted_cell.template
                        injected_template = injected_cell.template
                        probe_filename = "probe.prb"
                        probe_path = os.path.join(generation_directory, probe_filename)
                        probe = circusort.io.load_probe(probe_path)
                        # Plot templates with best FNR.
                        fig, ax = plt.subplots(ncols=2)
                        sorted_template.plot(ax=ax[0], output=best_fnr_paths[k], probe=probe,
                                             time_factor=25.0, voltage_factor=0.5)
                        injected_template.plot(ax=ax[1], output=best_fnr_paths[k], probe=probe,
                                               time_factor=25.0, voltage_factor=0.5)
                        plt.close(fig)

            # TODO plot the annoying templates.
            nb_annoying_templates = 5
            annoying_paths = {}
            for k in range(0, nb_annoying_templates):
                annoying_filename = "annoying_{}_{}.{}".format(configuration_name, k, image_format)
                annoying_paths[k] = os.path.join(output_directory, annoying_filename)
            if np.any([not os.path.isfile(path) for path in annoying_paths.values()]):
                # Retrieve the best matching between templates.
                t_min = 0.0  # s
                t_max = duration
                matches_cache_filename = "matches_{}.npz".format(configuration_name)
                matches_cache_path = os.path.join(output_directory, matches_cache_filename)
                matches = detected_cells.compute_matches(injected_cells, t_min=t_min, t_max=t_max,
                                                         path=matches_cache_path)
                # Find the annoying templates.
                # fnrs = matches.false_negative_rates
                # amplitudes = np.array([cell.template.peak_amplitude() for cell in detected_cells])
                # indices = np.where(np.logical_and(fnrs > np.median(fnrs), amplitudes > np.median(amplitudes)))[0]
                # np.random.shuffle(indices)
                indices = {
                    "64": np.array([3, 56, 32, 50, 59]),
                    "256": np.array([130, 154, 228, 207, 27]),
                    "1024": np.array([397, 593, 876, 1001, 525]),
                }
                indices = indices[configuration_name]
                for k in range(0, nb_annoying_templates):
                    if not os.path.isfile(annoying_paths[k]):
                        index = indices[k]
                        match = matches[index]
                        sorted_cell, injected_cell = match.get_cells()
                        sorted_template = sorted_cell.template
                        injected_template = injected_cell.template
                        probe_filename = "probe.prb"
                        probe_path = os.path.join(generation_directory, probe_filename)
                        probe = circusort.io.load_probe(probe_path)
                        # Plot annoying templates.
                        fig, ax = plt.subplots(ncols=2)
                        sorted_template.plot(ax=ax[0], output=annoying_paths[k], probe=probe,
                                             time_factor=25.0, voltage_factor=0.5)
                        injected_template.plot(ax=ax[1], output=annoying_paths[k], probe=probe,
                                               time_factor=25.0, voltage_factor=0.5)
                        plt.close(fig)

            # plt.show()  # Figures are saved to files and don't need to be shown.


if __name__ == '__main__':

    main()
