# -*- coding=utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

from collections import OrderedDict

import network

nb_rows = 3
nb_columns = 3
radius = 100.0  # µm
nb_cells_range = [27]
duration = 2.5 * 60.0  # s


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
        block_labels = {
            block_name: network.block_labels.get(block_name, block_name)
            for block_name in block_names
        }
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

            labels = [
                block_labels[block_name]
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
            ax_.boxplot(data, notch=True, whis=1.5, labels=labels,
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
            label = block_labels[block_name]
            ax_.plot(x, y, marker='o', label=label)
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

    # Validate clustering (if necessary).
    if args.pending_validation:

        for configuration in configurations:
            # Define template store path.
            name = configuration['general']['name']
            sorting_directory = os.path.join(directory, "sorting", name)
            template_store_filename = "templates.h5"
            template_store_path = os.path.join(sorting_directory, template_store_filename)
            # Load template store.
            from circusort.io import load_template_store
            template_store = load_template_store(template_store_path)
            # Print number of template in store.
            string = "There are {} templates in the store."
            message = string.format(template_store.nb_templates)
            print(message)
            # Plot the first templates of the store.
            templates = template_store.get()
            nb_rows_ = 6
            nb_columns_ = 6
            fig, ax = plt.subplots(nrows=nb_rows_, ncols=nb_columns_, figsize=(3.0 * 6.4, 2.0 * 4.8))
            for k, template in enumerate(templates):
                if k < nb_rows_ * nb_columns_:
                    i = k // nb_columns_
                    j = k % nb_columns_
                    with_xaxis = (i == (nb_rows_ - 1))
                    with_yaxis = (j == 0)
                    with_scale_bars = (i == (nb_rows_ - 1)) and (j == 0)
                    template.plot(ax=ax[i, j], probe=template_store.probe, with_title=False, with_xaxis=with_xaxis,
                                  with_yaxis=with_yaxis, with_scale_bars=with_scale_bars)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    main()
