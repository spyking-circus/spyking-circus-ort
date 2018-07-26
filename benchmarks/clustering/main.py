# -*- coding=utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import circusort

# from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import OrderedDict

import network

nb_rows = 3
nb_columns = 3
radius = 100.0  # µm
nb_cells_range = [27]
duration = 5.0 * 60.0  # s
preload_templates = False
nb_waveforms_clustering = 400


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

            network.sorting(name, with_precomputed_templates=preload_templates,
                            nb_waveforms_clustering=nb_waveforms_clustering)

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
            generation_directory = os.path.join(directory, "generation", name)
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

            # Extract templates from the store.
            templates = template_store.get()

            # # Plot the first templates of the store.
            # nb_rows_ = 6
            # nb_columns_ = 6
            # fig, ax = plt.subplots(nrows=nb_rows_, ncols=nb_columns_, figsize=(3.0 * 6.4, 2.0 * 4.8))
            # for k, template in enumerate(templates):
            #     if k < nb_rows_ * nb_columns_:
            #         i = k // nb_columns_
            #         j = k % nb_columns_
            #         title = "Template {}".format(k)
            #         with_xaxis = (i == (nb_rows_ - 1))
            #         with_yaxis = (j == 0)
            #         with_scale_bars = (i == (nb_rows_ - 1)) and (j == 0)
            #         template.plot(ax=ax[i, j], probe=template_store.probe, title=title, with_xaxis=with_xaxis,
            #                       with_yaxis=with_yaxis, with_scale_bars=with_scale_bars)

            # TODO match detected templates with generated templates.
            # TODO   am I missing some template (i.e. generated)
            # TODO   am I adding some template (i.e. detected)

            injected_cells = circusort.io.load_cells(generation_directory)
            injected_templates = [
                injected_cell.template
                for injected_cell in injected_cells
            ]
            similarities = np.array([
                [
                    template.similarity(injected_template)
                    for template in templates
                ]
                for injected_template in injected_templates
            ])

            def find_matches():

                nb_injected_templates = len(injected_templates)
                nb_detected_templates = len(templates)

                injected_indices = list(range(0, nb_injected_templates))
                detected_indices = list(range(0, nb_detected_templates))

                pairs = []
                for k in range(0, min(nb_injected_templates, nb_detected_templates)):
                    sub_s = similarities[injected_indices, :]
                    sub_s = sub_s[:, detected_indices]
                    i = np.argmax(sub_s)
                    i, j = np.unravel_index(i, sub_s.shape)
                    i, j = injected_indices[i], detected_indices[j]
                    pairs.append((i, j))
                    injected_indices.remove(i)
                    detected_indices.remove(j)

                missing_indices = np.array(injected_indices)
                additional_indices = np.array(detected_indices)

                return pairs, missing_indices, additional_indices

            pairs, missing_indices, additional_indices = find_matches()
            nb_figures = len(pairs) + len(missing_indices) + len(additional_indices)

            nb_rows_ = 6
            nb_columns_ = 6
            nb_subplots = nb_rows_ * nb_columns_
            fig, ax = plt.subplots(nrows=nb_rows_, ncols=nb_columns_, figsize=(3.0 * 6.4, 2.0 * 4.8))
            for k, pair in enumerate(pairs):
                if k < nb_subplots:
                    i = k // nb_columns_
                    j = k % nb_columns_
                    k_1, k_2 = pair
                    template_1 = injected_templates[k_1]
                    template_2 = templates[k_2]
                    title = "Matched templates g{} & d{}".format(k_1, k_2)
                    with_xaxis = (i == (nb_rows_ - 1))
                    with_yaxis = (j == 0)
                    with_scale_bars = (i == (nb_rows_ - 1)) and (j == 0)
                    template_1.plot(ax=ax[i, j], probe=template_store.probe, title=title, with_xaxis=with_xaxis,
                                    with_yaxis=with_yaxis, with_scale_bars=with_scale_bars, color='C0')
                    template_2.plot(ax=ax[i, j], probe=template_store.probe, title=title, with_xaxis=with_xaxis,
                                    with_yaxis=with_yaxis, with_scale_bars=False, color='C1')
            for k, index in enumerate(missing_indices):
                k_ = k + len(pairs)
                if k_ < nb_subplots:
                    i = k_ // nb_columns_
                    j = k_ % nb_columns_
                    template = injected_templates[index]
                    title = "Missing template {}".format(index)
                    with_xaxis = (i == (nb_rows_ - 1))
                    with_yaxis = (j == 0)
                    with_scale_bars = (i == (nb_rows_ - 1)) and (j == 0)
                    template.plot(ax=ax[i, j], probe=template_store.probe, title=title, with_xaxis=with_xaxis,
                                  with_yaxis=with_yaxis, with_scale_bars=with_scale_bars, color='C0')
            for k, index in enumerate(additional_indices):
                k_ = k + len(pairs) + len(missing_indices)
                if k_ < nb_subplots:
                    i = k_ // nb_columns_
                    j = k_ % nb_columns_
                    template = templates[index]
                    title = "Additional template {}".format(index)
                    with_xaxis = (i == (nb_rows_ - 1))
                    with_yaxis = (j == 0)
                    with_scale_bars = (i == (nb_rows_ - 1)) and (j == 0)
                    template.plot(ax=ax[i, j], probe=template_store.probe, title=title, with_xaxis=with_xaxis,
                                  with_yaxis=with_yaxis, with_scale_bars=with_scale_bars, color='C1')

            detected_templates = templates
            generated_templates = injected_templates

            fig, ax = plt.subplots()
            title = "Templates g26 & d21"
            generated_templates[26].plot(ax=ax, probe=template_store.probe, color='C0')
            detected_templates[21].plot(ax=ax, probe=template_store.probe, title=title, color='C1')
            #
            # fig, ax = plt.subplots()
            # title = "Templates d13 & g11 & d28"
            # detected_templates[13].plot(ax=ax, probe=template_store.probe, color='C2')
            # generated_templates[11].plot(ax=ax, probe=template_store.probe, color='C0')
            # detected_templates[28].plot(ax=ax, probe=template_store.probe, title=title, color='C1')
            #
            # fig, ax = plt.subplots()
            # title = "Template d13 vs d26+28"
            # detected_templates[13].plot(ax=ax, probe=template_store.probe, color='C2')
            # mixed_template = (detected_templates[26] + detected_templates[28]) * 0.5
            # mixed_template.plot(ax=ax, probe=template_store.probe, title=title, color='C1')
            #
            # fig, ax = plt.subplots()
            # title = "Templates d23 & g19 & d26"
            # detected_templates[23].plot(ax=ax, probe=template_store.probe, color='C2')
            # generated_templates[19].plot(ax=ax, probe=template_store.probe, color='C0')
            # detected_templates[26].plot(ax=ax, probe=template_store.probe, title=title, color='C1')
            #
            # fig, ax = plt.subplots()
            # title = "Templates d23 & g11 & d28"
            # detected_templates[23].plot(ax=ax, probe=template_store.probe, color='C2')
            # generated_templates[11].plot(ax=ax, probe=template_store.probe, color='C0')
            # detected_templates[28].plot(ax=ax, probe=template_store.probe, title=title, color='C1')
            #
            # fig, ax = plt.subplots()
            # title = "Templates d14 & g10 & d12"
            # detected_templates[14].plot(ax=ax, probe=template_store.probe, color='C1')
            # generated_templates[10].plot(ax=ax, probe=template_store.probe, color='C0')
            # detected_templates[12].plot(ax=ax, probe=template_store.probe, title=title, color='C2')
            #
            # fig, ax = plt.subplots()
            # title = "Templates d3 & g22 & d25"
            # detected_templates[3].plot(ax=ax, probe=template_store.probe, color='C1')
            # generated_templates[22].plot(ax=ax, probe=template_store.probe, color='C0')
            # detected_templates[25].plot(ax=ax, probe=template_store.probe, title=title, color='C2')

            # def order_similarities(similarities):
            #
            #     nb_injected_templates = len(injected_templates)
            #     nb_detected_templates = len(templates)
            #
            #     s = similarities.copy()
            #     for k in range(0, min(nb_injected_templates, nb_detected_templates)):
            #         sub_s = s[k:, k:]
            #         i = np.argmax(sub_s)
            #         i, j = np.unravel_index(i, sub_s.shape)
            #         i, j = i + k, j + k
            #         # Swap row.
            #         tmp = s[k, :].copy()
            #         s[k, :] = s[i, :]
            #         s[i, :] = tmp
            #         # Swap column.
            #         tmp = s[:, k].copy()
            #         s[:, k] = s[:, j]
            #         s[:, j]= tmp
            #
            #     if nb_injected_templates > nb_detected_templates:
            #         for k in range(nb_detected_templates, nb_injected_templates):
            #             sub_s = s[k:, :]
            #             i = np.argmax(sub_s)
            #             print(i)
            #             i, j = np.unravel_index(i, sub_s.shape)
            #             i, j = i + k, j
            #             # Swap row.
            #             tmp = s[k, :].copy()
            #             s[k, :] = s[i, :]
            #             s[i, :] = tmp
            #
            #     return s
            #
            # ordered_similarities = order_similarities(similarities)
            #
            # fig, ax = plt.subplots()
            # im = ax.imshow(ordered_similarities, cmap='coolwarm', vmin=-1.0, vmax=+1.0)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im, cax=cax)
            # ax.set_xlabel(u"detected templates")
            # ax.set_ylabel(u"injected templates")
            # ax.set_title(u"Template similarities")

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    main()
