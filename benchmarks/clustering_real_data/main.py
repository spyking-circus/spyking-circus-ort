import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import circusort

import network


# Define directories.
directory = network.directory
configuration_directory = os.path.join(directory, "configuration")
original_data_path = os.path.join(configuration_directory, "data.raw")
original_probe_path = os.path.join(configuration_directory, "probe.prb")
recording_directory = os.path.join(directory, "recording")
sorting_directory = os.path.join(directory, "sorting")
output_directory = os.path.join(directory, "output")


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--preparation', dest='pending_preparation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    parser.add_argument('--analysis', dest='pending_analysis', action='store_true', default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    if np.all([args_dict[key] is None for key in args_dict]):
        for key in args_dict:
            args_dict[key] = True
    else:
        for key in args_dict:
            args_dict[key] = args_dict[key] is True

    # Configuration.

    if args.pending_configuration:

        # Make configuration directory (if necessary).
        if not os.path.isdir(configuration_directory):
            os.makedirs(configuration_directory)

        # Check if original data exists.
        if not os.path.isfile(original_data_path):
            string = "original data file not found: {}"
            message = string.format(original_data_path)
            raise IOError(message)
        else:
            string = "original data file: {}"
            message = string.format(original_data_path)
            print(message)

        # Check if original probe exists.
        if not os.path.isfile(original_probe_path):
            string = "original probe file not found: {}"
            message = string.format(original_probe_path)
            raise IOError(message)
        else:
            string = "original probe file: {}"
            message = string.format(original_probe_path)
            print(message)

    # Preparation.

    if args.pending_preparation:

        # 1. Load original data.
        sampling_rate = 20e+3  # Hz
        dtype = 'uint16'
        gain = 0.1042  # µV / arb. unit
        record = circusort.io.load_record(original_data_path, original_probe_path, sampling_rate=sampling_rate,
                                          dtype=dtype, gain=gain)

        # 2. Create copied data.
        copied_data_path = os.path.join(recording_directory, "data.raw")
        copied_probe_path = os.path.join(recording_directory, "probe.prb")
        # # 1. Parameters for the 9 electrodes & 5 minutes version.
        # channels = np.array([133, 134, 161, 166, 201, 202, 229, 231, 232]) - 1
        # t_min, t_max = 2.0 * 60.0, 7.0 * 60.0
        # t_plot_min, t_plot_max = 2.0 * 60.0, 3.0 * 60.0
        # 2. Parameters for the 252 electrodes & 5 minutes version.
        channels = np.array(list(range(1, 127)) + list(range(129, 255))) - 1  # i.e. discard 127, 128, 255 and 256
        t_min, t_max = 2.0 * 60.0, 7.0 * 60.0
        t_plot_min, t_plot_max = 2.0 * 60.0, 2.2 * 60.0
        # # 3. Parameters for the 252 electrodes & 30 minutes version.
        # channels = np.array(list(range(1, 127)) + list(range(129, 255))) - 1  # i.e. discard 127, 128, 255 and 256
        # t_min, t_max = 2.0 * 60.0, 32.0 * 60.0
        # t_plot_min, t_plot_max = 2.0 * 60.0, 2.2 * 60.0
        record.copy(copied_data_path, copied_probe_path, channels=channels, t_min=t_min, t_max=t_max)

        # 3. Overview the copied data.
        # Make output directory (if necessary).
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        # Load the copied data.
        data = circusort.io.load_datafile(copied_data_path, sampling_rate, len(channels), dtype, gain=gain)
        # Plot an overview of the copied data.
        path = os.path.join(output_directory, "data_overview.pdf")
        data.plot(output=path, t_min=t_plot_min, t_max=t_plot_max)
        # Close plot.
        plt.close()
        # Display message.
        string = "Overview of the prepared data: {}"
        message = string.format(path)
        print(message)

    # Sorting.

    if args.pending_sorting:

        network.sorting()

    # Analysis.

    if args.pending_analysis:

        # Plot templates.

        probe_path = os.path.join(recording_directory, "probe.prb")
        probe = circusort.io.load_probe(probe_path)

        template_store_path = os.path.join(sorting_directory, "templates.h5")
        template_store = circusort.io.load_template_store(template_store_path)

        nb_templates = len(template_store)

        for template_id in range(0, nb_templates):
            template = template_store.get(template_id)
            title = "Template {}".format(template_id)
            path = os.path.join(output_directory, "template_{}.pdf".format(template_id))
            template.plot(probe=probe, title=title)
            plt.savefig(path)
            plt.close()

        # Plot auto-correlograms.

        spikes_path = os.path.join(sorting_directory, "spikes.h5")
        spikes = circusort.io.load_spikes(spikes_path)

        nb_cells = len(spikes)

        for cell_id in range(0, nb_cells):
            cell = spikes.get_cell(cell_id)
            train = cell.train
            nb_spikes = len(train)
            bin_counts, bin_edges = train.auto_correlogram()
            bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bar_heights = bin_counts
            bar_widths = bin_edges[1:] - bin_edges[:-1]
            plt.bar(bar_centers, bar_heights, bar_widths)
            plt.xlabel("lag (ms)")
            plt.ylabel("spike count")
            plt.title("Auto-correlogram of template {} ({} spikes)".format(cell_id, nb_spikes))
            path = os.path.join(output_directory, "autocorrelogram_{}.pdf".format(cell_id))
            plt.savefig(path)
            plt.close()

    # TODO complete?


if __name__ == '__main__':

    main()
