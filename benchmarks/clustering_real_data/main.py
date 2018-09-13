import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import circusort

import network


directory = network.directory
configuration_directory = os.path.join(directory, "configuration")
original_data_path = os.path.join(configuration_directory, "data.raw")
original_probe_path = os.path.join(configuration_directory, "probe.prb")
recording_directory = os.path.join(directory, "recording")


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', dest='pending_configuration', action='store_true', default=None)
    parser.add_argument('--preparation', dest='pending_preparation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_configuration is None and args.pending_preparation is None and args.pending_sorting is None:
        args.pending_configuration = True
        args.pending_preparation = True
        args.pending_sorting = True
    else:
        args.pending_configuration = args.pending_configuration is True
        args.pending_preparation = args.pending_preparation is True
        args.pending_sorting = args.pending_sorting is True

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

        sampling_rate = 20e+3  # Hz
        dtype = 'uint16'
        gain = 0.1042  # µV / arb. unit

        record = circusort.io.load_record(original_data_path, original_probe_path, sampling_rate=sampling_rate,
                                          dtype=dtype, gain=gain)

        channels = np.array([133, 134, 161, 166, 201, 202, 229, 231, 232]) - 1
        t_min, t_max = 2.0 * 60.0, 7.0 * 60.0
        copied_data_path = os.path.join(recording_directory, "data.raw")
        copied_probe_path = os.path.join(recording_directory, "probe.prb")
        record.copy(copied_data_path, copied_probe_path, channels=channels, t_min=t_min, t_max=t_max)

        data = circusort.io.load_datafile(copied_data_path, sampling_rate, len(channels), dtype, gain=gain)

        data.plot(t_min=2.0 * 60.0, t_max=3.0 * 60.0)
        plt.show()

    # Sorting.

    if args.pending_sorting:

        network.sorting()

    # TODO complete?

    raise NotImplementedError()


if __name__ == '__main__':

    main()
