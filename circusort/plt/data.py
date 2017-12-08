# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def plot_data(path, dtype='uint16', nb_channels=1, sampling_rate=20e+3,
              t_min=0.0, t_max=1.0, channels=None, output=None):
    """Plot data from file.

    Parameters:
        path: string
            The path to be used when loading the data.
        dtype: string (optional)
            The data type to be used when loading the data. The default value is 'uint16'.
        nb_channels: integer (optional)
            The number of channels to be used when loading the data. The default value is 1.
        sampling_rate: float (optional)
            The sampling rate to be used when loading the data. The default value is 20e+3.
        t_min: float (optional)
            The start time of the temporal window to be used when loading the data [s]. The default value is 0.0.
        t_max: float (optional)
            The end time of the temporal window to be used when loading the data [s]. The default value is 1.0.
        channels: none | list (optional)
            The channels to be used when loading the data. If is None then all the channels will be used. The default
            value is None.
        output: none | string (optional)
            The path to be used to save the figure. If is None then display the figure and block until the figure
            have been closed. The default value is None.
    """

    # Expand user's home directory (if necessary).
    path = os.path.expanduser(path)

    # Create the memory-map to the data stored in a binary file on disk.
    fp = np.memmap(path, dtype=dtype, mode='r')
    fp = np.reshape(fp, (-1, nb_channels))

    # Compute the start and end indices of the temporal window to be used when loading the data.
    i_min = int(np.ceil(t_min * sampling_rate))
    i_max = int(np.floor(t_max * sampling_rate)) + 1

    # Compute the number of samples which fall in the temporal window.
    nb_samples = i_max - i_min

    # Define the channels to be used when loading the data (if necessary).
    if channels is None:
        channels = list(range(0, nb_channels))
    nb_selected_channels = len(channels)

    # Load the data.
    data = fp[i_min:i_max, channels]
    data = data.astype(np.float)
    # TODO voltage rescaling to get values in µV.

    # Plot the figure.
    x = np.linspace(i_min, i_max, num=nb_samples, endpoint=False) / sampling_rate
    plt.style.use('seaborn-paper')
    if nb_selected_channels == 1:
        # Plot a single channel.
        plt.subplots()
        y = data[:, 0]
        plt.plot(x, y, color='C0')
        plt.xlabel("time (s)")
        plt.ylabel("voltage (arb. unit)")
    else:
        # Plot multiple channels.
        plt.subplots()
        y_spread = np.max(np.abs(data))
        y_scale = 0.5 / y_spread if y_spread > sys.float_info.epsilon else 1.0
        for j in range(0, data.shape[1]):
            y_offset = float(j)
            y = y_scale * data[:, j] + y_offset
            plt.plot(x, y, color='C0')
        # TODO add scale bar.
        locations = list(range(0, nb_selected_channels))
        labels = [str(channel) for channel in channels]
        plt.yticks(locations, labels)
        plt.xlabel("time (s)")
        plt.ylabel("channel")
    plt.title("{}".format(path))
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

    return
