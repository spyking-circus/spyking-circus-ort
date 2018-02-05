# -*- coding: utf-8 -*-

import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def plot_data(data_file, t_min=0.0, t_max=1.0, channels=None, ax=None, output=None):
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

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    # Plot the figure.
    snippet = data_file.get_snippet(t_min, t_max)

    x = np.linspace(t_min, t_max, num=len(snippet), endpoint=False)

    # Define the channels to be used when loading the data (if necessary).
    if channels is None:
        channels = list(range(data_file.nb_channels))
    else:
        snippet = snippet[:, channels]

    nb_selected_channels = len(channels)

    # Plot multiple channels.
    y_spread = np.max(np.abs(snippet))
    y_scale = 0.5 / y_spread if y_spread > sys.float_info.epsilon else 1.0
    for j in range(0, snippet.shape[1]):
        y_offset = float(j)
        y = y_scale * snippet[:, j] + y_offset
        ax.plot(x, y, color='C0')
    # TODO add scale bar.
    locations = list(range(0, nb_selected_channels))
    labels = [str(channel) for channel in channels]
    ax.set_yticks(locations, labels)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("channel")
    ax.set_title("{}".format(data_file.path))

    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)

    return
