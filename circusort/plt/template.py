# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from circusort.io.template import TemplateStore, TemplateComponent, Template


def plot_template(template_store, probe_file, templates, components=['first'], output=None):
    """Plot template from template store.

    Parameters:
        template_store: string
            The path of the template store to be used when loading the data.
        templates: none | list (optional)
            The templates to be used when loading the data. If is None then all the channels will be used. The default
            value is None.
        components: first | second | both
            The components that should be displayed
        output: none | string (optional)
            The path to be used to save the figure. If is None then display the figure and block until the figure
            have been closed. The default value is None.
    """

    # Expand user's home directory (if necessary).
    template_store = TemplateStore(os.path.expanduser(template_store))
    probe          = load_probe(os.path.expanduser(probe_file))

    # Define the channels to be used when loading the data (if necessary).
    if templates is None:
        templates = template_store.indices

    templates = template_store.get(templates)
    for t in templates:
        t.first_component


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
