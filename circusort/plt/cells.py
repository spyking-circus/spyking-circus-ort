# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gds
import numpy as np


def plot_reconstruction(cells, t_min, t_max, sampling_rate, data_file=None, ax=None, output=None, channels=None):

    sampling_rate = float(sampling_rate)
    gmin = int(t_min * sampling_rate)
    gmax = int(t_max * sampling_rate)

    nb_channels = cells[cells.ids[0]].template.first_component.nb_channels

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    result = np.zeros((gmax - gmin, nb_channels), dtype='float32')
    for c in cells:
        width = c.template.temporal_width
        half_width = width // 2
        sub_train = c.slice(t_min + half_width/sampling_rate, t_max - half_width/sampling_rate)
        t1 = sub_train.template.first_component.to_dense().T

        if sub_train.template.two_components:
           t2 = c.template.two_components.to_dense().T

        for spike, amp in zip(sub_train.train, sub_train.amplitude):
            offset = int(spike*sampling_rate) - gmin

            if c.template.two_components:
                result[int(offset - half_width):int(offset + half_width + 1), :] += amp[0] * t1 + amp[1] * t2
            else:
                result[int(offset - half_width):int(offset + half_width + 1), :] += amp * t1

    if data_file is not None:
        snippet = data_file.get_snippet(t_min, t_max)

    if channels is None:
        channels = range(data_file.nb_channels)

    y_spread = np.max(np.abs(snippet))
    y_scale = 0.5 / y_spread if y_spread > sys.float_info.epsilon else 1.0

    for j, i in enumerate(channels):
        y_offset = float(j)
        if data_file is not None:
            ax.plot(y_scale * snippet[:, i] + y_offset, '0.5')
        ax.plot(y_scale * result[:, i] + y_offset, 'r')

    ax.set_xlabel(u"Times (s)")
    ax.set_ylabel(u"# Cells")
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)



def raster_plot(cells, t_min, t_max, ax=None, output=None):

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    for count, c in enumerate(cells):
        sub_train = c.slice(t_min, t_max)
        ax.scatter(sub_train.train.times, [count]*len(sub_train.train), c='k', marker='|')

    ax.set_xlabel(u"Times (s)")
    ax.set_ylabel(u"# Cells")
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)

    return
