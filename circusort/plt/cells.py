# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gds
import numpy as np
import sys


def plot_reconstruction(cells, t_min, t_max, sampling_rate, data_file, ax=None, output=None, channels=None,
                        mads=None, peaks=None, filtered_data=None):

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

    snippet = data_file.get_snippet(t_min, t_max)

    if filtered_data is not None:
        filtered_snippet = filtered_data.get_snippet(t_min, t_max)
    else:
        filtered_snippet = None

    if channels is None:
        channels = range(data_file.nb_channels)

    x = np.linspace(t_min, t_max, num=len(snippet), endpoint=False)

    y_spread = np.max(np.abs(snippet))
    y_scale = 0.5 / y_spread if y_spread > sys.float_info.epsilon else 1.0

    for k, channel in enumerate(channels):
        y_offset = float(k)
        ax.plot(x, y_scale * snippet[:, channel] + y_offset, color='0.5')
        if filtered_snippet is not None:
            ax.plot(x, y_scale * filtered_snippet[:, channel] + y_offset, color='0.75')
        ax.plot(x, y_scale * result[:, channel] + y_offset, color='r')

    # Add MADs (if possible).
    if mads is not None:
        mads_snippet = mads.get_snippet(t_min, t_max)
        x = mads.get_snippet_times(t_min, t_max)
        for k, channel in enumerate(channels):
            y_offset = float(k)
            y = y_scale * (-7.0 * mads_snippet[:, channel]) + y_offset
            ax.plot(x, y, color='C0')

    # Add peaks (if possible).
    if peaks is not None:
        for k, channel in enumerate(channels):
            y_offset = float(k)
            x = peaks.get_times(t_min=t_min, t_max=t_max, channels=[channel])
            y = y_offset * np.ones_like(x)
            ax.scatter(x, y, color='C1', marker='.', zorder=3)

    ax.set_xlabel(u"Times (s)")
    ax.set_ylabel(u"Channels")
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)

    return


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


def plot_rates(cells, time_bin=1, ax=None, output=None):

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    rates = cells.rate(time_bin)
    cax = ax.imshow(rates, aspect='auto', origin='lower')
    cbar = fig.colorbar(cax)
    ax.set_xlabel(u"Times (s)")
    ax.set_ylabel(u"# Cells")

    x = ax.get_xticks()
    axis = ['%g' %i for i in np.linspace(cells.t_min, cells.t_max, len(x[1:]))]
    ax.set_xticks(x[1:], axis)

    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)

    return