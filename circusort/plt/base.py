import matplotlib.pyplot as plt
import numpy as np
import sys


def _plot_time_of_interest(data, time, window=10e-3, sampling_rate=20e+3, channels=None,
                           cells=None, probe=None, peaks=None, voltage_factor=0.5, ax=None, **kwargs):

    _ = kwargs

    if ax is None:
        _, ax = plt.subplots()

    t_min = time - window / 2.0
    t_max = time + window / 2.0
    k_min = int(t_min * sampling_rate)
    k_max = int(t_max * sampling_rate)

    snippet = data.get_snippet(t_min, t_max)

    if cells is not None:
        nb_channels = cells[cells.ids[0]].template.first_component.nb_channels
        result = np.zeros((k_max - k_min, nb_channels), dtype='float32')
        for c in cells:
            width = c.template.temporal_width
            half_width = width // 2
            sub_train = c.slice(t_min + half_width/sampling_rate, t_max - half_width/sampling_rate)
            t1 = sub_train.template.first_component.to_dense().T

            if sub_train.template.two_components:
                t2 = c.template.two_components.to_dense().T
            else:
                t2 = 0.0

            for spike, amp in zip(sub_train.train, sub_train.amplitude):
                offset = int(spike*sampling_rate) - k_min

                if c.template.two_components:
                    result[int(offset - half_width):int(offset + half_width + 1), :] += amp[0] * t1 + amp[1] * t2
                else:
                    result[int(offset - half_width):int(offset + half_width + 1), :] += amp * t1
    else:
        result = None

    if channels is None:
        channels = range(data.nb_channels)

    x = np.linspace(t_min, t_max, num=len(snippet), endpoint=False)

    y_spread = np.max(np.abs(snippet))
    y_scale = 0.5 / y_spread if y_spread > sys.float_info.epsilon else 1.0

    if probe is None:

        for k, channel in enumerate(channels):
            y_offset = float(k)
            ax.plot(x, y_scale * snippet[:, channel] + y_offset, color='0.5')
            # if filtered_snippet is not None:
            #     ax.plot(x, y_scale * filtered_snippet[:, channel] + y_offset, color='0.75')
            if result is not None:
                ax.plot(x, y_scale * result[:, channel] + y_offset, color='r')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel("time (s)")
        ax.set_ylabel("channel")
        ax.set_title("t = {} s".format(time))

    else:

        time_factor = 0.9 * (probe.minimum_interelectrode_distance / window)

        ax.set_aspect('equal')
        x_min, x_max = probe.x_limits
        y_min, y_max = probe.y_limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for k, channel in enumerate(channels):
            # Plot waveform.
            x_0, y_0 = probe.get_channel_position(channel)
            x_ = time_factor * (x - np.mean(x)) + x_0
            y_ = voltage_factor * snippet[:, channel] + y_0
            label = "waveform {}".format(channel)
            ax.plot(x_, y_, label=label, **kwargs)
            # Plot peaks (if necessary).
            if peaks is not None:
                times = peaks.get_times(t_min=t_min, t_max=t_max, channels=[channel])
                if len(times) > 0:
                    x_ = time_factor * (times - np.mean(x)) + x_0
                    y_ = np.zeros_like(x_) + y_0
                    ax.scatter(x_, y_, marker='|', color='C1')

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("t = {} s".format(time))

    return


def plot_time_of_interest(data, time, ax=None, path=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    _plot_time_of_interest(data, time, ax=ax, **kwargs)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path)
        plt.close(fig)

    return


def plot_times_of_interest(data, times, ax=None, path=None, **kwargs):

    nb_times = len(times)

    if ax is None:
        fig, ax = plt.subplots(ncols=nb_times)
    else:
        fig = ax.get_figure()

    if nb_times <= 0:
        pass
    elif nb_times == 1:
        time = times[0]
        plot_time_of_interest(data, time, ax=ax, **kwargs)
    else:
        for time, ax_ in zip(times, ax):
            plot_time_of_interest(data, time, ax=ax_, **kwargs)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path)

    return
