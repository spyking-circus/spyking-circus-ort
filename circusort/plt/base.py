import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_time_of_interest(data, time, window=10e-3, sampling_rate=20e+3, channels=None, cells=None, ax=None, **kwargs):

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
        plot_time_of_interest(data, time, ax=ax,)
    else:
        for time, ax_ in zip(times, ax):
            plot_time_of_interest(data, time, ax=ax_, **kwargs)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path)

    return
