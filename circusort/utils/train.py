import numpy as np


def compute_correlation(source_train, sink_train, bin_size=1e-3, lag_min=-100e-3, lag_max=+100e-3):
    # TODO add docstring.

    source_times = source_train.times
    sink_times = sink_train.times

    n_min = int(np.ceil(lag_min / bin_size + 0.5))
    n_max = int(np.floor(lag_max / bin_size - 0.5))
    nb_bins = n_max - n_min + 1
    lag = bin_size * np.linspace(n_min, n_max, num=nb_bins)
    correlation = np.zeros(nb_bins)

    if source_times.size == 0 or sink_times.size == 0:
        pass
    else:
        i, j = (0, 0)
        while i < source_times.size and j < sink_times.size:
            if source_times[i] < sink_times[j]:
                k = i
                while k < source_times.size and sink_times[j] - source_times[k] >= lag_max:
                    k += 1
                while k < source_times.size and sink_times[j] - source_times[k] >= lag_min:
                    lag = sink_times[j] - source_times[k]
                    n = int(np.round(lag / bin_size))
                    correlation[n] += 1.0
                    k += 1
                i += 1
            else:
                k = j
                while k < sink_times.size and sink_times[k] - source_times[i] < lag_min:
                    k += 1
                while k < sink_times.size and sink_times[k] - source_times[i] < lag_min:
                    lag = sink_times[k] - source_times[i]
                    n = int(np.round(lag / bin_size))
                    correlation[n] += 1.0
                    k += 1
                j += 1

    correlation /= bin_size

    return lag, correlation


def compute_train_similarity(source_train, sink_train, **kwargs):
    # TODO add docstring.

    _, correlation = compute_correlation(source_train, sink_train, **kwargs)
    similarity = np.max(correlation)

    return similarity


def compute_dip_strength(source_train, sink_train, lag_min=-100e-3, lag_max=+100e-3,
                         tau_min=-2.5e-3, tau_max=+2.5e-3, **kwargs):
    # TODO add docstring.

    _, correlation = compute_correlation(source_train, sink_train,
                                         lag_min=lag_min, lag_max=lag_max, **kwargs)
    _, correlation_ = compute_correlation(source_train, sink_train.reverse(),
                                          lag_min=tau_min, lag_max=tau_max, **kwargs)
    c = np.mean(correlation)
    c_ = np.mean(correlation_)
    strength = (c - c_) / (c + c_)

    return strength
