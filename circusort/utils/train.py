import numpy as np


def compute_correlation(source_train, sink_train, bin_size=1e-3, lag_min=-100e-3, lag_max=+100e-3, **kwargs):
    # TODO add docstring.

    _ = kwargs  # Discard additional keyword arguments.

    assert source_train.t_min == sink_train.t_min, "{} != {}".format(source_train.t_min, sink_train.t_min)
    assert source_train.t_max == sink_train.t_max, "{} != {}".format(source_train.t_max, sink_train.t_max)

    n_min = int(np.ceil(lag_min / bin_size + 0.5))
    n_max = int(np.floor(lag_max / bin_size - 0.5))
    nb_bins = n_max - n_min + 1
    lags = bin_size * np.linspace(n_min, n_max, num=nb_bins)
    lag_min = bin_size * float(n_min - 0.5)
    lag_max = bin_size * float(n_max + 0.5)
    correlations = np.zeros(nb_bins)

    if source_train.times.size == 0:
        pass
    elif sink_train.times.size == 0:
        pass
    else:
        # Sort times.
        source_times = np.sort(source_train.times)
        sink_times = np.sort(sink_train.times)
        # Iter through interesting pair of spikes.
        i, j = (0, 0)
        while i < source_times.size and j < sink_times.size:
            if source_times[i] < sink_times[j]:
                k = i
                while k < source_times.size and lag_max <= sink_times[j] - source_times[k]:
                    k += 1
                while k < source_times.size and lag_min <= sink_times[j] - source_times[k]:
                    lag = sink_times[j] - source_times[k]
                    n = int(np.round(lag / bin_size))
                    correlations[n - n_min] += 1.0
                    k += 1
                i += 1
            else:
                k = j
                while k < sink_times.size and sink_times[k] - source_times[i] < lag_min:
                    k += 1
                while k < sink_times.size and sink_times[k] - source_times[i] < lag_max:
                    lag = sink_times[k] - source_times[i]
                    n = int(np.round(lag / bin_size))
                    correlations[n - n_min] += 1.0
                    k += 1
                j += 1
        # Normalize correlations.
        correlations /= (source_train.t_max - source_train.t_min)

    return lags, correlations


def compute_reverted_correlation(source_train, sink_train, **kwargs):

    assert source_train.t_min == sink_train.t_min
    assert source_train.t_max == sink_train.t_max

    reverted_sink_train = sink_train.reverse()
    lags, correlations = compute_correlation(source_train, reverted_sink_train, **kwargs)

    return lags, correlations


def compute_train_similarity(source_train, sink_train, t_min=None, t_max=None, **kwargs):
    # TODO add docstring.

    source_train = source_train.slice(t_min=t_min, t_max=t_max)
    sink_train = sink_train.slice(t_min=t_min, t_max=t_max)
    _, correlations = compute_correlation(source_train, sink_train, **kwargs)
    similarity = np.max(correlations)

    return similarity


def compute_dip_strength(source_train, sink_train, lag_min=-100e-3, lag_max=+100e-3,
                         tau_min=-2.5e-3, tau_max=+2.5e-3, **kwargs):
    # TODO add docstring.

    _, correlations = compute_correlation(source_train, sink_train,
                                         lag_min=lag_min, lag_max=lag_max, **kwargs)
    _, correlations_ = compute_correlation(source_train, sink_train.reverse(),
                                          lag_min=tau_min, lag_max=tau_max, **kwargs)
    c = np.mean(correlations)
    c_ = np.mean(correlations_)
    strength = (c - c_) / (c + c_)

    return strength
