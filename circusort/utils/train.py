import numpy as np


def compute_correlation(source_train, sink_train, bin_size=1e-3, lag_min=-100e-3, lag_max=+100e-3, **kwargs):
    """Compute the cross-correlogram between two spike trains.

    Parameters:
        source_train: circusort.obj.Train
            The spike train to use as reference.
        sink_train: circusort.obj.Train
            The spike train to compare to the spike train of reference.
        bin_size: float (optional)
            The size of the bin to use [s]. The default value is 1e-3.
        lag_min: float (optional)
            The minimum of the lag window of interest [s]. The default value is -100e-3.
        lag_max: float (optional)
            The maximum of the lag window of interest [s]. The default value is +100e-3.
    Returns:
        lags: numpy.ndarray
            The lag values for each bin of the cross-correlogram. An array of shape: (nb_bins,).
        correlations: numpy.ndarray
            The correlation values for each bin of the cross-correlogram. An array of shape: (nb_bins,).
    """

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


def compute_reversed_correlation(source_train, sink_train, **kwargs):
    """Compute the reversed cross-correlogram between two spike trains.

    Parameters:
        source_train: circusort.obj.Train
            The spike train to use as reference.
        sink_train: circusort.obj.Train
            The spike train to reverse and compare to the spike train of reference.
    Returns:
        lags: numpy.ndarray
            The lag values for each bin of the cross-correlogram. An array of shape: (nb_bins,).
        correlations: numpy.ndarray
            The correlation values for each bin of the cross-correlogram. An array of shape: (nb_bins,).
    See also:
        circusort.utils.train.compute_correlation
    """

    assert source_train.t_min == sink_train.t_min, "{} != {}".format(source_train.t_min, sink_train.t_min)
    assert source_train.t_max == sink_train.t_max, "{} != {}".format(source_train.t_max, sink_train.t_max)

    reverted_sink_train = sink_train.reverse()
    lags, correlations = compute_correlation(source_train, reverted_sink_train, **kwargs)

    return lags, correlations


def compute_train_similarity(source_train, sink_train, t_min=None, t_max=None, **kwargs):
    """Compute the similarity between two spike trains.

    Parameters:
        source_train: circusort.obj.Train
            The spike train to use as reference.
        sink_train: circusort.obj.Train
            The spike train to compare to the spike train of reference.
        t_min: none | float
            The minimum of the time window of interest. The default value is None.
        t_max: none | float
            The maximum of the time window of interest. The default value is None.
    Return:
        similarity: float
            The similarity between the two spike trains as the maximum correlation between them.
    See also:
        circusort.utils.train.compute_correlation
    """

    source_train = source_train.slice(t_min=t_min, t_max=t_max)
    sink_train = sink_train.slice(t_min=t_min, t_max=t_max)
    _, correlations = compute_correlation(source_train, sink_train, **kwargs)
    similarity = np.max(correlations)

    return similarity


def compute_pic_strength(source_train, sink_train, lag_min=-100e-3, lag_max=+100e-3,
                         tau_min=-2.5e-3, tau_max=+2.5e-3, **kwargs):
    """Compute the strength of the pic of correlation between two spike trains around 0.0 ms lag.

    Parameters:
        source_train: circusort.obj.Train
            The spike train to use as reference.
        sink_train: circusort.obj.Train
            The spike train to compare to the spike train of reference.
        lag_min: float (optional)
            The minimum of the lag window of interest to estimate the baseline correlation [s]. The default value is
            -100e-3.
        lag_max: float (optional)
            The maximum of the lag window of interest to estimate the baseline correlation [s]. The default value is
            +100e-3.
        tau_min: float (optional)
            The minimum of the lag window of interest to estimate the pic of correlation [s]. The default value is
            -2.5e-3.
        tau_max: float (optional)
            The maximum of the lag window of interest to estimate the pic of correlation [s]. The default value is
            +2.5e-3.
    Return:
        strength: float
            The strength of the pic of correlation between the two spike trains around 0.0 ms lag.
    See also:
        circusort.utils.train.compute_correlation
        circusort.utils.train.compute_reversed_correlation
    """

    _, correlations = compute_correlation(source_train, sink_train,
                                          lag_min=tau_min, lag_max=tau_max, **kwargs)
    _, correlations_ = compute_reversed_correlation(source_train, sink_train,
                                                    lag_min=lag_min, lag_max=lag_max, **kwargs)
    c = np.mean(correlations)
    c_ = np.mean(correlations_)
    strength = (c - c_) / (c + c_)

    return strength
