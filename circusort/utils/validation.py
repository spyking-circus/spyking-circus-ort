import numpy as np
from circusort.obj.train import Train

def get_fp_fn_rate(spike_trains, target, jitter):
    '''Return the false positive and false negative rates for a given
    spike train (target) compared to a list of spike trains. All matches
    are established up to a certain jitter, expressed in time steps.
    
    The function returns a list of tuples (false positive rate, false negative rate) 
    for every input spike trains
    '''

    results = []

    assert isinstance(spike_trains, list), "spike_trains should be a list of spike trains"

    for spk in spike_trains:
        count = 0
        for spike in spk.times:
            idx = np.where(np.abs(target.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(spk) > 0:
            fp_rate = count/float(len(spk))
        else:
            fp_rate = 0

        count = 0
        for spike in target:
            idx = np.where(np.abs(spk.times - spike) < jitter)[0]
            if len(idx) > 0:
                count += 1
        if len(target) > 0:
            fn_rate = count/(float(len(target)))
        else:
            fn_rate = 0

        results += [[1 - fp_rate, 1 - fn_rate]]

    return results


def best_match(spike_trains, target, jitter):
    '''Return the best combination of spike trains that will reduce
    as much as possible the total error while reconstructing the target
    spike train, the error being defined as the mean between false 
    positive and false negatives. 
    The function returns a list of the selected units, and the corresponding
    errors [false postive, false negative]'''

    results = []

    assert isinstance(spike_trains, list), "spike_trains should be a list of spike trains"

    selection  = []
    error      = [1, 1]
    find_next  = True
    sel_spikes = []

    while (find_next == True):

        to_explore   = np.setdiff1d(spike_trains.ids, np.unique(selection))
        ids          = spike_trains.ids
            
        if len(to_explore) > 0:

            new_spike_trains = spike_trains.slice(to_explore + sel_spikes)

            local_errors = get_fp_fn_rate(new_spike_trains, target, jitter)
            print local_errors
            errors       = np.mean(local_errors, 1)

            if np.min(errors) <= np.mean(error):
                idx         = np.argmin(errors)
                selection  += [to_explore[idx]]
                error       = local_errors[idx]
                sel_spikes  = new_spike_trains[idx]
            else:
                find_next = False
        else:
            find_next = False

    return selection, error