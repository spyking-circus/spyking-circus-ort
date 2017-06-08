def append_hdf5(dataset, data):
    '''Append 1D-array to a HDF5 dataset.

    Parameters
    ----------
    dataset: ?
        HDF5 dataset.
    data: numpy.ndarray
        1D-array.
    '''

    old_size = len(dataset)
    new_size = old_size + len(data)
    new_shape = (new_size,)
    dataset.resize(new_shape)
    dataset[old_size:new_size] = data

    return
