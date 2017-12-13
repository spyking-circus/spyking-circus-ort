import h5py


class Train(object):
    # TODO add docstring

    def __init__(self, times):
        # TODO add docstring.

        self.times = times

    def save(self, path):
        """Save train to file.

        Parameters:
            path: string
                The path to the file in which to save the train.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('times', shape=self.times.shape, dtype=self.times.dtype, data=self.times)
        file_.close()

        return

    # TODO complete.
