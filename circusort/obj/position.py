import h5py


class Position(object):
    # TODO add docstring.

    def __init__(self, x, y):
        # TODO add docstring.

        self.x = x
        self.y = y

    def save(self, path):
        """Save position to file.

        Parameters:
            path: string
                The path to the file in which to save the position.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('x', shape=self.x.shape, dtype=self.x.dtype, data=self.x)
        file_.create_dataset('y', shape=self.y.shape, dtype=self.y.dtype, data=self.y)
        file_.close()

        return

    # TODO complete.
