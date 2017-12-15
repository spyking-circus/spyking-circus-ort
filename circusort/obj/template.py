import h5py


class Template(object):
    # TODO add docstring

    def __init__(self, channels, waveforms):
        # TODO add docstring.

        self.channels = channels
        self.waveforms = waveforms

    def save(self, path):
        """Save template to file.

        Parameters:
            path: string
                The path to file in which to save the template.
        """

        file_ = h5py.File(path, mode='w')
        file_.create_dataset('channels', shape=self.channels.shape, dtype=self.channels.dtype, data=self.channels)
        file_.create_dataset('waveforms', shape=self.waveforms.shape, dtype=self.waveforms.dtype, data=self.waveforms)
        file_.close()

        return

    # TODO complete.
