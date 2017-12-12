import numpy as np


class Cell(object):
    """Cell model.

    Attributes:
        template: tuple
            The template of the cell.
        train: numpy.ndarray
            The spike train of the cell.
        chunk_width: float
            The width of the chunks used to bin the train into chunk subtrains [s].
        subtrains: dictionary
            The chunk subtrains.
    """

    def __init__(self, template, train, position):
        """Initialization.

        Parameters:
            template: tuple
                The template of the cell.
            train: numpy.ndarray
                The spike train of the cell.
            position: numpy.ndarray
                The position of the cell.
        """

        self.template = template
        self.train = train
        self.position = position

        self.chunk_width = None
        self.subtrains = None

    def precompute_chunk_subtrains(self, chunk_width=51.2):
        """Precompute the chunk subtrains.

        Parameters:
            chunk_width: float (optional).
                The temporal width of the chunks [ms]. The default value is 51.2.
        """

        # Define the temporal width of the chunks.
        self.chunk_width = chunk_width * 1e-3
        # Define the subtrains associated to each chunk.
        subtrains = {}
        times = self.train
        chunk_indices = times / self.chunk_width
        chunk_indices = chunk_indices.astype(np.int)
        for chunk_index in np.unique(chunk_indices):
            is_in_chunk = (chunk_indices == chunk_index)
            subtrains[chunk_index] = times[is_in_chunk]
        self.subtrains = subtrains

        return self.subtrains

    def get_chunk_subtrain(self, chunk_index, chunk_width=51.2):
        """Get the specified chunk subtrain.

        Parameters:
            chunk_index: integer
                The index of the chunk to get.
            chunk_width: float (optional).
                The temporal width of the chunks [ms]. The default value is 51.2.
        """

        # Precompute the chunk subtrains (if necessary).
        if self.chunk_width != chunk_width:
            self.precompute_chunk_subtrains(chunk_width=chunk_width)

        # Get the specified chunk subtrain.
        default_subtrain = np.array([])
        ante_subtrain = self.subtrains.get(chunk_index - 1, default_subtrain)
        curr_subtrain = self.subtrains.get(chunk_index + 0, default_subtrain)
        post_subtrain = self.subtrains.get(chunk_index + 1, default_subtrain)
        subtrain = np.concatenate((ante_subtrain, curr_subtrain, post_subtrain))
        reference_time = float(chunk_index) * self.chunk_width
        subtrain = subtrain - reference_time

        return subtrain

    def get_template(self):
        """Get the template of the cell."""

        channels, waveforms = self.template
        nb_channels, nb_timestamps = waveforms.shape

        timestamps = np.arange(0, nb_timestamps) - (nb_timestamps - 1) // 2
        timestamps = timestamps[np.newaxis, :]
        timestamps = np.repeat(timestamps, repeats=nb_channels, axis=0)

        channels = channels[:, np.newaxis]
        channels = np.repeat(channels, repeats=nb_timestamps, axis=1)

        i = timestamps.flatten()
        j = channels.flatten()
        v = waveforms.flatten()

        return i, j, v
