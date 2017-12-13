import numpy as np
import os

from .parameter.cell import CellParameters


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

        self.parameters = CellParameters()

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
        times = self.train.times
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

        channels = self.template.channels
        waveforms = self.template.waveforms
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

    def save(self, directory):
        """Save the cell to file.

        Parameters:
        directory: string
            The directory in which to save the cell.
        """

        # Create the directory (if necessary).
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Save the template of the cell.
        template_path = os.path.join(directory, "template.h5")
        self.parameters.add('template', 'path', template_path)
        self.parameters.add('template', 'mode', 'default')
        template = self.template
        template.save(template_path)

        # Save the train of the cell.
        train_path = os.path.join(directory, "train.h5")
        self.parameters.add('train', 'path', train_path)
        self.parameters.add('train', 'mode', 'default')
        train = self.train
        train.save(train_path)

        # Save the position of the cell.
        position_path = os.path.join(directory, "position.h5")
        self.parameters.add('position', 'path', position_path)
        self.parameters.add('position', 'mode', 'default')
        position = self.position
        position.save(position_path)

        # Save the parameters of the cell.
        parameters_path = os.path.join(directory, "parameters.txt")
        parameters = self.parameters
        parameters.save(parameters_path)

        return
