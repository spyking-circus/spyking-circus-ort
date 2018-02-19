# -*- coding: utf-8 -*-

import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.io.parameter import get_cell_parameters
from circusort.utils.path import normalize_path


class Cell(object):
    """Cell model.

    Attributes:
        template: circusort.obj.Template
            The template of the cell.
        train: circusort.obj.Train
            The spike train of the cell.
        amplitude: none | circusort.obj.Amplitude
            The amplitude of the cells.
        position: none | circusort.obj.Position
            The position of the cells.
        chunk_width: float
            The width of the chunks used to bin the train into chunk subtrains [s].
        subtrains: dictionary
            The chunk subtrains.
    """

    def __init__(self, template, train, amplitude=None, position=None, parameters=None):
        """Initialization.

        Parameters:
            template: circusort.obj.Template
                The template of the cell.
            train: circusort.obj.Train
                The spike train of the cell.
            amplitude: none | circusort.obj.Amplitude (optional)
                The amplitude of the cell. The default value is None.
            position: none | circusort.obj.Position (optional)
                The position of the cell. The default value is None.
            parameters: none | circusort.obj.CellParameters (optional)
                The parameters of the cell. The default value is None.
        """

        self.template = template
        self.train = train
        self.amplitude = amplitude
        self.position = position

        if amplitude is not None:
            if len(self.train) > 0 and len(self.amplitude) > 0:
                assert self.train.t_min <= self.amplitude.t_min, "t_min for Train and Amplitude do not match"
                assert self.train.t_max >= self.amplitude.t_max, "t_max for Train and Amplitude do not match"

        self.parameters = get_cell_parameters() if parameters is None else parameters

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

    @property
    def t_min(self):

        return self.train.t_min

    @property
    def t_max(self):

        return self.train.t_max

    @property
    def mean_rate(self):

        return self.train.mean_rate

    def rate(self, time_bin=1):

        bins = np.arange(self.t_min, self.t_max, time_bin)
        x, y = np.histogram(self.train.times, bins=bins)

        return x/time_bin

    def slice(self, t_min=None, t_max=None):
        # TODO add docstring.

        if self.amplitude is not None:
            new_amplitude = self.amplitude.slice(t_min, t_max)
        else:
            new_amplitude = None

        cell = Cell(self.template, self.train.slice(t_min, t_max), new_amplitude, self.position)

        return cell

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
        self.template_path = os.path.join(directory, "template.h5")
        self.parameters.add('template', 'path', self.template_path)
        self.parameters.add('template', 'mode', 'default')
        self.template.save(self.template_path)

        # Save the train of the cell.
        self.train_path = os.path.join(directory, "train.h5")
        self.parameters.add('train', 'path', self.train_path)
        self.parameters.add('train', 'mode', 'default')
        self.train.save(self.train_path)

        # Save the amplitude of the cell (if necessary).
        if self.amplitude is not None:
            self.amplitude_path = os.path.join(directory, "amplitude.h5")
            self.parameters.add('amplitude', 'path', self.amplitude_path)
            self.parameters.add('amplitude', 'mode', 'default')
            self.amplitude.save(self.amplitude_path)

        # Save the position of the cell (if necessary).
        if self.position is not None:
            self.position_path = os.path.join(directory, "position.h5")
            self.parameters.add('position', 'path', self.position_path)
            self.parameters.add('position', 'mode', 'default')
            self.position.save(self.position_path)

        # Save the parameters of the cell.
        parameters_path = os.path.join(directory, "parameters.txt")
        self.parameters.save(parameters_path)
        # TODO correct the previous lines.

        return

    def plot(self, output=None, **kwargs):
        # TODO add docstring.

        if output is None:

            raise NotImplementedError()  # TODO complete.

        else:

            path = normalize_path(output)
            if os.path.isdir(path):
                if 'rate' in self.parameters['train']:
                    self.plot_rate(output=path, **kwargs)
                if 'x' in self.parameters['position'] and 'y' in self.parameters['position']:
                    self.plot_position(output=path, **kwargs)
                if self.amplitude is not None:
                    self.amplitude.plot(output=path, **kwargs)
                if self.position is not None:
                    self.position.plot(output=path, **kwargs)
                self.train.plot(output=path, **kwargs)
                self.template.plot(output=path, **kwargs)
            else:
                raise NotImplementedError()  # TODO complete.

        return

    def _plot_rate(self, ax, t_min=0.0, t_max=10.0, **kwargs):

        if 'rate' in self.parameters['train']:

            rate = self.parameters['train']['rate']
            if isinstance(rate, float):
                rate = eval("lambda t: {}".format(rate))
            elif isinstance(rate, (str, unicode)):
                rate = eval("lambda t: {}".format(rate), kwargs)
            else:
                message = "Unknown rate type: {}".format(type(rate))
                raise TypeError(message)
            rate = np.vectorize(rate)

            x = np.linspace(t_min, t_max, num=1000)
            y = rate(x)

            ax.plot(x, y)
            ax.set_xlim(t_min, t_max)
            ax.set_xlabel(u"time (s)")
            ax.set_ylabel(u"rate (Hz)")
            ax.set_title(u"Rate")

        else:

            raise NotImplementedError()  # TODO plot the empirical firing rate instead.

        return

    def plot_rate(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(1, 1)
            ax_ = plt.subplot(gs[0])
            self._plot_rate(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                plt.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "parameters_rate.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                plt.tight_layout()
                plt.savefig(path)
        else:
            self._plot_rate(ax, **kwargs)

        return

    def _plot_position(self, ax, **kwargs):
        # TODO add docstring.

        self.plot_x_position(ax=ax[0], **kwargs)
        self.plot_y_position(ax=ax[1], **kwargs)

        return

    def plot_position(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        if ax is None:
            fig = plt.figure()
            gs = gds.GridSpec(2, 1)
            ax_ = [plt.subplot(gs[i]) for i in [0, 1]]
            self._plot_position(ax_, **kwargs)
            gs.tight_layout(fig)
            if output is None:
                plt.show()
            else:
                path = normalize_path(output)
                if path[-4:] != ".pdf":
                    path = os.path.join(path, "parameters_position.pdf")
                directory = os.path.dirname(path)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                plt.savefig(path)
        else:
            self._plot_position(ax, **kwargs)

        return

    def _plot_x_position(self, ax, t_min=0.0, t_max=10.0, **kwargs):

        x = self.parameters['position']['x']
        if isinstance(x, float):
            x = eval("lambda t: {}".format(x))
        elif isinstance(x, (str, unicode)):
            x = eval("lambda t: {}".format(x), kwargs)
        else:
            message = "Unknown x type: {}".format(type(x))
            raise TypeError(message)
        x = np.vectorize(x)

        t = np.linspace(t_min, t_max, num=1000)
        x = x(t)

        ax.plot(t, x, color='C0')
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel(u"time (s)")
        ax.set_ylabel(u"x (µm)")
        ax.set_title(u"x-coordinate")

        return

    def plot_x_position(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        try:
            if ax is None:
                fig = plt.figure()
                gs = gds.GridSpec(1, 1)
                ax_ = plt.subplot(gs[0])
                self._plot_x_position(ax_, **kwargs)
                gs.tight_layout(fig)
                if output is None:
                    plt.show()
                else:
                    path = normalize_path(output)
                    if path[-4:] != ".pdf":
                        path = os.path.join(path, "parameters_x_position.pdf")
                    directory = os.path.dirname(path)
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    plt.savefig(path)
            else:
                self._plot_x_position(ax, **kwargs)
        except TypeError:
            pass

        return

    def _plot_y_position(self, ax, t_min=0.0, t_max=10.0, **kwargs):

        y = self.parameters['position']['y']
        if isinstance(y, float):
            y = eval("lambda t: {}".format(y))
        elif isinstance(y, (str, unicode)):
            y = eval("lambda t: {}".format(y), kwargs)
        else:
            message = "Unknown y type: {}".format(type(y))
            raise TypeError(message)
        y = np.vectorize(y)

        t = np.linspace(t_min, t_max, num=1000)
        y = y(t)

        ax.plot(t, y, color='C1')
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel(u"time (s)")
        ax.set_ylabel(u"y (µm)")
        ax.set_title(u"y-coordinate")

        return

    def plot_y_position(self, output=None, ax=None, **kwargs):
        # TODO add docstring.

        if output is not None and ax is None:
            plt.ioff()

        try:
            if ax is None:
                fig = plt.figure()
                gs = gds.GridSpec(1, 1)
                ax_ = plt.subplot(gs[0])
                self._plot_y_position(ax_, **kwargs)
                gs.tight_layout(fig)
                if output is None:
                    plt.show()
                else:
                    path = normalize_path(output)
                    if path[-4:] != ".pdf":
                        path = os.path.join(path, "parameters_y_position.pdf")
                    directory = os.path.dirname(path)
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    plt.savefig(path)
            else:
                self._plot_y_position(ax, **kwargs)
        except TypeError:
            pass

        return
