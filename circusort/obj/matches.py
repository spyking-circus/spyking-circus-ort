import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.obj.match import Match


class Matches(object):

    def __init__(self, cells_pred, cells_true, threshold=0.9, t_min=None, t_max=None, path=None):

        self._cells_pred = cells_pred
        self._cells_true = cells_true
        self._threshold = threshold
        self._t_min = t_min
        self._t_max = t_max
        self._path = path

        self._indices = None
        self._errors = None
        self._ordered_indices = None

        self._update()

    def __getitem__(self, i):

        if i < - self._nb_matches or self._nb_matches <= i:
            string = "index {} is out of bounds for matches with size {}"
            message = string.format(i, self._nb_matches)
            raise IndexError(message)
        elif i < 0:
            i = self._nb_matches - i
        else:
            pass
        j = self._indices[i]
        cell_pred = self._cells_pred[i]
        cell_true = self._cells_true[j]
        match = Match(cell_pred, cell_true, t_min=self._t_min, t_max=self._t_max)

        return match

    @property
    def _nb_matches(self):

        return self._cells_pred.nb_cells

    @property
    def false_negative_rates(self):

        rates = np.zeros(self._nb_matches, dtype=np.float)
        for i, j in enumerate(self._indices):
            if j == -1:
                rates[i] = 0.0
            else:
                cell_pred = self._cells_pred[i]
                cell_true = self._cells_true[j]
                train_pred = cell_pred.train
                train_true = cell_true.train
                rates[i] = train_pred.compute_false_negative_rate(train_true, t_min=self._t_min, t_max=self._t_max)

        return rates

    @property
    def _false_discovery_rates(self):

        rates = np.zeros(self._nb_matches, dtype=np.float)
        for i, j in enumerate(self._indices):
            if j == -1:
                rates[i] = 0.0
            else:
                cell_pred = self._cells_pred[i]
                cell_true = self._cells_true[j]
                train_pred = cell_pred.train
                train_true = cell_true.train
                rates[i] = train_pred.compute_false_discovery_rate(train_true, t_min=self._t_min, t_max=self._t_max)

        return rates

    @property
    def _true_positive_differences(self):

        differences = np.zeros(self._nb_matches, dtype=np.float)
        for i, j in enumerate(self._indices):
            if j == -1:
                differences[i] = 0.0
            else:
                cell_pred = self._cells_pred[i]
                cell_true = self._cells_true[j]
                train_pred = cell_pred.train
                train_true = cell_true.train
                tp_pred = train_pred.compute_true_positive(train_true, t_min=self._t_min, t_max=self._t_max)
                tp_true = train_true.compute_true_positive(train_pred, t_min=self._t_min, t_max=self._t_max)
                differences[i] = tp_pred - tp_true

        return differences

    @property
    def errors(self):

        errors = np.zeros(self._nb_matches, dtype=np.float)
        for i, j in enumerate(self._indices):
            if j == -1:
                errors[i] = +1.0
            else:
                cell_pred = self._cells_pred[i]
                cell_true = self._cells_true[j]
                train_pred = cell_pred.train
                train_true = cell_true.train
                errors[i] = train_pred.compute_difference(train_true, t_min=self._t_min, t_max=self._t_max)

        return errors

    def _update(self):

        if self._path is not None and os.path.isfile(self._path):

            self._load()

        else:

            similarities = self._cells_pred.compute_similarities(self._cells_true)

            self._indices = np.zeros(self._nb_matches, dtype=np.int)
            self._errors = np.zeros(self._nb_matches, dtype=np.float)

            for i, cell in enumerate(self._cells_pred):
                potential_indices = np.where(similarities[i, :] > self._threshold)[0]
                potential_cells = self._cells_true.slice_by_ids(potential_indices)
                nb_potential_cells = len(potential_cells)
                if nb_potential_cells > 0:
                    potential_errors = np.zeros(nb_potential_cells, dtype=np.float)
                    for k, potential_cell in enumerate(potential_cells):
                        if self._t_min is None:
                            t_min = max(cell.t_min, potential_cell.t_min)
                        else:
                            t_min = self._t_min
                        if self._t_max is None:
                            t_max = min(cell.t_max, potential_cell.t_max)
                        else:
                            t_max = self._t_max
                        train = cell.train.slice(t_min, t_max)
                        potential_train = potential_cell.train.slice(t_min, t_max)
                        potential_errors[k] = train.compute_difference(potential_train)
                    index = np.argmin(potential_errors)
                    self._indices[i] = potential_indices[index]
                    self._errors[i] = potential_errors[index]
                else:
                    self._indices[i] = -1
                    self._errors[i] = +1.0

            self._ordered_indices = similarities.ordered_indices

            self._save()

        return

    def plot(self, ax=None, ordering=False, path=None):

        if ax is None:
            fig, ax = plt.subplots(nrows=4)
        else:
            fig = ax.get_figure()

        self.plot_errors(ax=ax[0], ordering=ordering)
        self.plot_false_negative_rates(ax=ax[1], ordering=ordering)
        self.plot_false_discovery_rates(ax=ax[2], ordering=ordering)
        self.plot_true_positive_absolute_differences(ax=ax[3], ordering=ordering)

        fig.tight_layout()

        if path is not None:
            fig.savefig(path)

        return

    def plot_curve(self, ax=None, path=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(ncols=2)
        else:
            fig = ax.get_figure()

        x = 100.0 * self.false_negative_rates
        y = 100.0 * (1.0 - self._false_discovery_rates)

        for k, ax_ in enumerate(ax):
            ax_.scatter(x, y, s=5, **kwargs)

            if k == 0:
                ax_.plot([0.0, 100.0], [0.0, 100.0], color='black', linestyle='--')
                ax_.set_xlim(left=-5.0, right=+105.0)
                ax_.set_ylim(bottom=-5.0, top=+105.0)
                ax_.set_aspect('equal')
            else:
                xlim = ax_.get_xlim()
                ylim = ax_.get_ylim()
                ax_.plot([-5.0, +105.0], [-5.0, +105.0], color='black', linestyle='--')
                ax_.set_xlim(*xlim)
                ax_.set_ylim(*ylim)
                ax_.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

            ax_.set_xlabel("false negative rate (%)")
            ax_.set_ylabel("positive predicted value (%)")
            ax_.set_title("Matches curve")

        fig.tight_layout()

        if path is not None:
            fig.savefig(path)

        return

    def plot_errors(self, ax=None, ordering=False):

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(0, self._nb_matches)
        y = 100.0 * self._errors
        if ordering:
            y = y[self._ordered_indices]

        ax.bar(x, y, width=1.0)

        ax.set_xlim(-0.5, float(self._nb_matches - 1) + 0.5)
        ax.set_ylim(bottom=0.0)

        ax.set_xticks([])

        ax.set_xlabel("match")
        ax.set_ylabel("error (%)")
        ax.set_title("Match errors")

        return

    def plot_false_negative_rates(self, ax=None, ordering=False):

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(0, self._nb_matches)
        y = 100.0 * self.false_negative_rates
        if ordering:
            y = y[self._ordered_indices]

        ax.bar(x, y, width=1.0)

        ax.set_xlim(-0.5, float(self._nb_matches - 1) + 0.5)
        ax.set_ylim(bottom=0.0)

        ax.set_xticks([])

        ax.set_xlabel("match")
        ax.set_ylabel("rate (%)")
        ax.set_title("False negative rates")

        return

    def plot_false_discovery_rates(self, ax=None, ordering=False):

        if ax is None:
            _, ax = plt.subplots()

        x = np.arange(0, self._nb_matches)
        y = 100.0 * self._false_discovery_rates
        if ordering:
            y = y[self._ordered_indices]

        ax.bar(x, y, width=1.0)

        ax.set_xlim(-0.5, float(self._nb_matches - 1) + 0.5)
        ax.set_ylim(bottom=0.0)

        ax.set_xticks([])

        ax.set_xlabel("match")
        ax.set_ylabel("rate (%)")
        ax.set_title("False discovery rates")

        return

    def plot_true_positive_absolute_differences(self, ax=None, ordering=False):

        if ax is None:
            _, ax = plt.subplot()

        x = np.arange(0, self._nb_matches)
        y = np.abs(self._true_positive_differences)
        if ordering:
            y = y[self._ordered_indices]

        ax.bar(x, y, width=1.0)

        ax.set_xlim(-0.5, float(self._nb_matches - 1) + 0.5)
        ax.set_ylim(bottom=0.0)

        ax.set_xticks([])

        ax.set_xlabel("match")
        ax.set_ylabel("abs. diff.")
        ax.set_title("True positive abs. diff.")

        return

    def _save(self):

        if self._path is not None:
            self.save(self._path)

    def save(self, path):

        kwargs = {
            'indices': self._indices,
            'errors': self._errors,
            'order': self._ordered_indices,
        }

        np.savez(path, **kwargs)

        return

    def _load(self):

        with np.load(self._path) as file:
            self._indices = file['indices']
            self._errors = file['errors']
            self._ordered_indices = file['order']

        return
