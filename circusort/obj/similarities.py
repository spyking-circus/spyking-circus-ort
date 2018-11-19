from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

import matplotlib.pyplot as plt
import numpy as np


class Similarities(object):

    def __init__(self, cells_pred, cells_true):

        self._cells_pred = cells_pred
        self._cells_true = cells_true

        self._cmap = 'RdBu'

        self._similarities = None
        self._ordered_indices = None

        self._update()

    def __getitem__(self, i):

        return self._similarities.__getitem__(i)

    def __setitem__(self, i, v):

        return self._similarities.__setitem__(i, v)

    def _update(self):

        nb_cells_pred = self._cells_pred.nb_cells
        nb_cells_true = self._cells_true.nb_cells
        shape = (nb_cells_pred, nb_cells_true)

        self._similarities = np.zeros(shape, dtype=np.float)

        for i, cell_pred in enumerate(self._cells_pred):
            template_pred = cell_pred.template
            for j, cell_true in enumerate(self._cells_true):
                template_true = cell_true.template
                self._similarities[i, j] = template_pred.similarity(template_true)

        return

    @property
    def ordered_indices(self):

        # WARNING: this function is useful only when the detected templates are strictly equal to the generated
        # templates, otherwise it does not make any sense to run a hierarchical clustering on the similarity matrix.
        assert self._cells_true.nb_cells == self._cells_pred.nb_cells
        nb_cells = self._cells_true.nb_cells

        if self._ordered_indices is None:

            if nb_cells > 1:
                metric = 'correlation'
                # Define the distance matrix.
                distances = pdist(self._similarities, metric=metric)
                # Perform hierarchical/agglomerative clustering.
                linkages = linkage(distances, method='single', metric=metric)
                # Reorder templates.
                linkages_ordered = optimal_leaf_ordering(linkages, distances, metric=metric)
                # Extract ordered list.
                self._ordered_indices = leaves_list(linkages_ordered)
            else:
                self._ordered_indices = np.arange(0, nb_cells)

        return self._ordered_indices

    def plot(self, ax=None, ordering=False, path=None):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if ordering:
            x = np.copy(self._similarities)
            x = x[self.ordered_indices, :]
            x = x[:, self.ordered_indices]
        else:
            x = self._similarities

        im = ax.imshow(x, vmin=-1.0, vmax=+1.0, cmap=self._cmap)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.05)
        bar = fig.colorbar(im, ticks=[-1.0, 0.0, +1.0], cax=cax)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel("injected templates")
        ax.set_ylabel("detected templates")
        ax.set_title("Template similarities")

        bar.set_label("correlation")
        bar.ax.set_yticklabels(["$-1$", "$0$", "$+1$"])

        fig.tight_layout()

        if path is not None:
            fig.savefig(path)

        return
