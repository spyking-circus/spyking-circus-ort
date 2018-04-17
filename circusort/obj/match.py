class Match(object):
    # TODO add docstring.

    def __init__(self, cell_pred, cell_true, t_min=None, t_max=None):

        self._cell_pred = cell_pred
        self._cell_true = cell_true
        self._t_min = t_min
        self._t_max = t_max

    def collect_false_positives(self):
        # TODO add docstring.

        train_pred = self._cell_pred.train
        train_true = self._cell_true.train

        train_fp = train_pred.collect_false_positives(train_true)

        return train_fp

    def collect_false_negatives(self):
        # TODO add docstring.

        train_pred = self._cell_pred.train
        train_true = self._cell_true.train

        train_fn = train_pred.collect_false_negatives(train_true)

        return train_fn
