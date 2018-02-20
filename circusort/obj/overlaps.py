# -*- coding: utf-8 -*-
import numpy as np
import h5py
import scipy.sparse


class Overlaps(object):

    def __init__(self, template, target, _scols, size, temporal_width):

        self._scols = _scols
        self.size = size
        self.temporal_width = temporal_width
        self.new_indices = []
        self.overlaps = self._get_overlaps(template, target)

    @property
    def do_update(self):

        return len(self.new_indices) > 0

    def __len__(self):

        return self.overlaps.shape[0]

    def _get_overlaps(self, template, target):

        all_x = np.zeros(0, dtype=np.int32)
        all_y = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        for idelay in self._scols['delays']:
            tmp_1 = template[:, self._scols['left'][idelay]]
            tmp_2 = target[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            dx, dy = data.nonzero()
            ones = np.ones(len(dx), dtype=np.int32)
            all_x = np.concatenate((all_x, dx * target.shape[0] + dy))
            all_y = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self.temporal_width:
                tmp_1 = template[:, self._scols['right'][idelay]]
                tmp_2 = target[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                dx, dy = data.nonzero()
                ones = np.ones(len(dx), dtype=np.int32)
                all_x = np.concatenate((all_x, dx * target.shape[0] + dy))
                all_y = np.concatenate((all_y, (self.size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        shape = (target.shape[0] * template.shape[0], self.size)

        return scipy.sparse.csr_matrix((all_data, (all_x, all_y)), shape=shape)

    def update(self, template, target):

        new_overlaps = self._get_overlaps(template, target[self.new_indices])
        self.overlaps = scipy.sparse.vstack((self.overlaps, new_overlaps))
        self.new_indices = []

    def save(self, path):

        with h5py.File(path, mode='w') as file_:

            file_.create_dataset('data', data=self.overlaps.data, chunks=True)
            file_.create_dataset('indices', data=self.overlaps.indices, chunks=True)
            file_.create_dataset('indptr', data=self.overlaps.indptr, chunks=True)
            file_.attrs['new_indices'] = self.new_indices
            file_.attrs['temporal_width'] = self.temporal_width
            file_.attrs['size'] = self.size

        return