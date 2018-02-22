# -*- coding: utf-8 -*-
import numpy as np
import h5py
import scipy.sparse


class Overlaps(object):

    def __init__(self, _scols, size, temporal_width):

        self._scols = _scols
        self.size = size
        self.temporal_width = temporal_width
        self.indices_ = []
        self.overlaps = None

    @property
    def do_update(self):

        return len(self.indices_) > 0

    def __len__(self):

        return self.overlaps.shape[0]

    def _get_overlaps(self, template, target, non_zeros=None):

        all_x = np.zeros(0, dtype=np.int32)
        all_y = np.zeros(0, dtype=np.int32)
        all_data = np.zeros(0, dtype=np.float32)

        if non_zeros is not None:
            sub_target = target[non_zeros]
        else:
            sub_target = target

        for idelay in self._scols['delays']:
            tmp_1 = template[:, self._scols['left'][idelay]]
            tmp_2 = sub_target[:, self._scols['right'][idelay]]
            data = tmp_1.dot(tmp_2.T)
            dx, dy = data.nonzero()

            ones = np.ones(len(dx), dtype=np.int32)
            all_x = np.concatenate((all_x, dx * sub_target.shape[0] + dy))
            all_y = np.concatenate((all_y, (idelay - 1) * ones))
            all_data = np.concatenate((all_data, data.data))

            if idelay < self.temporal_width:
                tmp_1 = template[:, self._scols['right'][idelay]]
                tmp_2 = sub_target[:, self._scols['left'][idelay]]
                data = tmp_1.dot(tmp_2.T)
                dx, dy = data.nonzero()
                ones = np.ones(len(dx), dtype=np.int32)
                all_x = np.concatenate((all_x, dx * sub_target.shape[0] + dy))
                all_y = np.concatenate((all_y, (self.size - idelay) * ones))
                all_data = np.concatenate((all_data, data.data))

        if non_zeros is None:
            shape = (target.shape[0] * template.shape[0], self.size)
            return scipy.sparse.csr_matrix((all_data, (all_x, all_y)), shape=shape)
        else:
            shape_1 = (sub_target.shape[0] * template.shape[0], self.size)
            res_1 = scipy.sparse.csr_matrix((all_data, (all_x, all_y)), shape=shape_1)

            shape_2 = (target.shape[0] * template.shape[0], self.size)
            res_2 = scipy.sparse.csr_matrix(shape_2, dtype=np.float32)
            res_2[non_zeros] = res_1
            return res_2

    def update(self, template, target, non_zeros=None):

        if non_zeros is not None:
            indices = np.in1d(self.indices_, non_zeros)
            indices = np.where(indices == True)[0]
            if len(indices) > 0:
                non_zeros = np.arange(len(self.indices_))[indices]
            else:
                non_zeros = None
        
        new_overlaps = self._get_overlaps(template, target[self.indices_], non_zeros)
        self.overlaps = scipy.sparse.vstack((self.overlaps, new_overlaps))
        self.indices_ = []

    def initialize(self, template, target, non_zeros=None):

        self.overlaps = self._get_overlaps(template, target, non_zeros)

    def save(self, path):

        with h5py.File(path, mode='w') as file_:

            file_.create_dataset('data', data=self.overlaps.data, chunks=True)
            file_.create_dataset('indices', data=self.overlaps.indices, chunks=True)
            file_.create_dataset('indptr', data=self.overlaps.indptr, chunks=True)
            file_.attrs['indices'] = self.indices_
            file_.attrs['temporal_width'] = self.temporal_width
            file_.attrs['size'] = self.size

        return