# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.obj.cells import Cells

def plot_reconstruction(cells, t_min, t_max, sampling_rate, data_file=None):

    sampling_rate = float(sampling_rate)
    gmin = int(t_min * sampling_rate)
    gmax = int(t_max * sampling_rate)

    nb_channels = cells[cells.ids[0]].template.first_component.nb_channels

    result = np.zeros((gmax - gmin, nb_channels), dtype='float32')
    for c in cells:
        width = c.template.temporal_width
        half_width = width // 2
        sub_train = c.slice(t_min + half_width/sampling_rate, t_max - half_width/sampling_rate)
        t1 = sub_train.template.first_component.to_dense().T

        if sub_train.template.two_components:
           t2 = c.template.two_components.to_dense().T

        for spike, amp in zip(sub_train.train, sub_train.amplitude):
            offset = int(spike*sampling_rate) - gmin

            if c.template.two_components:
                result[int(offset - half_width):int(offset + half_width + 1), :] += amp[0] * t1 + amp[1] * t2
            else:
                result[int(offset - half_width):int(offset + half_width + 1), :] += amp * t1

    if data_file is not None:
        snippet = data_file.get_snippet(t_min, t_max)

    return result, snippet

