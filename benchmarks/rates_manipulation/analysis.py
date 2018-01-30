import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "rates_manipulation")
from circusort.io.spikes import load_spikes
from circusort.io.cells import load_cells
from circusort.io.template_store import load_template_store
from circusort.utils.train import compute_train_similarity, compute_pic_strength


fitted_spikes  = load_spikes(os.path.join(os.path.join(directory, 'sorting'), 'spikes.h5'))
found_templates = load_template_store(os.path.join(os.path.join(directory, 'sorting'), 'templates.h5'))
injected_cells = load_cells(os.path.join(directory, 'generation'))

similarities = [[] for i in xrange(len(injected_cells.cells))]

for count, cell in enumerate(injected_cells):
    template = cell.template.to_template(100)
    for t in found_templates:
        similarities[count] += [t.similarity(template)]

similarities = np.array(similarities)
similarity_thresh = 0.9


matches = [[] for i in xrange(len(injected_cells.cells))]
strengths = [[] for i in xrange(len(injected_cells.cells))]

for count, cell in enumerate(injected_cells):
    matches[count] = np.where(similarities[count] > d_thresh)[0]
    for item in matches[count]:
        sink_cell = fitted_spikes.get_unit(item)
        sink_cell.train.t_min = cell.train.t_min
        sink_cell.train.t_max = cell.train.t_max
        strengths[count] += [compute_pic_strength(cell.train, sink_cell.train)]
