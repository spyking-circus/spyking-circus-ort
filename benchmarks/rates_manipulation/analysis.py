import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import circusort


from circusort.io.spikes import load_spikes
from circusort.io.cells import load_cells
from circusort.io.template_store import load_template_store
from circusort.utils.train import compute_train_similarity, compute_pic_strength
from circusort.utils.validation import get_fp_fn_rate


generation_directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "rates_manipulation")
probe_path = os.path.join(generation_directory, "probe.prb")

similarity_thresh = 0.9

print('Loading data...')
fitted_spikes   = load_spikes(os.path.join(os.path.join(generation_directory, 'sorting'), 'spikes.h5')).to_cells()
found_templates = load_template_store(os.path.join(os.path.join(generation_directory, 'sorting'), 'templates.h5'))
injected_cells  = load_cells(os.path.join(generation_directory, 'generation'))

print fitted_spikes.t_max

injected_cells.set_t_min(fitted_spikes.t_min)
injected_cells.set_t_max(fitted_spikes.t_max)

print('Computing similarities...')
similarities = [[] for i in range(len(injected_cells))]

for count, cell in enumerate(injected_cells):
    template = cell.template.to_template(100)
    for t in found_templates:
        similarities[count] += [t.similarity(template)]

similarities = np.array(similarities)


print('Computing matches...')
matches = [[] for i in range(len(injected_cells))]
errors = [[] for i in range(len(injected_cells))]

for count, cell in enumerate(injected_cells):
    matches[count] = np.where(similarities[count] > similarity_thresh)[0]
    
    sink_cells = fitted_spikes.slice(matches[count])
    if len(sink_cells) > 0:
        errors[count] += np.mean(get_fp_fn_rate(sink_cells, cell, 2e-3), 1).tolist()
    else:
        errors[count] += []

res = []

for count, e in enumerate(errors):
    if len(e) > 0:
        idx  = matches[count][np.argmin(e)]
        emin = e[np.argmin(e)]
    else:
        idx = -1
        emin = -1
    res += [[idx, emin]]