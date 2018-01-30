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

similarity_thresh = 0.8

print('Loading data...')
fitted_spikes   = load_spikes(os.path.join(os.path.join(generation_directory, 'sorting'), 'spikes.h5')).to_cells()
found_templates = load_template_store(os.path.join(os.path.join(generation_directory, 'sorting'), 'templates.h5'))
injected_cells  = load_cells(os.path.join(generation_directory, 'generation'))

print fitted_spikes.t_max

injected_cells.set_t_min(0)
injected_cells.set_t_max(fitted_spikes.t_max)

print('Computing similarities...')
similarities = [[] for i in range(len(found_templates))]

for count, t in enumerate(found_templates):
    for cell in injected_cells:
        template = cell.template.to_template(100)
        similarities[count] += [t.similarity(template)]

similarities = np.array(similarities)


print('Computing matches...')
matches = [[] for i in range(len(found_templates))]
errors = [[] for i in range(len(found_templates))]

for count, cell in enumerate(fitted_spikes):
    matches[count] = np.where(similarities[count] > similarity_thresh)[0]
    
    sink_cells = injected_cells.slice(matches[count])
    errors[count] += np.mean(get_fp_fn_rate(sink_cells, cell, 2e-3), 1).tolist()

