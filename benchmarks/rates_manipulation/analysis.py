import os
import numpy as np

from circusort.io.spikes import load_spikes, spikes2cells
from circusort.io.cells import load_cells
from circusort.io.template_store import load_template_store
from circusort.utils.validation import get_fp_fn_rate
from circusort.io.datafile import load_datafile
from circusort.plt.cells import *

generation_directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "rates_manipulation")
probe_path = os.path.join(generation_directory, "probe.prb")

similarity_thresh = 0.8

print('Loading data...')
fitted_spikes = load_spikes(os.path.join(os.path.join(generation_directory, 'sorting'), 'spikes.h5'))
found_templates = load_template_store(os.path.join(os.path.join(generation_directory, 'sorting'), 'templates.h5'))
fitted_cells = spikes2cells(fitted_spikes, found_templates)
injected_cells = load_cells(os.path.join(generation_directory, 'generation'))


filename = os.path.join(os.path.join(generation_directory, 'generation'), 'data.raw')

data_file = load_datafile(filename, 20000, 100, 'int16')


print('Computing similarities...')
similarities = [[] for i in range(len(injected_cells))]

for count, cell in enumerate(injected_cells):
    template = cell.template
    for t in found_templates:
        similarities[count] += [t.similarity(template)]

similarities = np.array(similarities)


print('Computing matches...')
matches = [[] for i in range(len(injected_cells))]
errors = [[] for i in range(len(injected_cells))]

for count, cell in enumerate(injected_cells):
    matches[count] = np.where(similarities[count] > similarity_thresh)[0]

    sink_cells = fitted_cells.slice_by_ids(matches[count])

    gtmin, gtmax = np.inf, 0
    for c in sink_cells:
        if c.train.times.min() < gtmin:
            gtmin = c.train.times.min()
        if c.train.times.max() > gtmax:
            gtmax = c.train.times.max()

    mytrain = cell.train#
    print "Computing errors for cell %d in [%g,%g] with %d spikes" %(count, gtmin, gtmax, len(mytrain))

    if len(sink_cells) > 0:
        errors[count] += np.mean(get_fp_fn_rate([i.train for i in sink_cells], mytrain, 2e-3), 1).tolist()
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

from circusort.io.probe import load_probe
p=load_probe('/home/pierre/.spyking-circus-ort/benchmarks/rates_manipulation/generation/probe.prb')

injected_cells[0].template.plot(probe=p)
found_templates[res[0][0]].plot(probe=p)