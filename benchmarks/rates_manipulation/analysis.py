import os
# import numpy as np

from circusort.io.spikes import load_spikes, spikes2cells
from circusort.io.cells import load_cells
from circusort.io.template_store import load_template_store
# from circusort.utils.validation import get_fp_fn_rate
from circusort.io.datafile import load_datafile
from circusort.plt.cells import *
from circusort.plt.template import *
from circusort.io.probe import load_probe

data_path = "rates_manipulation"

p = load_probe('/home/pierre/.spyking-circus-ort/benchmarks/%s/generation/probe.prb' %data_path)

generation_directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", data_path)
probe_path = os.path.join(generation_directory, "probe.prb")

similarity_thresh = 0.9

print('Loading data...')
injected_cells = load_cells(os.path.join(generation_directory, 'generation'))
fitted_spikes = load_spikes(os.path.join(os.path.join(generation_directory, 'sorting'), 'spikes.h5'))
found_templates = load_template_store(os.path.join(os.path.join(generation_directory, 'sorting'), 'templates.h5'))
fitted_cells = spikes2cells(fitted_spikes, found_templates)
filename = os.path.join(os.path.join(generation_directory, 'generation'), 'data.raw')
data_file = load_datafile(filename, 20000, p.nb_channels, 'int16', 0.1042)


print('Computing similarities...')
similarities = injected_cells.compute_similarities(fitted_cells)


print('Computing matches...')
matches = injected_cells.compute_matches(fitted_cells)

# injected_cells[0].template.plot(probe=p)
# found_templates[matches[0][0]].plot(probe=p)
