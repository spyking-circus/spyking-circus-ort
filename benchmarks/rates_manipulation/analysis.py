import os
# import numpy as np

from circusort.io.spikes import load_spikes, spikes2cells
from circusort.io.cells import load_cells
from circusort.io.template_store import load_template_store
# from circusort.utils.validation import get_fp_fn_rate
from circusort.io.datafile import load_datafile
# from circusort.plt.cells import *
from circusort.io.probe import load_probe

p=load_probe('/home/pierre/.spyking-circus-ort/benchmarks/rates_manipulation_2/generation/probe.prb')

generation_directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "rates_manipulation_2")
probe_path = os.path.join(generation_directory, "probe.prb")

similarity_thresh = 0.9

print('Loading data...')
injected_cells = load_cells(os.path.join(generation_directory, 'generation'))
fitted_spikes = load_spikes(os.path.join(os.path.join(generation_directory, 'sorting'), 'spikes.h5'))
found_templates = load_template_store(os.path.join(os.path.join(generation_directory, 'sorting'), 'templates.h5'))
fitted_cells = spikes2cells(fitted_spikes, found_templates)
filename = os.path.join(os.path.join(generation_directory, 'generation'), 'data.raw')
data_file = load_datafile(filename, 20000, 100, 'int16', 0.1042)


print('Computing similarities...')

# similarities = [[] for i in range(len(injected_cells))]
#
# for count, cell in enumerate(injected_cells):
#     similarities[count] = cell.template.similarity(found_templates.get())
#
# similarities = np.array(similarities)

similarities = injected_cells.compute_similarities(fitted_cells)


print('Computing matches...')

# matches = [[] for i in range(len(injected_cells))]
# errors = [[] for i in range(len(injected_cells))]
#
# for count, cell in enumerate(injected_cells):
#     matches[count] = np.where(similarities[count] > similarity_thresh)[0]
#
#     sink_cells = fitted_cells.slice_by_ids(matches[count])
#
#     mytrain = cell.train.slice(sink_cells.t_min, sink_cells.t_max)
#
#     print "Computing errors for cell %d in [%g,%g] with %d spikes" %(count, mytrain.t_min, mytrain.t_max, len(mytrain))
#
#     if len(sink_cells) > 0:
#         errors[count] += [np.mean(get_fp_fn_rate([i.train for i in sink_cells], mytrain, 2e-3))]
#     else:
#         errors[count] += []
#
# res = []
#
# for count, e in enumerate(errors):
#     if len(e) > 0:
#         idx  = matches[count][np.argmin(e)]
#         emin = e[np.argmin(e)]
#     else:
#         idx = -1
#         emin = -1
#     res += [[idx, emin]]

matches = injected_cells.compute_matches(fitted_cells)

injected_cells[0].template.plot(probe=p)
found_templates[matches[0][0]].plot(probe=p)
