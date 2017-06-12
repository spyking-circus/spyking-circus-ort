# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import numpy

## In this example, we have 20 fixed neurons. 10 are active during the first 30s of the experiment, and then 10 new ones
## are appearing after 30s. The goal here is to study how the clsutering can handle such an discountinuous change


host = '127.0.0.1' # to run the test locally
data_path  = '/tmp/output.raw'
hdf5_path  = '/tmp/output.hdf5'
probe_file = 'mea_4.prb'


director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)

sampling_rate  = 20000
two_components = False
nb_channels    = 4

nb_cells = 10
cell_obj = '''
ans = {
    'r': (lambda r, a, d: (lambda t: (t > 500) * (r + a * np.sin(2.0 * np.pi * float(t) / d))))(r_ref, a, d),
}
'''
cells_args = [
    {
        'object': cell_obj,
        'globals': {},
        'locals': {
            'r_ref': 10.0, # reference firing rate (i.e. mean firing rate)
            'a': 8.0, # sinusoidal amplitude for firing rate modification
            'd': 10.0, # number of chunk per period
        },
    }
    for i in range(0, nb_cells)
]

cell_obj = '''
ans = {
    'r': (lambda r, a, d: (lambda t: (t < 500) * (r + a * np.sin(2.0 * np.pi * float(t) / d))))(r_ref, a, d),
}
'''

for i in xrange(nb_cells):
    cells_args += [
        {
            'object': cell_obj,
            'globals': {},
            'locals': {
                'r_ref': 10.0, # reference firing rate (i.e. mean firing rate)
                'a': 8.0, # sinusoidal amplitude for firing rate modification
                'd': 10.0, # number of chunk per period
            },
        }
    ]

generator     = manager.create_block('synthetic_generator', cells_args=cells_args, hdf5_path=hdf5_path, probe=probe_file)
filter        = manager.create_block('filter', cut_off=100)
whitening     = manager.create_block('whitening')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector', threshold=6)
pca           = manager.create_block('pca', nb_waveforms=100)
cluster       = manager.create_block('density_clustering', probe=probe_file, nb_waveforms=100, log_level=logging.DEBUG, two_components=two_components)
updater       = manager.create_block('template_updater', probe=probe_file, data_path='templates', nb_channels=nb_channels, log_level=logging.DEBUG)
fitter        = manager.create_block('template_fitter', log_level=logging.INFO, two_components=two_components)
writer        = manager.create_block('writer', data_path='/tmp/output.dat')
writer_2      = manager.create_block('spike_writer')
writer_3      = manager.create_block('peak_writer', neg_peaks='/tmp/peaks.dat')

director.initialize()

director.connect(generator.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster.get_input('mads')])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster.get_input('peaks'), fitter.get_input('peaks'), writer_3.input])
director.connect(pca.get_output('pcs'), cluster.get_input('pcs'))
director.connect(cluster.get_output('templates'), updater.get_input('templates'))
director.connect(updater.get_output('updater'), fitter.get_input('updater'))
director.connect(fitter.output, writer_2.input)

director.start()
director.sleep(duration=60.0)
director.stop()
