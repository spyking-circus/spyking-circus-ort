# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import argparse
import circusort
import logging
import numpy
import sys
from circusort.io.utils import generate_fake_probe

## In this example, we have 20 fixed neurons. 10 are active during the first 30s of the experiment, and then 10 new ones
## are appearing after 30s. The goal here is to study how the clustering can handle such an discountinuous change


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='local', choices=['local', 'remote'], help="distributed computation or not")
args = parser.parse_args()


if args.mode == 'local':
    master = '127.0.0.1'
    slaves = ['127.0.0.1', '127.0.0.1', '127.0.0.1']
elif args.mode == 'remote':
    master = '192.168.0.254'
    slaves = ['192.168.0.1', '192.168.0.2', '192.168.0.3']

data_path  = '/tmp/output.dat'
peak_path  = '/tmp/peaks.dat'
thres_path = '/tmp/thresholds.dat'
temp_path  = '/tmp/templates'

director  = circusort.create_director(host=master)
manager   = {}

for computer in slaves + [master]:
    manager[computer] = director.create_manager(host=computer)

sampling_rate  = 20000
two_components = True
nb_channels    = 4


probe_file    = generate_fake_probe(nb_channels, radius=2, prb_file='test.prb')

generator     = manager[master].create_block('fake_spike_generator', nb_channels=nb_channels)
filter        = manager[master].create_block('filter', cut_off=100)
whitening     = manager[master].create_block('whitening')
mad_estimator = manager[master].create_block('mad_estimator')
peak_detector = manager[master].create_block('peak_detector', threshold=5)
pca           = manager[master].create_block('pca', nb_waveforms=500)

cluster_1     = manager[slaves[0]].create_block('density_clustering', probe=probe_file, nb_waveforms=100, two_components=two_components, channels=range(0, nb_channels, 2))
cluster_2     = manager[slaves[1]].create_block('density_clustering', probe=probe_file, nb_waveforms=100, two_components=two_components, channels=range(1, nb_channels, 2))

updater       = manager[slaves[-1]].create_block('template_updater', probe=probe_file, data_path=temp_path, nb_channels=nb_channels)
fitter        = manager[slaves[-1]].create_block('template_fitter', log_level=logging.INFO, two_components=two_components)

writer        = manager[master].create_block('writer', data_path=data_path)
writer_2      = manager[master].create_block('spike_writer')
writer_3      = manager[master].create_block('peak_writer', neg_peaks=peak_path)
writer_4      = manager[master].create_block('writer', data_path=thres_path)

director.initialize()

director.connect(generator.output, filter.input)
director.connect(filter.output, whitening.input)
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), cluster_1.get_input('data'), cluster_2.get_input('data'), pca.get_input('data'), fitter.get_input('data'), writer.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), cluster_1.get_input('mads'), cluster_2.get_input('mads'), writer_4.input])
director.connect(peak_detector.get_output('peaks'), [pca.get_input('peaks'), cluster_1.get_input('peaks'), cluster_2.get_input('peaks'), fitter.get_input('peaks'), writer_3.input])
director.connect(pca.get_output('pcs'), [cluster_1.get_input('pcs'), cluster_2.get_input('pcs')])
director.connect(cluster_1.get_output('templates'), updater.get_input('templates'))
director.connect(cluster_2.get_output('templates'), updater.get_input('templates'))
director.connect(updater.get_output('updater'), fitter.get_input('updater'))
director.connect(fitter.output, writer_2.input)

director.start()
director.sleep(duration=100.0)
director.stop()

start_time = mad_estimator.start_step
stop_time  = fitter.counter

from utils.analyzer import Analyzer
r = Analyzer(writer_2.recorded_data, probe_file, temp_path, filtered_data=data_path, threshold_data=thres_path, start_time=start_time, stop_time=stop_time)
