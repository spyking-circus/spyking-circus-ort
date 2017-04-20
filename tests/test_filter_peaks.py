# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)

nb_channels = 10
nb_samples  = 1024

noise    = manager.create_block('fake_spike_generator', nb_channels=nb_channels, nb_samples=nb_samples)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter', cut_off=100)
writer_1 = manager.create_block('writer', data_path='/tmp/input.dat')
writer_2 = manager.create_block('writer', data_path='/tmp/mads.dat')
writer_3 = manager.create_block('peak_writer')
mad_estimator = manager.create_block('mad_estimator', threshold=6)
peak_detector = manager.create_block('peak_detector', sign_peaks='both')

manager.initialize()

manager.connect(noise.output, selector.input)
manager.connect(selector.output, filter.input)
manager.connect(filter.output, [mad_estimator.input, writer_1.input, peak_detector.get_input('data')])
manager.connect(mad_estimator.get_output('mads'), [peak_detector.get_input('mads'), writer_2.input])
manager.connect(peak_detector.get_output('peaks'), writer_3.input)
manager.start()
director.sleep(duration=5.0)

director.stop()

start_mad     = mad_estimator.start_step
neg_peak_file = writer_3.recorded_peaks['negative']
pos_peak_file = writer_3.recorded_peaks['positive']

import pylab, numpy

x1 = numpy.memmap('/tmp/input.dat', dtype=numpy.float32, mode='r')
x1 = x1.reshape(x1.size/nb_channels, nb_channels)

neg_peaks = numpy.fromfile(neg_peak_file, dtype=numpy.int32)
neg_peaks = neg_peaks.reshape(neg_peaks.size/2, 2)

pos_peaks = numpy.fromfile(pos_peak_file, dtype=numpy.int32)
pos_peaks = pos_peaks.reshape(pos_peaks.size/2, 2)

mads      = numpy.fromfile('/tmp/mads.dat', dtype=numpy.float32)
t_max     = mads.size/nb_channels
mads      = mads[:t_max*nb_channels].reshape(t_max, nb_channels)

channel_to_show = 0

t_stop    = (start_mad+10)*nb_samples
t_start   = (start_mad-1)*nb_samples
max_offset = x1[t_start:t_stop, channel_to_show].max()
min_offset = x1[t_start:t_stop, channel_to_show].min()

pylab.plot(numpy.arange(t_start, t_stop), x1[t_start:t_stop, channel_to_show])

idx = numpy.where((neg_peaks[:,1] < t_stop) & (neg_peaks[:,1] >= t_start) & (neg_peaks[:,0] == channel_to_show))
sub_peaks = neg_peaks[idx]
pylab.scatter(sub_peaks[:, 1], min_offset + sub_peaks[:, 0], c='r')

idx = numpy.where((pos_peaks[:,1] < t_stop) & (pos_peaks[:,1] >= t_start) & (pos_peaks[:,0] == channel_to_show))
sub_peaks = pos_peaks[idx]
pylab.scatter(sub_peaks[:, 1], max_offset + sub_peaks[:, 0], c='r')

res = numpy.zeros(0, dtype=numpy.float32)
for count, i in enumerate(xrange(start_mad-1, start_mad+10)):
    res = numpy.concatenate((res, mads[count, channel_to_show]*numpy.ones(nb_samples)))

pylab.plot(numpy.arange(t_start, t_stop), res, 'k--')
pylab.plot(numpy.arange(t_start, t_stop), -res, 'k--')

pylab.show()


