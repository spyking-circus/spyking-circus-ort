# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import settings
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host)

nb_channels = 10

noise    = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter', cut_off=100)
writer_1 = manager.create_block('writer', data_path='/tmp/input.dat')
writer_2 = manager.create_block('writer', data_path='/tmp/mads.dat')
#writer_3 = manager.create_block('writer', data_path='/tmp/peaks.dat')
mad_estimator = manager.create_block('mad_estimator')
peak_detector = manager.create_block('peak_detector')

manager.initialize()

manager.connect(noise.output, selector.input)
manager.connect(selector.output, filter.input)
manager.connect(filter.output, [mad_estimator.input, writer_1.input, peak_detector.get_input('data')])
manager.connect(mad_estimator.get_output('mads'), [peak_detector.get_input('mads'), writer_2.input])

manager.start()
director.sleep(duration=2.0)

director.stop()

import pylab, numpy

x1 = numpy.memmap('/tmp/input.dat', dtype=numpy.float32)
x1 = x1.reshape(x1.size/nb_channels, nb_channels)
x2 = numpy.memmap('/tmp/mads.dat', dtype=numpy.float32)

pylab.plot(x1[:, 0])
pylab.plot(x2)
pylab.show()
