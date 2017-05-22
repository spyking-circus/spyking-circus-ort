# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging


host = '127.0.0.1' # to run the test locally

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host, log_level=logging.INFO)

nb_channels = 10

noise    = manager.create_block('fake_spike_generator', nb_channels=nb_channels)
selector = manager.create_block('channel_selector')
filter   = manager.create_block('filter', cut_off=100)
writer_1 = manager.create_block('writer', data_path='/tmp/input.dat')
writer_2 = manager.create_block('writer', data_path='/tmp/output.dat')

manager.initialize()

manager.connect(noise.output, [writer_1.input, selector.input])
manager.connect(selector.output, filter.input)
manager.connect(filter.output, writer_2.input)

manager.start()

director.sleep(duration=2.0)
director.stop()

import pylab, numpy

x1 = numpy.memmap('/tmp/input.dat', dtype=numpy.float32, mode='r')
x1 = x1.reshape(x1.size/nb_channels, nb_channels)
x2 = numpy.memmap('/tmp/output.dat', dtype=numpy.float32, mode='r')
x2 = x2.reshape(x2.size/nb_channels, nb_channels)

pylab.plot(x1[:, 0])
pylab.plot(x2[:, 0])
pylab.show()