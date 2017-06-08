# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging


host = '127.0.0.1' # to run the test locally

nb_channels = 10

director = circusort.create_director(host=host, log_level=logging.INFO)
manager = director.create_manager(host=host, log_level=logging.INFO)

reader = manager.create_block('reader', data_path='/tmp/input.dat', nb_channels=nb_channels)
writer = manager.create_block('writer', data_path='/tmp/output.dat')

manager.initialize()

director.connect(reader.output, writer.input)

manager.start()  
director.sleep(duration=1)

import pylab, numpy

x1 = numpy.memmap('/tmp/input.dat', dtype=numpy.float32, mode='r')
x1 = x1.reshape(x1.size/nb_channels, nb_channels)
x2 = numpy.memmap('/tmp/output.dat', dtype=numpy.float32, mode='r')
x2 = x2.reshape(x2.size/nb_channels, nb_channels)

pylab.subplot(211)
pylab.plot(x1[:, 0])
pylab.subplot(212)
pylab.plot(x2[:, 0])
pylab.show()