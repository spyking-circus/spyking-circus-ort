# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging

host = '127.0.0.1' # to run the test locally
data_path_1 = '/tmp/output_1.raw'
data_path_2 = '/tmp/output_2.raw'

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host, log_level=logging.INFO)

generator = manager.create_block('synthetic_generator')
filter    = manager.create_block('filter', cut_off=100)
writer_1  = manager.create_block('writer', data_path=data_path_1)
writer_2  = manager.create_block('writer', data_path=data_path_2)

manager.initialize()

manager.connect(generator.output, [writer_1.input, filter.input])
manager.connect(filter.output, writer_2.input)

manager.start()

director.sleep(duration=2.0)
director.stop()



import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

nb_channels = 16

x1 = np.memmap(data_path_1, dtype='float', mode='r')
x1 = np.reshape(x1, (x1.size / nb_channels, nb_channels))
x2 = np.memmap(data_path_2, dtype='float', mode='r')
x2 = np.reshape(x2, (x2.size / nb_channels, nb_channels))


iref = 4 * 2000
imin = iref + 0
imax = iref + 20000
x = np.arange(imin, imax)
shape = (imax - imin, nb_channels)
y1 = np.zeros(shape)
y2 = np.zeros(shape)
for channel in range(0, nb_channels):
    y1[:, channel] = x1[imin:imax, channel].astype('float')
    y2[:, channel] = x2[imin:imax, channel].astype('float')
ymin = min([np.amin(y) for y in [y1, y2]])
ymax = max([np.amax(y) for y in [y1, y2]])
for channel in range(0, nb_channels):
    z1 = y1[:, channel] - ymin
    z2 = y2[:, channel] - ymin
    if ymin < ymax:
        z1 = z1 / (ymax - ymin) / 2.0
        z2 = z2 / (ymax - ymin) / 2.0
    offset = float(channel)
    plt.plot(x, z1 + offset + 0.5, color='C0')
    plt.plot(x, z2 + offset + 0.0, color='C1')
p1 = mpl.patches.Patch(color='C0', label='raw')
p2 = mpl.patches.Patch(color='C1', label='filtered')
plt.legend(handles=[p1, p2])
plt.show()
