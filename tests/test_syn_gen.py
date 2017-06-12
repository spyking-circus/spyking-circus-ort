# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import numpy

host = '127.0.0.1' # to run the test locally
data_path_1 = '/tmp/output_1.raw'
data_path_2 = '/tmp/output_2.raw'
hdf5_path = '/tmp/output.hdf5'
plot_path = '/tmp/output.pdf'

director  = circusort.create_director(host=host)
manager   = director.create_manager(host=host, log_level=logging.INFO)

nb_cells = 10
cell_obj = '''
ans = {
    # 'x': (lambda x: (lambda t: x))(xref),
    # 'y': (lambda y: (lambda t: y))(yref),
    # 'z': (lambda z: (lambda t: z))(zref),
    'r': (lambda r, a, d: (lambda t: r + a * np.sin(2.0 * np.pi * float(t) / d)))(rref, a, d),
}
'''
cells_args = [
    {
        'object': cell_obj,
        'globals': {},
        'locals': {
            'xref': +0.0, # reference x-coordinate
            'yref': +0.0, # reference y-coordinate
            'zref': +20.0, # reference z-coordinate
            'rref': +10.0, # reference firing rate (i.e. mean firing rate)
            'a': +8.0, # sinusoidal amplitude for firing rate modification
            'd': +10.0, # number of chunk per period
        },
    }
    for i in range(0, nb_cells)
]

generator = manager.create_block('synthetic_generator', cells_args=cells_args, hdf5_path=hdf5_path, probe='mea_16.prb')
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

x1 = np.memmap(data_path_1, dtype='float32', mode='r')
x1 = np.reshape(x1, (x1.size / nb_channels, nb_channels))
x2 = np.memmap(data_path_2, dtype='float32', mode='r')
x2 = np.reshape(x2, (x2.size / nb_channels, nb_channels))


sr = 20.0e+3 # Hz # sampling_rate
iref = 4 * 2000
imin = iref + 0
imax = iref + 1 * int(sr)
x = np.arange(imin, imax).astype('float32') / sr
shape = (imax - imin, nb_channels)
y1 = np.zeros(shape)
y2 = np.zeros(shape)
for channel in range(0, nb_channels):
    y1[:, channel] = x1[imin:imax, channel].astype('float32')
    y2[:, channel] = x2[imin:imax, channel].astype('float32')
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
plt.xlabel('time (s)')
plt.ylabel('channel')
plt.show()
plt.savefig(plot_path)
