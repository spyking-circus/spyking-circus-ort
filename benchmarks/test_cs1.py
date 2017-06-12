# Test to measure the computational efficiency of two operations in one block
# associated to one manager.

import circusort
import logging
import numpy

host = '127.0.0.1' # to run the test locally
data_path = '/tmp/output.raw'
hdf5_path = '/tmp/output.hdf5'

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

generator = manager.create_block('synthetic_generator', cells_args=cells_args, hdf5_path=hdf5_path)
filter    = manager.create_block('filter', cut_off=300)
writer_1  = manager.create_block('writer', data_path=data_path)

manager.initialize()

manager.connect(generator.output, filter.input)
manager.connect(filter.output, writer_1.input)

manager.start()

director.sleep(duration=2.0)
director.stop()
