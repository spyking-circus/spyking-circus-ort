import h5py
import numpy as np
import Queue
import scipy as sp
import scipy.signal
import threading
import time
import tempfile
import os

from circusort.block import block
from circusort import io
from circusort import utils



# TODO find if the communication mechanism between the main and background
# threads is necessary, i.e. is there another canonical way to stop the
# background thread (in a infinite loop) from the main thread?

class Synthetic_generator(block.Block):
    '''TODO add docstring'''

    name = "Synthetic Generator"

    params = {
        'dtype'         : 'float',
        'probe_filename': '~/spyking-circus/probes/mea_16.prb',
        'sampling_rate' : 20000.0,
        'nb_samples'    : 2000,
        'nb_cells'      : 10,
        'hdf5_path'     : None,
    }

    def __init__(self, cells_args=None, **kwargs):

        block.Block.__init__(self, **kwargs)
        self.cells_args = cells_args
        if self.cells_args is not None:
            self.nb_cells = len(self.cells_args)
        self.add_output('data')

    def _get_tmp_path(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name))
        tmp_file.close()
        return data_path

    def _initialize(self):
        '''TODO add docstring.'''

        # Retrieve the geometry of the probe.
        self.probe = io.Probe(self.probe_filename)
        self.nb_channels = self.probe.nb_channels
        self.fov = self.probe.field_of_view

        # Generate synthetic cells.
        self.cells = {}
        for c in range(0, self.nb_cells):
            x_ref = np.random.uniform(self.fov['x_min'], self.fov['x_max']) # um # cell x-coordinate
            y_ref = np.random.uniform(self.fov['y_min'], self.fov['y_max']) # um # cell y-coordinate
            z_ref = 20.0 # um # cell z-coordinate
            r_ref = 5.0 # Hz # cell firing rate
            # TODO remove the following lines.
            # cell_args = {
            #     'x': {'source': 'x_ref', 'globals': {'x_ref': x_ref}}
            #     'y': {'source': 'y_ref', 'globals': {'y_ref': y_ref}}
            #     'z': {'source': 'z_ref', 'globals': {'z_ref': z_ref}}
            #     'r': {'source': 'r_ref', 'globals': {'r_ref': r_ref}}
            # }
            cell_args = {
                'x': (lambda x: lambda t: x)(x_ref),
                'y': (lambda y: lambda t: y)(y_ref),
                'z': (lambda z: lambda t: z)(z_ref),
                'r': (lambda r: lambda t: r)(r_ref),
            }
            if self.cells_args is not None:
                cell_args.update(self.exec_kwargs(self.cells_args[c]))
            self.cells[c] = Cell(**cell_args)

        # Configure the data output of this block.
        self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))

        if self.hdf5_path is None:
            self.hdf5_path = self._get_tmp_path()

        self.hdf5_path = os.path.abspath(os.path.expanduser(self.hdf5_path))
        if not os.path.exists(self.hdf5_path):
            os.makedirs(self.hdf5_path)
        self.log.info('{n} records synthetic data into {k}'.format(k=self.hdf5_path, n=self.name))

        # Define and launch the background thread for data generation.
        ## First queue is used as a buffer for synthetic data.
        self.queue = Queue.Queue(maxsize=600)
        ## Second queue is a communication mechanism between the main and
        ## background threads in order to be able to stop the background thread.
        self.rpc_queue = Queue.Queue()
        ## Define the target function of the background thread.
        def syn_gen_target(rpc_queue, queue, nb_channels, probe, nb_samples, nb_cells, cells, hdf5_path):
            '''Synthetic data generation (background thread)'''
            mu = 0.0 # uV # noise mean
            sigma = 4.0 # uV # noise standard deviation
            spike_trains_buffer_ante = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            spike_trains_buffer_curr = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            spike_trains_buffer_post = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            hdf5_file = h5py.File(hdf5_path, 'w')
            for c in range(0, nb_cells):
                hdf5_cell = hdf5_file.create_group('cell_{}'.format(c))
                hdf5_cell.create_dataset('x', (0,), dtype='float', maxshape=(2**32,))
                hdf5_cell.create_dataset('y', (0,), dtype='float', maxshape=(2**32,))
                hdf5_cell.create_dataset('z', (0,), dtype='float', maxshape=(2**32,))
                hdf5_cell.create_dataset('r', (0,), dtype='float', maxshape=(2**32,))
                hdf5_cell.create_dataset('e', (0,), dtype='int', maxshape=(2**32,))
                hdf5_cell.create_dataset('spike_times', (0,), dtype='int', maxshape=(2**32,))
                s, u = self.cells[c].get_waveform()
                hdf5_cell_waveform = hdf5_cell.create_group('waveform')
                hdf5_s = hdf5_cell_waveform.create_dataset('x', s.shape, dtype=s.dtype)
                hdf5_s[...] = s
                hdf5_u = hdf5_cell_waveform.create_dataset('y', u.shape, dtype=u.dtype)
                hdf5_u[...] = u
            # Generate spikes for the third part of this buffer.
            chunk_number = 0
            for c in range(0, nb_cells):
                spike_trains_buffer_post[c] = cells[c].generate_spike_trains(chunk_number, nb_samples)
            while rpc_queue.empty(): # check if main thread requires a stop
                if not queue.full(): # limit memory consumption
                    # 1. Generate noise.
                    shape = (nb_samples, nb_channels)
                    data = np.random.normal(mu, sigma, shape)
                    # 2. Get spike trains.
                    spike_trains_buffer_ante = spike_trains_buffer_curr
                    spike_trains_buffer_curr = spike_trains_buffer_post
                    for c in range(0, nb_cells):
                        spike_trains_buffer_curr[c] = cells[c].generate_spike_trains(chunk_number + 1, nb_samples)
                    # 3. Reconstruct signal from spike trains.
                    for c in range(0, nb_cells):
                        # Get waveform.
                        i, j, v = cells[c].get_waveforms(chunk_number, probe)
                        # Get spike train.
                        spike_train = spike_trains_buffer_curr[c]
                        # Add waveforms into the data.
                        for t in spike_train:
                            b = np.logical_and(0 <= t + i, t + i < nb_samples)
                            data[t + i[b], j[b]] = data[t + i[b], j[b]] + v[b]
                            # TODO Manage edge effects.
                    # 4. Save spike trains in HDF5 file.
                    for c in range(0, nb_cells):
                        spike_times = spike_trains_buffer_curr[c] + chunk_number * nb_samples
                        utils.append_hdf5(hdf5_file['cell_{}/spike_times'.format(c)], spike_times)
                        r = cells[c].r(chunk_number)
                        utils.append_hdf5(hdf5_file['cell_{}/r'.format(c)], [r])
                        x = cells[c].x(chunk_number)
                        utils.append_hdf5(hdf5_file['cell_{}/x'.format(c)], [x])
                        y = cells[c].y(chunk_number)
                        utils.append_hdf5(hdf5_file['cell_{}/y'.format(c)], [y])
                        z = cells[c].z(chunk_number)
                        utils.append_hdf5(hdf5_file['cell_{}/z'.format(c)], [z])
                        e = cells[c].e(chunk_number, probe)
                        utils.append_hdf5(hdf5_file['cell_{}/e'.format(c)], [e])
                    # Finally, send data to main thread and update chunk number.
                    #data = np.transpose(data)
                    queue.put(data)
                    chunk_number += 1
            hdf5_file.close()
            return
        ## Define background thread for data generation.
        args = (self.rpc_queue, self.queue, self.nb_channels, self.probe, self.nb_samples, self.nb_cells, self.cells, self.hdf5_path)
        self.syn_gen_thread = threading.Thread(target=syn_gen_target, args=args)
        self.syn_gen_thread.deamon = True
        ## Launch background thread for data generation.
        self.log.info("{n} launches background thread for data generation".format(n=self.name))
        self.syn_gen_thread.start()

        return

    def _process(self):
        '''TODO add docstring.'''

        # Get data from background thread.
        data = self.queue.get()
        # Simulate duration between two data acquisitions.
        time.sleep(self.nb_samples / int(self.sampling_rate))
        # Send data.
        self.output.send(data)

        return

    def exec_kwargs(self, input_kwargs):
        '''Convert input keyword arguments into output keyword arguments.

        Parameter
        ---------
        input_kwargs: dict
            Dictionnary with the following keys: 'object', 'globals' and 'locals'.

        Return
        ------
        output_kwargs: dict
            Dictionnary with the following keys: 'x', 'y', 'z' and 'r'.
        '''

        # Define object (i.e. string or code object).
        obj_key = 'object'
        assert obj_key in input_kwargs
        assert isinstance(input_kwargs[obj_key], (str, unicode)), "current type is {}".format(type(input_kwargs[obj_key]))
        obj = input_kwargs[obj_key]

        # Define global dictionary.
        glb_key = 'globals'
        if glb_key in input_kwargs:
            glb = input_kwargs[glb_key]
        else:
            glb = {}
        # Add numpy to global namespace.
        glb['np'] = np
        glb['numpy'] = np

        # Define local dictionary.
        loc_key = 'locals'
        if loc_key in input_kwargs:
            loc = input_kwargs[loc_key]
        else:
            loc = {}

        # Execute dynamically the Python code.
        exec(obj, glb, loc)

        # Retrieve its result.
        output_kwargs = loc['ans']

        return output_kwargs

    def __del__(self):

        # Require a stop from the background thread.
        self.rpc_queue.put("stop")



class Cell(object):

    def __init__(self, x=None, y=None, z=None, r=None, t='default', sr=20.0e+3, rp=20.0e-3):
        '''TODO add docstring.

        Parameters
        ----------
        x: None | dict (default: None)
            Cell x-coordinate through time (i.e. chunk number).
        y: None | dict (default: None)
            Cell y-coordinate through time (i.e. chunk number).
        z: None | dict (default None)
            Cell z-coordinate through time (i.e. chunk number).
        r: None | dict (default None)
            Cell firing rate through time (i.e. chunk number).
        t: string (default: 'default')
            Cell type.
        sr: float (default: 20.0e+3 Hz)
            Sampling rate.
        rp: float (default: 20.0e-3 s)
            Refactory period.
        '''

        if x is None:
            self.x = lambda t: 0.0
        else:
            self.x = x
        if y is None:
            self.y = lambda t: 0.0
        else:
            self.y = y
        if z is None:
            self.z = lambda t: 20.0 # um
        else:
            self.z = z
        if r is None:
            self.r = lambda t: 5.0 # Hz
        else:
            self.r = r
        self.t = t # cell type
        self.sr = sr # sampling_rate
        self.rp = rp # refactory period

        self.buffered_spike_times = np.array([], dtype='float')

    def e(self, chunk_number, probe):
        '''Nearest electrode for the given chunk.

        Parameter
        ---------
        chunk_number: int
            Number of the current chunk.

        Return
        ------
        e: int
            Number of the electrode/channel which is the nearest to this cell.
        '''

        x = self.x(chunk_number)
        y = self.y(chunk_number)
        r = 100.0 # um # TODO get rid of this local parameter.
        c, d = probe.get_channels_around(x, y, r)

        e = c[np.argmin(d)]
        # NB: Only the first minimum is returned.

        return e

    def generate_spike_trains(self, chunk_number, nb_samples):
        '''TODO add docstring.'''

        scale = 1.0 / self.r(chunk_number)
        size = 1 + int(float(nb_samples) / self.sr / scale)

        spike_times = np.array([])

        last_spike_time = 0.0
        max_spike_time = float(nb_samples) / self.sr
        while last_spike_time < max_spike_time:
            # We need to generate some new spike times.
            spike_intervals = np.random.exponential(scale=scale, size=size)
            spike_intervals = spike_intervals[self.rp < spike_intervals]
            spike_times = np.concatenate([spike_times, last_spike_time + np.cumsum(spike_intervals)])
            if len(spike_times) > 0:
                last_spike_time = spike_times[-1]
        self.buffered_spike_times = spike_times[max_spike_time <= spike_times]

        spike_times = spike_times[spike_times < max_spike_time]
        spike_steps = spike_times * self.sr
        spike_steps = spike_steps.astype('int')

        return spike_steps

    def get_waveform(self):
        '''TODO add docstring.'''

        tau = 1.5e-3 # s # characteristic time
        amp = -40.0 # um # minimal voltage

        i_start = -20
        i_stop = +60
        steps = np.arange(i_start, i_stop + 1)
        times = steps.astype('float') / self.sr
        times = times - times[0]
        u = np.sin(4.0 * np.pi * times / times[-1])
        u = u * np.power(times * np.exp(- times / tau), 10.0)
        u = u * (amp / np.amin(u))

        return steps, u

    def get_waveforms(self, chunk_number, probe):
        '''TODO add docstring.'''

        steps, u = self.get_waveform()

        x = self.x(chunk_number)
        y = self.y(chunk_number)
        r = 100.0 # um # TODO get rid of this local parameter.
        channels, distances = probe.get_channels_around(x, y, r)

        z = self.z(chunk_number)
        distances = np.sqrt(np.power(distances, 2.0) + z ** 2)

        i = np.tile(steps, channels.size)
        j = np.repeat(channels, steps.size)
        v = np.zeros((steps.size, channels.size))
        half_distance = 45.0 # um
        for k in range(0, channels.size):
            coef = 1.0 / (1.0 + (distances[k] / half_distance) ** 2.0) # coefficient of attenuation
            v[:, k] = coef * u
        v = np.transpose(v)
        v = v.flatten()

        return i, j, v
