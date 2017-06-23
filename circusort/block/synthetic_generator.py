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
from circusort.io.synthetic import SyntheticStore



# TODO find if the communication mechanism between the main and background
# threads is necessary, i.e. is there another canonical way to stop the
# background thread (in a infinite loop) from the main thread?

class Synthetic_generator(block.Block):
    '''TODO add docstring'''

    name = "Synthetic Generator"

    params = {
        'dtype'         : 'float32',
        'probe'         : None,
        'sampling_rate' : 20000.0,
        'nb_samples'    : 1024,
        'nb_cells'      : 10,
        'hdf5_path'     : None,
    }

    def __init__(self, cells_args=None, cells_params=None, **kwargs):

        block.Block.__init__(self, **kwargs)
        self.cells_args = cells_args
        if self.probe == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = io.Probe(self.probe, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        if self.cells_args is not None:
            self.nb_cells = len(self.cells_args)
        if cells_params is None:
            self.cells_params = {}
        else:
            self.cells_params = cells_params
        self.add_output('data')

    def _get_tmp_path(self):
        tmp_file  = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name))
        tmp_file.close()
        return data_path

    def _initialize(self):
        '''TODO add docstring.'''

        # Retrieve the geometry of the probe.
        self.nb_channels = self.probe.nb_channels
        self.fov = self.probe.field_of_view

        # Generate synthetic cells.
        self.cells = {}
        for c in range(0, self.nb_cells):
            x_ref = np.random.uniform(self.fov['x_min'], self.fov['x_max']) # um # cell x-coordinate
            y_ref = np.random.uniform(self.fov['y_min'], self.fov['y_max']) # um # cell y-coordinate
            z_ref = 0.0 # um # cell z-coordinate
            r_ref = 5.0 # Hz # cell firing rate
            cell_args = {
                'x': eval("lambda t: %s" % x_ref),
                'y': eval("lambda t: %s" % y_ref),
                'z': eval("lambda t: %s" % z_ref),
                'r': eval("lambda t: %s" % r_ref),
            }
            
            if self.cells_args is not None:
                self.log.debug('{n} creates a cell with params {p}'.format(n=self.name, p=self.cells_args[c]))
                cell_args.update(self.exec_kwargs(self.cells_args[c], self.cells_params))
            self.cells[c] = Cell(**cell_args)

        # Configure the data output of this block.
        self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))

        if self.hdf5_path is None:
            self.hdf5_path = self._get_tmp_path()

        self.hdf5_path = os.path.abspath(os.path.expanduser(self.hdf5_path))
        self.log.info('{n} records synthetic data from {d} cells into {k}'.format(k=self.hdf5_path, n=self.name, d=self.nb_cells))

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

            self.synthetic_store = SyntheticStore(hdf5_path, 'w')

            for c in range(0, nb_cells):

                s, u = self.cells[c].get_waveform()

                params = {'cell_id'    : c, 
                          'waveform/x' : s,
                          'waveform/y' : u
                          }

                self.synthetic_store.add(params)

            # Generate spikes for the third part of this buffer.
            chunk_number = 0
            to_write     = {}
            frequency    = 100

            for c in range(0, nb_cells):
                spike_trains_buffer_post[c] = cells[c].generate_spike_trains(chunk_number, nb_samples)
                to_write[c] = {'cell_id' : c, 'x' : [], 'y' : [], 'z' : [], 'e' : [], 'r' : [], 'spike_times' : []}


            while rpc_queue.empty(): # check if main thread requires a stop
                if not queue.full(): # limit memory consumption
                    # 1. Generate noise.
                    shape = (nb_samples, nb_channels)
                    data = np.random.normal(mu, sigma, shape).astype(self.dtype)
                    # 2. Get spike trains.
                    spike_trains_buffer_ante = spike_trains_buffer_curr
                    spike_trains_buffer_curr = spike_trains_buffer_post
                    for c in range(0, nb_cells):
                        spike_trains_buffer_curr[c] = cells[c].generate_spike_trains(chunk_number + 1, nb_samples)

                        # 3. Reconstruct signal from spike trains.

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

                        spike_times = spike_trains_buffer_curr[c] + chunk_number * nb_samples

                        to_write[c]['x'] += [cells[c].x(chunk_number)]
                        to_write[c]['y'] += [cells[c].y(chunk_number)]
                        to_write[c]['z'] += [cells[c].y(chunk_number)]
                        to_write[c]['e'] += [cells[c].e(chunk_number, probe)]
                        to_write[c]['r'] += [cells[c].r(chunk_number)]
                        to_write[c]['spike_times'] += spike_times.tolist()

                        if chunk_number % frequency == 0:
                            self.synthetic_store.add(to_write[c])
                            to_write[c] = {'cell_id' : c, 'x' : [], 'y' : [], 'z' : [], 'e' : [], 'r' : [], 'spike_times' : []}

                    # Finally, send data to main thread and update chunk number.
                    #data = np.transpose(data)
                    queue.put(data)
                    chunk_number += 1
            self.synthetic_store.close()
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

    def exec_kwargs(self, input_kwargs, input_params):
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
        # obj_key = 'object'
        # assert obj_key in input_kwargs
        # assert isinstance(input_kwargs[obj_key], (str, unicode)), "current type is {}".format(type(input_kwargs[obj_key]))
        # obj = input_kwargs[obj_key]

        # # Define global dictionary.
        # glb_key = 'globals'
        # if glb_key in input_kwargs:
        #     glb = input_kwargs[glb_key]
        # else:
        #     glb = {}
        # Add numpy to global namespace.
        input_params['np'] = np
        input_params['numpy'] = np

        # # Define local dictionary.
        # loc_key = 'locals'
        # if loc_key in input_kwargs:
        #     loc = input_kwargs[loc_key]
        # else:
        #     loc = {}

        # # Execute dynamically the Python code.
        # exec(obj, glb, loc)

        # # Retrieve its result.
        # output_kwargs = loc['ans']

        for key in input_kwargs.keys():
            if type(input_kwargs[key]) == unicode:
                input_kwargs[key] = eval("lambda t: %s" % input_kwargs[key], input_params)
        return input_kwargs

    def __del__(self):

        # Require a stop from the background thread.
        self.rpc_queue.put("stop")



class Cell(object):

    variables = {'x'       : None,
                 'y'       : None,
                 'z'       : None,
                 'r'       : None,
                 't'       : 'default',
                 'sr'      : 20000, 
                 'rp'      : 5e-3,
                 'nn'      : 100, 
                 'hf_dist' : 50, 
                 'a_dist'  : 1}

    def __init__(self, x=None, y=None, z=None, r=None, t='default', sr=20.0e+3, rp=20.0e-3, nn=100, hf_dist = 45.0, a_dist=1.0):
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
        self.t  = t # cell type
        self.sr = sr # sampling_rate
        
        self.rp = rp # refactory period
        self.nn = nn
        self.hf_dist = hf_dist
        self.a_dist = a_dist

        self.buffered_spike_times = np.array([], dtype='float32')

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
        
        c, d = probe.get_channels_around(x, y, self.nn)

        e = c[np.argmin(d)]
        # NB: Only the first minimum is returned.

        return e

    def generate_spike_trains(self, chunk_number, nb_samples):
        '''TODO add docstring.'''

        if self.r(chunk_number) > 0.0:
            scale = 1.0 / self.r(chunk_number)
        else:
            scale = np.inf

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
        times = steps.astype('float32') / self.sr
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
        channels, distances = probe.get_channels_around(x, y, self.nn)

        z = self.z(chunk_number)
        distances = np.sqrt(np.power(distances, 2.0) + z ** 2)

        i = np.tile(steps, channels.size)
        j = np.repeat(channels, steps.size)
        v = np.zeros((steps.size, channels.size))
        for k in range(0, channels.size):
            coef = self.a_dist / (1.0 + (distances[k] / self.hf_dist) ** 2.0) # coefficient of attenuation
            v[:, k] = coef * u
        v = np.transpose(v)
        v = v.flatten()

        return i, j, v
