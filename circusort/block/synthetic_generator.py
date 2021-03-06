# -*- coding: utf-8 -*-

import numpy as np
try:
    from Queue import Queue  # Python 2 compatibility.
except ImportError:  # i.e. ModuleNotFoundError
    from queue import Queue  # Python 3 compatibility.
import tempfile
import threading
import time
import os
import json

from circusort.block import block
from circusort import io
from circusort.io import get_tmp_dirname
from circusort.io.synthetic import SyntheticStore

try:
    unicode  # Python 2 compatibility.
except NameError:
    unicode = str  # Python 3 compatibility.

# TODO find if the communication mechanism between the main and background
# threads is necessary, i.e. is there another canonical way to stop the
# background thread (in a infinite loop) from the main thread?


__classname__ = 'SyntheticGenerator'


class SyntheticGenerator(block.Block):
    """Generate a synthetic MEA recording.

    Arguments:
        cells_args: list
            List of dictionaries used as input arguments for the creation of the
            synthetic cells.
        cells_params: dict
            Dictionary of global input arguments for all these synthetic cells.
        hdf5_path: string
            HDF5 path.
        probe: string
            Probe path.
        log_level: integer
            Level for the associated logger.

    Attributes:
        dtype: type:
            Data type.
        probe: string
            Path to the location of the file describing the probe.
        sampling_rate: float
            Sampling rate [Hz]. The default value is 20e+3.
        nb_samples
            Chunk size.
        nb_cells: integer
            Number of cells. The default value is 10.
        hdf5_path: string
            Path to the location used to save the parameters of the generation.
        log_path: string (optional)
        seed: integer (optional)
            Seed for random generation. The default value is 42.
        is_realistic: boolean (optional)
            Send output buffers so that the sampling rate is verified. The default value is True.

    Output:
        data

    """

    name = "Synthetic Generator"

    params = {
        'working_directory': None,
        'dtype': 'float32',
        'probe': None,
        'sampling_rate': 20e+3,  # Hz
        'nb_samples': 1024,
        'nb_cells': 10,
        'hdf5_path': None,
        'log_path': None,
        'seed': 42,
        'is_realistic': True,
    }

    def __init__(self, cells_args=None, cells_params=None, **kwargs):

        # Preinitialize object with the base class.
        block.Block.__init__(self, **kwargs)

        self.sampling_rate = self.sampling_rate  # Useful to disable some PyCharm warnings.
        self.nb_samples = self.nb_samples  # Useful to disable some PyCharm warnings.
        self.seed = self.seed  # Useful to disable some PyCharm warnings.
        self.is_realistic = self.is_realistic  # Useful to disable some PyCharm warnings.

        self.working_directory = self.working_directory  # Useful to disable some PyCharm warnings.
        if self.working_directory is None:
            self.mode = 'default'
        else:
            self.mode = 'preconfigured'

        if self.mode == 'default':

            # Save class parameters.
            self.cells_args = cells_args
            if cells_params is None:
                self.cells_params = {}
            else:
                self.cells_params = cells_params
            # Compute the number of cells.
            if self.cells_args is not None:
                self.nb_cells = len(self.cells_args)
            # Open the probe file.
            if self.probe is None:
                self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
            else:
                self.probe = io.load_probe(self.probe, logger=self.log)
                self.log.info('{n} reads the probe layout'.format(n=self.name))

            # TODO log/save input keyword argument to file.
            self.log_path = self.log_path  # Useful to disable some PyCharm warnings.
            if self.log_path is not None:
                log_kwargs = {k: self.params[k] for k in ['nb_samples']}
                with open(self.log_path, 'w') as log_file:
                    json.dump(log_kwargs, log_file, sort_keys=True, indent=4)

        elif self.mode == 'preconfigured':

            parameters_path = os.path.join(self.working_directory, "parameters.txt")
            parameters = io.get_data_parameters(parameters_path)
            # TODO get rid of the following try ... except ...
            try:
                self.duration = parameters['general']['duration']
            except KeyError:
                self.duration = 60.0  # s
            probe_path = os.path.join(self.working_directory, "probe.prb")
            self.probe = io.load_probe(probe_path, logger=self.log)

        self._number = -1

        # Add data output.
        self.add_output('data', structure='dict')

    @staticmethod
    def _resolve_hdf5_path():

        tmp_dirname = get_tmp_dirname()
        tmp_filename = "synthetic.hdf5"
        tmp_path = os.path.join(tmp_dirname, tmp_filename)

        return tmp_path

    @staticmethod
    def _get_tmp_path():

        tmp_file = tempfile.NamedTemporaryFile()
        data_path = os.path.join(tempfile.gettempdir(), os.path.basename(tmp_file.name))
        tmp_file.close()

        return data_path

    def _initialize(self):

        # Seed the random generator.
        np.random.seed(self.seed)

        # Retrieve the geometry of the probe.
        self.nb_channels = self.probe.nb_channels
        self.fov = self.probe.field_of_view

        if self.mode == 'default':

            # Generate synthetic cells.
            self.cells = {}
            for c in range(0, self.nb_cells):
                x_ref = np.random.uniform(self.fov['x_min'], self.fov['x_max'])  # µm  # cell x-coordinate
                y_ref = np.random.uniform(self.fov['y_min'], self.fov['y_max'])  # µm  # cell y-coordinate
                z_ref = 0.0  # um  # cell z-coordinate
                r_ref = 5.0  # Hz  # cell firing rate
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

        elif self.mode == 'preconfigured':

            self.cells = io.load_cells(self.working_directory, mode='by cells', probe=self.probe)
            self.nb_cells = self.cells.nb_cells
            self.dtype = 'int16'  # TODO set default data type to 16 bit signed (or unsigned) integer.

        # Configure the data output of this block.
        self.output.configure(dtype=self.dtype, shape=(self.nb_samples, self.nb_channels))

        if self.hdf5_path is None:
            self.hdf5_path = self._get_tmp_path()

        self.hdf5_path = os.path.abspath(os.path.expanduser(self.hdf5_path))
        info_msg = "{n} records synthetic data from {d} cells into {k}"
        self.log.info(info_msg.format(k=self.hdf5_path, n=self.name, d=self.nb_cells))

        # Define and launch the background thread for data generation.
        # # First queue is used as a buffer for synthetic data.
        self.queue = Queue(maxsize=600)
        # # Second queue is a communication mechanism between the main and
        # # background threads in order to be able to stop the background thread.
        self.rpc_queue = Queue()

        # # Define background thread for data generation.
        if self.mode == 'default':
            args = (self.rpc_queue, self.queue, self.nb_channels, self.probe,
                    self.nb_samples, self.cells, self.hdf5_path)
            self.syn_gen_thread = threading.Thread(target=syn_gen_target, args=args)
        elif self.mode == 'preconfigured':
            args = (self.rpc_queue, self.queue, self.nb_channels, self.nb_samples,
                    self.sampling_rate, self.duration, self.cells)
            self.syn_gen_thread = threading.Thread(target=pre_syn_gen_target, args=args)
        self.syn_gen_thread.deamon = True
        # # Launch background thread for data generation.
        self.log.info("{n} launches background thread for data generation".format(n=self.name))
        self.syn_gen_thread.start()

        return

    def _process(self):
        """Process one buffer of data."""

        # Get data from background thread.
        data = self.queue.get()
        self._number += 1

        if isinstance(data, str) and data == 'EOS':
            # Stop processing block.
            self.stop_pending = True
        else:
            if self.is_realistic:
                # Simulate duration between two data acquisitions.
                time.sleep(float(self.nb_samples) / self.sampling_rate)
            # Prepare output packet.
            packet = {
                'number': self._number,
                'payload': data,
            }
            # Send output packet.
            self.get_output('data').send(packet)

        return

    @staticmethod
    def exec_kwargs(input_kwargs, input_params):
        """Convert input keyword arguments into output keyword arguments.

        Parameter:
            input_kwargs: dict
                Dictionary with the following keys: 'object', 'globals' and 'locals'.

        Return:
            output_kwargs: dict
                Dictionary with the following keys: 'x', 'y', 'z' and 'r'.
        """

        # Add numpy to global namespace.
        input_params['np'] = np
        input_params['numpy'] = np

        # TODO add comment.
        for key in input_kwargs.keys():
            if type(input_kwargs[key]) == unicode and key != 't':
                input_kwargs[key] = eval("lambda t: %s" % input_kwargs[key], input_params)

        return input_kwargs

    def __del__(self):

        # Require a stop from the background thread.
        self.rpc_queue.put("stop")


class Cell(object):
    """Cell object

    Parameters
    ----------
    x: None | dict, optional
        Cell x-coordinate through time (i.e. chunk number). The default value is None.
    y: None | dict, optional
        Cell y-coordinate through time (i.e. chunk number). The default value is None.
    z: None | dict, optional
        Cell z-coordinate through time (i.e. chunk number). The default value is None.
    r: None | dict, optional
        Cell firing rate through time (i.e. chunk number). The default value is None.
    s: float, optional
        Temporal shift of the first spike. The default value is 0.0.
    t: string, optional
        Cell type. The default value is 'default'.
    sr: float, optional
        Sampling rate [Hz]. The default value is 20.0e+3.
    rp: float, optional
        Refractory period [s]. The default value is 20.0e-3.
    nn: float, optional
        Radius used to identify the neighboring channels. The default value is 100.0.
    hf_dist: float, optional
        First parameter for the attenuation of the spike waveforms. The default value is 45.0.
    a_dist: float, optional
        Second parameter for the attenuation of the spike waveforms. The default value is 1.0.

    """

    variables = {
        'x': None,
        'y': None,
        'z': None,
        'r': None,
        's': 0.0,
        't': 'default',
        'sr': 20000,
        'rp': 5e-3,
        'nn': 100,
        'hf_dist': 50,
        'a_dist': 1,
    }

    def __init__(self, x=None, y=None, z=None, r=None, s=0.0, t='default', sr=20.0e+3,
                 rp=20.0e-3, nn=100.0, hf_dist=45.0, a_dist=1.0):

        if x is None:
            self.x = lambda _: 0.0
        else:
            self.x = x
        if y is None:
            self.y = lambda _: 0.0
        else:
            self.y = y
        if z is None:
            self.z = lambda _: 20.0  # um
        else:
            self.z = z
        if r is None:
            self.r = lambda _: 5.0  # Hz
        else:
            self.r = r
        self.s = s  # temporal shift of the first spike
        self.t = t  # cell type
        self.sr = sr  # sampling_rate

        self.rp = rp  # refractory period
        self.nn = nn
        self.hf_dist = hf_dist
        self.a_dist = a_dist

        self.buffered_spike_times = np.array([], dtype='float32')
        self._waveform = None
        self._indices = None

    def e(self, chunk_number, probe):
        """Nearest electrode for the given chunk.

        Parameter
        ---------
        chunk_number: int
            Number of the current chunk.

        Return
        ------
        e: int
            Number of the electrode/channel which is the nearest to this cell.
        """

        x = self.x(chunk_number)
        y = self.y(chunk_number)

        c, d = probe.get_channels_around(x, y, self.nn)

        e = c[np.argmin(d)]
        # NB: Only the first minimum is returned.

        return e

    def generate_spike_trains(self, chunk_number, nb_samples):
        """Generate spike trains

        Parameters
        ----------
        chunk_number: integer
            Identifier of the current chunk.
        nb_samples
            Number of samples per buffer.
        """

        if self.t == 'default':

            # Take refractory period into account to achieve the wanted firing rate.
            r = self.r(chunk_number)
            if self.rp > 0.0:
                r_max = 1.0 / self.rp  # maximal firing rate with refractory periods
                msg = "A firing rate of {} Hz is impossible with a refractory period of {} ms"
                assert r < r_max, msg.format(r, self.rp)
                r = r / (1.0 - r * self.rp)
            if r > 0.0:
                scale = 1.0 / r
            else:
                scale = np.inf

            size = 1 + int(float(nb_samples) / self.sr / scale)

            if self.buffered_spike_times.size == 0:
                spike_times = np.array([])
                last_spike_time = 0.0
            else:
                spike_times = self.buffered_spike_times
                last_spike_time = spike_times[-1]
            max_spike_time = float(nb_samples) / self.sr
            while last_spike_time < max_spike_time:
                # We need to generate some new spike times.
                spike_intervals = np.random.exponential(scale=scale, size=size)
                spike_intervals = spike_intervals[self.rp < spike_intervals]
                spike_times = np.concatenate([spike_times, last_spike_time + np.cumsum(spike_intervals)])
                if len(spike_times) > 0:
                    last_spike_time = spike_times[-1]
            self.buffered_spike_times = spike_times[max_spike_time <= spike_times] - max_spike_time

            spike_times = spike_times[spike_times < max_spike_time]
            spike_steps = spike_times * self.sr
            spike_steps = spike_steps.astype('int')

        elif self.t == 'periodic':

            if self.r(chunk_number) > 0.0:
                scale = 1.0 / self.r(chunk_number)
                if self.buffered_spike_times.size == 0:
                    spike_times = np.array([])
                    last_spike_time = self.s - float(chunk_number * nb_samples) * self.sr
                else:
                    spike_times = self.buffered_spike_times
                    last_spike_time = spike_times[-1]
                max_spike_time = float(nb_samples) / self.sr
                while last_spike_time < max_spike_time:
                    # We need to generate some new spike times.
                    nb_spikes = int(np.floor((max_spike_time - last_spike_time) / scale) + 1)
                    spike_intervals = np.array(nb_spikes * [scale])
                    # TODO add a warning when the firing period is less or equal than the refractory period.
                    spike_times = np.concatenate([spike_times, last_spike_time + np.cumsum(spike_intervals)])
                    if len(spike_times) > 0:
                        last_spike_time = spike_times[-1]
                self.buffered_spike_times = spike_times[max_spike_time <= spike_times] - max_spike_time

                spike_times = spike_times[spike_times < max_spike_time]
                spike_steps = spike_times * self.sr
                spike_steps = spike_steps.astype('int')

            else:
                spike_steps = np.array([], dtype='int')

        else:

            raise NotImplementedError("unknown cell type '{}'".format(self.t))

        return spike_steps

    @property
    def waveform(self):
        """Get spike waveform"""

        if self._waveform is None:
            tau = 1.5e-3  # s  # characteristic time
            amp = -80.0  # um  # minimal voltage

            i_start = -20
            i_stop = +60
            steps = np.arange(i_start, i_stop + 1)
            times = steps.astype('float32') / self.sr
            times = times - times[0]
            u = np.sin(4.0 * np.pi * times / times[-1])
            u = u * np.power(times * np.exp(- times / tau), 10.0)
            u = u * (amp / np.amin(u))
            self._waveform = steps, u

        return self._waveform

    def get_waveforms(self, chunk_number, probe, sparse_factor=0.5):
        """Get spike waveforms

        Parameters
        ----------
        chunk_number: integer
            Number of the current chunk.
        probe: circusort.io.Probe
            Description of the probe.
        sparse_factor: percentage of channels that are set to 0 randomly (default is 0.5)
        """

        steps, u = self.get_waveform()

        x = self.x(chunk_number)
        y = self.y(chunk_number)
        channels, distances = probe.get_channels_around(x, y, self.nn)

        z = self.z(chunk_number)
        distances = np.sqrt(np.power(distances, 2.0) + z ** 2)

        i = np.tile(steps, channels.size)
        j = np.repeat(channels, steps.size)
        v = np.zeros((steps.size, channels.size))

        self._indices = np.arange(channels.size)

        if sparse_factor > 0:
            if self._indices is None:
                self._indices = np.random.permutation(self._indices)[:int(channels.size*sparse_factor)]

        for k in self._indices:
            coef = self.a_dist / (1.0 + (distances[k] / self.hf_dist) ** 2.0)  # coefficient of attenuation
            v[:, k] = coef * u
        v = np.transpose(v)
        v = v.flatten()

        return i, j, v


def syn_gen_target(rpc_queue, queue, nb_channels, probe, nb_samples, cells, hdf5_path):
    """Synthetic data generation (background thread)."""

    mu = 0.0  # uV  # noise mean
    sigma = 4.0  # uV  # noise standard deviation
    nb_cells = len(cells)

    spike_trains_buffer_ante = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
    spike_trains_buffer_curr = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
    spike_trains_buffer_post = {c: np.array([], dtype='int') for c in range(0, nb_cells)}

    synthetic_store = SyntheticStore(hdf5_path, 'w')

    for c in range(0, nb_cells):
        s, u = cells[c].get_waveform()

        params = {
            'cell_id': c,
            'waveform/x': s,
            'waveform/y': u
        }

        synthetic_store.add(params)

    # Generate spikes for the third part of this buffer.
    chunk_number = 0
    to_write = {}
    frequency = 100

    for c in range(0, nb_cells):
        spike_trains_buffer_post[c] = cells[c].generate_spike_trains(chunk_number, nb_samples)
        to_write[c] = {
            'cell_id': c,
            'x': [],
            'y': [],
            'z': [],
            'e': [],
            'r': [],
            'spike_times': [],
        }

    while rpc_queue.empty():  # check if main thread requires a stop
        if not queue.full():  # limit memory consumption
            # 1. Generate noise.
            shape = (nb_samples, nb_channels)
            data = np.random.normal(mu, sigma, shape).astype(np.float32)
            # 2. Get spike trains.
            spike_trains_buffer_ante = spike_trains_buffer_curr.copy()
            spike_trains_buffer_curr = spike_trains_buffer_post.copy()
            for c in range(0, nb_cells):
                spike_trains_buffer_post[c] = cells[c].generate_spike_trains(chunk_number + 1, nb_samples)

                # 3. Reconstruct signal from spike trains.

                # Get waveform.
                i, j, v = cells[c].get_waveforms(chunk_number, probe)
                # Get current spike train.
                spike_train = spike_trains_buffer_curr[c]
                # Add waveforms into the data.
                for t in spike_train:
                    b = np.logical_and(0 <= t + i, t + i < nb_samples)
                    data[t + i[b], j[b]] = data[t + i[b], j[b]] + v[b]
                # Get previous spike train.
                spike_train = spike_trains_buffer_ante[c] - nb_samples
                # Add waveforms into the data.
                for t in spike_train:
                    b = np.logical_and(0 <= t + i, t + i < nb_samples)
                    data[t + i[b], j[b]] = data[t + i[b], j[b]] + v[b]
                # Get post spike train.
                spike_train = spike_trains_buffer_post[c] + nb_samples
                # Add waveforms into the data.
                for t in spike_train:
                    b = np.logical_and(0 <= t + i, t + i < nb_samples)
                    data[t + i[b], j[b]] = data[t + i[b], j[b]] + v[b]

                # 4. Save spike trains in HDF5 file.

                spike_times = spike_trains_buffer_curr[c] + chunk_number * nb_samples
                # # TODO remove following lines.
                # if len(spike_times) > 0 and chunk_number < 50:
                #     print("{} + {} x {} = {}".format(spike_train, chunk_number, nb_samples, spike_times))

                to_write[c]['x'] += [cells[c].x(chunk_number)]
                to_write[c]['y'] += [cells[c].y(chunk_number)]
                to_write[c]['z'] += [cells[c].z(chunk_number)]
                to_write[c]['e'] += [cells[c].e(chunk_number, probe)]
                to_write[c]['r'] += [cells[c].r(chunk_number)]
                to_write[c]['spike_times'] += spike_times.tolist()

                if chunk_number % frequency == 0:
                    synthetic_store.add(to_write[c])
                    to_write[c] = {
                        'cell_id': c,
                        'x': [],
                        'y': [],
                        'z': [],
                        'e': [],
                        'r': [],
                        'spike_times': [],
                    }

            # Finally, send data to main thread and update chunk number.
            # data = np.transpose(data)
            queue.put(data)
            chunk_number += 1

    # We write the remaining data for the cells.
    for c in range(0, nb_cells):
        synthetic_store.add(to_write[c])

    synthetic_store.close()

    return


def pre_syn_gen_target(rpc_queue, queue, nb_channels, nb_samples_per_chunk, sampling_rate, duration, cells):
    """Preconfigured synthetic data generation (background thread).

    Parameters:
        rpc_queue: Queue.Queue
        queue: Queue.Queue
        nb_channels: integer
        nb_samples_per_chunk: integer
        sampling_rate: float
        duration: float
        cells: dictionary
    """

    mu = 0.0  # µV  # noise mean
    sigma = 4.0  # µV  # noise standard deviation
    chunk_width = float(nb_samples_per_chunk) / sampling_rate * 1e+3  # ms  # chunk width
    data_width = duration * sampling_rate  # data width
    nb_samples = int(data_width)
    nb_chunks = nb_samples // nb_samples_per_chunk
    nb_samples_last_chunk = nb_samples % nb_samples_per_chunk
    quantum_width = 0.1042  # µV / AD
    dtype = np.int16  # output data type
    dtype_info = np.iinfo(dtype)

    chunk_number = 0
    while chunk_number < nb_chunks:
        if not queue.full():  # limit memory consumption
            # a. Generate some gaussian noise.
            shape = (nb_samples_per_chunk, nb_channels)
            data = np.random.normal(loc=mu, scale=sigma, size=shape).astype(np.float32)
            # b. Inject waveforms.
            for cell in cells:
                # Collect subtrain with spike events which fall inside the current chunk.
                subtrain = cell.get_chunk_subtrain(chunk_number, chunk_width=chunk_width)
                # Retrieve the template to inject.
                i_ref, j, v = cell.template.synthetic_export
                # Inject the template to the correct spatiotemporal locations.
                for t in subtrain:
                    k = int(t * sampling_rate)
                    i = i_ref + k
                    m = np.logical_and(0 <= i, i < nb_samples_per_chunk)
                    data[i[m], j[m]] = data[i[m], j[m]] + v[m]
            # c. Quantize the data.
            data = data / quantum_width
            data = np.round(data)
            data[data < dtype_info.min] = dtype_info.min
            data[data > dtype_info.max] = dtype_info.max
            data = data.astype(dtype)
            # d. Send data to main thread.
            queue.put(data)
            # e. Update chunk number.
            chunk_number += 1

    # Manage last (partial) chunk (if necessary).
    if nb_samples_last_chunk > 0:
        while chunk_number == nb_chunks:
            if not queue.full():  # limit memory consumption
                # a. Generate some gaussian noise.
                shape = (nb_samples_last_chunk, nb_channels)
                data = np.random.normal(loc=mu, scale=sigma, size=shape).astype(np.float32)
                # b. Inject waveforms.
                for cell in cells:
                    # Collect subtrain with spike events which fall inside the current chunk.
                    subtrain = cell.get_chunk_subtrain(chunk_number, chunk_width=chunk_width)
                    # Retrieve the template to inject.
                    i_ref, j, v = cell.template.synthetic_export
                    # Inject the template to the correct spatiotemporal locations.
                    for t in subtrain:
                        k = int(t * sampling_rate)
                        i = i_ref + k
                        m = np.logical_and(0 <= i, i < nb_samples_last_chunk)
                        data[i[m], j[m]] = data[i[m], j[m]] + v[m]
                # c. Add trailing zeros.
                shape = (nb_samples_per_chunk - nb_samples_last_chunk, nb_channels)
                zeros = np.zeros(shape)
                data = np.concatenate((data, zeros))
                # d. Quantize the data.
                data = data / quantum_width
                data = np.round(data)
                data[data < dtype_info.min] = dtype_info.min
                data[data > dtype_info.max] = dtype_info.max
                data = data.astype(dtype)
                # e. Send data to main thread.
                queue.put(data)
                # f. Update chunk number.
                chunk_number += 1

    # Send the signal for the end of the stream.
    queue.put('EOS')

    return
