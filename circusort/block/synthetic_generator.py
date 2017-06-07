import numpy as np
import Queue
import scipy as sp
import scipy.signal
import threading
import time

from circusort.block import block
from circusort import io



# TODO find if the communication mechanism between the main and background
# threads is necessary, i.e. is there another canonical way to stop the
# background thread (in a infinite loop) from the main thread?

class Synthetic_generator(block.Block):
    '''TODO add docstring'''

    name = "Synthetic Generator"

    params = {
        'dtype'         : 'float',
        'probe_filename': '~/spyking-circus/probes/mea_16.prb',
        # 'nb_channels'   : 16, # TODO remove this parameter (redundancy).
        'sampling_rate' : 20000.0,
        'nb_samples'    : 2000,
        'nb_cells'      : 10,
    }

    def __init__(self, **kwargs):

        block.Block.__init__(self, **kwargs)
        self.add_output('data')

    def _initialize(self):
        '''TODO add docstring.'''

        # Retrieve the geometry of the probe.
        self.probe = io.Probe(self.probe_filename)
        self.log.info("probe: {}".format(self.probe))
        self.nb_channels = self.probe.nb_channels
        self.log.info("nb_channels: {}".format(self.nb_channels))
        self.fov = self.probe.field_of_view
        self.log.info("field_of_view: {}".format(self.fov))

        # Generate synthetic cells.
        self.cells = {}
        for k in range(0, self.nb_cells):
            t = 'default' # cell type
            x = np.random.uniform(self.fov['x_min'], self.fov['x_max']) # cell x-coordinate
            y = np.random.uniform(self.fov['y_min'], self.fov['y_max']) # cell y-coordinate
            self.cells[k] = Cell(t, x, y)

        # Configure the data output of this block.
        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))

        # Define and launch the background thread for data generation.
        ## First queue is used as a buffer for synthetic data.
        self.queue = Queue.Queue(maxsize=600)
        ## Second queue is a communication mechanism between the main and
        ## background threads in order to be able to stop the background thread.
        self.rpc_queue = Queue.Queue()
        ## Define the target function of the background thread.
        def syn_gen_target(rpc_queue, queue, nb_channels, nb_samples, nb_cells, cells):
            '''Synthetic data generation (background thread)'''
            mu = 0.0 # uV # noise mean
            sigma = 10.0 # uV # noise standard deviation
            spike_trains_buffer_ante = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            spike_trains_buffer_curr = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            spike_trains_buffer_post = {c: np.array([], dtype='int') for c in range(0, nb_cells)}
            # Generate spikes for the third part of this buffer.
            for c in range(0, nb_cells):
                spike_trains_buffer_post[c] = cells[c].generate_spike_trains(nb_samples)
            while rpc_queue.empty(): # check if main thread requires a stop
                if not queue.full(): # limit memory consumption
                    # First, generate noise.
                    shape = (nb_channels, nb_samples)
                    data = np.random.normal(mu, sigma, shape)
                    # Second, update spike trains.
                    spike_trains_buffer_ante = spike_trains_buffer_curr
                    spike_trains_buffer_curr = spike_trains_buffer_post
                    for c in range(0, nb_cells):
                        spike_trains_buffer_post[c] = cells[c].generate_spike_trains(nb_samples)
                    # Third, reconstruct signal from spike trains.
                    for c in range(0, nb_cells):
                        spike_train = spike_trains_buffer_curr[c]
                        for t in spike_train:
                            data[:, t] = float(c) * 40.0
                    # Finally, send data to main thread.
                    queue.put(data)
            # TODO complete.
            return
        ## Define background thread for data generation.
        args = (self.rpc_queue, self.queue, self.nb_channels, self.nb_samples, self.nb_cells, self.cells)
        self.syn_gen_thread = threading.Thread(target=syn_gen_target, args=args)
        self.syn_gen_thread.deamon = True
        ## Launch background thread for data generation.
        self.log.info("Launch background thread for data generation...")
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

    def __del__(self):

        # Require a stop from the background thread.
        self.rpc_queue.put("stop")



class Cell(object):

    def __init__(self, t, x, y, sr=20000.0):
        '''TODO add docstring.'''
        self.t = t # cell type
        self.x = x # cell x-coordinate
        self.y = y # cell y-coordinate

        self.sampling_rate = sr

        # Define the waveform associated to the cell.
        self.n_t_ante = 10
        self.n_t_post = 20
        self.n_t = self.n_t_ante + self.n_t_post
        t = np.linspace(-self.n_t_ante, self.n_t_post, num=self.n_t, endpoint=False)
        t = t + float(self.n_t_ante)
        t = t / self.sampling_rate
        self.waveform = t * np.exp(- 4000.0 * t)

        self.buffered_spike_times = np.array([], dtype='float')

    def generate_spike_trains(self, nb_samples):
        '''TODO add docstring.'''

        # scale = 0.5
        scale = 10.0
        size = 1 + int(float(nb_samples) / self.sampling_rate / scale)
        refactory_period = 20.0e-3 # s
        spike_times = np.array([])

        last_spike_time = 0.0
        max_spike_time = float(nb_samples) / self.sampling_rate
        while last_spike_time < max_spike_time:
            # We need to generate some new spike times.
            spike_intervals = np.random.exponential(scale=scale, size=size)
            spike_intervals = spike_intervals[refactory_period < spike_intervals]
            spike_times = np.concatenate([spike_times, last_spike_time + np.cumsum(spike_intervals)])
            if len(spike_times) > 0:
                last_spike_time = spike_times[-1]
        self.buffered_spike_times = spike_times[max_spike_time <= spike_times]
        spike_times = spike_times[spike_times < max_spike_time]
        spike_steps = spike_times * self.sampling_rate
        spike_steps = spike_steps.astype('int')

        return spike_steps
