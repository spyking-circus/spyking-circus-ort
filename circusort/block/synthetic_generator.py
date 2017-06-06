import numpy as np
import Queue
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
        'sampling_rate' : 20000,
        'nb_samples'    : 2000,
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

        self.output.configure(dtype=self.dtype, shape=(self.nb_channels, self.nb_samples))

        # First queue is used as a buffer for synthetic data.
        self.queue = Queue.Queue(maxsize=600)
        # Second queue is a communication mechanism between the main and
        # background threads in order to be able to stop the background thread.
        self.rpc_queue = Queue.Queue()

        def syn_gen_target(rpc_queue, queue, nb_channels, nb_samples):
            '''Synthetic data generation (background thread)'''
            while rpc_queue.empty(): # check if main thread requires a stop
                if not queue.full(): # limit memory consumption
                    data = np.random.randn(nb_channels, nb_samples)
                    queue.put(data)
            # TODO complete.
            return

        # Declare background thread for data generation.
        args = (self.rpc_queue, self.queue, self.nb_channels, self.nb_samples)
        self.syn_gen_thread = threading.Thread(target=syn_gen_target, args=args)
        self.syn_gen_thread.deamon = True

        # Launch background thread for data generation.
        self.log.info("Launch background thread for data generation...")
        self.syn_gen_thread.start()

        return

    def _process(self):
        '''TODO add docstring.'''

        # Get data from background thread.
        data = self.queue.get()
        # Simulate duration between two data acquisitions.
        time.sleep(self.nb_samples / self.sampling_rate)
        # Send data.
        self.output.send(data)

        return

    def __del__(self):

        # Require a stop from the background thread.
        self.rpc_queue.put("stop")
