import numpy as np
import os
from multiprocessing import Queue

from circusort.block.block import Block
from circusort.block.ort_displayer.gui_process import GUIProcess
from circusort.io.probe import load_probe
from circusort.io.template_store import load_template_store
from circusort.io.spikes import load_spikes

_ALL_BLOCKING_PIPES = ['data']
_ALL_NONBLOCKING_PIPES = ['templates', 'spikes', 'peaks']

_ALL_PIPES_ = ['params', 'number'] + _ALL_BLOCKING_PIPES #+ _ALL_NONBLOCKING_PIPES

__classname__ = "OrtDisplayer"


class OrtDisplayer(Block):
    """Peak displayer"""

    name = "OrtDisplayer"

    params = {
        'probe_path': None,
    }

    def __init__(self, **kwargs):
        """Initialization"""

        Block.__init__(self, **kwargs)

        for pipe in _ALL_PIPES_:
            self.add_input(pipe, structure='dict')
    
        self.all_queues = {}
        self._probe_path = self.probe_path
        
        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._sampling_rate = None
        self._number = 0

        return

    def _initialize(self):

        for pipe in _ALL_PIPES_:
            self.all_queues[pipe] = Queue()
        
        self._qt_process = GUIProcess(self.all_queues)
        self._qt_process.start()

        return


    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, sampling_rate=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels
        self._sampling_rate = sampling_rate

        self.all_queues['params'].put({
            'nb_samples': self._nb_samples,
            'sampling_rate': self._sampling_rate, 
            'probe_path' : self._probe_path
        })

        return


    def _process(self):

        self._measure_time(label='start', period=10)

        for pipe in _ALL_BLOCKING_PIPES:
            if pipe in _ALL_PIPES_:
                data_packet = self.get_input(pipe).receive()
                self.all_queues[pipe].put(data_packet['payload'])
                self._number = data_packet['number']

        self.all_queues['number'].put(self._number)

        for pipe in _ALL_NONBLOCKING_PIPES:
            if pipe in _ALL_PIPES_:
                data_packet = self.get_input(pipe).receive(blocking=False)
                if data_packet is not None:
                    self.all_queues[pipe].put(data_packet['payload'])

        self._measure_time(label='end', period=10)

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self._nb_samples) / self._sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return