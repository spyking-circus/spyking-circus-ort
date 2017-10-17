import matplotlib.pyplot as plt
import numpy as np
import os

from circusort import io
from circusort.io.synthetic import SyntheticStore


class Results(object):
    """Results of the scenario"""
    # TODO complete docstring.

    def __init__(self, generator, spike_writer, probe_path, temp_path):

        # Save raw input arguments.
        self.generator = generator
        self.spike_writer = spike_writer
        self.probe_path = probe_path
        self.temp_path = temp_path

        # TODO implement the function 'load_generation'.
        # # Retrieve generation.
        # self.gen = io.load_generation(generator)
        gen_path = os.path.abspath(self.generator.hdf5_path)
        self.gen = SyntheticStore(gen_path)

        # TODO correct the following line.
        self.sampling_rate = 20.0e+3  # [Hz]

        # Retrieve spike attributes.
        # # Retrieve spike times.
        spike_steps_path = self.spike_writer.recorded_data['spike_times']
        self.spike_steps = np.fromfile(spike_steps_path, dtype=np.int32)
        self.spike_times = self.spike_steps.astype(np.float32) / self.sampling_rate
        # # Retrieve spike templates.
        spike_templates_path = self.spike_writer.recorded_data['templates']
        self.spike_templates = np.fromfile(spike_templates_path, dtype=np.int32)
        # # Retrieve spike amplitudes.
        spike_amplitudes_path = self.spike_writer.recorded_data['amplitudes']
        self.spike_amplitudes = np.fromfile(spike_amplitudes_path, dtype=np.float32)

        # Retrieve probe.
        self.probe = io.Probe(self.probe_path)

    def compare_spike_trains(self):
        """Compare spike trains.

        """

        # TODO retrieve the generated spike train.
        generated_spike_train = self.gen.get(variables='spike_times')
        generated_spike_train = generated_spike_train[u'0']['spike_times']
        generated_spike_train = generated_spike_train.astype(np.float32) / self.sampling_rate

        # TODO retrieve the inferred spike trains.
        inferred_spike_trains = {}
        for template_id in np.unique(self.spike_templates):
            inferred_spike_train = self.spike_times[self.spike_templates == template_id]
            inferred_spike_trains[template_id] = inferred_spike_train

        # TODO plot spike trains to compare them visually.
        plt.figure()
        # Plot inferred spike trains.
        for k, template_id in enumerate(inferred_spike_trains):
            x = [t for t in inferred_spike_trains[template_id]]
            y = [float(k) for _ in x]
            plt.scatter(x, y, c='b')
        # Plot generated spike train.
        x = [t for t in generated_spike_train]
        y = [float(len(inferred_spike_trains)) for _ in x]
        plt.scatter(x, y, c='r')
        plt.xlabel("time (arb. unit)")
        plt.ylabel("spike train")
        plt.title("Spike trains comparison")
        plt.show()

        return
