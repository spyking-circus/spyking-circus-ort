# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys

from . import generate_probe, save_probe, load_probe


def pregenerate(working_directory=None, probe_file=None, template_directory=None, train_directory=None,
                duration=60.0, sampling_rate=20e+3, **kwargs):
    """Pregenerate signal."""
    # TODO complete docstring.

    # Find/generate probe.
    if probe_file is None:
        # Check if there is a probe file in the working directory.
        probe_file = os.path.join(working_directory, "config", "generation", "probe.prb")
        # TODO check if there is any .prb file not only a probe.prb file.
        if os.path.isfile(probe_file):
            # Load the probe.
            probe = load_probe(probe_file)
        else:
            # Generate the probe.
            probe = generate_probe()
    else:
        # Check if the probe file exists.
        if os.path.isfile(probe_file):
            # Load the probe.
            probe = load_probe(probe_file)
        else:
            # Raise an error.
            message = "No such probe file: {}".format(probe_file)
            raise OSError(message)

    # Save probe.
    probe_file = os.path.join(working_directory, "generation", "probe.prb")
    save_probe(probe_file, probe)

    # Find/generate templates.
    if template_directory is None:
        # Check if there is a template directory in the working directory.
        template_directory = os.path.join(working_directory, "config", "generation", "templates")
        if os.path.isdir(template_directory):
            # Load the templates.
            templates = load_templates(template_directory)
        else:
            # Generate the templates.
            templates = generate_templates(probe=probe)
    else:
        # Check if the template directory exists.
        if os.path.isdir(template_directory):
            # Load the templates.
            templates = load_templates(template_directory)
        else:
            # Raise an error.
            message = "No such template directory: {}".format(template_directory)
            raise OSError(message)

    # Save templates.
    template_directory = os.path.join(working_directory, "generation", "templates")
    save_templates(template_directory, templates)

    # Find/generate trains.
    if train_directory is None:
        # Check if there is a train directory in the working directory.
        train_directory = os.path.join(working_directory, "config", "generation", "trains")
        if os.path.isdir(train_directory):
            # Load the trains.
            trains = load_trains(train_directory)
        else:
            # Generate the trains.
            trains = generate_trains()
    else:
        # Check if the train directory exists.
        if os.path.isdir(train_directory):
            # Load the trains.
            trains = load_trains(train_directory)
        else:
            # Raise and error.
            message = "No such train directory: {}".format(train_directory)
            raise OSError(message)

    # Save trains.
    train_directory = os.path.join(working_directory, "generation", "trains")
    save_trains(train_directory, trains)

    # Find/generate signal.
    data_path = os.path.join(working_directory, "generation", "data.raw")
    nb_channels = probe.nb_channels
    nb_samples = int(duration * sampling_rate)
    shape = (nb_channels * nb_samples,)
    f = np.memmap(data_path, dtype=np.int16, mode='w+', shape=shape)
    nb_chunk_samples = 1024
    nb_chunks = nb_samples / nb_chunk_samples
    for k in range(0, nb_chunks):
        i_start = (k + 0) * nb_chunk_samples * nb_channels
        i_end = (k + 1) * nb_chunk_samples * nb_channels
        size = i_end - i_start
        data = np.random.normal(scale=10.0, size=size)
        data *= (32767.0 / 500.0)  # 16 bit integer: from -32768 to 32767.
        f[i_start:i_end] = data.astype(np.int16)
    if nb_samples % nb_chunk_samples > 0:
        i_start = nb_chunks * nb_chunk_samples * nb_channels
        i_end = nb_samples * nb_channels
        size = i_end - i_start
        data = np.random.normal(scale=10.0, size=size)
        data *= (32767.0 / 500.0)  # 16 bit integer: from -32768 to 32767.
        f[i_start:i_end] = data.astype(np.int16)
    for k, times in trains.iteritems():
        channels, waveforms = templates[k]
        for time in times:
            pass
            # TODO insert waveforms at corresponding channels and time.
            # steps = range(time - width / 2.0, time + width / 2.0)
            # waveforms = interpolate(waveforms, initial_steps, steps)
            # f[channels, steps] = waveforms

    return


def generate_waveform(width=5.0e-3, amplitude=-80.0, sampling_rate=20e+3):
    """Generate a waveform.

    Parameters:
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        amplitude: float (optional)
            Voltage amplitude [µV]. The default value is -80.0.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.

    Return:
        waveform: np.array
            Generated waveform.
    """

    i_start = - int(width * sampling_rate / 2.0)
    i_stop = + int(width * sampling_rate / 2.0)
    steps = np.arange(i_start, i_stop + 1)
    times = steps.astype('float32') / sampling_rate
    waveform = - np.cos(times / (width / 2.0) * (1.5 * np.pi))
    if np.amin(waveform) < - sys.float_info.epsilon:
        waveform /= - np.amin(waveform)
        waveform *= amplitude

    return waveform


def generate_templates(nb_templates=3, probe=None,
                       centers=None, max_amps=None,
                       radius=None, width=5.0e-3, sampling_rate=20e+3):
    """Generate templates.

    Parameters:
        nb_templates: none | integer (optional)
            Number of templates to generate. The default value is 3.
        probe: none | circusort.io.Probe
            Description of the probe (e.g. spatial layout). The default value is None.
        centers: none | list (optional)
            Coordinates of the centers (spatially) of the templates [µm]. The default value is None.
        max_amps: none | float (optional)
            Maximum amplitudes of the templates [µV]. The default value is None.
        radius: none | float (optional)
            Radius of the signal horizon [µm]. The default value is None.
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.

    Return:
        templates: dictionary
            Generated templates.
    """

    if probe is None:
        probe = generate_probe()

    if centers is None:
        centers = [(0.0, 0.0) for _ in range(0, nb_templates)]
        # TODO generate 'good' random centers.

    if max_amps is None:
        max_amps = [-80.0 for _ in range(0, nb_templates)]
        # TODO generate 'good' random maximum amplitudes.

    if radius is None:
        radius = probe.radius

    nb_samples = 1 + 2 * int(width * sampling_rate / 2.0)

    templates = {}
    for k in range(0, nb_templates):
        # Get distance to the nearest electrode.
        center = centers[k]
        nearest_electrode_distance = probe.get_nearest_electrode_distance(center)
        # Get channels before signal horizon.
        x, y = center
        channels, distances = probe.get_channels_around(x, y, radius + nearest_electrode_distance)
        # Declare waveforms.
        nb_electrodes = len(channels)
        shape = (nb_electrodes, nb_samples)
        waveforms = np.zeros(shape, dtype=np.float)
        # Initialize waveforms.
        amplitude = max_amps[k]
        waveform = generate_waveform(width=width, amplitude=amplitude, sampling_rate=sampling_rate)
        for i, distance in enumerate(distances):
            gain = (1.0 + distance / 100.0) ** -2.0
            waveforms[i, :] = gain * waveform
        # Store template.
        template = (channels, waveforms)
        templates[k] = template

    return templates


def save_templates(directory, templates):

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for k, template in templates.iteritems():
        channels, waveforms = template
        filename = "{}.h5".format(k)
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='w')
        f.create_dataset('channels', shape=channels.shape, dtype=channels.dtype, data=channels)
        f.create_dataset('waveforms', shape=waveforms.shape, dtype=waveforms.dtype, data=waveforms)
        f.close()

    return


def load_templates(directory):

    if not os.path.isdir(directory):
        message = "No such template directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()

    templates = {}

    for k, filename in enumerate(filenames):
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='r')
        channels = f['channels']
        waveforms = f['waveforms']
        f.close()
        template = (channels, waveforms)
        templates[k] = template

    return templates


def generate_trains(nb_trains=3, duration=60.0, rate=1.0, refractory_period=1.0e-3):
    """Generate trains.

    Parameters:
        nb_trains: integer (optional)
            Number of trains. The default value is 3.
        duration: float (optional)
            Train duration [s]. The default value is 60.0.
        rate: float (optional)
            Spike rate [Hz]. The default value is 1.0.
        refractory_period: float (optional)
            Refractory period [s]. The default value is 1.0e-3.
    """

    trains = {}

    for k in range(0, nb_trains):
        scale = 1.0 / rate
        time = 0.0
        train = []
        while time < duration:
            size = int((duration - time) * rate) + 1
            intervals = np.random.exponential(scale=scale, size=size)
            times = time + np.cumsum(intervals)
            train.append(times[times < duration])
            time = times[-1]
        train = np.concatenate(train)
        trains[k] = train

    return trains


def save_trains(directory, trains):
    """Save trains to files.

    Parameters:
        directory: string
        trains: dictionary
    """
    # TODO complete docstring.

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for k, times in trains.iteritems():
        filename = "{}.h5".format(k)
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='w')
        f.create_dataset('times', shape=times.shape, dtype=times.dtype, data=times)
        f.close()

    return


def load_trains(directory):
    """Load trains from files.

    Parameter:
        directory: string
    """

    if not os.path.isdir(directory):
        message = "No such train directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()

    trains = {}

    for k, filename in enumerate(filenames):
        path = os.path.join(directory, filename)
        f = h5py.File(path, mode='r')
        times = f['times']
        f.close()
        train = times
        trains[k] = train

    return trains
