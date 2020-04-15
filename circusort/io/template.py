# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys

from circusort.obj.template import Template, TemplateComponent
from circusort.obj.position import Position

if sys.version_info.major == 3:
    unicode = str


def generate_waveform(width=5.0e-3, amplitude=80.0, sampling_rate=20e+3):
    """Generate a waveform.

    Parameters:
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        amplitude: float (optional)
            Voltage amplitude [µV]. The default value is 80.0.
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
    gaussian = np.exp(- (times / (width / 4.0)) ** 2.0)
    waveform = np.multiply(waveform, gaussian)
    if np.min(waveform) < - sys.float_info.epsilon:
        waveform /= np.abs(np.min(waveform))
        waveform *= amplitude

    return waveform


def generate_template(probe=None, position=(0.0, 0.0), amplitude=80.0, radius=None,
                      width=5.0e-3, sampling_rate=20e+3, sparse_factor=0.5, mode='default', **kwargs):
    """Generate a template.

    Parameters:
        probe: circusort.obj.Probe
            Description of the probe (e.g. spatial layout).
        position: tuple | circusort.obj.Position (optional)
            Coordinates of position of the center (spatially) of the template [µm]. The default value is (0.0, 0.0).
        amplitude: float (optional)
            Maximum amplitude of the template [µV]. The default value is 80.0.
        radius: none | float (optional)
            Radius of the signal horizon [µm]. The default value is None.
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.
        sparse_factor: float (optional)
            Between 0-1: The number of channels that will be randomly set to 0. Default is 0.5
        mode: string (optional)
            Mode of generation. The default value is 'default'.

    Return:
        template: tuple
            Generated template.
    """

    assert probe is not None
    if isinstance(position, Position):
        position = position.get_initial_position()
    radius = probe.radius if radius is None else radius
    _ = kwargs

    if mode == 'default':

        # Compute the number of sampling times.
        nb_samples = 1 + 2 * int(width * sampling_rate / 2.0)
        # Get distance to the nearest electrode.
        nearest_electrode_distance = probe.get_nearest_electrode_distance(position)
        # Get channels before signal horizon.
        x, y = position
        channels, distances = probe.get_channels_around(x, y, radius + nearest_electrode_distance)
        # Declare waveforms.
        nb_electrodes = len(channels)
        shape = (nb_electrodes, nb_samples)
        waveforms = np.zeros(shape, dtype=np.float)
        # Initialize waveforms.
        waveform = generate_waveform(width=width, amplitude=amplitude, sampling_rate=sampling_rate)

        indices = np.arange(len(channels))
        if sparse_factor is not None:
            indices = np.random.permutation(indices)[:int(len(channels)*sparse_factor)]

        for i in indices:
            gain = (1.0 + distances[i] / 40.0) ** -2.0
            waveforms[i, :] = gain * waveform

        # Define template.
        first_component = TemplateComponent(waveforms, channels, probe.nb_channels, amplitudes=[0.8, 1.2])
        template = Template(first_component, channel=None, creation_time=0)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return template


def save_template(path, template):
    """Save template to file.

    Parameters:
        path: string
            The path to file in which to save the template.
        template: tuple
            The template to save.
    """

    template.save(path)

    return


def load_template(path):
    """Load template.

    Parameter:
        path: string
            Path from which to load the template.

    Return:
        template: tuple
            Template. The first element of the tuple contains the support of the template (i.e. channels). The second
            element contains the corresponding waveforms.
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        message = "No such template file: {}".format(path)
        raise IOError(message)

    f = h5py.File(path, mode='r')
    channels = f['indices'][()]
    waveforms = f['waveforms/1'][()]
    amplitudes = f['amplitudes'][()]
    channel = f.attrs['channel']
    creation_time = f.attrs['creation_time']
    nb_channels = f.attrs['nb_channels']
    compressed = f.attrs['compressed']

    first_component = TemplateComponent(waveforms, channels, nb_channels, amplitudes)
    second_component = None
    if '2' in f.get('waveforms').keys():
        waveforms = f['waveforms/2'][()]
        second_component = TemplateComponent(waveforms, channels, nb_channels)

    f.close()
    template = Template(first_component, channel, second_component, creation_time=creation_time,
                        compressed=compressed, path=path)

    return template


def get_template(path=None, **kwargs):
    """Get template.
    Parameter:
        path: none | string (optional)
            The path to use to get the template. The default value is None.
    Return:
        template: tuple
            The template to get.
    See also:
        circusort.io.generate_template (for additional parameters)
    """

    if not isinstance(path, (str, unicode)):
        template = generate_template(**kwargs)
    elif not os.path.isfile(path):
        template = generate_template(**kwargs)
    else:
        try:
            template = load_template(path)
        except IOError:
            template = generate_template(**kwargs)

    return template


def load_component_from_dict(template_dict, indices, nb_channels):

    if nb_channels is None:
        nb_channels = np.max(indices)

    waveforms = np.array(template_dict['wav'], dtype=np.float32)
    amplitudes = np.array(template_dict['amp'], dtype=np.float32)
    component = TemplateComponent(waveforms, indices, nb_channels, amplitudes)

    return component


def load_template_from_dict(template_dict, probe):

    channel = int(template_dict['channel'])
    creation_time = int(template_dict['time'])

    if 'compressed' in template_dict:
        indices = np.array(template_dict['compressed'], dtype=np.int32)
        compressed = True
    else:
        indices = probe.edges[channel]
        compressed = False

    first_component = load_component_from_dict(template_dict['0'], indices, probe.nb_channels)

    if '2' in template_dict:
        second_component = load_component_from_dict(template_dict['1'], indices, probe.nb_channels)
    else:
        second_component = None

    template = Template(first_component, channel=channel, second_component=second_component,
                        creation_time=creation_time, compressed=compressed)

    return template
