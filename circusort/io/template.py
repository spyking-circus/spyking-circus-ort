# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys

from circusort.io import generate_probe

from circusort.obj.template import Template
from circusort.obj.position import Position


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


def generate_positions(nb_cells, probe):
    """Generate the positions of the cells.

    Parameters:
        nb_cells: integer
            The number of cells.
        probe: circusort.obj.Probe
            The probe.
    """

    fov = probe.field_of_view
    x_min = fov['x_min']
    x_max = fov['x_max']
    y_min = fov['y_min']
    y_max = fov['y_max']

    positions = []
    for _ in range(0, nb_cells):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        position = (x, y)
        positions.append(position)

    return positions


def generate_amplitudes(nb_cells):
    """Generate the amplitudes of the cells."""

    amplitudes = np.random.normal(80.0, scale=2.5, size=nb_cells)

    return amplitudes


def generate_template(probe=None, position=(0.0, 0.0), amplitude=80.0, radius=None,
                      width=5.0e-3, sampling_rate=20e+3, mode='default', **kwargs):
    """Generate a template.

    Parameters:
        probe: circusort.obj.Probe
            Description of the probe (e.g. spatial layout).
        position: tuple (optional)
            Coordinates of position of the center (spatially) of the template [µm]. The default value is (0.0, 0.0).
        amplitude: float (optional)
            Maximum amplitude of the template [µV]. The default value is 80.0.
        radius: none | float (optional)
            Radius of the signal horizon [µm]. The default value is None.
        width: float (optional)
            Temporal width [s]. The default value is 5.0e-3.
        sampling_rate: float (optional)
            Sampling rate [Hz]. The default value is 20e+3.
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
        for i, distance in enumerate(distances):
            gain = (1.0 + distance / 40.0) ** -2.0
            waveforms[i, :] = gain * waveform
        # Define template.
        template = Template(channels, waveforms)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return template


def generate_templates(nb_templates=3, probe=None,
                       positions=None, max_amps=None,
                       radius=None, width=5.0e-3, sampling_rate=20e+3):
    """Generate templates.

    Parameters:
        nb_templates: none | integer (optional)
            Number of templates to generate. The default value is 3.
        probe: none | circusort.io.Probe
            Description of the probe (e.g. spatial layout). The default value is None.
        positions: none | list (optional)
            Coordinates of position of the centers (spatially) of the templates [µm]. The default value is None.
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
            Generated dictionary of templates.
    """

    if probe is None:
        probe = generate_probe()

    if positions is None:
        positions = generate_positions(nb_templates, probe)

    if max_amps is None:
        max_amps = generate_amplitudes(nb_templates)

    if radius is None:
        radius = probe.radius

    templates = {}
    for k in range(0, nb_templates):
        position = positions[k]
        amplitude = max_amps[k]
        template = generate_template(probe=probe, position=position, amplitude=amplitude,
                                     radius=radius, width=width, sampling_rate=sampling_rate)
        templates[k] = template

    return templates


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


def save_templates(directory, templates, mode='default'):
    """Save templates.

    Parameters:
        directory: string
            Directory in which to save the templates.
        templates: dictionary
            Dictionary of templates.
        mode: string (optional)
            The mode to use to save the templates. Either 'default', 'by templates' or 'by cells'.
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if mode == 'default':

        raise NotImplementedError()  # TODO complete.

    elif mode == 'by templates':

        template_directory = os.path.join(directory, "templates")
        if not os.path.isdir(template_directory):
            os.makedirs(directory)
        for k, template in templates.iteritems():
            filename = "{}.h5".format(k)
            path = os.path.join(template_directory, filename)
            save_template(path, template)

    elif mode == 'by cells':

        for k, template in templates.iteritems():
            cell_directory = os.path.join(directory, "cells", "{}".format(k))
            if not os.path.isdir(cell_directory):
                os.makedirs(cell_directory)
            filename = "template.h5"
            path = os.path.join(cell_directory, filename)
            save_template(path, template)

    else:

        message = "Unknown mode value: {}".format(mode)
        raise ValueError(message)

    return


def list_templates(directory):
    """List template paths contained in the specified directory.

    Parameter:
        directory: string
            Directory from which to list the templates.

    Return:
        paths: list
            List of template paths found in the specified directory.
    """

    if not os.path.isdir(directory):
        message = "No such template directory: {}".format(directory)
        raise OSError(message)

    filenames = os.listdir(directory)
    filenames.sort()
    paths = [os.path.join(directory, filename) for filename in filenames]

    return paths


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
        raise OSError(message)

    f = h5py.File(path, mode='r')
    channels = f.get('channels').value
    waveforms = f.get('waveforms').value
    f.close()
    template = Template(channels, waveforms)

    return template


def load_templates(directory):
    """Load templates.

    Parameter:
        directory: string
            Directory from which to load the templates.

    Return:
        templates: dictionary
            Dictionary of templates.
    """

    # TODO implement two mode:
    # TODO   one with templates in multiple HDF5 file inside one directory
    # TODO   one with templates in one HDF5 file

    directory = os.path.expanduser(directory)
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        message = "No such template directory: {}".format(directory)
        raise OSError(message)

    paths = list_templates(directory)

    templates = {
        k: load_template(path)
        for k, path in enumerate(paths)
    }

    return templates


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

    if path is None:
        template = generate_template(**kwargs)
    elif not os.path.isfile(path):
        template = generate_template(**kwargs)
    else:
        try:
            template = load_template(path)
        except OSError:
            template = generate_template(**kwargs)

    return template
