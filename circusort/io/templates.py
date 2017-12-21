# -*- coding: utf-8 -*-

import numpy as np
import os

from circusort.io.probe import generate_probe
from circusort.io.template import generate_template, save_template, load_template
from circusort.io.template_store import load_template_store
from circusort.utils.path import normalize_path


def generate_positions(nb_cells, probe):
    """Generate the positions of the cells.

    Parameters:
        nb_cells: integer
            The number of cells.
        probe: circusort.obj.Probe
            The probe.
    Return:
        positions: list
            The generated positions of the cells.
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
    """Generate the amplitudes of the cells.

    Parameter:
        nb_cells: integer
            The number of cells.
    Return:
        amplitudes: numpy.ndarray
            The generated amplitudes of the cells.
    """

    amplitudes = np.random.normal(80.0, scale=2.5, size=nb_cells)

    return amplitudes


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


def _load_templates_from_file(path):

    template_store = load_template_store(path)
    templates = {
        k: template_store[k].to_template()  # TODO be able to avoid the call to to_template.
        for k in template_store
    }

    return templates


def _load_templates_from_directory(path):

    paths = list_templates(path)
    templates = {
        k: load_template(path)
        for k, path in enumerate(paths)
    }

    return templates


def load_templates(path):
    """Load templates.

    Parameter:
        path: string
            The path to the location from which to load the templates. Either a path to a directory (which contains
            multiple HDF5 files) or a HDF5 file.
    Return:
        templates: dictionary
            The dictionary of loaded templates.
    """

    path = normalize_path(path)
    if os.path.isfile(path):
        templates = _load_templates_from_file(path)
    elif os.path.isdir(path):
        templates = _load_templates_from_directory(path)
    else:
        message = "No such template file or directory: {}".format(path)
        raise OSError(message)

    return templates
