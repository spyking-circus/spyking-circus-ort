# -*- coding: utf-8 -*-

import numpy as np
import os

from . import generate_probe, save_probe, load_probe


def pregenerate(working_directory=None, probe_file=None, template_directory=None, **kwargs):

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
            templates = generate_templates()
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

    # TODO find/generate noise
    # e.g. "config/generation/noise.h5"

    # TODO find/generate spike trains
    # e.g. "config/generation/trains/train_0.h5

    # TODO find/generate signal
    # e.g. "config/generation/matching.h5"

    return


def load_templates(directory):

    # TODO load templates from all the .h5 files contained in the given directory.

    templates = np.array([])

    return templates


def generate_templates(probe=None, center=None, radius=None,
                       width=None, sampling_rate=None,
                       max_amp=None):
    """Generate templates.

    Parameters:
        probe: none | probe (optional)
            Description of the probe (e.g. spatial layout).
        center: none | tuple (optional)
            Coordinate of the center (spatially) of the template [µm x µm].
        radius: none | float (optional)
            Radius of the signal horizon [µm].
        width: none | float (optional)
            Temporal width [ms].
        sampling_rate: none | float (optional)
            Sampling rate [Hz].
        max_amp: none | float (optional)
            Maximum amplitude [µV].

    Return:
        templates: np.array
            Generated templates.
    """

    # TODO generate templates (according to which set of parameters)?

    templates = np.array([])

    return templates


def save_templates(directory, templates):

    if not os.path.isdir(directory):
        os.makedirs(directory)

    # TODO save one .h5 file for each template in `templates`.

    return
