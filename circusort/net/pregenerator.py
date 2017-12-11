import os

from circusort import io
from circusort.base import create_director


def find_or_generate_probe(probe_path=None, working_directory=None):
    """Find or generate probe to use during the pregeneration.

    Parameters:
        probe_path: none | string (optional)
            Path to the probe file. The default value is None.
        working_directory: none | string (optional)
            Path to the working directory. The default value is None.

    Return:
        probe: circusort.io.Probe
            Found or generated probe.
    """

    if probe_path is None:
        if working_directory is None:
            working_directory = os.path.join("~", ".spyking-circus-ort")
            working_directory = os.path.expanduser(working_directory)
        # Check if there is a probe file in the working directory.
        probe_path = os.path.join(working_directory, "config", "generation", "probe.prb")
        # TODO check if there is any .prb file not only a probe.prb file.
        if os.path.isfile(probe_path):
            # Load the probe.
            probe = io.load_probe(probe_path)
        else:
            # Generate the probe.
            probe = io.generate_probe()
    else:
        # Check if the probe file exists.
        if os.path.isfile(probe_path):
            # Load the probe.
            probe = io.load_probe(probe_path)
        else:
            # Raise an error.
            message = "No such probe file: {}".format(probe_path)
            raise OSError(message)

    return probe


def save_probe(working_directory, probe):
    """Save probe to use during the pregeneration.

    Parameters:
        working_directory: string
            Path to the working directory in which to save the probe.
        probe: circusort.io.Probe
            Probe.
    """

    probe_path = os.path.join(working_directory, "generation", "probe.prb")
    io.save_probe(probe_path, probe)

    return


def find_or_generate_templates(template_directory=None, probe=None, working_directory=None):
    """Find or generate templates to use during the pregeneration.

    Parameters:
        template_directory: none | string (optional)
            Path to the template directory from which to load the templates. The default value is None.
        probe: none | circusort.io.Probe (optional)
            Probe. The default value is None.
        working_directory: none | string (optional)
            Path to the working directory from which to load the templates and in which to save the templates. The
            default value is None.

    Return:
        templates: dictionary
            Found or generated dictionary of templates.
    """

    if template_directory is None:
        if working_directory is None:
            working_directory = os.path.join("~", ".spyking-circus-ort")
            working_directory = os.path.expanduser(working_directory)
        # Check if there is a template directory in the working directory.
        template_directory = os.path.join(working_directory, "config", "generation", "templates")
        if os.path.isdir(template_directory):
            # Load the templates.
            templates = io.load_templates(template_directory)
        else:
            # Generate the templates.
            templates = io.generate_templates(probe=probe)
    else:
        # Check if the template directory exists.
        if os.path.isdir(template_directory):
            # Load the templates.
            templates = io.load_templates(template_directory)
        else:
            # Raise an error.
            message = "No such template directory: {}".format(template_directory)
            raise OSError(message)

    return templates


def save_templates(working_directory, templates):
    """Save templates to use during the pregeneration.

    Parameters:
        working_directory: string
            Path to the working directory in which to save the templates.
        templates: dictionary
            Dictionary of templates to save.
    """

    template_directory = os.path.join(working_directory, "generation", "templates")
    io.save_templates(template_directory, templates)

    return


def find_or_generate_trains(train_directory=None, working_directory=None):
    """Find or generate trains to use during the pregeneration.

    Parameters:
        train_directory: none | string (optional)
            Path to the train directory from which to load the trains. The default value is None.
        working_directory: none | string (optional)
            Path to the working directory from which to load the trains and in which to save the trains. The default
            value is None.

    Return:
        trains: dictionary
            Found or generated dictionary of trains.
    """

    if train_directory is None:
        if working_directory is None:
            working_directory = os.path.join("~", ".spyking-circus-ort")
            working_directory = os.path.expanduser(working_directory)
        # Check if there is a train directory in the working directory.
        train_directory = os.path.join(working_directory, "config", "generation", "trains")
        if os.path.isdir(train_directory):
            # Load the trains.
            trains = io.load_trains(train_directory)
        else:
            # Generate the trains.
            trains = io.generate_trains()
    else:
        # Check if the train directory exists.
        if os.path.isdir(train_directory):
            # Load the trains.
            trains = io.load_trains(train_directory)
        else:
            # Raise and error.
            message = "No such train directory: {}".format(train_directory)
            raise OSError(message)

    return trains


def save_trains(working_directory, trains):
    """Save trains to use during the pregeneration.

    Parameters:
        working_directory: string
            Path to the working directory in which to save the trains.
        trains: dictionary
            Dictionary of trains to save.
    """

    train_directory = os.path.join(working_directory, "generation", "trains")
    io.save_trains(train_directory, trains)

    return


def pregenerator(working_directory=None, probe_path=None, template_directory=None, train_directory=None):
    """Pregenerate synthetic signal.

    Parameters:
        working_directory: none | string (optional)
        probe_path: none | string (optional)
        template_directory: none | string (optional)
        train_directory: none | string (optional)
    """
    # TODO complete docstring.

    # Retrieve probe.
    probe = find_or_generate_probe(probe_path, working_directory)
    save_probe(working_directory, probe)

    # Retrieve templates.
    templates = find_or_generate_templates(template_directory, probe, working_directory)
    save_templates(working_directory, templates)

    # Retrieve trains.
    trains = find_or_generate_trains(train_directory, working_directory)
    save_trains(working_directory, trains)

    # Generate signal.
    host = '127.0.0.1'  # TODO correct IP address? & transform into a keyword argument?
    data_path = os.path.join(working_directory, "data.raw")
    director = create_director(host=host)
    manager = director.create_manager(host=host)
    generator = manager.create_block('synthetic_generator', working_directory=working_directory)
    writer = manager.create_block('writer', data_path=data_path)
    director.initialize()
    director.connect(generator.output, writer.input)
    director.start()
    director.join()
    director.stop()

    return
