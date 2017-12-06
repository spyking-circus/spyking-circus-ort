import os

from circusort import io
from circusort.base import create_director


def find_or_generate_probe(probe_path, working_directory):
    """Find or generate probe to use during the pregeneration."""
    # TODO complete docstring.

    if probe_path is None:
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
    """Save probe to use during the pregeneration."""
    # TODO complete docstring.

    probe_path = os.path.join(working_directory, "generation", "probe.prb")
    io.save_probe(probe_path, probe)

    return


def find_or_generate_templates(template_directory, probe, working_directory):
    """Find or generate templates to use during the pregeneration."""
    # TODO complete docstring.

    if template_directory is None:
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
    """Save templates to use during the pregeneration."""
    # TODO complet docstring.

    template_directory = os.path.join(working_directory, "generation", "templates")
    io.save_templates(template_directory, templates)

    return


def find_or_generate_trains(train_directory, working_directory):
    """Find or generate trains to use during the pregeneration."""
    # TODO add docstring.

    if train_directory is None:
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

    train_directory = os.path.join(working_directory, "generation", "trains")
    io.save_trains(train_directory, trains)

    return


def pregenerator(working_directory=None, probe_path=None, template_directory=None, train_directory=None):
    """Pregenerate synthetic signal."""
    # TODO complete docstring.

    # Get probe.
    probe = find_or_generate_probe(probe_path, working_directory)
    save_probe(working_directory, probe)

    # Get templates.
    templates = find_or_generate_templates(template_directory, probe, working_directory)
    save_templates(working_directory, templates)

    # Get trains.
    trains = find_or_generate_trains(train_directory, working_directory)
    save_trains(working_directory, trains)

    # TODO uncomment the following lines.
    # # Generate signal.
    # host = '127.0.0.1'  # TODO keyword argument?
    # director = create_director(host=host)
    # manager = director.create_manager(host=host)
    # generator = manager.create_block('synthetic_generator')  # TODO give some parameters.
    # writer = manager.create_block('writer')  # TODO give some parameters.
    # director.initialize()
    # director.connect(generator.output, writer.input)
    # director.start()
    # director.sleep(duration=60.0)  # TODO test if we can remove this line.
    # director.join()  # TODO test if it returns after the completion of the pregeneration.

    raise NotImplementedError()
