import os

from circusort import io
from circusort.base import create_director

def find_or_generate_probe(path=None, directory=None):
    """Find or generate probe to use during the pregeneration.

    Parameters:
        path: none | string (optional)
            Path to the probe file. The default value is None.
        directory: none | string (optional)
            Path to the probe directory. The default value is None.

    Return:
        probe: circusort.io.Probe
            Found or generated probe.
    """

    if path is None:
        if directory is None:
            directory = os.path.join("~", ".spyking-circus-ort", "probes")
            directory = os.path.expanduser(directory)
        # Check if there is a probe file in the directory.
        path = os.path.join(directory, "probe.prb")
        # TODO check if there is any .prb file not only a probe.prb file.
        if os.path.isfile(path):
            # Load the probe.
            probe = io.load_probe(path)
        else:
            # Generate the probe.
            probe = io.generate_probe()
    else:
        # Check if the probe file exists.
        if os.path.isfile(path):
            # Load the probe.
            probe = io.load_probe(path)
        else:
            # Raise an error.
            message = "No such probe file: {}".format(path)
            raise OSError(message)

    return probe


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
        template_directory = os.path.join(working_directory, "configuration", "generation", "templates")
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
        train_directory = os.path.join(working_directory, "configuration", "generation", "trains")
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


def find_or_generate_parameters(path=None, working_directory=None):
    """Find or generate parameters to use during the generation.

    Parameters:
        path: none | string (optional)
            Path to the parameters file. The default value is None.
        working_directory: none | string (optional)
            Path to the working directory. The default value is None.

    Return:
        parameters: dictionary
            Found or generated parameters.
    """

    if path is None:
        if working_directory is None:
            working_directory = os.path.join("~", ".spyking-circus-ort")
            working_directory = os.path.expanduser(working_directory)
        # Check if there is a parameters file in the working directory.
        path = os.path.join(working_directory, "configuration", "generation", "parameters.txt")
        # TODO check if there is any other file not only a "parameters.txt" file.
        if os.path.isfile(path):
            # Load the parameters.
            parameters = io.load_parameters(path)
        else:
            # Generate the parameters.
            parameters = io.generate_parameters()
    else:
        # Check if the parameters file exists.
        if os.path.isfile(path):
            # Load the parameters.
            parameters = io.load_parameters(path)
        else:
            # Raise an error.
            message = "No such parameters file: {}".format(path)
            raise OSError(message)

    return parameters


def save_parameters(working_directory, parameters):
    """Save parameters to use during the generation.

    Parameters:
        working_directory: string
            Path to the working directory in which to save the parameters file.
        parameters: dictionary
            Parameters to save.
    """

    path = os.path.join(working_directory, "generation", "parameters.txt")
    io.save_parameters(path, parameters)

    return


def pregenerator(working_directory=None, configuration_directory=None, generation_directory=None):
    """Pregenerate synthetic signal.

    Parameters:
        working_directory: none | string (optional)
        configuration_directory: none | string (optional)
        generation_directory: none | string (optional)
    """
    # TODO complete docstring.

    # Define the paths to the configuration and generation directories (if necessary).
    if working_directory is None:
        working_directory = os.getcwd()
    if configuration_directory is None:
        configuration_directory = os.path.join(working_directory, "configuration")
    if generation_directory is None:
        generation_directory = os.path.join(working_directory, "generation")

    # Get parameters.
    parameters = io.get_data_parameters(configuration_directory)

    probe = io.get_probe(**parameters['probe'])
    probe.save(generation_directory)
    probe.plot(generation_directory)
    parameters['probe'] = probe.get_parameters()

    cells = io.get_cells(probe=probe, **parameters['cells'])
    cells.save(generation_directory)
    cells.plot(generation_directory, probe=probe)
    parameters['cells'] = cells.get_parameters()

    # Save parameters.
    parameters.save(generation_directory)

    # Generate signal.
    host = '127.0.0.1'  # TODO correct IP address? & transform into a keyword argument?
    data_path = os.path.join(generation_directory, "data.raw")
    director = create_director(host=host)
    manager = director.create_manager(host=host)
    generator = manager.create_block('synthetic_generator', working_directory=generation_directory, is_realistic=False)
    writer = manager.create_block('writer', data_path=data_path)
    director.initialize()
    director.connect(generator.output, writer.input)
    director.start()
    director.join()
    director.stop()

    return
