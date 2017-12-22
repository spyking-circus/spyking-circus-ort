import argparse
import os

import circusort

from logging import DEBUG


# Parse command line.
parser = argparse.ArgumentParser()
parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
parser.add_argument('--introspection', dest='pending_introspection', action='store_true', default=None)
args = parser.parse_args()
if args.pending_generation is None and args.pending_sorting is None and args.pending_introspection is None:
    args.pending_generation = True
    args.pending_sorting = True
    args.pending_introspection = True
else:
    args.pending_generation = args.pending_generation is True
    args.pending_sorting = args.pending_sorting is True
    args.pending_introspection = args.pending_introspection is True


# Define the working directory.
directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling_0")
directory = os.path.expanduser(directory)
if not os.path.isdir(directory):
    os.makedirs(directory)
sorting_directory = os.path.join(directory, "sorting")
introspection_directory = os.path.join(directory, "introspection")


# Generate data (if necessary).
if args.pending_generation:

    circusort.net.pregenerator(working_directory=directory)


# Sort data (if necessary).
if args.pending_sorting:

    # Define parameters
    host = '127.0.0.1'  # i.e. run the test locally
    nb_channels = 16
    nb_samples = 1024
    sampling_rate = 20e+3  # Hz

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    if not os.path.isdir(introspection_directory):
        os.makedirs(introspection_directory)

    # Define keyword arguments.
    reader_kwargs = {
        'data_path': os.path.join(directory, "data.raw"),
        'dtype': 'int16',
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'introspection_path': introspection_directory,
    }
    signal_writer_kwargs = {
        'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
        'introspection_path': introspection_directory,
    }

    # Define the elements of the Circus network.
    director = circusort.create_director(host=host)
    manager = director.create_manager(host=host)
    reader = manager.create_block('reader', **reader_kwargs)
    writer = manager.create_block('writer', log_level=DEBUG, **signal_writer_kwargs)
    # Initialize the elements of the Circus network.
    director.initialize()
    # Connect the elements of the Circus network.
    director.connect(reader.output, writer.input)
    # or  # director.connect(reader.output, [reader.input])  # if the previous line does not work.
    # Launch the network.
    director.start()
    director.join()
    director.destroy()


# Introspect sorting (if necessary).
if args.pending_introspection:

    # Load time measurements from disk.
    measurements = circusort.io.load_time_measurements(introspection_directory, name='file_reader_1')
    print(measurements)

    # TODO evaluate real-time performances.
    # TODO plot real-time performances.

    raise NotImplementedError()  # TODO complete.
