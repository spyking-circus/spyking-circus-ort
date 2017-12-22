import argparse
import os

import circusort

from logging import DEBUG


# TODO correct the following lines.
# Parse command line.
parser = argparse.ArgumentParser()
parser.add_argument('--no-realism', dest='is_realistic', action='store_false', default=True)
parser.add_argument('--no-sorting', dest='skip_sorting', action='store_true', default=False)
parser.add_argument('--init-temp-dict', dest='init_temp_dict', action='store_true')
args = parser.parse_args()


# Define parameters
host = '127.0.0.1'  # i.e. run the test locally
nb_channels = 16
nb_samples = 1024
sampling_rate = 20e+3  # Hz
directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling_0")
directory = os.path.expanduser(directory)
if not os.path.isdir(directory):
    message = "Directory does not exist: {}".format(directory)
    raise OSError(message)
sorting_directory = os.path.join(directory, "sorting")
if not os.path.isdir(sorting_directory):
    os.makedirs(sorting_directory)
# TODO complete.


# Define keyword arguments.
reader_kwargs = {
    'data_path': os.path.join(directory, "data.raw"),
    'dtype': 'int16',
    'nb_channels': nb_channels,
    'nb_samples': nb_samples,
    'sampling_rate': sampling_rate,
    'is_realistic': args.is_realistic,
}
signal_writer_kwargs = {
    'data_path': os.path.join(sorting_directory, "data_filtered.raw"),
}


# Define the elements of the Circus network.
director = circusort.create_director(host=host)
manager = director.create_manager(host=host)
reader = manager.create_block('reader', **reader_kwargs)
writer = manager.create_block('writer', log_level=DEBUG, **signal_writer_kwargs)
# Initialize the elements of the Circus network.
director.initialize()
# Connect the elements of the Circus network.
director.connect(reader.output, reader.input)
# or  # director.connect(reader.output, [reader.input])  # if the previous line does not work.


# Analyze the real-time performances.
# TODO complete.
