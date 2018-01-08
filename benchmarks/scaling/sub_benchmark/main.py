import argparse
import os

import circusort

from logging import DEBUG


def main():

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
    configuration_directory = os.path.join(directory, "configuration")

    # Load configurations.
    configurations = circusort.io.get_configurations(configuration_directory)

    # Process each configuration.
    for configuration in configurations:

        name = configuration['general']['name']

        configuration_directory = os.path.join(directory, "configuration", name)
        generation_directory = os.path.join(directory, "generation", name)
        sorting_directory = os.path.join(directory, "sorting", name)
        introspection_directory = os.path.join(directory, "introspection", name)
        output_directory = os.path.join(directory, "output", name)

        # Generate data (if necessary).
        if args.pending_generation:

            circusort.net.pregenerator(configuration_directory=configuration_directory,
                                       generation_directory=generation_directory)

        # Sort data (if necessary).
        if args.pending_sorting:

            # Load generation parameters.
            parameters = circusort.io.get_data_parameters(generation_directory)

            # Define parameters.
            host = '127.0.0.1'  # i.e. run the test locally
            dtype = parameters['general']['dtype']
            nb_channels = parameters['probe']['nb_channels']
            nb_samples = parameters['general']['buffer_width']
            sampling_rate = parameters['general']['sampling_rate']

            # Create directories (if necessary).
            if not os.path.isdir(sorting_directory):
                os.makedirs(sorting_directory)
            if not os.path.isdir(introspection_directory):
                os.makedirs(introspection_directory)

            # Define keyword arguments.
            reader_kwargs = {
                'data_path': os.path.join(generation_directory, "data.raw"),
                'dtype': dtype,
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

            # Load generation parameters.
            parameters = circusort.io.get_data_parameters(generation_directory)

            # Define parameters.
            nb_samples = parameters['general']['buffer_width']
            sampling_rate = parameters['general']['sampling_rate']
            duration_buffer = float(nb_samples) / sampling_rate
            nb_channels = parameters['probe']['nb_channels']

            # Load time measurements from disk.
            measurements_reader = circusort.io.load_time_measurements(introspection_directory, name='file_reader_1')
            durations_reader = measurements_reader['end'] - measurements_reader['start']
            speed_factors_reader = duration_buffer / durations_reader
            measurements_writer = circusort.io.load_time_measurements(introspection_directory, name='file_writer_1')
            durations_writer = measurements_writer['end'] - measurements_writer['start']
            speed_factors_writer = duration_buffer / durations_writer

            # TODO evaluate real-time performances.
            data = [speed_factors_reader, speed_factors_writer]
            labels = ['reader', 'writer']

            # Print the number of observations for each dataset.
            for data_, label in zip(data, labels):
                n = len(data_)
                print("{}: n={}".format(label, n))

            # TODO plot real-time performances.
            import matplotlib.pyplot as plt

            flierprops = {
                'marker': 's',
                'markersize': 1,
                'markerfacecolor': 'k',
                'markeredgecolor': 'k',
            }
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            output_path = os.path.join(output_directory, "real_time_performances.pdf")

            plt.ioff()
            plt.style.use('seaborn-paper')

            fig, ax = plt.subplots(1, 1)
            ax.boxplot(data, notch=True, whis=1.5, labels=labels, flierprops=flierprops)
            ax.set_ylabel("speed factor")
            ax.set_title("Real-time performances ({} electrodes)".format(nb_channels))
            fig.tight_layout()
            fig.savefig(output_path)

            plt.close(fig)


if __name__ == '__main__':

    main()
