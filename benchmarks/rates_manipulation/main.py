import argparse
import os

import circusort

from logging import DEBUG, INFO

from circusort.io.cells import list_cells


nb_rows = 10
nb_columns = 10
nb_cells = 100
duration = 10 * 60# s
preload_templates = False
nb_waveforms_clustering = 100


def main():

    # Parse command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', dest='pending_generation', action='store_true', default=None)
    parser.add_argument('--sorting', dest='pending_sorting', action='store_true', default=None)
    args = parser.parse_args()
    if args.pending_generation is None and args.pending_sorting is None:
        args.pending_generation = True
        args.pending_sorting = True
    else:
        args.pending_generation = args.pending_generation is True
        args.pending_sorting = args.pending_sorting is True

    # Define the working directory.
    directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "rates_manipulation")
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    configuration_directory = os.path.join(directory, "configuration")
    if not os.path.isdir(configuration_directory):
        os.makedirs(configuration_directory)
        # TODO remove the following commented lines.
        # # Define probe path.
        # probe_path = os.path.join(configuration_directory, "probe.prb")
        # # Generate probe.
        # probe = circusort.io.generate_probe(mode='mea', nb_rows=nb_rows, nb_columns=nb_columns)
        # # Save probe.
        # probe.save(probe_path)
        # Define cells directory.
        cells_directory = os.path.join(configuration_directory, "cells")
        # Generate configuration.
        kwargs = {
            'general': {
                'duration': duration,
            },
            'probe': {
                'mode': 'mea',
                'nb_rows': nb_rows,
                'nb_columns': nb_columns,
            },
            'cells': {
                'mode': "default",
                'nb_cells': nb_cells,
                'path': cells_directory,
            }
        }
        configuration = circusort.io.generate_configuration(**kwargs)
        # Save configuration.
        configuration.save(configuration_directory)
        # Create cells directory.
        os.makedirs(cells_directory)
        # cells_parameters = circusort.io.generate_cells_parameters()  # TODO enable this line.
        # cells_parameters.save(cells_directory)  # TODO enable this line.
        # For each cell...
        for k in range(0, nb_cells):
            # Define cell directory.
            cell_directory = os.path.join(cells_directory, str(k))
            cell_parameters = [
                ('train', [
                    ('rate', "2 + 5.0*(t > %g)" %((k + 0.5)*(2*duration/3.)/float(nb_cells))),
                ]),
                ('position', []),  # TODO be able to remove this line.
                ('template', []),  # TODO be able to remove this line.
            ]
            cell_parameters = circusort.obj.Parameters(cell_parameters)
            cell_parameters.save(cell_directory)

    # Define directories.
    generation_directory = os.path.join(directory, "generation")
    sorting_directory = os.path.join(directory, "sorting")

    # Generate data (if necessary).
    if args.pending_generation:

        circusort.net.pregenerator(configuration_directory=configuration_directory,
                                   generation_directory=generation_directory)

    # Sort data (if necessary).
    if args.pending_sorting:

        # Load generation parameters.
        parameters = circusort.io.get_data_parameters(generation_directory)
        introspect_path = os.path.join(directory, 'introspection')
        # Define parameters.
        host = '127.0.0.1'  # i.e. run the test locally
        dtype = parameters['general']['dtype']
        nb_channels = parameters['probe']['nb_channels']
        nb_samples = parameters['general']['buffer_width']
        sampling_rate = parameters['general']['sampling_rate']
        threshold_factor = 7.0
        probe_path = os.path.join(generation_directory, "probe.prb")
        probe = circusort.io.load_probe(probe_path)
        precomputed_template_paths = [
            os.path.join(e, 'template.h5')
            for e in list_cells(os.path.join(generation_directory, 'cells'))
        ]

        # Create sorting directory (if necessary).
        if not os.path.isdir(sorting_directory):
            os.makedirs(sorting_directory)

        # Define keyword arguments.
        reader_kwargs = {
            'name': "reader",
            'data_path': os.path.join(generation_directory, "data.raw"),
            'dtype': dtype,
            'nb_channels': nb_channels,
            'nb_samples': nb_samples,
            'sampling_rate': sampling_rate,
            'is_realistic': True,
        }
        filter_kwargs = {
            'name': "filter",
            'cut_off': 0.1,  # Hz
            'log_level': DEBUG,
        }
        mad_kwargs = {
            'name': "mad",
            'time_constant': 10.0,
            'log_level': DEBUG,
        }
        detector_kwargs = {
            'name': "detector",
            'threshold_factor': threshold_factor,
            'sampling_rate': sampling_rate,
            'log_level': DEBUG,
        }
        pca_kwargs = {
            'name': "pca",
            'nb_waveforms': 10000,
            'log_level': DEBUG,
        }
        cluster_kwargs = {
            'name': "cluster",
            'threshold_factor': threshold_factor,
            'sampling_rate': sampling_rate,
            'nb_waveforms': nb_waveforms_clustering,
            'probe_path': probe_path,
            'two_components': False,
            'log_level': INFO
        }
        if preload_templates:
            cluster_kwargs['channels'] = []

        updater_kwargs = {
            'name': "updater",
            'probe_path': probe_path,
            'data_path': os.path.join(sorting_directory, "templates.h5"),
            'sampling_rate': sampling_rate,
            'nb_samples': nb_samples,
            'log_level': DEBUG,
        }

        if preload_templates:
            updater_kwargs['precomputed_template_paths'] = precomputed_template_paths


        fitter_kwargs = {
            'name': "fitter",
            'sampling_rate': sampling_rate,
            'log_level': DEBUG,
            'introspection_path': introspect_path
        }
        writer_kwargs = {
            'name': "writer",
            'data_path': os.path.join(sorting_directory, "spikes.h5"),
            'sampling_rate': sampling_rate,
            'nb_samples': nb_samples,
            'log_level': DEBUG,
        }

        # Define the elements of the network.
        director = circusort.create_director(host=host)
        manager = director.create_manager(host=host)
        reader = manager.create_block('reader', **reader_kwargs)
        filter_ = manager.create_block('filter', **filter_kwargs)
        mad = manager.create_block('mad_estimator', **mad_kwargs)
        detector = manager.create_block('peak_detector', **detector_kwargs)
        pca = manager.create_block('pca', **pca_kwargs)
        cluster = manager.create_block('density_clustering', **cluster_kwargs)
        updater = manager.create_block('template_updater', **updater_kwargs)
        fitter = manager.create_block('template_fitter', **fitter_kwargs)
        writer = manager.create_block('spike_writer', **writer_kwargs)
        # Initialize the elements of the network.
        director.initialize()
        # Connect the elements of the network.
        director.connect(reader.output, [
            filter_.input
        ])
        director.connect(filter_.output, [
            mad.input,
            detector.get_input('data'),
            pca.get_input('data'),
            cluster.get_input('data'),
            fitter.get_input('data'),
        ])
        director.connect(mad.output, [
            detector.get_input('mads'),
            cluster.get_input('mads'),
        ])
        director.connect(detector.get_output('peaks'), [
            pca.get_input('peaks'),
            cluster.get_input('peaks'),
            fitter.get_input('peaks'),
        ])
        director.connect(pca.get_output('pcs'), [
            cluster.get_input('pcs'),
        ])
        director.connect(cluster.get_output('templates'), [
            updater.get_input('templates'),
        ])
        director.connect(updater.get_output('updater'), [
            fitter.get_input('updater'),
        ])
        director.connect(fitter.output, [
            writer.input,
        ])
        # Launch the network.
        director.start()
        director.join()
        director.destroy()

    return


if __name__ == '__main__':

    main()
