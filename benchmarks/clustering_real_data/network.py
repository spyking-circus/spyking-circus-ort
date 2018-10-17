import os

import circusort

from logging import INFO
from logging import DEBUG


user_directory = os.path.expanduser("~")
circus_directory = os.path.join(user_directory, ".spyking-circus-ort")
directory = os.path.join(circus_directory, "benchmarks", "clustering_real_data")

# # 1. Parameters for the 9 electrodes & 5 minutes version.
# nb_channels = 9
# hosts = {
#     'master': '127.0.0.1',
# }
# hosts_keys = [  # ordered
#     'master',
# ]
# managers_keys = {
#     'reader': 'master',
#     'filter': 'master',
#     #'writer': 'master',
#     'mad': 'master',
#     'detector': 'master',
#     'peak_writer': 'master',
#     'pca': 'master',
#     'cluster': 'master',
#     # 'cluster_writer': 'master',
#     'updater': 'master',
#     # 'updater_writer': 'master',
#     'fitter': 'master',
#     'spike_writer': 'master',
# }
# 2-3. Parameters for the 252 electrodes & 5 or 30 minutes version.
nb_channels = 252
hosts = {
    'master': '192.168.0.254',
    'slave_1': '192.168.0.1',
    'slave_2': '192.168.0.2',
    'slave_3': '192.168.0.3',
}
hosts_keys = [  # ordered
    'master',
    'slave_1',
    'slave_2',
    'slave_3',
]
managers_keys = {
    'reader': 'master',
    'filter': 'slave_1',
    # 'writer': 'slave_1',
    'mad': 'slave_1',
    'detector': 'slave_1',
    'peak_writer': 'slave_1',
    'pca': 'slave_1',
    'cluster': 'slave_2',
    # 'cluster_writer': 'slave_2',
    'updater': 'slave_2',
    # 'updater_writer': 'slave_2',
    'fitter': 'slave_3',
    'spike_writer': 'master',
}


def sorting(nb_waveforms_clustering=1000):
    """Create the sorting network."""

    # Define directories.
    recording_directory = os.path.join(directory, "recording")
    sorting_directory = os.path.join(directory, "sorting")
    # introspection_directory = os.path.join(directory, "introspection")
    log_directory = os.path.join(directory, "log")
    output_directory = os.path.join(directory, "output")
    debug_directory = os.path.join(directory, "debug")

    # Define parameters.
    dtype = 'uint16'
    nb_samples = 1024
    sampling_rate = 20e+3
    threshold_factor = 7.0
    alignment = True
    spike_width = 5.0  # ms
    spike_jitter = 1.0  # ms
    spike_sigma = 2.75  # µV
    probe_path = os.path.join(recording_directory, "probe.prb")
    nb_fitters = 2

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    # if not os.path.isdir(introspection_directory):
    #     os.makedirs(introspection_directory)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # Define keyword arguments.
    director_kwargs = {
        'log_path': os.path.join(log_directory, "log.txt"),
        'log_level': INFO,
    }
    reader_kwargs = {
        'name': "reader",
        'data_path': os.path.join(recording_directory, "data.raw"),
        'dtype': dtype,
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'speed_factor': 1.0,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    filter_kwargs = {
        'name': "filter",
        'cut_off': 500.0,  # Hz
        'order': 1,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    # writer_kwargs = {
    #     'name': "writer",
    #     'data_path': os.path.join(recording_directory, "filtered_data.raw"),
    #     # 'introspection_path': introspection_directory,
    #     'log_level': DEBUG,
    # }
    mad_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    detector_kwargs = {
        'name': "detector",
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    peak_writer_kwargs = {
        'name': "peak_writer",
        'data_path': os.path.join(sorting_directory, "peaks.h5"),
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    pca_kwargs = {
        'name': "pca",
        'spike_width': spike_width,
        'spike_jitter': spike_jitter,
        'spike_sigma': spike_sigma,
        'alignment': alignment,
        'nb_waveforms': 1000,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    cluster_kwargs = {
        'name': "cluster",
        'threshold_factor': threshold_factor,
        'alignment': alignment,
        'sampling_rate': sampling_rate,
        'spike_width': spike_width,
        'spike_jitter': spike_jitter,
        'spike_sigma': spike_sigma,
        'nb_waveforms': nb_waveforms_clustering,
        'nb_waveforms_tracking': 100000,  # i.e. block tracking
        'probe_path': probe_path,
        'two_components': False,
        'local_merges': 3,
        'debug_plots': debug_directory,
        # 'debug_ground_truth_templates': precomputed_template_paths,
        # 'introspection_path': introspection_directory,
        'log_level': INFO,
    }
    # cluster_writer_kwargs = {
    #     'name': "cluster_writer",
    #     'output_directory': sorting_directory,
    #     # 'introspection_path': introspection_directory,
    #     'log_level': DEBUG,
    # }
    updater_bis_kwargs = {
        'name': "updater_bis",
        'probe_path': probe_path,
        'templates_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_path': os.path.join(sorting_directory, "overlaps.p"),
        # 'precomputed_template_paths': precomputed_template_paths if with_precomputed_templates else None,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    # updater_writer_kwargs = {
    #     'name': "updater_writer",
    #     'output_directory': sorting_directory,
    #     # 'introspection_path': introspection_directory,
    #     'log_level': DEBUG,
    # }
    fitter_bis_kwargs = {
        'name': "fitter_bis",
        'degree': nb_fitters,
        # 'templates_init_path': os.path.join(sorting_directory, "templates.h5"),
        # 'overlaps_init_path': os.path.join(sorting_directory, "overlaps.p"),
        'sampling_rate': sampling_rate,
        'discarding_eoc_from_updater': True,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    spike_writer_kwargs = {
        'name': "spike_writer",
        'data_path': os.path.join(sorting_directory, "spikes.h5"),
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        # 'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=hosts['master'], **director_kwargs)
    managers = {
        key: director.create_manager(host=hosts[key])
        for key in hosts_keys
    }
    reader = managers[managers_keys['reader']].create_block('reader', **reader_kwargs)
    filter_ = managers[managers_keys['filter']].create_block('filter', **filter_kwargs)
    # writer = managers[managers_keys['writer']].create_block('writer', **writer_kwargs)
    mad = managers[managers_keys['mad']].create_block('mad_estimator', **mad_kwargs)
    detector = managers[managers_keys['detector']].create_block('peak_detector', **detector_kwargs)
    peak_writer = managers[managers_keys['peak_writer']].create_block('peak_writer', **peak_writer_kwargs)
    pca = managers[managers_keys['pca']].create_block('pca', **pca_kwargs)
    cluster = managers[managers_keys['cluster']].create_block('density_clustering', **cluster_kwargs)
    # cluster_writer = managers[managers_keys['cluster_writer']].create_block('cluster_writer', **cluster_writer_kwargs)
    updater = managers[managers_keys['updater']].create_block('template_updater_bis', **updater_bis_kwargs)
    # updater_writer = managers[managers_keys['updater_writer']].create_block('updater_writer', **updater_writer_kwargs)
    fitter = managers[managers_keys['fitter']].create_network('fitter_bis', **fitter_bis_kwargs)
    spike_writer = managers[managers_keys['spike_writer']].create_block('spike_writer', **spike_writer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.output, [
        filter_.get_input('data'),
    ])
    director.connect(filter_.output, [
        # writer.get_input('data'),
        mad.get_input('data'),
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
        peak_writer.get_input('peaks'),
        pca.get_input('peaks'),
        cluster.get_input('peaks'),
        fitter.get_input('peaks'),
    ])
    director.connect(pca.get_output('pcs'), [
        cluster.get_input('pcs'),
    ])
    director.connect(cluster.get_output('templates'), [
        # cluster_writer.get_input('templates'),
        updater.get_input('templates'),
    ])
    director.connect(updater.get_output('updater'), [
        # updater_writer.get_input('updater'),
        fitter.get_input('updater'),
    ])
    director.connect_network(fitter)
    director.connect(fitter.get_output('spikes'), [
        spike_writer.get_input('spikes'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
