import os

import circusort

from collections import OrderedDict
from logging import DEBUG, INFO


name = "network_4"

directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling", "electrodes", name)
directory = os.path.expanduser(directory)

nb_filters = 4
nb_detectors = 4
nb_fitters = 4

block_names = [
    "reader",
] + [
    "filter_{}".format(k)
    for k in range(0, nb_filters)
] + [
    "mad",
] + [
    "detector_{}".format(k)
    for k in range(0, nb_detectors)
] + [
    # "pca",
    # "cluster",
    # "updater",
] + [
    "fitter_fitter_bis_{}".format(k)
    for k in range(0, nb_fitters)
] + [
    "writer",
]
block_groups = {
    "reader": ["reader"],
    "filter (x{})".format(nb_filters): [
        "filter_{}".format(k)
        for k in range(0, nb_filters)
    ],
    "mad": ["mad"],
    "detector (x{})".format(nb_detectors): [
        "detector_{}".format(k)
        for k in range(0, nb_detectors)
    ],
    # "pca": ["pca"],
    # "cluster": ["cluster"],
    # "updater": ["updater"],
    "fitter (x{})".format(nb_fitters): [
        "fitter_fitter_bis_{}".format(k)
        for k in range(0, nb_fitters)
    ],
    "writer": ["writer"],
}
block_nb_buffers = {
    key: 1
    for key in block_names
}
for k in range(0, nb_fitters):
    block_nb_buffers["fitter_fitter_bis_{}".format(k)] = nb_fitters


def sorting(configuration_name):
    """Create the 1st sorting subnetwork.

    Parameter:
        configuration_name: string
            The name of the configuration (i.e. context).
    """

    # Define directories.
    generation_directory = os.path.join(directory, "generation", configuration_name)
    sorting_directory = os.path.join(directory, "sorting", configuration_name)
    log_directory = os.path.join(directory, "log", configuration_name)
    introspection_directory = os.path.join(directory, "introspection", configuration_name)

    # Load generation parameters.
    parameters = circusort.io.get_data_parameters(generation_directory)

    # Define parameters.
    # hosts = OrderedDict([
    #     ('master', '192.168.0.254'),
    #     ('slave_1', '192.168.0.1'),
    #     ('slave_2', '192.168.0.4'),
    #     ('slave_3', '192.168.0.7'),
    # ])
    hosts = OrderedDict([
        ('master', '192.168.0.254'),
        ('slave_1', '192.168.0.1'),
        ('slave_2', '192.168.0.2'),
        ('slave_3', '192.168.0.3'),
    ])
    dtype = parameters['general']['dtype']
    nb_channels = parameters['probe']['nb_channels']
    nb_samples = parameters['general']['buffer_width']
    sampling_rate = parameters['general']['sampling_rate']
    threshold_factor = 7.0
    alignment = True
    spike_width = 5.0  # ms
    spike_jitter = 1.0  # ms
    spike_sigma = 2.75  # µV
    probe_path = os.path.join(generation_directory, "probe.prb")
    precomputed_template_paths = [
        cell.template.path
        for cell in circusort.io.load_cells(generation_directory)
    ]

    # Create directories (if necessary).
    if not os.path.isdir(sorting_directory):
        os.makedirs(sorting_directory)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
    if not os.path.isdir(introspection_directory):
        os.makedirs(introspection_directory)

    # Define keyword arguments.
    director_kwarg = {
        'log_path': os.path.join(log_directory, "log.txt"),
        'log_level': INFO,
    }
    reader_kwargs = {
        'name': "reader",
        'data_path': os.path.join(generation_directory, "data.raw"),
        'dtype': dtype,
        'nb_channels': nb_channels,
        'nb_samples': nb_samples,
        'sampling_rate': sampling_rate,
        'is_realistic': True,
        'speed_factor': 1.0,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    filter_kwargs = {
        'name': "filter",
        'degree': nb_filters,
        'cut_off': 1.0,  # Hz
        'order': 1,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    mad_kwargs = {
        'name': "mad",
        'time_constant': 10.0,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    detector_kwargs = {
        'name': "detector",
        'degree': nb_detectors,
        'threshold_factor': threshold_factor,
        'sampling_rate': sampling_rate,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    pca_kwargs = {
        'name': "pca",
        'spike_width': spike_width,
        'spike_jitter': spike_jitter,
        'spike_sigma': spike_sigma,
        'nb_waveforms': 2000,
        'introspection_path': introspection_directory,
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
        'nb_waveforms': 100000,
        'probe_path': probe_path,
        'two_components': False,
        'local_merges': 3,
        'introspection_path': introspection_directory,
        'log_level': INFO,
    }
    updater_bis_kwargs = {
        'name': "updater",
        'probe_path': probe_path,
        'templates_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_path': os.path.join(sorting_directory, "overlaps.p"),
        'precomputed_template_paths': precomputed_template_paths,
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }
    fitter_bis_kwargs = {
        'name': "fitter",
        'degree': nb_fitters,
        'templates_init_path': os.path.join(sorting_directory, "templates.h5"),
        'overlaps_init_path': os.path.join(sorting_directory, "overlaps.p"),
        'sampling_rate': sampling_rate,
        'discarding_eoc_from_updater': True,
        'introspection_path': introspection_directory,
        'introspection_factor': 1.0 / float(nb_fitters),
        'log_level': DEBUG,
    }
    writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(sorting_directory, "spikes.h5"),
        'sampling_rate': sampling_rate,
        'nb_samples': nb_samples,
        'introspection_path': introspection_directory,
        'log_level': DEBUG,
    }

    # Define the elements of the network.
    director = circusort.create_director(host=hosts['master'], **director_kwarg)
    managers = OrderedDict([
        (key, director.create_manager(host=host))
        for key, host in iter(hosts.items())
    ])
    reader = managers['master'].create_block('reader', **reader_kwargs)
    filter_ = managers['slave_1'].create_network('filter', **filter_kwargs)
    mad = managers['slave_1'].create_block('mad_estimator', **mad_kwargs)
    detector = managers['slave_2'].create_network('peak_detector', **detector_kwargs)
    pca = managers['slave_2'].create_block('pca', **pca_kwargs)
    cluster = managers['slave_2'].create_block('density_clustering', **cluster_kwargs)
    updater = managers['slave_2'].create_block('template_updater_bis', **updater_bis_kwargs)
    fitter = managers['slave_3'].create_network('fitter_bis', **fitter_bis_kwargs)
    writer = managers['master'].create_block('spike_writer', **writer_kwargs)
    # Initialize the elements of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(reader.get_output('data'), [
        filter_.get_input('data'),
    ])
    director.connect_network(filter_)
    director.connect(filter_.get_output('data'), [
        mad.get_input('data'),
        detector.get_input('data'),
        pca.get_input('data'),
        cluster.get_input('data'),
        fitter.get_input('data'),
    ])
    director.connect(mad.get_output('mads'), [
        detector.get_input('mads'),
        cluster.get_input('mads'),
    ])
    director.connect_network(detector)
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
    director.connect_network(fitter)
    director.connect(fitter.get_output('spikes'), [
        writer.get_input('spikes'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()
