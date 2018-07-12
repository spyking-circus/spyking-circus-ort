import os

import circusort

from logging import DEBUG


sco_directory = os.path.join("~", ".spyking-circus-ort")
sco_directory = os.path.expanduser(sco_directory)
directory = os.path.join(sco_directory, "examples", "listen_n_qt_display")

probe_path = os.path.join(sco_directory, "probes", "mea_256.prb")


def main():

    # Define directories.
    output_directory = os.path.join(directory, "output")
    log_directory = os.path.join(directory, "log")

    # Create log directory (if necessary).
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)

    # Define parameters.
    hosts = {
        'master': '127.0.0.1',
    }

    # Define keyword arguments.
    director_kwargs = {
        'log_path': os.path.join(log_directory, "log.txt"),
    }
    listener_kwargs = {
        'name': "listener",
        'acq_host': '192.168.0.253',
        'acq_port': 40006,
        'acq_dtype': 'uint16',
        'acq_nb_samp': 2000,
        'acq_nb_chan': 261,
        'dtype': 'float32',
        'log_level': DEBUG,
    }
    qt_displayer_kwargs = {
        'name': "displayer",
        'probe_path': probe_path,
        'log_level': DEBUG,
    }
    writer_kwargs = {
        'name': "writer",
        'data_path': os.path.join(output_directory, "data.h5"),
        'nb_samples': 1024,
        'sampling_rate': 20e+3,
        'log_level': DEBUG,
    }

    # Define the element of the network.
    director = circusort.create_director(host=hosts['master'], **director_kwargs)
    manager = director.create_manager(host=hosts['master'])
    listener = manager.create_block('listener', **listener_kwargs)
    qt_displayer = manager.create_block('qt_displayer', **qt_displayer_kwargs)
    writer = manager.create_block('writer', **writer_kwargs)
    # Initialize the element of the network.
    director.initialize()
    # Connect the elements of the network.
    director.connect(listener.get_output('data'), [
        qt_displayer.get_input('data'),
        writer.get_input('data'),
    ])
    # Launch the network.
    director.start()
    director.join()
    director.destroy()

    return


if __name__ == '__main__':

    main()
