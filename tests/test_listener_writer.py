# Test to check the ability to pull data from the USB-MEA256-System of
# Multi Channel Systems MCS GmbH

import argparse
import logging

import circusort


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='local', choices=['local', 'remote'],
                    help="distributed computation or not")
args = parser.parse_args()


# Set input arguments.

if args.mode == 'local':
    master = '192.168.0.254'
elif args.mode == 'remote':
    master = '192.168.0.254'
    slaves = [
        '192.168.0.1',
        '192.168.0.2',
    ]

acq_host = '192.168.0.253'
acq_port = 40006
acq_dtype = 'uint16'
acq_nb_samp = 2000
acq_nb_chan = 261
data_path = '/tmp/data.raw'
sleep_duration = 100.0  # s


if args.mode == 'local':

    # Set circusort network

    director = circusort.create_director(host=master)
    manager = director.create_manager(host=master)

    listener = manager.create_block('listener', acq_host=acq_host, acq_port=acq_port,
                                    acq_dtype=acq_dtype, acq_nb_samp=acq_nb_samp,
                                    acq_nb_chan=acq_nb_chan,log_level=logging.DEBUG)
    writer = manager.create_block('writer', data_path=data_path)

elif args.mode == 'remote':

    # Set circusort network

    director = circusort.create_director(host=master)
    manager = {}
    for machine in slaves:
        manager[machine] = director.create_manager(host=machine)

    listener = manager[slaves[0]].create_block('listener', acq_host=acq_host, acq_port=acq_port,
                                               acq_dtype=acq_dtype, acq_nb_samp=acq_nb_samp,
                                               acq_nb_chan=acq_nb_chan, log_level=logging.DEBUG)
    writer = manager[slaves[1]].create_block('writer', data_path=data_path)

director.initialize()

director.connect(listener.output, writer.input)

# Start acquisition

director.start()
director.sleep(duration=sleep_duration)
director.stop()
