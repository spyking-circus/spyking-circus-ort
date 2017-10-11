# Test to check the ability to pull data from the USB-MEA256-System of
# Multi Channel Systems MCS GmbH

import circusort


# Set input arguments.

master = '192.168.0.254'
slaves = [
    '192.168.0.1',
    '192.168.0.2',
]

acq_host = '192.168.0.253'
acq_port = 4006
acq_dtype = 'uint16'
acq_nb_chan = 261
data_path = '/tmp/data.raw'
sleep_duration = 100.0  # s


# Set circusort network

director = circusort.create_director(host=master)
manager = {}
for machine in slaves:
    manager[machine] = director.create_manager(host=machine)

listener = manager[slaves[0]].create_block('listener', host=acq_host, port=acq_port,
                                           dtype=acq_dtype, nb_chan=acq_nb_chan)
writer = manager[slaves[1]].create_block('writer', data_path=data_path)

director.initialize()

director.connect(listener.output, writer.input)


# Start acquisition

director.start()
director.sleep(duration=sleep_duration)
director.stop()
