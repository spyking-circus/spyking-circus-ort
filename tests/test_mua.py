# Test to check the ability to analyze the multi-unit activity (MUA) for data
# pulled from the USB-MEA256-System of Multi Channel Systems MCS GmbH.

import logging

import circusort


# Set input arguments.

master = '192.168.0.254'
slaves = [
    '192.168.0.1',
    '192.168.0.2',
    # '192.168.0.3',  # this machine shows a higher latency
    '192.168.0.4',
    '192.168.0.5',
    '192.168.0.6',
]

log_level = logging.DEBUG  # log level used for all the blocks

acq_host = '192.168.0.253'
acq_port = 40006
acq_dtype = 'uint16'
acq_nb_samp = 2000
acq_nb_chan = 261

cut_off = 100.0  # Hz  # cutoff frequency used during filtering

epsilon = 1.0e-2

threshold = 5  # threshold used during peak detection

peaks_path = '/tmp/peaks.raw'  # path used to save all the detected peaks

sleep_duration = 30.0  # s


# Set Circus network.

director = circusort.create_director(host=master)

manager = {}
for machine in [master] + slaves:
    manager[machine] = director.create_manager(host=machine)

listener = manager[master].create_block('listener', acq_host=acq_host, acq_port=acq_port,
                                        acq_dtype=acq_dtype, acq_nb_samp=acq_nb_samp,
                                        acq_nb_chan=acq_nb_chan, log_level=log_level)
filtering = manager[slaves[1]].create_block('filter', cut_off=cut_off, log_level=log_level)
# whitening = manager[slaves[2]].create_block('whitening', log_level=log_level)
mad_estimator = manager[slaves[3]].create_block('mad_estimator', epsilon=epsilon,
                                                log_level=log_level)
peak_detector = manager[slaves[4]].create_block('peak_detector', threshold=threshold,
                                                log_level=log_level)
peak_writer = manager[master].create_block('peak_writer', neg_peaks=peaks_path,
                                           log_level=log_level)


director.initialize()


director.connect(listener.output, filtering.input)
# director.connect(filtering.output, whitening.input)
# director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data')])
director.connect(filtering.output, [mad_estimator.input, peak_detector.get_input('data')])
director.connect(mad_estimator.output, peak_detector.get_input('mads'))
director.connect(peak_detector.get_output('peaks'), peak_writer.input)


# Start acquisition

director.start()
director.sleep(duration=sleep_duration)
director.stop()
