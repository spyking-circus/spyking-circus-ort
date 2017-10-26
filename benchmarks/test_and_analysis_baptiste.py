import h5py
import logging
# import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os

import circusort


# Test

host = '127.0.0.1'

director = circusort.create_director(host=host)
manager = director.create_manager(host=host)

sampling_rate = 20.0e+3  # Hz
# two_components = True
nb_samples = 1024  # number of samples per buffer
# nb_channels = 16
simulation_duration = 10.0  # s

cells_args = [
    {
        'r': 'r_ref',
        's': 0.99 * float(nb_samples) / sampling_rate,
        't': 'periodic',
    }
]
cells_params = {
    # 'r_ref': 100.0,  # Hz
    'r_ref': sampling_rate / float(nb_samples),  # Hz
}
hdf5_path = None
# probe_path = "mea_16_copy.prb"
probe_path = "mea_4_copy.prb"
tmp_dirname = circusort.io.get_tmp_dirname()
generator_path = os.path.join(tmp_dirname, "generator.dat")
filter_path = os.path.join(tmp_dirname, "filter.dat")
whitening_path = os.path.join(tmp_dirname, "whitening.dat")
peak_detector_path = os.path.join(tmp_dirname, "peak_detector.dat")
# peak_fitter_path = os.path.join(tmp_dirname, "peak_fitter.dat")
mad_estimator_path = os.path.join(tmp_dirname, "mad_estimator.dat")

generator = manager.create_block('synthetic_generator', cells_args=cells_args, cells_params=cells_params,
                                 hdf5_path=hdf5_path, probe=probe_path, log_level=logging.DEBUG)
filter_1 = manager.create_block('filter', cut_off=100, log_level=logging.DEBUG)
whitening = manager.create_block('whitening', log_level=logging.DEBUG)
mad_estimator = manager.create_block('mad_estimator', log_level=logging.DEBUG)
peak_detector = manager.create_block('peak_detector', threshold=5, log_level=logging.DEBUG)
# peak_fitter = manager.create_block('peak_detector', threshold=5, safety_time=0)
writer = manager.create_block('writer', data_path=generator_path)
writer_2 = manager.create_block('peak_writer', neg_peaks=peak_detector_path)
# writer_3 = manager.create_block('peak_writer', neg_peaks=peak_fitter_path)
writer_4 = manager.create_block('writer', data_path=mad_estimator_path)
writer_5 = manager.create_block('writer', data_path=filter_path)
writer_6 = manager.create_block('writer', data_path=whitening_path)


director.initialize()


director.connect(generator.output, [filter_1.input, writer.input])
director.connect(filter_1.output, [whitening.input, writer_5.input])
# director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'),
#                                     peak_fitter.get_input('data'), writer_6.input])
director.connect(whitening.output, [mad_estimator.input, peak_detector.get_input('data'), writer_6.input])
# director.connect(mad_estimator.output, [peak_detector.get_input('mads'), peak_fitter.get_input('mads'), writer_4.input])
director.connect(mad_estimator.output, [peak_detector.get_input('mads'), writer_4.input])
director.connect(peak_detector.get_output('peaks'), [writer_2.input])
# director.connect(peak_fitter.get_output('peaks'), [writer_3.input])


director.start()
director.sleep(duration=10.0)
director.stop()


# director.destroy()


# Analysis

# 1. We want to load the spike times for each synthetic cell.

synthetic_store = h5py.File(generator.hdf5_path, mode='r')
print("Path to synthetic store: {}".format(generator.hdf5_path))

cell_ids = synthetic_store.keys()
nb_cells = len(cell_ids)
print("Number of cells found in the synthetic store: {}".format(nb_cells))

cell_id = cell_ids[0]
cell = synthetic_store[cell_id]
spike_elecs = cell[u'e'][:]
spike_times = cell[u'spike_times'][:]

synthetic_store.close()

spike_elec = spike_elecs[0]
print("Spike electrode: {}".format(spike_elec))


# 2. We want to load the detected peak times.
peak_data = np.fromfile(peak_detector_path, dtype=np.int32)
# peak_data = np.fromfile(peak_fitter_path, dtype=np.int32)
peak_elecs = peak_data[0:None:+2]
peak_times = peak_data[1:None:+2]

b = (peak_elecs == spike_elec)
peak_elecs = peak_elecs[b]
peak_times = peak_times[b]

time_offset = whitening.start_step * nb_samples
peak_times = np.array([peak_time + time_offset for peak_time in peak_times])


# 3. We want to load the raw signal.
probe = circusort.io.Probe(probe_path)
trace_data = np.fromfile(generator_path, dtype=np.float32)
trace_data = np.reshape(trace_data, (-1, probe.nb_channels))


# 4. We want to load the filtered signal.
filtered_signal = np.fromfile(filter_path, dtype=np.float32)
filtered_signal = np.reshape(filtered_signal, (-1, probe.nb_channels))


# 5. We want to load the whitened signal.
whitened_signal = np.fromfile(whitening_path, dtype=np.float32)
whitened_signal = np.reshape(whitened_signal, (-1, probe.nb_channels))


# 6. We want to load the MADs.
mads_data = np.fromfile(mad_estimator_path, dtype=np.float32)
mads = np.reshape(mads_data, (-1, probe.nb_channels))


nb_buffers = trace_data.shape[0] / nb_samples


# # a. Select buffer with the leftmost spike time.
# k = np.argmin(spike_times % nb_samples)
# i = spike_times[k] / nb_samples
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{}:{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# # b. Select buffer with the rightmost spike time.
# k = np.argmax(spike_times % nb_samples)
# i = spike_times[k] / nb_samples
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{}:{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# c. Select buffer with the first peak time.
# assert peak_detector.start_step == peak_fitter.start_step, "peak_detector.start_step == {} != peak_fitter.start_step
# == {}".format(peak_detector.start_step, peak_fitter.start_step)
i = whitening.start_step + peak_detector.start_step
i_min = max(0, i - 9) * nb_samples
i_max = min(nb_buffers, i + 10) * nb_samples
print("Chunk of interest: {} [{}:{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# # d. Select buffer where the MAD start.
# start_mad = mad_estimator.start_step
# i = start_mad
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{}:{}]".format(i, i*nb_samples, (i+1)*nb_samples))


print("Start filtering: {}".format(filter_1.start_step))
print("Start whitening: {}".format(whitening.start_step))
print("Start MADs: {}".format(mad_estimator.start_step))
print("Start peak detector: {}".format(peak_detector.start_step))
# print("Start peak fitter: {}".format(peak_fitter.start_step))


plt.figure(figsize=(12, 9))

# Plot spans for each buffer.
k_min = i_min / nb_samples
k_max = i_max / nb_samples
for k in range(k_min, k_max, 2):
    x_min = (k + 0) * nb_samples - 0.5
    x_max = (k + 1) * nb_samples - 0.5
    p = plt.axvspan(x_min, x_max, facecolor='gray', alpha=0.25)

# Plot spans for each spike time.
spike_times_bis = spike_times[np.logical_and(i_min <= spike_times, spike_times < i_max)]
for x in spike_times_bis:
    x_min = x - 20 - 0.5
    x_max = x + 60 + 0.5
    plt.axvspan(x_min, x_max, facecolor='C0', alpha=0.25)
    plt.axvline(x, color='C0', linestyle='--')

# Plot spans for each peak time.
peak_times_bis = peak_times[np.logical_and(i_min <= peak_times, peak_times < i_max)]
for x in peak_times_bis:
    x_min = x - 20 - 0.5
    x_max = x + 60 + 0.5
    plt.axvspan(x_min, x_max, facecolor='C2', alpha=0.25)
    plt.axvline(x, color='C2', linestyle='--')

# Plot raw voltage trace for each channel.
x = np.arange(i_min - 0.5, i_max + 0.5)
y_scale = 0.02
for channel_id in range(probe.nb_channels):
    y = trace_data[i_min:i_max, channel_id]
    y = np.append(y, y[-1])
    y = y * y_scale
    y_offset = channel_id
    plt.step(x, y + y_offset, where='post', color='C0')

# Plot filtered voltage trace for each channel.
x = np.arange(i_min - 0.5, i_max + 0.5)
y_scale = 0.02
for channel_id in range(probe.nb_channels):
    y = filtered_signal[i_min:i_max, channel_id]
    y = np.append(y, y[-1])
    y = y * y_scale
    y_offset = channel_id
    plt.step(x, y + y_offset, where='post', color='C1')

# Plot whitened voltage trace for each channel.
j_min = i_min - whitening.start_step * nb_samples
j_max = i_max - whitening.start_step * nb_samples
x = np.arange(i_min - 0.5, i_max + 0.5)
y_scale = 0.02
for channel_id in range(probe.nb_channels):
    y = whitened_signal[j_min:j_max, channel_id]
    y = np.append(y, y[-1])
    y = y * y_scale
    y_offset = channel_id
    plt.step(x, y + y_offset, where='post', color='C2')

# Plot thresholds for each channel.
k_min = (i_min - whitening.start_step * nb_samples) / nb_samples
k_max = (i_max - whitening.start_step * nb_samples) / nb_samples
for channel_id in range(0, probe.nb_channels):
    thresh = np.zeros(0, dtype=np.float32)
    for k in range(k_min, k_max):
        if k < mad_estimator.start_step:
            thresh = np.concatenate((thresh, np.zeros(nb_samples)))
        else:
            thresh = np.concatenate((thresh, mads[k - mad_estimator.start_step, channel_id] * np.ones(nb_samples)))
    thresh *= y_scale
    y_offset = channel_id
    plt.plot(np.arange(i_min, i_max), +thresh + y_offset, c='gray', ls='--')
    plt.plot(np.arange(i_min, i_max), -thresh + y_offset, c='gray', ls='--')

raw_line = mlines.Line2D([], [], color='C0', label="raw")
filtered_line = mlines.Line2D([], [], color='C1', label="filtered")
whitened_line = mlines.Line2D([], [], color='C2', label="whitened")
plt.legend(handles=[raw_line, filtered_line, whitened_line], loc=0)

plt.xlabel("time (bin)")
plt.ylabel("channel")
plt.tight_layout()

plt.show()
