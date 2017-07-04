import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import circusort


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
print("Spike times: {}".format(spike_times[100:200]))

synthetic_store.close()

spike_elec = spike_elecs[0]
print("Spike electrode: {}".format(spike_elec))


# 2. We want to load the detected peak times.
peak_data = np.fromfile(peak_detector_path, dtype=np.int32)
peak_elecs = peak_data[0:None:+2]
peak_times = peak_data[1:None:+2]

# TODO remove following line.
print("Peak times: {}".format(np.sort(peak_times)[0:100]))

b = (peak_elecs == spike_elec)
peak_elecs = peak_elecs[b]
peak_times = peak_times[b]

# TODO remove following lines.
peak_lag = +0
peak_times = peak_times + peak_lag * 1024

# TODO remove following line.
print("Peak times: {}".format(np.sort(peak_times)[0:100]))

# 3. We want to load the raw signal.
probe = circusort.io.Probe(probe_path)
trace_data = np.fromfile(generator_path, dtype=np.float32)
trace_data = np.reshape(trace_data, (-1, probe.nb_channels))

# 4. We want to load the filtered signal.
filtered_signal = np.fromfile(filter_path, dtype=np.float32)
filtered_signal = np.reshape(filtered_signal, (-1, probe.nb_channels))

# # 5. We want to load the whitened signal.
# whitened_signal = np.fromfile(whitening_path, dtype=np.float32)
# whitened_signal = np.reshape(whitened_signal, (-1, probe.nb_channels))

# 6. We want to load the MADs.
mads_data = np.fromfile(mad_estimator_path, dtype=np.float32)
mads = np.reshape(mads_data, (-1, probe.nb_channels))


nb_samples = 1024 # number of samples per buffer
nb_buffers = trace_data.shape[0] / nb_samples


# # a. Select buffer with the leftmost spike time.
# k = np.argmin(spike_times % nb_samples)
# i = spike_times[k] / nb_samples
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{},{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# # b. Select buffer with the rightmost spike time.
# k = np.argmax(spike_times % nb_samples)
# i = spike_times[k] / nb_samples
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{},{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# c. Select buffer with the first peak time.
# assert peak_detector.start_step == peak_fitter.start_step, "peak_detector.start_step == {} != peak_fitter.start_step == {}".format(peak_detector.start_step, peak_fitter.start_step)
i = peak_detector.start_step
i_min = max(0, i - 9) * nb_samples
i_max = min(nb_buffers, i + 10) * nb_samples
print("Chunk of interest: {} [{},{}]".format(i, i*nb_samples, (i+1)*nb_samples))

# # d. Select buffer where the MAD start.
# start_mad = mad_estimator.start_step
# i = start_mad
# i_min = max(0, i - 4) * nb_samples
# i_max = min(nb_buffers, i + 5) * nb_samples
# print("Chunk of interest: {} [{},{}]".format(i, i*nb_samples, (i+1)*nb_samples))


start_mad = mad_estimator.start_step
print("Start MADs: {}".format(start_mad))
print("Start peak detector: {}".format(peak_detector.start_step))
print("Start peak fitter: {}".format(peak_fitter.start_step))


plt.figure(figsize=(12, 9))

j_min = 0 * nb_samples
j_max = 800 * nb_samples

spike_times_bis = spike_times[np.logical_and(j_min <= spike_times, spike_times <= j_max)]
peak_times_bis = peak_times[np.logical_and(j_min <= peak_times, peak_times <= j_max)]

bins = np.arange(j_min, j_max, nb_samples)

ax = plt.subplot(2, 1, 1)
plt.hist(spike_times_bis, bins)
# plt.ylim(0, 6)
plt.title("number of spikes detected per buffer")
plt.xlabel("buffer index")
plt.ylabel("number of spikes")

ax = plt.subplot(2, 1, 2)
plt.hist(peak_times_bis, bins)
# plt.ylim(0, 6)
plt.title("number of peaks detected per buffer")
plt.xlabel("buffer index")
plt.ylabel("number of peaks")


# import sys
# sys.exit(0)


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
peak_times_bis = peak_times
peak_times_bis = peak_times_bis[np.logical_and(i_min <= peak_times_bis, peak_times_bis < i_max)]
for x in peak_times_bis:
    x_min = x - 20 - 0.5
    x_max = x + 60 + 0.5
    plt.axvspan(x_min, x_max, facecolor='C1', alpha=0.25)
    plt.axvline(x, color='C1', linestyle='--')

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

# # Plot whitened voltage trace for each channel.
# x = np.arange(i_min - 0.5, i_max + 0.5)
# y_scale = 0.02
# for channel_id in range(probe.nb_channels):
#     y = whitened_signal[i_min:i_max, channel_id]
#     y = np.append(y, y[-1])
#     y = y * y_scale
#     y_offset = channel_id
#     plt.step(x, y + y_offset, where='post', color='C2')

# Plot thresholds for each channel.
for channel_id in range(0, probe.nb_channels):
    thresh = np.zeros(0, dtype=np.float32)
    for k in range(k_min, k_max):
        if k < start_mad:
            thresh = np.concatenate((thresh, np.zeros(nb_samples)))
        else:
            thresh = np.concatenate((thresh, mads[k - start_mad, channel_id] * np.ones(nb_samples)))
    thresh *= y_scale
    y_offset = channel_id
    plt.plot(np.arange(i_min, i_max), +thresh + y_offset, c='gray', ls='--')
    plt.plot(np.arange(i_min, i_max), -thresh + y_offset, c='gray', ls='--')

plt.xlabel("time (bin)")
plt.ylabel("channel")
plt.tight_layout()

plt.show()
