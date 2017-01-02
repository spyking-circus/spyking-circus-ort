import matplotlib.pyplot as plt

import circusort



path = "/data/tmp/synthetic.raw"

# Generate data
desc = circusort.io.generate.synthetic_grid(path)

# Load data
data = circusort.io.load.raw_binary(path, desc.nb_channels, desc.length, desc.sampling_rate)

# Plot data
nb_channels = data.shape[1]
plt.figure()
plt.subplot(1, 1, 1)
for channel in range(0, nb_channels):
    i_start = 20000
    i_end = i_start + 10000
    plt.plot(data[i_start:i_end, channel] + 1.0e-2 * float(channel), color='blue')
plt.show()
