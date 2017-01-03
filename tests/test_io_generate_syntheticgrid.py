import matplotlib.pyplot as plt
import numpy as np

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
t_start = 0.0
t_end = 1.0
i_start = int(t_start * desc.sampling_rate)
i_end = int(t_end * desc.sampling_rate)
for channel in range(0, nb_channels):
    x = np.arange(i_start, i_end).astype('float') / desc.sampling_rate
    y = data[i_start:i_end, channel] + 20.0e-3 * float(channel)
    plt.plot(x, y, color='blue')
yticks = [20.0e-3 * float(channel) for channel in range(0, nb_channels)]
ylabels = [str(channel) for channel in range(0, nb_channels)]
plt.yticks(yticks, ylabels)
plt.xlabel(r"time $(s)$")
plt.ylabel(r"channel")
plt.title(r"Extracellular recordings")
plt.show()
