import matplotlib.pyplot as plt
import numpy as np



def traces(data, channels_id=None, t_start=None, sampling_rate=None):
    nb_time_steps, nb_channels = data.shape
    if channels_id is not None:
        assert(nb_channels == len(channels_id))
    x = np.arange(0, nb_time_steps)
    plt.figure()
    for channel in range(0, nb_channels):
        offset = 0.02 * float(channel)
        y = data[:, channel] + offset
        print(x.shape)
        print(y.shape)
        plt.plot(x, y)
    plt.suptitle(r"Extracellular traces")
    plt.show()
    return
