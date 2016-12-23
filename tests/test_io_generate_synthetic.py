import circusort


path = "/data/tmp/synthetic.raw"
data = circusort.io.generate.synthetic(path)


import matplotlib.pyplot as plt

for channel in range(0, 3):
    i_start = 20000
    i_end = i_start + 10000
    plt.plot(data[i_start:i_end, channel] + 1.0e-2 * float(channel), color='blue')
plt.show()
