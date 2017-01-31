import circusort



path = "/data/tmp/test.raw"
force = False

if not circusort.io.isdata(path) or force:
    # Generate data if necessary
    print("Generate data at '{}'...".format(path))
    circusort.io.generate.default(path, visualization=True)
    print("Done.")

# Load data
print("Load data from '{}'...".format(path))
data = circusort.io.load(path)
print("Done.".format(path))


circusort.plot.traces(data)

# import matplotlib.pyplot as plt
#
# plt.figure()
# for k in range(0, data.shape[1]):
#     plt.plot(data[:, k] + 0.03 * float(k))
# plt.show()
