import circusort



path = "/data/tmp/test.raw"
force = True

if not circusort.io.isdata(path) or force:
    # Generate data if necessary
    print("Generate data at '{}'...".format(path))
    circusort.io.generate.default(path)
    print("Done.")

# Load data
print("Load data from '{}'...".format(path))
data = circusort.io.load(path)
print("Done.".format(path))
