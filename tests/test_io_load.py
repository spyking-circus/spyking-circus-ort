import circusort



path = "/data/tmp/test.raw"
force = False

if not circusort.io.isdata(path) or force:
    # Generate data if necessary
    circusort.io.generate.default(path)

# Load data
data = circusort.io.load(path)
