import os

import circusort


# TODO retrieve circusort's default parameters

# TODO define the missing parameters

# TODO define the working directory


# import circusort
# circusort.io.pregenerate()
# # or
import circusort
params = circusort.io.parse_parameters()
circusort.io.pregenerate(**params)
# # or
# import os
# import circusort
# path = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling")
# path = os.path.expanduser(path)
# circusort.io.pregenerate(path=path)
# # or
# import os
# import circusort
# path = os.path.join("~", ".spyking-circus-ort", "benchmarks", "scaling")
# path = os.path.expanduser(path)
# circusort.io.generate.synthetic(path)

# TODO generate/find templates

# TODO generate/find noise

# TODO generate/find spike trains

# TODO generate signal (if necessary)

