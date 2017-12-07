import circusort as cco


params = cco.io.parse_parameters()
# or params = cco.io.parse_parameters('pregeneration')  # TODO enable this line of code.

cco.net.pregenerator(**params)
# or cco.net.pregenerator(**params.pregeneration)  # TODO enable this line of code.
# or cco.net.pregenerator(**params['pregeneration'])  # TODO enable this line of code.
