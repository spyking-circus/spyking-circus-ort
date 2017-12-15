import matplotlib.pyplot as plt

from circusort.obj.probe import Probe


def plot_probe(probe, output=None):
    # TODO add docstring.

    if isinstance(probe, (str, unicode)):
        raise NotImplementedError()  # TODO complete.
    elif isinstance(probe, Probe):
        pass
    else:
        message = "Unknown probe type: {}".format(type(probe))
        raise TypeError(message)

    x = probe.x
    y = probe.y

    plt.subplots()
    plt.scatter(x, y)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

    return
