import matplotlib.pyplot as plt
import matplotlib.gridspec as gds
from circusort.obj.probe import Probe


def plot_probe(probe, ax=None, output=None):
    # TODO add docstring.

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    if isinstance(probe, (str, unicode)):
        raise NotImplementedError()  # TODO complete.
    elif isinstance(probe, Probe):
        pass
    else:
        message = "Unknown probe type: {}".format(type(probe))
        raise TypeError(message)

    x = probe.x
    y = probe.y

    ax.scatter(x, y)
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)


    return
