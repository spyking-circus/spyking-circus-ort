import numpy as np

from circusort.utils.template import compute_template_similarity
from circusort.utils.train import compute_train_similarity, compute_pic_strength


def find_matching(source_cells, sink_cells, **kwargs):
    # TODO add docstring.

    nb_source_cells = source_cells.nb_cells
    nb_sink_cells = sink_cells.nb_cells
    shape = (nb_source_cells, nb_sink_cells)

    # Compute the matrix of template similarities.
    template_similarities = np.zeros(shape)
    for i, source_cell in enumerate(source_cells):
        source_template = source_cell.template
        for j, sink_cell in enumerate(sink_cells):
            sink_template = sink_cell.template
            similarity = compute_template_similarity(source_template, sink_template)
            template_similarities[i, j] = similarity

    # Compute the matrix of train similarities.
    train_similarities = np.zeros(shape)
    for i, source_cell in enumerate(source_cells):
        source_train = source_cell.train.slice(**kwargs)
        for j, sink_cell in enumerate(sink_cells):
            sink_train = sink_cell.train.slice(**kwargs)
            similarity = compute_train_similarity(source_train, sink_train)
            train_similarities[i, j] = similarity

    # Compute the matrix of pic strengths.
    pic_strengths = np.zeros(shape)
    for i, source_cell in enumerate(source_cells):
        source_train = source_cell.train.slice(**kwargs)
        for j, sink_cell in enumerate(sink_cells):
            sink_train = sink_cell.train.slice(**kwargs)
            strength = compute_pic_strength(source_train, sink_train)
            pic_strengths[i, j] = strength

    # TODO remove the following temporary lines.
    import matplotlib.pyplot as plt
    plt.ioff()
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(template_similarities)
    ax[1].imshow(train_similarities)
    ax[2].imshow(pic_strengths)
    from circusort.utils.train import compute_correlation, compute_reversed_correlation
    fig, ax = plt.subplots(3, 3, sharex='all', sharey='all')
    for i in range(0, 3):
        for j in range(0, 3):
            source_train = source_cells[i].train.slice(**kwargs)
            sink_train = sink_cells[j].train.slice(**kwargs)
            lag, correlation = compute_correlation(source_train, sink_train, **kwargs)
            ax[i, j].plot(lag, correlation)
            lag, correlation = compute_reversed_correlation(source_train, sink_train, **kwargs)
            ax[i, j].plot(lag, correlation)
    fig.tight_layout()
    plt.show()

    # TODO use the correlations between templates?
    # TODO use the correlations between ISIs?
    # TODO use the correlations between spike trains?

    # raise NotImplementedError()  # TODO complete.

    return
