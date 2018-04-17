# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time

from circusort.obj.template_store import TemplateStore
from circusort.obj.overlaps_store import OverlapsStore


def precompute_overlap_(store, genuine=False):

    if genuine:

        store.precompute_overlaps()

        answer = None

    else:

        start_times = np.zeros(store.nb_templates, dtype=np.float)
        end_times = np.zeros(store.nb_templates, dtype=np.float)

        for index in range(store.nb_templates):
            # Print message.
            string = "{}/{}"
            message = string.format(index, store.nb_templates)
            print(message)
            # Measure start time.
            start_times[index] = time.time()
            # Precompute the overlaps.
            store.get_overlaps(index, component='1')
            if store.two_components:
                store.get_overlaps(index, component='2')
            # Measure end time.
            end_times[index] = time.time()

        answer = {
            'indices': np.arange(0, store.nb_templates),
            'durations': end_times - start_times,
        }

    return answer


def plot_answer(ans, path=None):

    x = ans['indices']
    y = ans['durations']

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("index")
    ax.set_ylabel("duration (s)")
    ax.set_title("Overlap computation times")

    if path is not None:
        fig.savefig(path)

    return


def main():

    # Define the working directory.
    directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "overlaps_store")
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Define the path to the templates.
    templates_filename = "templates.h5"
    templates_path = os.path.join(directory, templates_filename)
    if not os.path.isfile(templates_path):
        string = "Provide a file  of templates named {} in {}"
        message = string.format(templates_filename, templates_path)
        raise NotImplementedError(message)

    # Initialize the templates store.
    template_store = TemplateStore(templates_path, mode='r')

    # Initialize the overlaps store.
    overlaps_store = OverlapsStore(template_store)

    # Print info message.
    string = "The overlaps store is initialized with {} templates from {}"
    message = string.format(overlaps_store.nb_templates, templates_path)
    print(message)

    # Define the path to the answer.
    ans_filename = "answer.p"
    ans_path = os.path.join(directory, ans_filename)

    # Get the answer.
    if not os.path.isfile(ans_path):
        # Precompute all the overlaps.
        ans = precompute_overlap_(overlaps_store)
        # Save the answer.
        with open(ans_path, mode='wb') as ans_file:
            pickle.dump(ans, ans_file)
    else:
        # Load the answer.
        with open(ans_path, mode='rb') as ans_file:
            ans = pickle.load(ans_file)

    # # Define the path to the overlaps.
    # overlaps_filename = "overlaps.h5"
    # overlaps_path = os.path.join(directory, overlaps_filename)

    # Save all the overlaps.
    # overlaps_store.save_internal_overlaps_dictionary(overlaps_path)

    # Define the path to the plot.
    plot_filename = "answer.pdf"
    plot_path = os.path.join(directory, plot_filename)

    # Plot the answer.
    plot_answer(ans, path=plot_path)

    return


if __name__ == '__main__':

    main()
