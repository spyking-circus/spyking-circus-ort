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
    ax.scatter(x, y, s=10)
    ax.set_xlabel("template index")
    ax.set_ylabel("duration (s)")
    ax.set_title("Execution times (overlaps)")

    if path is not None:
        fig.savefig(path)

    return


def main():

    # Define the working directory.
    directory = os.path.join("~", ".spyking-circus-ort", "benchmarks", "overlaps_store")
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Define the path to the answer.
    ans_filename = "answer.p"
    ans_path = os.path.join(directory, ans_filename)

    if not os.path.isfile(ans_path):

        # Define the path to the input templates.
        input_templates_filename = "input_templates.h5"
        input_templates_path = os.path.join(directory, input_templates_filename)
        if not os.path.isfile(input_templates_path):
            string = "Provide a file of input templates named {} in {}"
            message = string.format(input_templates_filename, directory)
            raise IOError(message)

        # Initialize the input template store.
        input_template_store = TemplateStore(input_templates_path, mode='r')

        start_times = np.zeros(input_template_store.nb_templates)
        end_times = np.zeros(input_template_store.nb_templates)

        # Define the path to the output templates.
        templates_filename = "templates.h5"
        templates_path = os.path.join(directory, templates_filename)

        # Define the path to the probe.
        probe_filename = "probe.prb"
        probe_path = os.path.join(directory, probe_filename)
        if not os.path.isfile(probe_path):
            string = "Provide a probe file names {} in {}"
            message = string.format(probe_filename, directory)
            raise IOError(message)

        # Create the output template store.
        template_store = TemplateStore(templates_path, probe_file=probe_path, mode='w')
        template = input_template_store[0]
        template_store.add(template)
        if not os.path.isfile(templates_path):
            string = "Create a file of templates named {} in {}."
            message = string.format(templates_filename, directory)
            raise NotImplementedError(message)
        del template_store

        # Initialize the output template store.
        template_store = TemplateStore(templates_path, mode='r+')

        # Initialize the output overlaps store.
        start_times[0] = time.time()
        overlaps_store = OverlapsStore(template_store)
        overlaps_store.precompute_overlaps()
        end_times[0] = time.time()

        # Print info message.
        string = "The overlaps store is initialized with {} templates from {}"
        message = string.format(overlaps_store.nb_templates, input_templates_path)
        print(message)

        for i in range(1, input_template_store.nb_templates):
            print("{}/{}".format(i, input_template_store.nb_templates))
            template = input_template_store[i]
            template_store.add(template)
            start_times[i] = time.time()
            overlaps_store.update([i])
            _ = overlaps_store.get_overlaps(i, component='1')
            end_times[i] = time.time()

        ans = {
            'indices': np.arange(0, input_template_store.nb_templates),
            'durations': end_times - start_times,
        }

        with open(ans_path, mode='wb') as ans_file:
            pickle.dump(ans, ans_file)

    else:

        with open(ans_path, mode='rb') as ans_file:
            ans = pickle.load(ans_file)

    # Define the path to the plot.
    plot_filename = "answer.pdf"
    plot_path = os.path.join(directory, plot_filename)

    # Plot the answer.
    plot_answer(ans, path=plot_path)

    # Print the total execution time.
    total_duration = np.sum(ans['durations'])
    print("total duration: {}".format(total_duration))

    return


if __name__ == '__main__':

    main()
