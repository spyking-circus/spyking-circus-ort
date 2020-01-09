# -*- coding: utf-8 -*-
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
import numpy as np
import os

from circusort.obj.template_store import TemplateStore

def plot_templates_from_list(templates, probe, ax=None, component='first', output=None, x_bar=1.0, y_bar=20.0, show_scale_bar=True):
    """Plot template from template store.

    Parameters:
        template: a list of templates
        probe: the probe used to represent this template
        component: string 
            can be first | second | both 
        output: none | string (optional)
            The path to be used to save the figure. If is None then display the figure and block until the figure
            have been closed. The default value is None.
        show_scale_bar : boolean
            If we want to display the scale bar. default is True
        x_bar: float
            x-scale bar length (in ms). The default value is 1.0.
        y_bar: float
            y-scale bar length (in µV). The default value is 20.0.
    """

    assert component in ['first', 'second', 'both']

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    scl = 0.9 * (probe.field_of_view['d'] / 2.0)
    # Plot the generated template.
    x_scl = scl

    if component is 'first':
        data_1 = []
        for t in templates:
            data_1 += [t.first_component.to_dense()]
        y_scl = scl * (1.0 / np.max(np.abs(data_1)))
    elif component is 'second':
        data_2 = []
        for t in templates:
            data_2 = [t.second_component.to_dense()]
        y_scl = scl * (1.0 / np.max(np.abs(data_2)))
    elif component is 'both':
        data_1 = []
        data_2 = []
        for t in templates:
            data_1 += [t.first_component.to_dense()]
            data_2 += [t.second_component.to_dense()]
        y_scl = scl * (1.0 / max(np.max(np.abs(data_1)), np.max(np.abs(data_2))))

    for count in range(len(templates)):
        color = 'C{}'.format(count % 10)

        for k in range(0, probe.nb_channels):
            x_prb, y_prb = probe.positions[:, k]

            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=templates[0].temporal_width)
            if component in ['first', 'both']:
                y = y_prb + y_scl * data_1[count][k, :]
            if component in ['second', 'both']:
                y = y_prb + y_scl * data_2[count][k, :]
            ax.plot(x, y, c=color)
    # Plot scale bars.
    x_bar_ = x_scl * (x_bar * 1e-3 * 20e+3) / (float(templates[0].temporal_width) / 2.0)
    if show_scale_bar:
        ax.plot([0.0, x_bar_], 2 * [0.0], c='black')
        ax.annotate(u"{} ms".format(x_bar), xy=(x_bar_, 0.0))
        y_bar_ = y_scl * y_bar
        ax.plot(2 * [0.0], [0.0, y_bar_], c='black')
        ax.annotate(u"{} µV".format(y_bar), xy=(0.0, y_bar_))
    ax.set_xlabel(u"x (µm)")
    ax.set_ylabel(u"y (µm)")
    ax.axis('scaled')
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)
    return


def plot_templates(template_store, ax=None, indices=None, component='first', output=None, x_bar=1.0, y_bar=20.0, show_scale_bar=True):
    """Plot template from template store.

    Parameters:
        template_store: string or TemplateStore object
            The path or the template store to be used when loading the data.
        indices: none | list (optional)
            The indices to be used when loading the data. If is None then all the channels will be used. The default
            value is None.
        component: string 
            can be first | second | both 
        output: none | string (optional)
            The path to be used to save the figure. If is None then display the figure and block until the figure
            have been closed. The default value is None.
        show_scale_bar : boolean
            If we want to display the scale bar. default is True
        x_bar: float
            x-scale bar length (in ms). The default value is 1.0.
        y_bar: float
            y-scale bar length (in µV). The default value is 20.0.
    """

    # Expand user's home directory (if necessary).
    if not isinstance(template_store, TemplateStore):
        template_store = TemplateStore(os.path.expanduser(template_store))
    
    probe = template_store.probe

    assert component in ['first', 'second', 'both']

    # Define the indices to be used when loading the data (if necessary).
    if indices is None:
        indices = template_store.indices

    if not np.iterable(indices):
        indices = [indices]
    
    assert len(indices) > 0

    templates = template_store.get(indices)

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    scl = 0.9 * (probe.field_of_view['d'] / 2.0)
    # Plot the generated template.
    x_scl = scl

    if component is 'first':
        data_1 = []
        for t in templates:
            data_1 += [t.first_component.to_dense()]
        y_scl = scl * (1.0 / np.max(np.abs(data_1)))
    elif component is 'second':
        data_2 = []
        for t in templates:
            data_2 = [t.second_component.to_dense()]
        y_scl = scl * (1.0 / np.max(np.abs(data_2)))
    elif component is 'both':
        data_1 = []
        data_2 = []
        for t in templates:
            data_1 += [t.first_component.to_dense()]
            data_2 += [t.second_component.to_dense()]
        y_scl = scl * (1.0 / max(np.max(np.abs(data_1)), np.max(np.abs(data_2))))

    for count, i in enumerate(indices):
        color = 'C{}'.format(i % 10)
        for k in range(0, probe.nb_channels):
            x_prb, y_prb = probe.positions[:, k]
            x = x_prb + x_scl * np.linspace(-1.0, +1.0, num=t.temporal_width)
            if component in ['first', 'both']:
                y = y_prb + y_scl * data_1[count][k, :]
            if component in ['second', 'both']:
                y = y_prb + y_scl * data_2[count][k, :]
            ax.plot(x, y, c=color)
    # Plot scale bars.
    x_bar_ = x_scl * (x_bar * 1e-3 * 20e+3) / (float(template_store.temporal_width) / 2.0)
    if show_scale_bar:
        ax.plot([0.0, x_bar_], 2 * [0.0], c='black')
        ax.annotate(u"{} ms".format(x_bar), xy=(x_bar_, 0.0))
        y_bar_ = y_scl * y_bar
        ax.plot(2 * [0.0], [0.0, y_bar_], c='black')
        ax.annotate(u"{} µV".format(y_bar), xy=(0.0, y_bar_))
    ax.set_xlabel(u"x (µm)")
    ax.set_ylabel(u"y (µm)")
    ax.axis('scaled')
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)
    return

def plot_center_of_mass(template_store, indices=None, ax=None, output=None, with_labels=True):
    """Plot center of mass from templates in a template store.

    Parameters:
        template_store: string or TemplateStore object
            The path or the template store to be used when loading the data.
        indices: none | list (optional)
            The indices to be used when loading the data. If is None then all the channels will be used. The default
            value is None.
        output: none | string (optional)
            The path to be used to save the figure. If is None then display the figure and block until the figure
            have been closed. The default value is None.
        with_labels: boolean
            To display the number of the templates, on top of center of mass
    """

    # Expand user's home directory (if necessary).
    if not isinstance(template_store, TemplateStore):
        template_store = TemplateStore(os.path.expanduser(template_store))
    
    probe = template_store.probe

    # Define the indices to be used when loading the data (if necessary).
    if indices is None:
        indices = template_store.indices

    if not np.iterable(indices):
        indices = [indices]
    
    assert len(indices) > 0

    templates = template_store.get(indices)

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    scl = 0.9 * (probe.field_of_view['d'] / 2.0)
    # Plot the generated template.
    x_scl = scl

    ax.scatter(probe.positions[0], probe.positions[1], c='k')

    for i, template in enumerate(templates):
        color = 'C{}'.format(i % 10)
        x, y = template.center_of_mass(probe)
        ax.scatter([x], [y], c=color)
        if with_labels:
            ax.text(x, y, str(i))

    ax.set_xlabel(u"x (µm)")
    ax.set_ylabel(u"y (µm)")
    ax.axis('scaled')
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)
    return


def plot_templates_on_channels(template_store, channels, ax=None, component='first', output=None,
                               x_bar=1.0, y_bar=20.0, show_scale_bar=True):

    # Expand user's home directory (if necessary).
    if not isinstance(template_store, TemplateStore):
        template_store = TemplateStore(os.path.expanduser(template_store))
    
    if not np.iterable(channels):
        channels = [channels]

    indices = []
    for channel in channels:
        if channel in template_store.templates_per_channels:
            indices += template_store.templates_per_channels[channel]

    plot_templates(template_store, ax, indices, component, output, x_bar, y_bar, show_scale_bar)

    return


def plot_templates_history(template_store, ax=None, output=None):

    # Expand user's home directory (if necessary).
    if not isinstance(template_store, TemplateStore):
        template_store = TemplateStore(os.path.expanduser(template_store))

    _ = template_store.probe
    indices = template_store.indices
    templates = template_store.get(indices)

    plt.style.use('seaborn-paper')

    if output is not None:
        plt.ioff()

    if ax is None:
        fig = plt.figure()
        gs = gds.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    for count, t in enumerate(templates):
        color = 'C{}'.format(indices[count] % 10)
        ax.scatter([t.creation_time], [indices[count]], c=color, marker='|')

    ax.set_xlabel(u"Time [step]")
    ax.set_ylabel(u"Indices")
    # plt.axis('scaled')
    gs.tight_layout(fig)

    if output is not None:
        plt.savefig(output)

    return
