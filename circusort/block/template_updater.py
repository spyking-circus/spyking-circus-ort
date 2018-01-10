import numpy as np
import os
import tempfile

from circusort.block.block import Block
from circusort.io.probe import load_probe
from circusort.obj.template_store import TemplateStore, TemplateComponent, Template
from circusort.utils.overlaps import TemplateDictionary


class Template_updater(Block):
    """Template updater

    Attributes:
        probe_path: string (optional)
        radius: float (optional)
        cc_merge: float (optional)
        cc_mixture: float (optional)
        data_path: string (optional)
    """
    # TODO complete docstring.

    name = "Template updater"

    params = {
        'probe_path': None,
        'radius': None,
        'cc_merge': 0.95,
        'cc_mixture': None,
        'data_path': None,
        'sampling_rate': 20e+3,
        'nb_samples': 1024,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        # The following lines are useful to avoid some PyCharm's warnings.
        self.probe_path = self.probe_path
        self.radius = self.radius
        self.cc_merge = self.cc_merge
        self.cc_mixture = self.cc_mixture
        self.sampling_rate = self.sampling_rate
        self.nb_samples = self.nb_samples

        if self.probe_path is None:
            message = "{}: the probe file must be specified!".format(self.name)
            self.log.error(message)
        else:
            self.probe = load_probe(self.probe_path, radius=self.radius, logger=self.log)
            message = "{} reads the probe layout".format(self.name)
            self.log.info(message)
        self.add_input('templates')
        self.add_output('updater', 'dict')

        self.two_components = None

    def _initialize(self):

        # Initialize path to save the templates.
        if self.data_path is None:
            self.data_path = self._get_tmp_path()
        else:
            self.data_path = os.path.expanduser(self.data_path)
            self.data_path = os.path.abspath(self.data_path)

        # Create the corresponding directory if it does not exist.
        data_directory, _ = os.path.split(self.data_path)
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        # Create object to handle templates.
        self.template_store = TemplateStore(self.data_path, self.probe_path, mode='w')
        self.template_dictionary = TemplateDictionary(self.template_store, cc_merge=self.cc_merge,
                                                      cc_mixture=self.cc_mixture)

        # Log path.
        message = "{} records templates into {}".format(self.name, self.data_path)
        self.log.info(message)

        return

    @staticmethod
    def _get_tmp_path():

        tmp_directory = tempfile.gettempdir()
        tmp_basename = "templates.h5"
        tmp_path = os.path.join(tmp_directory, tmp_basename)

        return tmp_path

    def _guess_output_endpoints(self):

        return

    def _data_to_templates(self, data):

        all_templates = []
        for key in data['dat'].keys():
            for channel in data['dat'][key].keys():
                ichannel = int(channel)
                templates = np.array(data['dat'][key][channel]).astype(np.float32)
                amplitudes = np.array(data['amp'][key][channel]).astype(np.float32)

                if self.two_components:
                    templates2 = np.array(data['two'][key][channel]).astype(np.float32)
                else:
                    templates2 = None

                for count in xrange(len(templates)):
                    first_component = TemplateComponent(templates[count],
                                                        self.template_store.mappings[ichannel],
                                                        self.template_store.nb_channels,
                                                        amplitudes[count])
                    if self.two_components:
                        second_component = TemplateComponent(templates2[count],
                                                             self.template_store.mappings[ichannel],
                                                             self.template_store.nb_channels)
                    else:
                        second_component = None

                    all_templates += [
                        Template(first_component, ichannel, second_component, creation_time=int(data['offset']))
                    ]

                if len(templates) > 0:
                    string = "{} received {} {} templates from electrode {}"
                    message = string.format(self.name, len(templates), key, channel)
                    self.log.debug(message)

        return all_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)

        if data is not None:

            self._measure_time('start', frequency=1)

            if not self.is_active:
                self._set_active_mode()
                if self.two_components is None:
                    self.two_components = 'two' in data

            templates = self._data_to_templates(data)
            accepted, nb_duplicates, nb_mixtures = self.template_dictionary.add(templates)

            if nb_duplicates > 0:
                message = "{} rejected {} duplicated templates".format(self.name, nb_duplicates)
                self.log.debug(message)
            if nb_mixtures > 0:
                message = "{} rejected {} composite templates".format(self.name, nb_mixtures)
                self.log.debug(message)
            if len(accepted) > 0:
                message = "{} accepted {} templates".format(self.name, len(accepted))
                self.log.debug(message)

            # message = "{} saved templates {}".format(self.name, accepted)
            # self.log.debug(message)
            output = {
                'templates_file': self.template_store.file_name,
                'indices': accepted
            }
            self.output.send(output)

            self._measure_time('end', frequency=1)

        return

    def _introspect(self):
        # TODO add docstring.

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return
