from .block import Block
import numpy as np
import os
import tempfile
from circusort.io.probe import load_probe
import scipy.sparse

# from circusort.io.utils import save_pickle
from circusort.io.template import TemplateStore, TemplateComponent, Template
from circusort.utils.overlaps import TemplateDictionary

class Template_updater(Block):
    """Template updater

    Attributes:
        spike_width: float (optional)
        probe_file: string (optional)
        radius: None (optional)
        sampling_rate: float (optional)
        cc_merge: float (optional)
        data_path: string (optional)
    """
    # TODO complete docstring.

    name = "Template updater"

    params = {
        'spike_width': 5.,  # ms
        'probe_file': None,
        'radius': None,  # um
        'sampling_rate': 20000,  # Hz
        'cc_merge': 0.95,
        'data_path': None,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        if self.probe_file == None:
            self.log.error('{n}: the probe file must be specified!'.format(n=self.name))
        else:
            self.probe = load_probe(self.probe_file, radius=self.radius, logger=self.log)
            self.log.info('{n} reads the probe layout'.format(n=self.name))
        self.add_input('templates')
        self.add_output('updater', 'dict')

    def __del__(self):
        self.template_store.close()

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
        self.two_components = None
        self.template_store = TemplateStore(self.data_path, self.probe_file, mode='w')
        #self.template_dictionary = TemplateDictionary(self.template_store, cc_merge)

        # Log path.
        info_msg = "{} records templates into {}"
        self.log.info(info_msg.format(self.name, self.data_path))

        return

    def _get_tmp_path(self):

        tmp_directory = tempfile.gettempdir()
        tmp_basename  = 'templates.h5'
        tmp_path      = os.path.join(tmp_directory, tmp_basename)
        return tmp_path

    def _guess_output_endpoints(self):
        return

    def _data_to_templates(self, data):
        all_templates = []
        for key in data['dat'].keys():
            for channel in data['dat'][key].keys():
                ichannel   = int(channel)
                templates  = np.array(data['dat'][key][channel]).astype(np.float32)
                amplitudes = np.array(data['amp'][key][channel]).astype(np.float32)

                if self.two_components:
                    templates2 = np.array(data['two'][key][channel]).astype(np.float32)

                for count in xrange(len(templates)):
                    self.log.debug('{n} received {s} {t} templates from electrode {k}'.format(n=self.name, s=len(templates), t=key, k=channel))
                    
                    first_component = TemplateComponent(templates[count], amplitudes[count][0], self.template_store.mappings[ichannel], self.nb_channels)
                    if self.two_components:
                        second_component = TemplateComponent(templates2[count], amplitudes[count][1], self.template_store.mappings[ichannel], self.nb_channels)

                    all_templates += [Template(first_component, ichannel, second_component, creation_time=int(data['offset']))]

        return all_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)
        if data is not None:

            if not self.is_active:
                self._set_active_mode()
                if self.two_components is None:
                    self.two_components = data.has_key('two')

                templates = self._data_to_templates(data)
                indices   = self.template_dictionary.add(templates)
                #self.template_store.add(templates)

                #self.output.send({'templates_file' : self.template_store.file_name, 'indices' : indices})
        return
