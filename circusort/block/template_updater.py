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
        probe_file: string (optional)
        radius: float (optional)
        cc_merge: float (optional)
        cc_mixture: float (optional)
        data_path: string (optional)
    """
    # TODO complete docstring.

    name = "Template updater"

    params = {
        'probe_file' : None,
        'radius'     : None,
        'cc_merge'   : 0.95,
        'cc_mixture' : 0.95,
        'data_path'  : None,
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
        self.template_store      = TemplateStore(self.data_path, self.probe_file, mode='w')
        self.template_dictionary = TemplateDictionary(self.template_store, cc_merge=self.cc_merge, cc_mixture=self.cc_mixture)

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
                    first_component = TemplateComponent(templates[count], self.template_store.mappings[ichannel], self.template_store.nb_channels, amplitudes[count])
                    if self.two_components:
                        second_component = TemplateComponent(templates2[count], self.template_store.mappings[ichannel], self.template_store.nb_channels)

                    all_templates += [Template(first_component, ichannel, second_component, creation_time=int(data['offset']))]

                if len(templates) > 0:
                    self.log.debug('{n} received {s} {t} templates from electrode {k}'.format(n=self.name, s=len(templates), t=key, k=channel))

        return all_templates

    def _process(self):

        data = self.inputs['templates'].receive(blocking=False)
        if data is not None:

            if not self.is_active:
                self._set_active_mode()
                if self.two_components is None:
                    self.two_components = data.has_key('two')

            templates = self._data_to_templates(data)
            accepted, nb_duplicates, nb_mixtures = self.template_dictionary.add(templates)

            if nb_duplicates > 0:
                self.log.debug('{n} rejected {s} duplicated templates'.format(n=self.name, s=nb_duplicates))
            if nb_mixtures > 0:
                self.log.debug('{n} rejected {s} composite templates'.format(n=self.name, s=nb_mixtures))
            if len(accepted) > 0:
                self.log.debug('{n} accepted {t} templates'.format(n=self.name, t=len(accepted)))

            #self.log.debug('{n} saved templates {k}'.format(n=self.name, k=accepted))
            self.output.send({'templates_file' : self.template_store.file_name, 'indices' : accepted})
        return
