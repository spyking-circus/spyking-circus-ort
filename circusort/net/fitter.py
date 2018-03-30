from circusort.net.network import Network


class Fitter(Network):
    """Fitter"""
    # TODO complete docstring.

    name = "Fitter network"

    params = {
        'degree': 2,
        'nb_samples': 1024,
    }

    def __init__(self, *args, **kwargs):

        Network.__init__(self, *args, **kwargs)

        # The following lines are useful to avoid some Pycharm's warnings.
        self.degree = self.degree
        self.nb_samples = self.nb_samples

    def _create_blocks(self):
        """Create the blocks of the network."""

        demultiplexer_kwargs = {
            'name': 'demultiplexer',
            'input_specs': [
                {
                    'name': "data",
                    'structure': 'array',
                    'policy': 'hard_blocking',
                },
                {
                    'name': "peaks",
                    'structure': 'dict',
                    'policy': 'soft_blocking',
                },
                {
                    'name': "updater",
                    'structure': 'dict',
                    'policy': 'non_blocking_broadcast',
                },
            ],
            'degree': self.degree,
            'overlap': 1,
            'nb_samples': self.nb_samples,
            'log_level': self.log_level,
        }
        demultiplexer_kwargs.update({
            key: value
            for key, value in self.params.iteritems()
            if key in ['introspection_path']
        })
        fitters_kwargs = {
            k: {
                'name': "{} fitter {}".format(self.name, k),
                '_nb_fitters': self.degree,
                '_fitter_id': k,
                'log_level': self.log_level,
            }
            for k in range(0, self.degree)
        }
        for k in range(0, self.degree):
            fitters_kwargs[k].update({
                key: value
                for key, value in self.params.iteritems()
                if key not in ['degree', 'nb_samples']
            })
        multiplexer_kwargs = {
            'name': 'multiplexer',
            'output_specs': [
                {
                    'name': "spikes",
                    'structure': 'dict',
                }
            ],
            'degree': self.degree,
            'log_level': self.log_level,
        }
        multiplexer_kwargs.update({
            key: value
            for key, value in self.params.iteritems()
            if key in ['introspection_path']
        })

        demultiplexer = self._create_block('demultiplexer', **demultiplexer_kwargs)
        fitters = {
            k: self._create_block('fitter', **fitters_kwargs[k])
            for k in range(0, self.degree)
        }
        multiplexer = self._create_block('multiplexer', **multiplexer_kwargs)

        # Register the network inputs.
        self._add_input('data', demultiplexer.get_input('data'))
        self._add_input('peaks', demultiplexer.get_input('peaks'))
        self._add_input('updater', demultiplexer.get_input('updater'))
        # Register the network outputs.
        self._add_output('spikes', multiplexer.get_output('spikes'))
        # Register the network blocks.
        self._add_block('demultiplexer', demultiplexer)
        self._add_block('fitters', fitters)
        self._add_block('multiplexer', multiplexer)

        return

    @staticmethod
    def _get_name(name, k):

        name = "{}_{}".format(name, k)

        return name

    def _connect(self):

        demultiplexer = self.get_block('demultiplexer')
        fitters = self.get_block('fitters')
        multiplexer = self.get_block('multiplexer')

        for k in range(0, self.degree):
            # Extract k-th fitter.
            fitter = fitters[k]
            # Connect demultiplexer to k-th fitter.
            for input_name in ['data', 'peaks', 'updater']:
                output_name = self._get_name(input_name, k)
                self.manager.connect(
                    demultiplexer.get_output(output_name),
                    [fitter.get_input(input_name)]
                )
            # Connect k-th fitter to multiplexer.
            for output_name in ['spikes']:
                input_name = self._get_name(output_name, k)
                self.manager.connect(
                    fitter.get_output(output_name),
                    [multiplexer.get_input(input_name)]
                )

        return
