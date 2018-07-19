import os
import pickle
import time

from circusort.block.block import Block


__classname__ = "ClusterWriter"


class ClusterWriter(Block):

    name = "Cluster writer"

    params = {
        'output_directory': None,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        self.add_input('templates', structure='dict')

        # The following line is useful to disable a PyCharm warning.
        self.output_directory = self.output_directory

        self._output_counter = 0
        self._sleep_duration = 0.01  # s

    def _initialize(self):

        # Create output directory (if necessary).
        if self.output_directory is not None and not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)
            # Log debug message.
            string = "{} creates output directory {}"
            message = string.format(self.name_and_counter, self.output_directory)
            self.log.debug(message)

        return

    def _save_templates(self, templates):

        # Define output path.
        output_filename = "templates_{}.pkl".format(self._output_counter)
        output_path = os.path.join(self.output_directory, output_filename)
        # Pickle template update.
        with open(output_path, mode='wb') as output_file:
            pickle.dump(templates, output_file)
        # Update output counter.
        self._output_counter += 1
        # Log debug message.
        string = "{} saves templates in {}"
        message = string.format(self.name_and_counter, output_path)
        self.log.debug(message)

        return

    def _process(self):

        templates_packet = self.get_input('templates').receive(blocking=False)

        self._measure_time(label='start', frequency=10)

        if templates_packet is None:
            # Wait before entering next loop.
            time.sleep(self._sleep_duration)
        else:
            # Log debug message.
            string = "{} receives templates"
            message = string.format(self.name_and_counter)
            self.log.debug(message)
            # Extract templates.
            templates = templates_packet['payload']
            # Save templates.
            if self.output_directory is not None:
                self._save_templates(templates)

        self._measure_time(label='end', frequency=10)

        return
