import os
import pickle
import time

from circusort.block.block import Block


__classname__ = "UpdaterWriter"


class UpdaterWriter(Block):

    name = "Updater writer"

    params = {
        'output_directory': None,
    }

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

        self.add_input('updater', structure='dict')

        # The following line is useful to disable a PyCharm warning.
        self.output_directory = self.output_directory

        self._output_counter = 0
        self._sleep_duration = 0.01  # s

    def _initialize(self):

        # Create output directory if necessary.
        if self.output_directory is not None and not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)
            # Log debug message.
            string = "{} creates output directory {}"
            message = string.format(self.name_and_counter, self.output_directory)
            self.log.debug(message)

        return

    def _save_update(self, update):

        # Define output file.
        output_filename = "template_update_{}.pkl".format(self._output_counter)
        output_path = os.path.join(self.output_directory, output_filename)
        # Pickle template update.
        with open(output_path, mode='wb') as output_file:
            pickle.dump(update, output_file)
        # Update output counter.
        self._output_counter += 1
        # Log debug message.
        string = "{} saves template update in {}"
        message = string.format(self.name_and_counter, output_path)
        self.log.debug(message)

        return

    def _process(self):

        updater_packet = self.get_input('updater').receive(blocking=False)

        self._measure_time(label='start', frequency=10)

        if updater_packet is None:
            # Wait before entering next loop.
            time.sleep(self._sleep_duration)
        else:
            # Log debug message.
            string = "{} receive updater"
            message = string.format(self.name_and_counter)
            self.log.debug(message)
            # Extract template update.
            update = updater_packet['payload']
            # Save template update.
            if self.output_directory is not None:
                self._save_update(update)

        self._measure_time(label='end', frequency=10)

        return
