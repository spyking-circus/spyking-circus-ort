import time

from circusort.block.block import Block


__classname__ = "UpdaterWriter"


class UpdaterWriter(Block):

    name = "Updater writer"

    params = {}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('updater', structure='dict')

    def _initialize(self):

        return

    def _process(self):

        updater_packet = self.get_input('updater').receive(blocking=False)

        self._measure_time(label='start', frequency=10)

        if updater_packet is None:
            duration = 0.01  # s
            time.sleep(duration)
        else:
            updater = updater_packet['payload']
            # TODO remove the 3 following lines.
            string = "{} receives updater: {}"
            message = string.format(self.name, updater)
            self.log.debug(message)
            # TODO remove the 2 following lines.
            duration = 0.01  # s
            time.sleep(duration)

        self._measure_time(label='end', frequency=10)

        return
