import time

from circusort.block.block import Block


__classname__ = "ClusterWriter"


class ClusterWriter(Block):

    name = "Cluster writer"

    params = {}

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)
        self.add_input('templates', structure='dict')

    def _initialize(self):

        return

    def _process(self):

        templates_packet = self.get_input('templates').receive(blocking=False)

        self._measure_time(label='start', frequency=10)

        if templates_packet is None:
            duration = 0.01  # s
            time.sleep(duration)
        else:
            templates = templates_packet['payload']
            # TODO remove the 3 following lines.
            string = "{} receives templates: {}"
            message = string.format(self.name, templates)
            self.log.debug(message)
            # TODO remove the 2 following lines.
            duration = 0.01  # s
            time.sleep(duration)

        self._measure_time(label='end', frequency=10)

        return
