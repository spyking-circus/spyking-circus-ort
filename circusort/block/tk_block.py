import traceback

from circusort.block.block import Block
from circusort.base.endpoint import EOCError, LOCError


class TkBlock(Block):

    def __init__(self, **kwargs):

        Block.__init__(self, **kwargs)

    def _initialize(self):

        raise NotImplementedError()

    def _tk_initialize(self):

        raise NotImplementedError()

    def run(self):

        if not self.ready:
            self.initialize()

        self._tk_initialize()

        # Log debug message.
        string = "{} is running"
        message = string.format(self.name)
        self.log.debug(message)

        self.running = True
        self._set_start_step()

        try:

            if self.nb_steps is not None:
                while self.counter < self.nb_steps:
                    self._process()
                    self.counter += 1
                    # if numpy.mod(self.counter, self.check_interval) == 0:
                    #     self._check_real_time_ratio()
            else:
                try:
                    while self.running and not self.stop_pending:
                        self._process()
                        self.counter += 1
                except (LOCError, EOCError):
                    for output in self.outputs.itervalues():
                        output.send_end_connection()
                    self.stop_pending = True
                    self.running = False
                if self.running and self.stop_pending and self.nb_inputs == 0:
                    # In this condition, the block is a source block.
                    for output in self.outputs.itervalues():
                        output.send_end_connection()
                    self.running = False
                try:
                    while self.running and self.stop_pending:
                        self._process()
                        self.counter += 1
                        # if numpy.mod(self.counter, self.check_interval) == 0:
                        #     self._check_real_time_ratio()
                except (LOCError, EOCError):
                    for output in self.outputs.itervalues():
                        output.send_end_connection()
                    self.running = False

        except Exception as e:  # i.e. unexpected exception

            # Send EOC signal through each output.
            for output in self.outputs.itervalues():
                output.send_end_connection()
            # Switch running flag.
            self.running = False
            # Log exception name and trace.
            exception_name = e.__class__.__name__
            exception_trace = traceback.format_exc()
            # Log debug message.
            string = "{} in block {}: {}"
            message = string.format(exception_name, self.name, exception_trace)
            self.log.error(message)

        self._tk_finalize()

        # Log debug message.
        string = "{} is stopped"
        message = string.format(self.name)
        self.log.debug(message)

        self._introspect()
        if self.introspection_path is not None:
            self._save_introspection()

        return

    def _process(self):

        raise NotImplementedError()

    def _tk_finalize(self):

        raise NotImplementedError()
