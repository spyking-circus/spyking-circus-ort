import h5py
import numpy as np
import tempfile
import os

from circusort.block.block import Block


class Writer(Block):
    """Writer.

    Attribute:
        data_path: none | string
        dataset_name: none | string
        mode: string
        nb_samples: integer
        sampling_rate: float
    """
    # TODO complete docstring

    name = "File writer"

    params = {
        'data_path': None,
        'dataset_name': None,
        'mode': None,
        'nb_samples': 1024,
        'sampling_rate': 20e+3,
    }

    def __init__(self, **kwargs):
        """Initialization.

        Parameter:
            data_path: none | string (optional)
                The path to the file to use to write the data.
                The default value is None.
            dataset_name: none | string (optional)
                The name to use for the HDF5 dataset.
                The default value is None
            mode: none | string (optional)
                The mode to use to write into the file.
                The default value is None.
            nb_samples: integer (optional)
                The number of sampling times for each buffer.
                The default value is 1024.
            sampling_rate: float (optional)
                The sampling rate used to record the data.
                The default value is 20e+3.
        """

        Block.__init__(self, **kwargs)
        self.add_input('data', structure='dict')

        # Lines useful to remove some PyCharm warnings.
        self.data_path = self._get_temp_file() if self.data_path is None else self.data_path
        self.dataset_name = 'dataset' if self.dataset_name is None else self.dataset_name
        self.mode = 'default' if self.mode is None else self.mode
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate

        self._dtype = None
        self._nb_samples = None
        self._nb_channels = None
        self._previous_number = None
        self._raw_file = None
        self._h5_file = None
        self._h5_dataset = None

    @staticmethod
    def _get_temp_file(mode='default'):
        """Get a path to a temporary file.

        Parameter:
            mode: none | string
                The mode to use to write into the file.
        """

        # Retrieve mode.
        if mode in ['default', 'raw']:
            extension = ".raw"
        elif mode in ['hdf5', 'h5']:
            extension = ".h5"
        else:
            # Raise value error.
            string = "Unknown mode value: {}"
            message = string.format(mode)
            raise ValueError(message)

        directory = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile() as file_:
            name = os.path.basename(file_.name)
            filename = name + extension
        data_path = os.path.join(directory, filename)

        return data_path

    def _initialize(self):

        # Retrieve extension.
        extension = os.path.splitext(self.data_path)[1]
        if extension in [".raw", ".dat", ".bin"]:
            self.mode = 'raw'
        elif extension == ".h5":
            self.mode = 'hdf5'
        else:
            # Raise value error.
            string = "Unknown extension value: {}"
            message = string.format(extension)
            raise ValueError(message)

        # Log info message.
        string = "{} records data into {}"
        message = string.format(self.name, self.data_path)
        self.log.info(message)

        # Retrieve mode.
        if self.mode == 'raw':
            self._raw_file = open(self.data_path, mode='wb')
            # TODO remove the following line?
            self.recorded_keys = {}
        elif self.mode == 'hdf5':
            self._h5_file = h5py.File(self.data_path, mode='w', swmr=True)
        else:
            # Raise value error.
            string = "Unknown mode value: {}"
            message = string.format(self.mode)
            raise ValueError(message)

        return

    def _configure_input_parameters(self, dtype=None, nb_samples=None, nb_channels=None, **kwargs):

        self._dtype = dtype
        self._nb_samples = nb_samples
        self._nb_channels = nb_channels

        return

    def _write(self, batch, number=None):

        if isinstance(batch, np.ndarray):
            if number is not None:
                if self._previous_number is not None:
                    # Fill missing batches with zeros.
                    for _ in range(self._previous_number + 1, number):
                        zeroed_batch = np.zeros(batch.shape, dtype=batch.dtype)
                        self._write_aux(zeroed_batch)
                self._previous_number = number
            self._write_aux(batch)
        else:
            # Log error message.
            string = "{} can only write arrays"
            message = string.format(self.name)
            self.log.error(message)

        return

    def _write_aux(self, batch):

        if self.mode == 'raw':
            self._raw_file.write(batch.tostring())
        elif self.mode == 'hdf5':
            if self.dataset_name in self._h5_file:
                dataset = self._h5_file[self.dataset_name]
                shape = dataset.shape
                shape_ = (shape[0] + batch.shape[0], shape[1])
                dataset.resize(shape_)
                dataset[shape[0]:, ...] = batch
                dataset.flush()
            else:
                max_shape = batch.ndim * (None,)
                dataset = self._h5_file.create_dataset(self.dataset_name, data=batch,
                                                       chunks=True, maxshape=max_shape)
                dataset.flush()
        else:
            # Raise value error.
            string = "Unknown mode value: {}"
            message = string.format(self.mode)
            raise ValueError(message)

        return

    def _process(self):

        data_packet = self.get_input('data').receive()
        number = data_packet['number']
        batch = data_packet['payload']

        self._measure_time(label='start', period=10)

        self._write(batch, number=number)

        self._measure_time(label='end', period=10)

        return

    def _introspect(self):

        nb_buffers = self.counter - self.start_step
        start_times = np.array(self._measured_times.get('start', []))
        end_times = np.array(self._measured_times.get('end', []))
        durations = end_times - start_times
        data_duration = float(self.nb_samples) / self.sampling_rate
        ratios = data_duration / durations

        min_ratio = np.min(ratios) if ratios.size > 0 else np.nan
        mean_ratio = np.mean(ratios) if ratios.size > 0 else np.nan
        max_ratio = np.max(ratios) if ratios.size > 0 else np.nan

        # Log info message.
        string = "{} processed {} buffers [speed:x{:.2f} (min:x{:.2f}, max:x{:.2f})]"
        message = string.format(self.name, nb_buffers, mean_ratio, min_ratio, max_ratio)
        self.log.info(message)

        return

    def __del__(self):
        """Deletion."""

        if self.mode == 'raw':
            self._raw_file.close()
        elif self.mode == 'hdf5':
            self._h5_file.close()
        else:
            # Raise value error.
            string = "Unknown mode value: {}"
            message = string.format(self.mode)
            raise ValueError(message)
