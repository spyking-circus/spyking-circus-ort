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
                The path to the file to use to write the data. The default value is None.
            dataset_name: none | string (optional)
                The name to use for the HDF5 dataset. The default value is None
            mode: none | string (optional)
                The mode to use to write into the file. The default value is None.
            nb_samples: integer (optional)
                The number of sampling times for each buffer. The default value is 1024.
            sampling_rate: float (optional)
                The sampling rate used to record the data. The default value is 20e+3.
        """

        Block.__init__(self, **kwargs)
        self.add_input('data')

        # Lines useful to remove some PyCharm warnings.
        self.data_path = self._get_temp_file() if self.data_path is None else self.data_path
        self.dataset_name = 'dataset' if self.dataset_name is None else self.dataset_name
        self.mode = 'default' if self.mode is None else self.mode
        self.nb_samples = self.nb_samples
        self.sampling_rate = self.sampling_rate

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

        if mode in ['default', 'raw']:
            extension = ".raw"
        elif mode in ['hdf5', 'h5']:
            extension = ".h5"
        else:
            message = "Unknown mode value: {}".format(mode)
            raise ValueError(message)

        directory = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile() as file_:
            name = os.path.basename(file_.name)
            filename = name + extension
        data_path = os.path.join(directory, filename)

        return data_path

    def _initialize(self):
        # TODO add docstring.

        extension = os.path.splitext(self.data_path)[1]
        if extension == ".raw":
            self.mode = 'raw'
        elif extension == ".h5":
            self.mode = 'hdf5'
        else:
            message = "Unknown extension value: {}".format(extension)
            raise ValueError(message)

        message = "{} records data into {}".format(self.name, self.data_path)
        self.log.info(message)

        if self.mode == 'raw':
            self._raw_file = open(self.data_path, mode='wb')
            # TODO remove the following line?
            self.recorded_keys = {}
        elif self.mode == 'hdf5':
            self._h5_file = h5py.File(self.data_path, mode='w', swmr=True)
        else:
            message = "Unknown mode value: {}".format(self.mode)
            raise ValueError(message)

        return

    def _process(self):
        # TODO add docstring.

        batch = self.input.receive()

        self._measure_time(label='start', frequency=100)  # TODO check location.

        if self.input.structure == 'array':
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
                message = "Unknown mode value: {}".format(self.mode)
                raise ValueError(message)
        else:
            message = "{} can only write arrays".format(self.name)
            self.log.error(message)

        self._measure_time(label='end', frequency=100)

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

    def __del__(self):
        """Deletion."""

        if self.mode == 'raw':
            self._raw_file.close()
        elif self.mode == 'hdf5':
            self._h5_file.close()
        else:
            message = "Unknown mode value: {}".format(self.mode)
            raise ValueError(message)
