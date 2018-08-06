import os

from circusort.obj.record import Record


def load_record(data_path, probe_path, sampling_rate=20e+3, dtype='int16', gain=0.1042):
    """Load data record.

    Arguments:
        data_path: string
            Path from which to load the data.
        probe_path: string
            Path from which to load the probe.
        sampling_rate: float (optional)
            The sampling rate [Hz].
            The default value is 20e+3.
        dtype: float (optional)
            The data type.
            The default value is int16.
        gain: float (optional)
            The data gain.
            The default value is 0.1042.
    Return:
        record: circusort.obj.Record
            Data record.
    """

    # Check if data path exists.
    data_path = os.path.expanduser(data_path)
    data_path = os.path.abspath(data_path)
    if not os.path.isfile(data_path):
        string = "No such data file: {}"
        message = string.format(data_path)
        raise IOError(message)

    # Check if probe path exists.
    probe_path = os.path.expanduser(probe_path)
    probe_path = os.path.abspath(probe_path)
    if not os.path.isfile(probe_path):
        string = "No such probe file: {}"
        message = string.format(probe_path)
        raise IOError(message)

    # Initialize object.
    record = Record(data_path, probe_path, sampling_rate=sampling_rate, dtype=dtype, gain=gain)

    return record
