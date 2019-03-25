from circusort.obj.template_store import TemplateStore
from circusort.utils.path import normalize_path


def load_template_store(path, probe_file=None):
    """Load a template store from disk.

    Parameter:
        path: string
            The path to the HDF5 file from which to load the template store.
    Return:
        store: circusort.obj.TemplateStore
    """

    path = normalize_path(path)
    store = TemplateStore(path, probe_file=probe_file, mode='r')

    return store
