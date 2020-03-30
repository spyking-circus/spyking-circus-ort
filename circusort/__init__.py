import os

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

if os.environ['DISPLAY'] == ':0':
    import matplotlib
    matplotlib.use('Agg')  # i.e. non-interactive backend
    import matplotlib.pyplot as plt
    assert plt.get_backend().lower() == 'agg'

from . import block
from . import cli
from . import io
from . import net
from . import obj
from . import plt
from . import utils

from .base import create_director


__version__ = '0.0.1'
