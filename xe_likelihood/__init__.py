
__author__ = """XENON Experiment"""
__email__ = 'knut.dundas.moraa@columbia.edu'
__version__ = '0.1.0'

import numpy as np

from .demo import demonstration
from .binwise_inference import BinwiseInference
from .spectrum import Spectrum
from .data import ancillary_data

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

xenon1t_published = np.loadtxt(pkg_resources.open_text(ancillary_data, "xenon1t_1ty.csv"),
                                delimiter=",", skiprows=1)