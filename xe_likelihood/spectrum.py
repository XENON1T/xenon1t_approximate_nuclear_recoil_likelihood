import logging
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as sps
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

logger = logging.getLogger('xe_bin_logger')
logging.basicConfig(format='%(levelname)s:\t%(funcName)s\t | %(message)s')


class Spectrum:
    """
    Utility class to handle dR/dE inputs and produce a callable function.
    This is the differential rate as a function of the true deposited energy
    in the detector, in the range 1-70 keV, with units in keV
    """

    def __init__(self, rate, scaling_factor=1, title=''):
        assert callable(rate), 'Spectrum rate must be a callable.'
        self.rate = rate
        self.scaling_factor = scaling_factor
        self.title = title
        logger.info('Spectrum ' + title + ' loaded with scaling factor ', scaling_factor)

    @classmethod
    def from_histogram(cls, ebins, rates, title='from histogram'):
        """ 
        Produces callable dR/dE spectrum from a histogram
        """
        norm = np.sum(rates)
        pdf = sps.rv_histogram((rates, ebins)).pdf
        return cls(pdf, title=title, scaling_factor=norm)

    @classmethod
    def from_sample(cls, energies, rates, title='from sample'):
        """ 
        Produces callable dR/dE spectrum from arrays of energy and rate.
        """
        rate = interp1d(energies, rates, bounds_error=False, fill_value=0.)
        return cls(rate, title=title)

    @classmethod
    def from_pdf(cls, func, title='from pdf'):
        """
        Produces callable dR/dE spectrum from a scipy function
        """
        return cls(func, title=title)

    @classmethod
    def from_csv(cls, file, delimiter=',', column_energy=0, column_drde=1, title='from csv'):
        """
        Produces callable dR/dE spectrum from a scipy function.
        """
        d = np.loadtxt(file, delimiter=delimiter)
        rate = interp1d(d[:, column_energy], d[:, column_drde], bounds_error=False, fill_value=0.)
        return cls(rate, title=title)

    @classmethod
    def from_wimp(cls, mass=50):
        if not mass in [50, 100]:
            raise ValueError('Only 50GeV and 100GeV wimp spectra are stored.')
        from .data import ancillary_data

        fname = f"dRdE_wimp_SI_{mass}GeV_10-45.csv"
        file = pkg_resources.open_text(ancillary_data, fname)
        return cls.from_csv(file, title=f'{mass} GeV, SI WIMP, x1e-45 tonne*year')
    
    def set_title(self, title):
        self.title = title

    def __call__(self, e):
        return self.rate(e) * self.scaling_factor
