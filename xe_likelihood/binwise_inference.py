import json
import logging
import logging
import sys
import warnings

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as join_path

from .hist2d import Hist2DCollection
from .spectrum import Spectrum
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


warnings.simplefilter("ignore", lineno=72)
warnings.simplefilter("ignore", lineno=2116)
warnings.simplefilter("ignore", lineno=2117)

if ('ipykernel' in sys.modules):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger('xe_bin_logger')
logging.basicConfig(format='%(levelname)s:\t%(funcName)s\t | %(message)s')


class BinwiseInference:
    """ 
    Class containing information required for inference. A class instance needs 
    to be initialised for each experiment. An Experiment is defined by:
        * A binned migration matrix (to transform true recoil spectrum 
          to reconstructed spectrum)
        * A likelihood matrix (provided per detector run, or per MC 
          simulation instance). 
    Constructors for XENON1T and XENONNT using the relevant variables 
    are provided. Data validations are preformed.
    """
    _spectrum = None
    _spectrum_in_reconstructed_bins = None
    _srm_max = None
    _llrs = None
    _uls  = None

    def __init__(self, name,
                migration_matrix,
                likelihood_matrix,
                threshold_function=None,
                spectrum = None,
                ):
        """ 
        Experiment initialized by reading a threshold file, migration matrix, 
        and a likelihood matrix. Perform a check that the x-axis of the 
        likelihood function is identical to the y-axis of the migration matrix. 
        This represents the binning of the reconstructed energy spectrum.
            :spectrum: callable true nuclear recoil energy spectrum, in units of 1/keV
            :toymc_n:  integer, number of toyMC to consider in projections            
        """
        self.name = name
        self.migration_matrix = migration_matrix 
        self.likelihood_matrix = likelihood_matrix
        self.threshold_function = threshold_function
        self.spectrum = spectrum

    @classmethod
    def from_xenon1t_sr(cls, spectrum=None):
        """
        Initializes an instance of the XENON1T Experiment 
            :spectrum: callable true nuclear recoil energy spectrum, in units of 1/keV
        """
        from .data import xe1t_nr_1ty_result

        migration_matrix = Hist2DCollection.from_package(xe1t_nr_1ty_result, "XENON1T_binwise_NR_MM.json",
                                                         name="XENON1T Migration-Matrix")
        likelihood_matrix = Hist2DCollection.from_package(xe1t_nr_1ty_result, "XENON1T_binwise_NR_LL.json",
                                                         name="XENON1T Likelihood-Matrix")
        
        threshold_data = json.load(pkg_resources.open_text(xe1t_nr_1ty_result, 
                                            "threshold_using_lines_90pct.json"))
        x = threshold_data['X']['Data']
        y = threshold_data['Y']['Data']
        threshold_function = interp1d(x, y, fill_value=sps.chi2(1).isf(0.1), bounds_error=False)
        logger.info("Loaded XENON1T Experiment, with llr curves.")

        return cls(name="XENON1T",
                migration_matrix=migration_matrix,
                likelihood_matrix=likelihood_matrix,
                threshold_function=threshold_function,
                spectrum=spectrum)

    @classmethod
    def from_xenonnt_mc(cls, spectrum=None):
        """
        Initializes an instance of the XENONnT Experiment 
            :spectrum: callable true nuclear recoil energy spectrum, in units of 1/keV
            :toymc_n:  integer, number of toyMC to consider in projections 
        """
        from .data import xent_nr_20ty_projection

        migration_matrix = Hist2DCollection.from_package(xent_nr_20ty_projection, "XENONnT_binwise_NR_MM.json",
                                                         name="XENONnT Migration-Matrix")
        likelihood_matrix = Hist2DCollection.from_package(xent_nr_20ty_projection, "XENONnT_binwise_NR_LL_toyMC.json",
                                                         name="XENONnT Likelihood-Matrix")
        
        return cls(name="XENONnT",
                migration_matrix=migration_matrix,
                likelihood_matrix=likelihood_matrix,
                threshold_function=None,
                spectrum=spectrum)
    
    @property
    def true_energy_bins(self):
        return self.migration_matrix.x_bins
    
    @property
    def reconstructed_energy_bins(self):
        return self.migration_matrix.y_bins

    @property
    def true_energy_edges(self):
        return self.migration_matrix.x_edges

    @property
    def reconstructed_energy_edges(self):
        return self.migration_matrix.y_edges

    @property
    def signal_expectations_bins(self):
        return self.likelihood_matrix.y_bins

    @property
    def signal_expectations_edges(self):
        return self.likelihood_matrix.y_edges

    @property
    def spectrum(self):
        if self._spectrum is None:
            raise RuntimeError('Spectrum has not been set.')
        return self._spectrum
    
    @spectrum.setter
    def spectrum(self, spectrum):
        """ 
        Loads a spectrum in true nuclear recoil, and transforms it to the reconstructued energy 
        spectrum using the migration matrix. Resets previously calculated log-likelihod ratio
        and upper limit values.
            :spectrum: callable true nuclear recoil energy spectrum, in units of 1/(keV*ty)
        """
        if spectrum is None:
            self._spectrum = spectrum
            return

        if not callable(spectrum):
            raise ValueError('Spectrum must be a callable.')

        if not isinstance(spectrum, Spectrum):
            spectrum = Spectrum.from_pdf(spectrum, "")

        self._spectrum = spectrum
        self._llrs = None
        self._uls  = None
        self._srm_max = None
        self._spectrum_in_reconstructed_bins = None

    @property
    def binned_signal_model(self):
        return np.asarray([quad(self.spectrum, ed, eu)[0] for ed, eu in 
                        zip(self.true_energy_edges[:-1], self.true_energy_edges[1:])])
    @property
    def spectrum_in_reconstructed_bins(self):
        if self._spectrum_in_reconstructed_bins is None:
            self._spectrum_in_reconstructed_bins = np.matmul(self.binned_signal_model, self.migration_matrix[0])
        return self._spectrum_in_reconstructed_bins

    @property
    def sum_spectrum_in_reconstructed_bins(self):
        return np.sum(self.spectrum_in_reconstructed_bins)

    @property
    def run_count(self):
        return len(self.likelihood_matrix)

    @property
    def srm_max(self):
        if self._srm_max is None:
            self._srm_max = self.signal_expectations_bins[-1]/self.spectrum_in_reconstructed_bins.max()
        return self._srm_max

    @property
    def llrs(self):
        if self._llrs is None:
            self.compute_likelihood_ratios()
        return self._llrs

    @property
    def uls(self):
        if self._uls is None:
            self.compute_uls()
        return self._uls

    def compute_likelihood_ratio(self, run=0, spectrum=None):
        """
        Compute the log-likelihod ratio using the  binwise likelihood profiles.
            :spectrum:  callable true nuclear recoil energy spectrum, in units of 1/keV
            :run: integer, index of run/toyMC to consider in projections  
        """
        if spectrum is not None:
            self.spectrum = spectrum
        
        assert run < self.run_count, "Only {:d} data results are available".format(self.run_count)
        logger.info("Constructing {:d}/{:d} likelihood curves for {:s}".format(run, self.run_count, self.name))
        
        srms    = np.linspace(0, self.srm_max, 1000)
        llrs    = np.zeros(len(srms))
        likelihood_matrix = self.likelihood_matrix[run]
        for i, mu_bin in enumerate(self.spectrum_in_reconstructed_bins):
            f_llr_bin = interp1d(self.signal_expectations_bins, likelihood_matrix[i, :],
                                 bounds_error=False, fill_value=np.inf)
            llrs += f_llr_bin(mu_bin * srms)
        llrs -= llrs.min()
        f_llr = interp1d(srms, llrs, bounds_error=False, fill_value=np.inf)
        return f_llr
        
    def compute_likelihood_ratios(self, spectrum=None):
        """
        Compute likelihood ratios for all toyMCs.
            :spectrum:  callable true nuclear recoil energy spectrum, in units of 1/keV
        """
        if spectrum is not None:
            self.spectrum = spectrum
        f_llrs = []
        disable_tqdm = self.run_count < 5
        for n in tqdm(range(self.run_count), disable=disable_tqdm, desc="Computing likelihood ratio curves"):
            f_llrs.append(self.compute_likelihood_ratio(run=n))
        self._llrs = f_llrs

    def compute_ul(self, cl=0.1, asymptotic=True, run=0):
        """
        Computes upper limit using the binwise likelihood arguments.
            :cl:          float, confidence level to be used for the asymptotic threshold
            :spectrum:    callable true nuclear recoil energy spectrum, in units of 1/keV
            :asymptotic:  bool, if False, will  use the optional toyMC-generated Neyman threshold
            :run: integer, index of toyMC to consider in projections
        returns ul: float, the multiple of the input spectrum at which the upper limit is
        """
        
        logger.info("Computing {:.1f} cl limit, asymptotic? {:d}".format(cl, int(asymptotic)))
        assert asymptotic or self.threshold_function is not None, "No function defined for nonasymptotic inference "
        if asymptotic:
            threshold = lambda x: sps.chi2(1).isf(cl)  # 2-sided asymptotic threshold
        else:
            assert self.threshold_function is not None, "No threshold function defined"
            mutot = self.sum_spectrum_in_reconstructed_bins
            threshold = lambda x: self.threshold_function(x * mutot)

        f_llr = self.compute_likelihood_ratio(run=run)

        fmin = minimize(f_llr, [0], method="Powell")
        xmin = fmin["x"]
        llmin = fmin["fun"]

        llz = lambda x: f_llr(x) - llmin - threshold(x)
        ul = brentq(llz, xmin, f_llr.x[-1])
        logger.info("ul={:.2e}".format(ul))
        return ul

    def compute_uls(self, cl=0.1, spectrum=None, asymptotic=True, nmax=float('inf')):
        """
        Compute upper limits using the binwise likelihood
            :cl:          float, confidence level to be used for the asymptotic threshold
            :spectrum:    callable true nuclear recoil energy spectrum, in units of 1/keV
            :asymptotic:  bool, if False, will  use the optional toyMC-generated Neyman threshold
        returns ul: array of floats, the multiple of the input spectrum at which the upper limit is
        """
        if spectrum is not None:
            self.spectrum = spectrum

        nruns = min(self.run_count, nmax)
        uls = np.zeros(nruns)

        logger.info("Computing {:d} {:.1f} cl limits, asymptotic? {:d}".format(nruns, cl, int(asymptotic)))
        disable_tqdm = nruns < 5

        for n in tqdm(range(nruns), disable=disable_tqdm, desc="Computation of {:.1f} CL UL ".format(cl)):
            uls[n] = self.compute_ul(cl, asymptotic=asymptotic, run=n)
        self._uls = uls
        return uls

    def compute_ul_percentiles(self, quantiles=sps.norm().cdf([-1, 0, 1])):
        """
        Returns percentiles of the limits for sensitivity computations.
            :quantiles: 
        """
        return ([np.percentile(self.uls, 100. * q) for q in quantiles])

    def plot_migration_matrix(self, run=0, show=False):
        """
        Displays run specific migration matrix
        """
        self.migration_matrix.plot(idx=run, show=show)
        if show:
            plt.show()

    def plot_likelihood_matrix(self, run=0, show=False):
        """
        Displays an individual likelihood matrix for the experiment.
            :run: integer, Index of run/toyMC likelihood to display.
        """
       
        self.likelihood_matrix.plot(idx=run, show=False)
        plt.plot(self.reconstructed_energy_bins, self.spectrum_in_reconstructed_bins / 10, label="Ereco")
        if show:
            plt.show()

    def plot_spectrum(self, spectrum=None, show=False):
        """
        Displays input true nuclear recoil spectrum and recosntrcuted energy spectru
        calculated using migration matrix.
        """
        if spectrum is not None:
            self.spectrum = spectrum
        plt.plot(self.true_energy_bins, self.binned_signal_model, label="E$_\mathrm{true}$")
        plt.plot(self.reconstructed_energy_bins, self.spectrum_in_reconstructed_bins, label="E$_\mathrm{rec}$")
        plt.title("Recoil energy " + self.spectrum.title)
        plt.xlabel("Energy [KeV]")
        plt.legend(loc="upper right")
        plt.ylabel("Normalized Differential Recoil [dR/dE]")
        if (show):
            plt.show()  # block=False

    def plot_threshold_function(self, show=False):
        """ 
        Displys the non-asymptotic threshold function, if provided (e.g. for XENON1T)
        """
        plt.title("Threshold function")
        if self.threshold_function is not None:
            plt.plot(self.threshold_function.x, self.threshold_function.y, color="k")
        if (show):
            plt.show()
            
    def plot_llr_curve(self, run=0, show=False):
        if run >= self.run_count:
            run = self.run_count - 1
        llr = self.compute_likelihood_ratio(run=run)
        plt.title("LLR curve " + str(run) + " " + self.name + " " + self.spectrum.title)
        plt.plot(llr.x, llr.y, color="k")
        plt.xlabel("Signal Multiple") 
        plt.ylabel("Likelihood Ratio")
        if (show):
            plt.show()

    def plot_loglikelihood_ratio(self, run=0, show=False):
        """
        Displays log-likelihood ratio curve.
            :run: integer, Index of toyMC likelihood to display.
        """
        if (run >= self.run_count):
            run = self.run_count - 1
        llr = self.compute_likelihood_ratio(run=run)
        plt.title("LLR curve " + str(run) + " " + self.name + " " + self.spectrum.title)
        plt.plot(llr.x, llr.y, color="k",
                 label="LLR curve #" + str(run))
        x_range = llr.x
        plt.plot(x_range, sps.chi2(1).isf(0.1) * np.ones(len(x_range)), color="gray", linestyle="--",
                 label="Asymptotic")
        if self.threshold_function is not None:
            plt.plot(self.threshold_function.x / self.sum_spectrum_in_reconstructed_bins,
                     self.threshold_function.y, color="gray", linestyle="-", label="Threshold Function")
        upper_limit = self.compute_uls(cl=0.1, asymptotic=True)
        print("Calculating asymptotic upper limit for log ratio plot:", upper_limit)
        plt.scatter([upper_limit], [llr(upper_limit)], color="magenta",
                    label="Asymptotic 0.1 limit=" + str(upper_limit[0]))
        if self.threshold_function is not None:
            upper_limit_nonasymptotic = self.compute_uls(cl=0.1, asymptotic=False)
            print("Calculating nonasymptotic upper limit for log ratio plot:", upper_limit_nonasymptotic)
            plt.scatter([upper_limit_nonasymptotic], [llr(upper_limit_nonasymptotic)],
                        color="cyan",
                        label="Non-asymptotic 0.1 limit=" + str(upper_limit_nonasymptotic))
        plt.xlabel("Signal Multiple") 
        plt.ylabel("Likelihood Ratio")
        plt.legend(loc="upper left")
        if (show):
            plt.show()

    def plot_summary(self, save=None, show=False):
        """
        Displays sumamry plot of all steps of the inference.
            :fname:       string, Filename of summary plot
            :run: integer, Index of toyMC likelihood to display.
        """
        if isinstance(save, str):
            fname = save
        else:
            fname = self.name + '_inference'
        if "." not in fname:
            fname = fname + ".png"
        plt.figure(figsize=(28, 14))
        plt.subplot(243)
        self.plot_spectrum()
        plt.subplot(244)
        self.plot_threshold_function()
        plt.subplot(247)
        self.plot_migration_matrix()
        plt.subplot(248)
        self.plot_likelihood_matrix()
        plt.subplot(221)
        self.plot_loglikelihood_ratio()
        plt.subplot(223)
        # if (len(self.llrs) > 1):
        #     self.compute_ul(cl=0.1)
        self.plot_histogram()
        if save and fname:
            plt.savefig(fname)
            print("Summary images saved to " + fname)
        if show:
            plt.show()
        

    def plot_histogram(self):
        v = self.uls
        assert len(v) > 0, "Not enough values found."
        n, bins, patches = plt.hist(v, density=False, facecolor='g', alpha=0.75,
                                    bins=np.logspace(np.log10(min(v) / 10), np.log10(max(v) * 10), 50))
        plt.xlabel('MC ' + self.name + ' 0.1 limits.')
        plt.ylabel('#')
        plt.title('Sensitivity for ' + self.name + " N=" + str(len(v)))
        plt.xscale('log')
        plt.ylim(0, max(n) * 1.1)
        plt.xscale('log')
        plt.grid(True)
