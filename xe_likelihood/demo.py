import scipy.stats as sps
import numpy as np
import logging
import sys
from xe_likelihood.binwise_inference import BinwiseInference
import time

logger = logging.getLogger('xe_bin_logger')
logging.basicConfig(format='%(levelname)s:\t%(funcName)s\t | %(message)s')

if ('ipykernel' in sys.modules):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def demonstration(experiment="xenon1t", show=True, savefig=False):
    """
    Runs the limit for a sample spectrum for XENON1T, 
    or computes the sensitivity for a sample spectrum 
    for XENONnT. Uses a 50 GeV/c^2 WIMP spectrum by
    default.
        :experiment: str, xenon1t or xenonnt
        :show:       bool, if True make plots of the likelihood profile
        :savefg: bool or path to save figure
    """
    import wimprates
    import numericalunits as nu
    nu.GeVm = nu.GeV / (nu.c0 ** 2)
    def wimprate_wimp_spectrum(e, M=50, interaction="SI"):
        return wimprates.rate_wimp(e * nu.keV, mw=M * nu.GeVm, sigma_nucleon=1e-45 * nu.cm ** 2,
                                   interaction=interaction) * (nu.keV * (1000 * nu.kg) * nu.year)

    wimp_mass = 50 #GeV/c^2
    test_spectrum = lambda e: wimprate_wimp_spectrum(e, wimp_mass, "SI")
    time0 = time.time()
    if experiment == "xenon1t":
        
        xenon_inference = BinwiseInference.from_xenon1t_sr(spectrum=test_spectrum)
        ul = xenon_inference.compute_ul(run=0)
        time1 = time.time()
        xenon_inference.plot_summary(show=show, save=savefig)
        print(f"Upper limit: {ul}")

    if experiment == "xenonnt":
        xenon_inference = BinwiseInference.from_xenonnt_mc(spectrum=test_spectrum)
        xenon_inference.compute_ul()
        print("Upper limit percentiles: ", xenon_inference.compute_ul_percentiles())
        time1 = time.time()
        xenon_inference.plot_summary(show=show, save=savefig)

    print("Time to compute inference: ",  time1 - time0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--experiment", type=str, default="xenonnt")
    args = parser.parse_args()
    demonstration(experiment=args.experiment, show=True)