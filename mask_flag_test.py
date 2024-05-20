import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import emcee
import astropy.units as u
from astropy.io import fits
import os
import corner
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps, trapz
import pyccl as ccl
from scipy.signal import convolve2d as conv2d
from configparser import ConfigParser
from pixell import enmap, utils
from helpers import *
import profiles
import importlib
import warnings
import emcee
import time
from cluster_data import *
from plottery.plotutils import colorscale

warnings.filterwarnings(
    "ignore",
    message="Data has no positive values, and therefore cannot be log-scaled.",
    category=UserWarning,
)

# loading config.ini
current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path + "/config.ini"
config = ConfigParser()
config.optionxform = str

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

data_path = config["FILES"]["DATA_PATH"]
profile_stacked_model = config["STACKED_HALO_MODEL"]["profile"]
profiles_module = importlib.import_module("profiles")
MCMC_func = importlib.import_module("MCMC_functions")
with_pool = bool(int(config["EXTRACT"]["WITH POOL"]))
completeness_file = config["FILES"]["COMPLETENESS"]
cosmological_model = dict(config["COSMOLOGICAL MODEL"])
cosmological_model = {
    key: float(cosmological_model[key]) for key in list(cosmological_model.keys())
}

prior_parameters = dict(config["STACKED_HALO_MODEL"])
prior_parameters = {
    key: list(prop2arr(prior_parameters[key], dtype=str))
    for key in list(prior_parameters.keys())
}
prior_parameters = list(prior_parameters.values())


nwalkers = int(config["STACKED_HALO_MODEL"]["nwalkers"])
nsteps = int(config["STACKED_HALO_MODEL"]["nsteps"])

if not os.path.exists(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"])

if not os.path.exists(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"])

match = False
data_mask_ratio = 0.2
skip = 1




def main():
    DR6 = config["FILES"]["DR6-ACT-map"]
    DR5 = config["FILES"]["DR5-ACT-map"]
    DES_Y3 = config["FILES"]["Y3-REDMAPPER"]
    MASK_DR6 = config["FILES"]["MASK_DR6-ACT-map"]
    width, w_units = prop2arr(config["EXTRACT"]["width"], dtype=str)
    width = np.deg2rad(float(width)) if w_units == "deg" else float(width)
    ufrom, uto = prop2arr(config["EXTRACT"]["CHANGE_UNIT"], dtype=str)
    zmin, zmax = prop2arr(config["EXTRACT"]["redshift"], dtype=np.float64)
    rewrite = bool(config["EXTRACT"]["REWRITE"])
    # load data
    print(f"Loading DATA from \033[92m{data_path}\033[0m.")
    szmap = enmap.read_map(data_path + DR6)
    dr5 = fits.open(data_path + DR5)[1].data
    redmapper = fits.open(data_path + DES_Y3)[1].data
    redmapper = redmapper[(redmapper["Z"] > zmin) & (redmapper["Z"] < zmax)]
    redmapper_RA, redmapper_dec = redmapper["RA"], redmapper["DEC"]
    mask = enmap.read_map(data_path + MASK_DR6)
    R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
    R_units = config["CLUSTER PROPERTIES"]["r_units"]
    FWHM,FWHM_units = prop2arr(config["CLUSTER PROPERTIES"]["FWHM"],dtype = str)
    FWHM = np.deg2rad(float(FWHM)) / 60 if FWHM_units == "arcmin" else float(FWHM)
    try:
        R_profiles = R_profiles * getattr(u, R_units)
    except:
        print(
            f"The units \033[92m{R_units}\033[0m doesn't exist in \033[95m{u}\033[0m."
        )
    if uto == "rad":
        print(f"Changing units from \033[92m{ufrom}\033[0m to \033[92m{uto}\033[0m.")
        redmapper_RA[redmapper_RA > 180] = redmapper_RA[redmapper_RA > 180] - 360
        redmapper_RA = np.deg2rad(redmapper_RA)
        redmapper_dec = np.deg2rad(redmapper_dec)
    if match == True:
        dr5 = redmapper[(dr5["redshift"] > zmin) & (dr5["redshift"] < zmax)]
        dr5ra, dr5dec = dr5["RADeg"], dr5["decDeg"]
        dr5ra[dr5ra > 180] = dr5ra[dr5ra > 180] - 360
        dr5ra, dr5dec = np.deg2rad(dr5ra), np.deg2rad(dr5dec)
        p1 = np.array([[dr5ra[i], dr5dec[i]] for i in range(len(dr5ra))])
        p2 = np.array([[redmapper_RA[i], szDec[i]] for i in range(len(szRA))])
        distances, index = KD(p1).query(p2)
        distances2, index2 = KD(p2).query(p1)
        ind = np.unique(index[distances < r0])
        ind2 = np.unique(index2[distances2 < r0])
    ra, dec = redmapper_RA, redmapper_dec
    masks = []
    div = []
    richness_list = []
    for i in range(len(ra)):
        box = [
            [dec[i] - width / 2.0, ra[i] - width / 2.0],
            [dec[i] + width / 2.0, ra[i] + width / 2.0],
              ]
        smask = mask.submap(box)
        masks.append(smask)
    N_total = len(ra)
    ratios = np.linspace(0,0.8,3)
    for ratio in ratios:
        NAs = 0
        NRs = 0
        rich = []
        for j in range(len(masks)):
            mask = masks[j]
            zeros = np.size(mask[mask != 1])
            size = np.size(mask)
            if zeros  >= (1 - ratio)*size:
                NRs+=1
            else:
                rich.append(redmapper[j]["LAMBDA_CHISQ"])
        div.append(N_total - NRs)
        richness_list.append(rich)
    fig,ax = plt.subplots()
    ax.plot(ratios,div,color='black')
    ax.grid(True)
    ax.set(xlabel="ratio",ylabel="$N_{total} / N_{accepted}$",title="mask test")
    fig.savefig("mask_test.png")
    plt.close()
    fig,ax = plt.subplots()
    for i in range(len(ratios)):
        ax.hist(richness_list[i],histtype='step',log=True,label=f"ratio = {ratios[i]}")
    ax.set(ylabel="N_clusters",xlabel="richness")
    ax.legend()
    fig.savefig("maks_hist.png")
if __name__ == "__main__":
    main()
