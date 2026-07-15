import inspect
import h5py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Value, shared_memory, Manager
# Standard Library
import os
import warnings
import importlib
from time import time
from configparser import ConfigParser

# Third-Party Libraries
import numpy as np
from icecream import ic
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner
from tqdm import tqdm
from PIL import Image
from lmfit import Model, Parameters
from sklearn.covariance import LedoitWolf

# Astropy
import astropy
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as planck18
from astropy.wcs import WCS
from astropy.io.fits import Header
from astropy import constants as const
from astropy.table import Table

# Scipy
from scipy.linalg import block_diag
from scipy.interpolate import griddata, interp1d, RectBivariateSpline, RegularGridInterpolator,UnivariateSpline
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.special import erf, erfc, erfinv
from scipy.integrate import simpson as simp, trapezoid as trapz
from scipy.spatial import cKDTree as KD
from scipy.signal import convolve2d as conv2d
from scipy.stats.kde import gaussian_kde
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.special import j0
from scipy.signal import fftconvolve
#aditionals libraries
import numbers
from collections.abc import Sequence

#numba
from numba import jit, njit

# Cosmology & CMB Tools
import pyccl as ccl
from pixell import enmap, utils, reproject

# MCMC Sampling
import emcee

# Custom Modules
from config import *
from helpers import *
import profiles
from profiles import *


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
grouped_clusters_path = config["FILES"]["GROUPED_CLUSTERS_PATH"]
profile_stacked_model = config["STACKED_HALO_MODEL"]["profile"]
profiles_module = importlib.import_module("profiles")
MCMC_func = importlib.import_module("MCMC_functions")
helpers = importlib.import_module("helpers")
with_pool = str2bool(config["EXTRACT"]["WITH POOL"])
redshift_bins = prop2arr(config["EXTRACT"]["REDSHIFT BINS"],dtype=np.float64)
completeness_file = config["FILES"]["COMPLETENESS"]
cosmological_model = dict(config["COSMOLOGICAL MODEL"])
cosmological_model = {
    key: float(cosmological_model[key]) for key in list(cosmological_model.keys())
}
match = str2bool(config["EXTRACT"]["MATCH"])
if match:
    r_match = float(int(config["EXTRACT"]["R_MATCH"])) * u.arcmin

match_agn = str2bool(config["EXTRACT"]["MATCH_AGN"])
if match_agn:
    r_agn_match = float((config["EXTRACT"]["R_AGN_MATCH"])) * u.arcmin


nwalkers = int(config["STACKED_HALO_MODEL"]["nwalkers"])
nsteps = int(config["STACKED_HALO_MODEL"]["nsteps"])
measured_propertie = config["CLUSTER PROPERTIES"]["measured_propertie"]
if not os.path.exists(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"])

if not os.path.exists(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"])
if not os.path.exists(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + '/profiles'):
    os.mkdir(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + '/profiles')

data_mask_ratio = float(config["EXTRACT"]["MASK_RATIO"])

width, w_units = prop2arr(config["EXTRACT"]["width"], dtype=str)
width = np.deg2rad(float(width)) if w_units == "deg" else float(width)

from numba import njit, prange
import numpy as np

def generate_j0_interpolator(xmin = 0, xmax = 1e10, nx = 100000):
    xv = np.linspace(xmin, xmax, nx)
    Jv = j0(xv)
    @njit(fastmath = True)
    def j0_interpolator(x):
        return np.interp(x, xv, Jv)
    return j0_interpolator

def p2h(k, m, z, M, r, R, PRmz, dndM, Pk, bm, bM, h):
    u = np.zeros((len(k), len(z), len(m)))
    for i, zi in enumerate(z):
        for j, mj in enumerate(m):
            lnR, lnP, y2 = loglog_spline_prepare(R, PRmz[:,i,j])
            u[:,i,j] = h.transform(lnR, lnP, y2, k, direction = "forward")
    PhP = np.trapz(dndM[None,:,:] * bm[None,:,:] * u, x = m, axis = 2)
    Pgm = (PhP * Pk.T)[:,None,:] * bM.T
    X = np.geomspace(1e-3, 1e4, 50)
    xlos = np.linspace(1e-2, 0.99, 50)
    P2halo = np.zeros((len(r), len(M), len(z)))
    for i, mi in enumerate(M):
        for j, zj in enumerate(z):
            lnK, lnPgm, y2 = loglog_spline_prepare(k, Pgm[:,i,j])
            xi = h.transform(lnK, lnPgm, y2, X, direction = "inverse")
            _f = interp1d(X, xi, bounds_error = False, fill_value = 0.)
            P2halo[:, i, j] = 2 * r[:,0,i,j] * np.trapz(
                _f(r[:,0,i,j,None]/xlos[None,:]) * (1/(xlos**2 * np.sqrt(1 - xlos**2))[None,:])
                    ,x = xlos, axis = 1) if r.ndim == 4 else 2 * r[:,i,j] * np.trapz(
                _f(r[:,i,j,None]/xlos[None,:]) * (1/(xlos**2 * np.sqrt(1 - xlos**2))[None,:])
                    ,x = xlos, axis = 1)
    return P2halo
    

@njit(fastmath=True, cache=True)
def compute_two_halo_term(j0_interpolator, PRMz, Rgrid, R2halo, dndM2halo, bM2halo,
                          bMobs, Pk, R_Mpc, k2halo, M_arr2halo, PllM, weights, log = False):

    Nlambda_true = PllM.shape[0]
    Nlambda_obs = PllM.shape[1]
    Nr_obs = R_Mpc.shape[0]
    Nm_obs = bMobs.shape[1]
    Nm_2halo = bM2halo.shape[1]
    Nz_obs = bMobs.shape[0]
    Nr_2halo = Rgrid.shape[0]
    Nk = len(k2halo)

    uRM = np.zeros((Nk, Nz_obs, Nm_2halo))
    for k in prange(Nk):
        for zi in range(Nz_obs):
            for m2 in range(Nm_2halo):
                s = 0.0
                for r in range(Nr_2halo - 1):
                    if log:
                        dr = np.log(R2halo[r+1]) - np.log(R2halo[r])
                    else:
                        dr = R2halo[r + 1] - R2halo[r]
                    x0 = R2halo[r]*k2halo[k]
                    x1 = R2halo[r + 1]*k2halo[k]
                    j0_0 = j0_interpolator(x0)
                    j0_1 = j0_interpolator(x1)
                    f0 = 2*np.pi * Rgrid[r, zi, m2] * j0_0 * PRMz[r, zi, m2]
                    f1 = 2*np.pi * Rgrid[r + 1, zi, m2] * j0_1 * PRMz[r + 1, zi, m2]
                    if log:
                        f0 *= R2halo[r]
                        f1 *= R2halo[r + 1]
                    current_term = 0.5 * (f0 + f1) * dr
                    s += current_term if np.isnan(current_term) == False else 0 
                uRM[k, zi, m2] = s
    mass_int = np.zeros((Nk, Nz_obs))
    for k in prange(Nk):
        for zi in range(Nz_obs):
            s = 0.0
            for m2 in range(Nm_2halo - 1):
                if log:
                    dM = np.log(M_arr2halo[m2 + 1]) - np.log(M_arr2halo[m2])
                else:
                    dM = M_arr2halo[m2 + 1] - M_arr2halo[m2]
                f0 = dndM2halo[zi, m2] * bM2halo[zi, m2] * uRM[k, zi, m2]
                f1 = dndM2halo[zi, m2 + 1] * bM2halo[zi, m2 + 1] * uRM[k, zi, m2 + 1]
                if log:
                    f0 *= M_arr2halo[m2]
                    f1 *= M_arr2halo[m2 + 1]
                current_term = 0.5*(f0 + f1) * dM
                s += current_term if np.isnan(current_term) == False else 0
            mass_int[k, zi] = s
    PhP = np.zeros((Nz_obs, Nk, Nm_obs))
    for zi in range(Nz_obs):
        for k in range(Nk):
            for m in range(Nm_obs):
                PhP[zi, k, m] = Pk[zi, k] * bMobs[zi, m] * mass_int[k, zi]
    xhi_P = np.zeros((Nr_obs, Nm_obs, Nz_obs))
    for r in prange(Nr_obs):
        for mr in range(Nm_obs):
            for zi in range(Nz_obs):
                s = 0.0
                for k in range(Nk - 1):
                    if log: 
                        dk = np.log(k2halo[k+1]) - np.log(k2halo[k])
                    else:
                        dk  = k2halo[k+1] - k2halo[k]
                    x0  = R_Mpc[r, mr, zi] * k2halo[k] if R_Mpc.ndim == 3 else R_Mpc[r, 0, mr, zi] * k2halo[k]
                    x1  = R_Mpc[r, mr, zi] * k2halo[k+1] if R_Mpc.ndim == 3 else R_Mpc[r, 0, mr, zi] * k2halo[k+1]
                    j0_0 = j0_interpolator(x0)
                    j0_1 = j0_interpolator(x1) 
                    f0  = PhP[zi,k,mr]   * j0_0 * k2halo[k]
                    f1  = PhP[zi,k+1,mr] * j0_1 * k2halo[k+1]
                    if log:
                        f0 *= k2halo[k]
                        f1 *= k2halo[k + 1]
                    current_term = 0.5*(f0+f1)*dk
                    s += current_term if np.isnan(current_term) == False else 0
                xhi_P[r,mr,zi] = s / (2*np.pi)
    out = np.zeros((Nr_obs, Nlambda_true, Nlambda_obs, Nm_obs, Nz_obs))
    for r in prange(Nr_obs):
        for l in range(Nlambda_true):
            for o in range(Nlambda_obs):
                for mr in range(Nm_obs):
                    for zi in range(Nz_obs):
                        out[r,l,o,mr,zi] = PllM[l,o,mr,zi] * xhi_P[r,mr,zi]

    return out
@njit(parallel=True, fastmath=True, cache=True)
def compute_two_halo_term_3d(PRMz, Rgrid, R2halo, dndM2halo, bM2halo,
                          bMobs, Pk, R_Mpc, k2halo, M_arr2halo, PllM, weights):

    Nlambda_true = PllM.shape[0]
    Nlambda_obs = PllM.shape[1]
    Nr_obs = R_Mpc.shape[0]
    Nm_obs = bMobs.shape[1]
    Nm_2halo = bM2halo.shape[1]
    Nz_obs = bMobs.shape[0]
    Nr_2halo = Rgrid.shape[0]
    Nr_los = PRMz.shape[0]
    Nk = len(k2halo)

    uRM = np.zeros((Nr_los, Nk, Nz_obs, Nm_2halo))
    for k in prange(Nk):
        for rl in range(Nr_los):
            for zi in range(Nz_obs):
                for m2 in range(Nm_2halo):
                    s = 0.0
                    for r in range(Nr_2halo - 1):
                        dr = R2halo[r + 1] - R2halo[r]
                        x0 = R2halo[r]*k2halo[k]
                        x1 = R2halo[r + 1]*k2halo[k]
                        st0 = np.sin(x0)/x0 if x0 != 0. else 1.
                        st1 = np.sin(x1)/x1 if x1 != 0. else 1.
                        f0 = 4*np.pi * Rgrid[r, m2, zi]**2 * st0 * PRMz[rl ,r, m2, zi]
                        f1 = 4*np.pi * Rgrid[r + 1, m2, zi]**2 * st1 * PRMz[rl ,r + 1, m2, zi]
                        s += 0.5 * (f0 + f1) * dr
                    uRM[rl, k, zi, m2] = s
    mass_int = np.zeros((Nr_los, Nk, Nz_obs))
    for k in prange(Nk):
        for rl in range(Nr_los):
            for zi in range(Nz_obs):
                s = 0.0
                for m2 in range(Nm_2halo - 1):
                    dM = M_arr2halo[m2 + 1] - M_arr2halo[m2]
                    f0 = dndM2halo[zi, m2] * bM2halo[zi, m2] * uRM[rl, k, zi, m2]
                    f1 = dndM2halo[zi, m2 + 1] * bM2halo[zi, m2 + 1] * uRM[rl ,k, zi, m2 + 1]
                    s += 0.5*(f0 + f1) * dM
                mass_int[rl, k, zi] = s
    PhP = np.zeros((Nr_los, Nz_obs, Nk, Nm_obs))
    for zi in range(Nz_obs):
        for rl in range(Nr_los):
            for k in range(Nk):
                for m in range(Nm_obs):
                    PhP[rl, zi, k, m] = Pk[zi, k] * bMobs[zi, m] * mass_int[rl, k, zi]
    xhi_P = np.zeros((Nr_los, Nr_obs, Nm_obs, Nz_obs))
    for r in prange(Nr_obs):
        for rl in range(Nr_los):
            for mr in range(Nm_obs):
                for zi in range(Nz_obs):
                    s = 0.0
                    for k in range(Nk - 1):
                        dk  = k2halo[k+1] - k2halo[k]
                        x0  = R_Mpc[r, mr, zi] * k2halo[k] if R_Mpc.ndim == 3 else R_Mpc[r, 0, mr, zi] * k2halo[k]
                        x1  = R_Mpc[r, mr, zi] * k2halo[k+1] if R_Mpc.ndim == 3 else R_Mpc[r, 0, mr, zi] * k2halo[k+1]
                        st0 = np.sin(x0)/x0 if x0 != 0.0 else 1.0
                        st1 = np.sin(x1)/x1 if x1 != 0.0 else 1.0
                        f0  = PhP[rl,zi,k,mr]   * st0 * k2halo[k]**2
                        f1  = PhP[rl,zi,k+1,mr] * st1 * k2halo[k+1]**2
                        s  += 0.5*(f0+f1)*dk
                    xhi_P[rl, r,mr,zi] = s / (2*np.pi**2)
    return xhi_P

#@check_none
class sz_cluster:
    def __init__(
        self,
        RA,
        DEC,
        richness,
        richness_error,
        r,
        imap,
        mask,
        z,
        z_err,
        box,
        ID="NO - ID",
        data_mask_ratio=0.2,
    ):
        if RA is not None:
            self.RA = RA
            self.DEC = DEC
            self.richness = richness
            self.richness_err = richness_error
            self.cluster_radius = r
            self.imap = np.copy(imap)
            self.mask = np.copy(mask)
            self.box = np.copy(box)
            self.z = float(z)
            self.z_err = float(z_err)
            self.ID = 0 if ID == None else ID
            self.output_path = (
                data_path
                + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]
                + "redmapper_ID="
                + str(self.ID)
            )
            self.total_SNr_map = np.mean(self.imap) ** 2 / np.std(self.imap) ** 2
            if os.path.exists(self.output_path) == False:
                os.mkdir(self.output_path)
        else:
            pass
            
    def agn(self, agn = None, from_path = False):
        if from_path == True:
            if os.path.exists(self.output_path) == True:
                agn = pd.read_csv(self.output_path + "/agn.csv").to_dict()
                self.match_with_agn = True
            else:
                self.match_with_agn = False
        if agn is None:
            self.match_with_agn = False
            return
        self.RA_agn = agn["RA"]
        self.DEC_agn = agn["DEC"]
        self.XNAME_agn = agn["XNAME"]
        self.RNAME_agn = agn["RNAME"]
        self.LOBE1_agn = agn["LOBE1"]
        self.LOBE2_agn = agn["LOBE2"]
        self.match_with_agn = True
    def save_agn(self):
        if self.match_with_agn == True:
            try:
                match_dict = {}
                keys, values = list(self.__dict__.keys()),list(self.__dict__.values())
                for i,key in enumerate(keys):
                    if key.split('_')[-1] == "agn":
                        match_dict[key.replace('_agn','')] = values[i]
                if "agn.csv" in os.listdir(self.output_path):
                    os.remove(self.output_path + '/agn.csv')
                pd.DataFrame(match_dict).to_csv(f"{self.output_path}/agn.csv", index = False)
            except AttributeError as e:
                print("AttributeError: '{}' object has no attribute '{}'".format(type(e).__name__, e.args[0].split("'")[1]))
    def match(self,match = None):
        #match data from DR5
        if match is None:
            self.matched = False
            return
        self.RADeg_match = match["RADeg"]
        self.decDeg_match = match["decDeg"]
        self.name_match  = match["name"]
        self.y_c_match = match["y_c"] # x 10^-4
        self.err_y_c_match = match["err_y_c"]
        self.redshift_match = match["redshift"]
        self.redshiftErr_match = match["redshiftErr"]
        self.M500c_match = match["M500c"] #from Arnaud et al 2010 / units 10^14 Msun
        self.M500cCal_match = match["M500cCal"] #calibration using weak lensing
        self.M500c_errMinus_match, self.M500c_errPlus_match = match["M500c_errMinus"],match["M500c_errPlus"]
        self.M500cCal_errMinus_match, self.M500cCal_errPlus_match = match["M500cCal_errMinus"],match["M500cCal_errPlus"]
        self.matched = True

    def save_match(self):
        if self.matched == True:
            try:
                match_dict = {}
                keys, values = list(self.__dict__.keys()),list(self.__dict__.values())
                for i,key in enumerate(keys):
                    if key.split('_')[-1] == "match":
                        match_dict[key.replace('_match','')] = values[i]
                if "match.csv" in os.listdir(self.output_path):
                    os.remove(self.output_path + '/match.csv')
                pd.DataFrame(match_dict).to_csv(f"{self.output_path}/match.csv", index = False)
            except AttributeError as e:
                print("AttributeError: '{}' object has no attribute '{}'".format(type(e).__name__, e.args[0].split("'")[1]))
    def plot(self, save = False, add_circles = False, plot_signal_centroid = False, imshow_unit = "arcmin",
            plot_signal = False, r_units = "arcmin", patchsize = 0.8, pixel_size = 0.5, signal_propt = "imap",
            show_cluster_information = None, cluster_information_names = None, show_contours = True,
            output = None, **kwargs):
        default_fig_kwargs = (
            ("figsize", (12,8)),
        )
        default_errorbar_kwargs = (
            ("color", "black"),
            ("fmt", "o"),
            ("capsize", 3),
        )
        default_suptitle_kwargs = (
            ("t", "profile and it signal on the map"),
        )
        default_ax_profile_kwargs = (
            ("ylabel", measured_propertie),
            ("xlabel", f"$R$ ({r_units})"),
            ("yscale", "log"),
        )
        default_ax_imshow_kwargs = (
            ("xlabel", f"$\Delta$DEC ({imshow_unit})"),
            ("ylabel", f"$\Delta$RA ({imshow_unit})")
        )
        extent = np.array((-patchsize/2, patchsize/2,-patchsize/2,patchsize/2))
        extent = 60*extent if imshow_unit == "arcmin" else extent

        default_imshow_kwargs = (
            ("extent", extent),
            ("origin", "lower"),
            ("interpolation", "nearest"),
            ("cmap", "turbo")
        )
        default_cbar_kwargs = (
            ("label",measured_propertie),
        )
        default_bbox_kwargs = (
            ("boxstyle", "round"),
            ("facecolor", "grey"),
            ("edgecolor", "black"),
            ("alpha", 0.8),
        )
        default_contour_kwargs = (
            ("colors","white"),
            ("extent", extent),
            ("levels",np.logspace(-5, np.log10(np.max(self.__dict__[str(signal_propt)])), 5))
        )

        fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)
        errorbar_kwargs = set_default(kwargs.pop("errorbar_kwargs",{}), default_errorbar_kwargs)
        ax_profile_kwargs = set_default(kwargs.pop("ax_profile_kwargs",{}), default_ax_profile_kwargs)
        ax_imshow_kwargs = set_default(kwargs.pop("ax_imshow_kwargs",{}), default_ax_imshow_kwargs)
        imshow_kwargs = set_default(kwargs.pop("imshow_kwargs",{}), default_imshow_kwargs)
        cbar_kwargs = set_default(kwargs.pop("cbar_kwargs",{}), default_cbar_kwargs)
        suptitle_kwargs = set_default(kwargs.pop("suptitle_kwargs",{}), default_suptitle_kwargs)
        bbox_kwargs = set_default(kwargs.pop("bbox_kwargs",{}), default_bbox_kwargs)
        contour_kwargs = set_default(kwargs.pop("contour_kwargs",{}), default_contour_kwargs)

        if plot_signal == True:
            fig, (ax_prof, ax_im) = plt.subplots(1,2, **fig_kwargs)
            ax_im.imshow(self.__dict__[str(signal_propt)], **imshow_kwargs)
            ax_im.set(**ax_imshow_kwargs)
            ax_im.set_aspect("auto")
        else:
            fig, ax_prof = plt.subplots(**fig_kwargs)

        if show_contours == True and plot_signal == True:
            try:
                ax_im.contour(self.__dict__[str(signal_propt)], **contour_kwargs)
            except ValueError:
                pass
        ax_prof.errorbar(self.R, self.profile, yerr = self.errors, **errorbar_kwargs)
        ax_prof.set(**ax_profile_kwargs)
        fig.suptitle(**suptitle_kwargs)
        if show_cluster_information is not None:
            if np.iterable(show_cluster_information) == True:
                propt_dict = self.__dict__
                text = []
                for i,p in enumerate(show_cluster_information):
                    try:
                        text.append(str(cluster_information_names[i]) + r" $= %.3f$" % propt_dict[p])
                    except:
                        text.append(str(p) + r" $= %.3f$" % propt_dict[p])
                text = '\n'.join(text)
                ax_text = ax_im if 'ax_im' in locals() else ax_prof
                default_text_kwargs = (
                    ("va","top"),
                    ("ha","left"),
                    ("x", 0.05),
                    ("y", 0.98),
                    ("fontsize", 20),
                    ("transform", ax_text.transAxes),
                    ("bbox", bbox_kwargs)
                )
                text_kwargs = set_default(kwargs.pop("text_kwargs",{}), default_text_kwargs)
                ax_text.text(s = text, **text_kwargs)
        if save == True:
            output = f"{self.output_path}/redmapper_ID={self.ID}_y_compton_map-DR6.png" if output is None else output
            fig.tight_layout()
            fig.savefig(output, transparent = True)

    def save(self, force = False):
        if os.path.exists(self.output_path) == False:
            os.mkdir(self.output_path)
        about_cluster = {
                "RA": self.RA,
                "DEC": self.DEC,
                "richness": self.richness,
                "richness_err": self.richness_err,
                "redshift": self.z,
                "redshift_err": self.z_err,
            }
        pd.DataFrame(about_cluster, index=[0]).to_csv(
                f"{self.output_path}/about_redmapper_ID={self.ID}.csv"
            )
        np.save(f"{self.output_path}/imap.npy", self.imap)
        np.save(f"{self.output_path}/mask.npy", self.mask)
        np.save(f"{self.output_path}/box.npy", self.box)
        if hasattr(self, "profile") == True:
            data = np.zeros((5, len(self.profile)))
            data[0] = self.R
            data[1] = self.profile
            data[2] = self.errors
            data[4] = self.SNr
            np.save(f"{self.output_path}/profile.npy", data)

    def generate_profile(
        self,
        from_path=False,
        r=[0, 300, 700],
        wcs = None,
        method_func=np.mean,
        center="DES",
        full_data=None,
        t_error = 'area',
    ):
        if from_path == False:
            if len(self.__dict__) == 0:
                raise Empty_Data("The cluster_data is totally empty!")
            else:
                theta = self.theta
                data = self.imap
                R_bins, profile, err, data = radial_binning(data, r, wcs = wcs)
                SNr = np.array([profile[i]/err[i] for i in range(len(err))])
                circles = []  # circles in the plot
                z = self.z
                limits = self.x[0][0].value,self.x[-1][-1].value,self.y[0][0].value,self.y[-1][-1].value
                self.limits = limits
                self.profile = profile
                self.errors = err
                self.circles = circles
                self.R = R_bins
                self.SNr = SNr
                self.total_SNr = np.sqrt(np.sum(SNr**2))

        elif from_path == True:
            if os.path.exists(self.output_path):
                profile_data = np.load(f"{self.output_path}/profile.npy")
                self.R = profile_data[0]
                self.profile = profile_data[1]
                self.errors = profile_data[2]
                self.SNr = profile_data[4]
                if full_data == True:
                    other_data = pd.read_csv(
                        f"{self.output_path}/about_redmapper_ID={self.ID}.csv"
                    )
                    self.RA = other_data.loc[0]["RA"]
                    self.DEC = other_data.loc[0]["DEC"]
                    self.richness = other_data.loc[0]["richness"]
                    self.z = other_data.loc[0]["redshift"]
                    self.total_SNr = other_data.loc[0]["total_SNr"]
                    self.imap = np.load(self.output_path+"/imap.npy")
                    self.mask = np.load(self.output_path+"/mask.npy")
                    self.box = np.load(self.output_path+"/box.npy")
            if "match.csv" in os.listdir(self.output_path):
                match_csv = pd.read_csv(self.output_path + "/match.csv")
                self.match(match_csv)

    def __str__(self):
        dr5 = False if self.matched == False else f"matched with {self.name_match}"
        if hasattr(self, "profile") == False:
            return f"""Cluster data:
* (RA,DEC): \033[92m({self.RA},{self.DEC}\033[0m)
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* imap shape: \033[92m{np.shape(self.imap)}\033[0m
* mask flag value: \033[92m{self.MASK_FLAG}\033[0m
* output path: \033[92m{self.output_path}\033[0m
* DR5 : \033[92m{dr5}\033[0m
"""

        else:
            return f"""Cluster data:
* (RA,DEC): \033[92m({self.RA},{self.DEC}\033[0m)
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* imap shape: \033[92m{np.shape(self.imap)}\033[0m
* mask flag value: \033[92m{self.MASK_FLAG}\033[0m
* output path: \033[92m{self.output_path}\033[0m
* DR5 : \033[92m{dr5}\033[0m
* R: \033[92m{self.R}\033[0m
* profile_shape: \033[92m{np.shape(self.profile)}\033[0m
"""

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError("Can only add sz_cluster objects.")
        if np.array_equal(self.R, other.R) == False:
            raise ValueError("R values must be the same.")
        if (hasattr(self, "profile") and hasattr(other, "profile")) == False:
            raise ValueError("Both must have a defined profile")
        profiles = (
            [self.profile, other.profile] if self.profile is not None else other.profile
        )
        new_grouped_cluster = grouped_clusters(
            self.R,
            self.theta,
            self.x,
            self.y,
            profiles,
            [self.errors, other.errors],
            [self.richness, other.richness],
            [self.z, other.z],
            [self.richness_err, other.richness_err],
            [self.z_err, other.z_err],
            [self.mask, other.mask],
            [self.imap, other.imap],
            [self.RA, other.RA],
            [self.DEC, other.DEC],
            [self.ID, other.ID],
            [self.box, other.box],
        )
        return new_grouped_cluster
    def calculate_Y(self, R, plot = False):
        if hasattr(R,'unit'):
            if hasattr(self,"theta") and R.unit == u.arcmin:
                r = self.theta
            elif hasattr(self,"cluster_radius") and R.unit == u.kpc:
                r = self.cluster_radius
            else:
                return 0
            dr = np.where(r < R)
            ydr = self.imap[dr]
            Y = cosmo.angular_diameter_distance(self.z)**(-2) * simps(ydr)
            dr = r[dr].flatten()
            h = dr[0] - dr[1]
            d2y = np.gradient(np.gradient(dr))
            err = np.abs(h**4 / 180 * (dr[-1]) * (d2y[-1]/2))
            return Y.value
        else:
            return 0

        return Y
    @classmethod
    def empty(self):
        arguments = list(inspect.signature(self.__init__).parameters.keys())[1::]
        dic = {n:None for n in arguments}
        return self(**dic)
    @classmethod
    def load_from_path(self, path):
        if os.path.exists(path) == True and len(os.listdir(path)) != 0:
            output = self.empty()
            output.output_path = path
            ID = path.split("=")[-1] if len(path.split("=")) > 1 else 1
            output.ID = str(ID)
            output.generate_profile(from_path = True)
            df = pd.read_csv(f"{output.output_path}/about_redmapper_ID={output.ID}.csv")
            output.RA = df["RA"][0]
            output.DEC = df["DEC"][0]
            output.richness = df["richness"][0]
            output.richness_err = df["richness_err"][0]
            output.z = df["redshift"][0]
            output.z_err = df["redshift_err"][0]
            output.imap = np.load(f"{output.output_path}/imap.npy")
            output.mask = np.load(f"{output.output_path}/mask.npy")
            output.box = np.load(f"{output.output_path}/box.npy")
            smap = output.imap
            shape = np.shape(smap)
            center = shape[0]//2, shape[1]//2
            pixel_width = np.rad2deg(width) / np.shape(smap)[0] * 60
            x,y = np.indices(np.shape(smap))
            theta = np.sqrt(((x - center[0])*pixel_width)**2 + (((y - center[1]))*pixel_width)**2) * u.arcmin
            output.theta = theta
            output.x = x * u.arcmin
            output.y = y * u.arcmin
            return output
        else:
            return None

class AutoCastAttr:
    def __init__(self, dtype=np.float32, **kwargs):

        super().__setattr__('dtype', dtype)

        for name, val in kwargs.items():
            setattr(self, name, val)

    def __setattr__(self, name, val):

        try: 
            if name == 'dtype' or name.startswith('_'):
                return super().__setattr__(name, val)

            dt = object.__getattribute__(self, 'dtype')

            if isinstance(val, np.ndarray):
                val = val.astype(dt)
            elif isinstance(val, Sequence) and not isinstance(val, str):
                if all(isinstance(x, numbers.Number) for x in val):
                    val = np.array(val, dtype=dt)
            elif isinstance(val, numbers.Number):
                val = dt(val)
            return super().__setattr__(name, val)
        except:
            return super().__setattr__(name, val)

class grouped_clusters(AutoCastAttr):
    def __init__(
        self,
        R,
        theta,
        x,
        y,
        profiles,
        errors,
        richness,
        z,
        richness_err = [],
        z_err = [],
        mask = [],
        imap = [],
        ra = [],
        dec = [],
        ID = [],
        box = [],
        output_path = None,
        dtype = np.float32,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.dtype = dtype 
        self.R = R
        self.theta = theta
        self.x = x
        self.y = y
        self.profiles = (
            profiles.tolist() if isinstance(profiles, np.ndarray) else profiles or []
        )
        self.errors = (
            errors.tolist() if isinstance(errors, np.ndarray) else errors or []
        )
        self.richness = (
            richness.tolist() if isinstance(richness, np.ndarray) else richness or []
        )
        self.richness_err = (
            richness_err.tolist() if isinstance(richness_err, np.ndarray) else richness_err or []
        )
        self.z = (
            z.tolist() if isinstance(z, np.ndarray) else z or []
        )
        self.z_err = (
            z_err.tolist() if isinstance(z_err, np.ndarray) else z_err or []
        )
        self.mask = (
            mask.tolist() if isinstance(mask, np.ndarray) else mask or []
        )
        self.imap = (
            imap.tolist() if isinstance(imap, np.ndarray) else imap or []
        )
        self.ra = (
            ra.tolist() if isinstance(ra, np.ndarray) else ra or []
        )
        self.dec = (
            dec.tolist() if isinstance(dec, np.ndarray) else dec or []
        )
        self.ID = (
            ID.tolist() if isinstance(ID, np.ndarray) else ID or []
        )
        self.box = (
            box.tolist() if isinstance(box, np.ndarray) else box or []
        )
        if output_path is None:
            if len(self.richness) > 0 and len(self.z) > 0 and self.richness is not None:
                self.output_path = (
                    data_path
                    + config["FILES"]["GROUPED_CLUSTERS_PATH"]
                    + f"GROUPED_CLUSTER_RICHNESS={np.round(np.min(self.richness))}-{np.round(np.max(self.richness))}"
                    + f"REDSHIFT={np.round(np.min(self.z),2)}-{np.round(np.max(self.z),2)}"
                )
                self.N = len(self.richness)
            elif (len(self.richness) == 0 or len(self.z) == 0) and self.richness is not None:
                self.output_path = (
                    data_path
                    + config["FILES"]["GROUPED_CLUSTERS_PATH"]
                    + f"GROUPED_CLUSTER_RICHNESS={np.round(self.richness)}"
                    + f"REDSHIFT={np.round(self.z,2)}"
                )
            elif self.richness is None:
                self.output_path = ""
                self.N = 0
    @classmethod
    def load_from_path(self, path, load_from_h5 = True, dtype = np.float32):
        c = self.empty(dtype = dtype)
        c.output_path = path
        if load_from_h5 == True:
            c.load_from_h5()
            c.output_path = path
        return c
    @classmethod
    def empty(self, dtype = np.float32):
        arguments = list(inspect.signature(self.__init__).parameters.keys())[1::]
        dic = {n:None for n in arguments}
        dic["dtype"] = dtype
        return self(**dic)
    def load_correlation_matrix(self, corr_file, format = "npy"):
        if format == "npy":
            corr_matrix = np.load(corr_file)
        self.corr_matrix = corr_matrix
        self.corr_source_file = corr_file
        self.corr_file_format = format
    def __len__(self):
        return len(self.richness)
    def __add__(self, other):
        if not isinstance(other, (sz_cluster, grouped_clusters)):
            raise ValueError("Can only add sz_cluster or grouped_clusters.")
        if isinstance(other, sz_cluster):
            other = grouped_clusters(
                self.R,
                self.theta,
                self.x,
                self.y,
                [other.profile],
                [other.errors],
                [other.richness],
                [other.z],
                [other.richness_err],
                [other.z_err],
                [other.mask],
                [other.imap],
                [other.RA],
                [other.DEC],
                [other.ID],
                [other.box]           
            )
        new_profiles = list(self.profiles) + [
            profile for profile in other.profiles if profile is not None
        ]
        new_errors = list(self.errors) + list(other.errors)
        new_richness = list(self.richness) + list(other.richness)
        new_richness_err = list(self.richness_err) + list(other.richness_err)
        new_redshift = list(self.z) + list(other.z)
        new_redshift_err = list(self.z_err) + list(other.z_err)
        try:
            new_mask = list(self.mask) + list(other.mask)
        except:
            new_mask = [[],[]]
        try:
            new_imap = self.imap + other.imap
        except:
            new_imap = [[],[]]
        new_ra = list(self.ra) + list(other.ra)
        new_dec = list(self.dec) + list(other.dec)
        new_ID = list(self.ID) + list(other.ID)
        try:
            new_box = list(self.box) + list(other.box)
        except:
            new_box = [[],[]]
        return grouped_clusters(
            self.R,
            self.theta,
            self.x,
            self.y,
            new_profiles,
            new_errors,
            new_richness,
            new_redshift,
            new_richness_err,
            new_redshift_err,
            new_mask,
            new_imap,
            new_ra,
            new_dec,
            new_ID,
            new_box
        )
    def __str__(self):
        if len(self.profiles) < 4:
            return f"""Grouped cluster data:
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* R: \033[92m{self.R}\033[0m
* profile_shape: \033[92m{np.shape(self.profiles)}\033[0m
"""
        else:
            return f"""Grouped cluster data:
* richness: [\033[92m{np.min(np.round(self.richness))},{np.max(np.round(self.richness))}\033[0m]
* redshift: [\033[92m{np.min(np.round(self.z,2))},{np.max(np.round(self.z,2))}\033[0m]
* R: \033[92m{self.R}\033[0m
* profile_shape: \033[92m{np.shape(self.profiles)}\033[0m
"""

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = key.step if key.step is not None else 1
            key = list(range(key.start, key.stop, step))
        if type(key) == int:
            i = key
            output = sz_cluster(
                self.ra[i],
                self.dec[i],
                self.richness[i],
                self.richness_err[i],
                self.R,
                self.imap[i] if len(self.imap) > 1 else [],
                self.mask[i] if len(self.mask) > 1 else [],
                self.z[i],
                self.z_err[i],
                self.box[i] if len(self.box) > 1 else [],
                self.ID[i] if len(self.ID) > 1 else [],
            )
            smap = self.imap[i]
            shape = np.shape(smap)
            center = shape[0]//2, shape[1]//2
            pixel_width = np.rad2deg(width) / np.shape(smap)[0] * 60
            x,y = np.indices(np.shape(smap))
            theta = np.sqrt(((x - center[0])*pixel_width)**2 + (((y - center[1]))*pixel_width)**2) * u.arcmin
            output.theta = theta
            output.R = (theta.to(u.radian) * cosmo.angular_diameter_distance(self.z[i])).value * u.kpc * 1000
            output.x = x * u.arcmin
            output.y = y * u.arcmin
            try:
                output.profile = self.profiles[i]
                output.R = self.R
                output.errors = self.errors[i]
            except:
                output.generate_profile(r = R_profiles)
        elif np.iterable(key):
            if np.all([type(k)==bool for k in key]):
                if len(key) != len(self):
                    raise Exception("The list of bools must have the same lenght of the cluster sample.")
                else:
                    
                    output = []
                    for i,k in enumerate(key):
                        if k == True:
                            output.append(sz_cluster(
                                self.ra[i],
                                self.dec[i],
                                self.richness[i],
                                self.richness_err[i],
                                self.R,
                                self.imap[i] if len(self.imap) > 1 else [],
                                self.mask[i] if len(self.mask) > 1 else [],
                                self.z[i],
                                self.z_err[i],
                                self.box[i] if len(self.box) > 1 else [],
                                self.ID[i] if len(self.ID) > 1 else [],

                            ))
                            smap = self.imap[i]
                            shape = np.shape(smap)
                            center = shape[0]//2, shape[1]//2
                            pixel_width = np.rad2deg(width) / np.shape(smap)[0] * 60
                            x,y = np.indices(np.shape(smap))
                            theta = np.sqrt(((x - center[0])*pixel_width)**2 + (((y - center[1]))*pixel_width)**2) * u.arcmin
                            output[-1].theta = theta
                            output[-1].R = (theta.to(u.radian) * cosmo.angular_diameter_distance(self.z[i])).value * u.kpc * 1000
                            output[-1].x = x * u.arcmin
                            output[-1].y = y * u.arcmin
                            try:
                                output[-1].profile = self.profiles[i]
                                output[-1].R = R_profiles
                                output[-1].errors = self.errors[i]
                            except:
                                output[-1].generate_profile(r =  self.R)
            elif np.all([isinstance(k, (int, np.integer)) for k in key]):
                output = []
                for i in key:
                    output.append(sz_cluster(
                        self.ra[i],
                        self.dec[i],
                        self.richness[i],
                        self.richness_err[i],
                        self.R,
                        self.imap[i] if len(self.imap) > 1 else [],
                        self.mask[i] if len(self.mask) > 1 else [],
                        self.z[i],
                        self.z_err[i],
                        self.box[i] if len(self.box) > 1 else [],
                        self.ID[i] if len(self.ID) > 1 else [], 
                    ))
                    smap = self.imap[i]
                    shape = np.shape(smap)
                    center = shape[0]//2, shape[1]//2
                    pixel_width = np.rad2deg(width) / np.shape(smap)[0] * 60
                    x,y = np.indices(np.shape(smap))
                    theta = np.sqrt(((x - center[0])*pixel_width)**2 + (((y - center[1]))*pixel_width)**2) * u.arcmin
                    output[-1].theta = theta
                    output[-1].x = x * u.arcmin
                    output[-1].y = y * u.arcmin
                    try:
                        output[-1].profile = self.profiles[i]
                        output[-1].errors = self.errors[i]
                        output[-1].R = self.R
                    except:
                        output[-1].generate_profile(r = self.R)
            output = np.sum(output)                               
        return output
    def match(self, match_radius, catalog, pos_columns = ["ra","dec"], cat_name = "CATALOG", saved_keys = None, save = False, method = "astropy_search_around_sky",
              cosmo = astropy.cosmology.Planck18, return_results = False):
        ra,dec = self.ra * u.deg,self.dec * u.deg
        z2dist = astropy.coordinates.Distance(z = self.z, cosmology = cosmo)
        clusters_coords = SkyCoord(ra,dec) #, distance = z2dist)
        
        ra_cat,dec_cat = catalog[pos_columns[0]] * u.deg,catalog[pos_columns[1]] * u.deg
        if len(pos_columns) > 2:
            z_cat = catalog[pos_columns[2]]
            zc2dist = astropy.coordinates.Distance(z = z_cat, cosmology = cosmo)
            cat_coords = SkyCoord(ra_cat,dec_cat)#, distance = zc2dist)
        else:
            cat_coords = SkyCoord(ra_cat,dec_cat)
        if method == "astropy_search_around_sky":
            idx1, idx2, sep2d,_  = clusters_coords.search_around_sky(cat_coords, match_radius)
            if hasattr(self, "match_dict") == False:
                self.match_dict = {}
            self.match_dict[cat_name] = {}
            self.match_dict[cat_name]["RA"] = ra_cat[idx1]
            self.match_dict[cat_name]["DEC"] = dec_cat[idx1]
            self.match_dict[cat_name]["match_idx1"] = idx1
            self.match_dict[cat_name]["match_idx2"] = idx2
            if saved_keys is not None:
                if np.iterable(saved_keys) == True:
                    for key in saved_keys:
                        try:
                            self.match_dict[cat_name][key] = catalog[key][idx1] 
                        except:
                            print(f"key {key} couldn't be saved.")
        
        if return_results == True:
            return idx1,idx2, sep2d, _
    def map_of_matched(self, catalog, indx, wcs, match_img, box_size = 1, share_plot = True, imap = None, contour_map = True, output = None, **kwargs):
        match = self.match_dict[catalog]
        ra_match, dec_match = match["RA"][indx], match["DEC"][indx]
        indx_cluster = match["match_idx2"][indx]
        ra_cluster, dec_cluster = self.ra[indx_cluster], self.dec[indx_cluster]
        print(f"(RA, DEC) cluster = {ra_cluster},{dec_cluster}")
        print(f"(RA, DEC) match = {ra_match},{dec_match}")
        if hasattr(self, "imap") == False or len(self.imap) <= 1:
            if imap is not None:
                box = [[dec_match.value - box_size/2, ra_match.value - box_size/2], [dec_match.value + box_size/2, ra_match.value + box_size/2]]
                smap = imap.submap(np.deg2rad(box))
        else:
            smap = self.imap[indx_cluster]
        box = [[dec_match.value - box_size/2, ra_match.value - box_size/2], [dec_match.value + box_size/2, ra_match.value + box_size/2]]
        N,M = np.shape(match_img)[0],np.shape(match_img)[1]
        px,py = np.arange(1, N + 1), np.arange(1, M + 1)
        pxv, pyv = np.meshgrid(px,py, indexing = 'ij')
        pixels = np.column_stack((pxv.flatten(), pyv.flatten()))
        sky_coords = wcs.pixel_to_world(pixels[:,0],pixels[:,1])
        ra,dec = sky_coords.ra.degree, sky_coords.dec.degree
        ra_reshape = ra.reshape((N, M))
        dec_reshape = dec.reshape((N,M))
        coords = np.stack((ra_reshape, dec_reshape), axis = -1)
        extent = [dec_reshape.min(), dec_reshape.max(), ra_reshape.min(), dec_reshape.max()]
        fig = plt.figure()
        ax = plt.axes()
        box = [[dec_match.value - box_size/2, ra_match.value - box_size/2], [dec_match.value + box_size/2, ra_match.value + box_size/2]]
        ax.imshow(smap, cmap = "RdBu_r", origin = "lower", extent = (box[0][0],box[1][0], box[0][1], box[1][1]), interpolation = "bilinear")
        ax.contourf(match_img, cmap = "grey", alpha = 0.3, vmin = np.mean(match_img) - 3*np.std(match_img), extent = extent)
        ax.scatter(coords[N//2, M//2][1],coords[N//2, M//2][0], marker = "o", label = f"{catalog}")
        ax.set(xlim = (box[0][0],box[1][0]), ylim = (box[0][1], box[1][1]))
        if output is not None:
            fig.savefig(output)
    def hist_with_match(self, attr, cat_name, compute_weights = False, normalize = True, smooth = False, sigma = 2, 
                output = None, fig = None, ax = None, return_results = False, compute_diff = False, **kwargs):
        default_ax_kwargs = (
            ("xlabel", attr),
            ("ylabel", "N of clusters"),
            ("title", f"{cat_name} {attr} distribution"),
            ("yscale", "log"),
        )
        default_fig_kwargs = (
            ("figsize", (8,6)),
        )

        default_hist_entire_kwargs = (
            ("color", "red"),
            ("fill", True),
            ("alpha", 0.5),
            ("label", "Entire data"),

        )
        default_hist_subsample_kwargs = (
            ("color", "blue"),
            ("fill", False),
            ("alpha", 0.8),
            ("label", f"Distribution of {attr} in {cat_name}")
        )

        default_hist_computation_kwargs = (
            ("density", False),
            ("bins", 30)
        )
        ax_kwargs = set_default(kwargs.pop("ax_kwargs", {}), default_ax_kwargs)
        fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)
        hist_entire_kwargs = set_default(kwargs.pop("hist_entire_kwargs",{}), default_hist_entire_kwargs)
        hist_subsample_kwargs = set_default(kwargs.pop("hist_subsample_kwargs",{}), default_hist_subsample_kwargs)
        hist_computation_kwargs = set_default(kwargs.pop("hist_computation_kwargs",{}), default_hist_computation_kwargs)
        attr_dict = self.__dict__
        if hasattr(self, attr):
            attr_arr = attr_dict[attr]
            if np.iterable(attr_arr):
                match_dict = self.match_dict[cat_name]
                subsample = attr_arr[match_dict["match_idx2"]]
                if fig is None and ax is None:
                    fig = plt.figure(**fig_kwargs)
                    ax = plt.axes()
                elif fig is None and ax is not None:
                    fig = ax.get_figure()
                elif fig is not None and ax is None:
                    ax = plt.axes()
                counts_entire_data, bins_entire_data = np.histogram(attr_arr, **hist_computation_kwargs)
                counts_subsample, bins_subsample = np.histogram(subsample, **hist_computation_kwargs)
                counts_entire_data = counts_entire_data/np.max(counts_entire_data) if normalize == True else counts_entire_data
                counts_subsample = counts_subsample/np.max(counts_subsample) if normalize == True else counts_subsample
                ax.stairs(counts_entire_data, bins_entire_data, **hist_entire_kwargs)  
                ax.stairs(counts_subsample, bins_subsample, **hist_subsample_kwargs)
                ax.set(**ax_kwargs)
                if compute_weights == True:
                    counts_entire_data, bins_entire_data = np.histogram(attr_arr, **hist_computation_kwargs)
                    counts_subsample, bins_subsample = np.histogram(subsample, **hist_computation_kwargs)
                    counts_entire_data = gaussian_filter1d(counts_entire_data, sigma) if smooth == True else counts_entire_data
                    counts_subsample = gaussian_filter1d(counts_subsample, sigma) if smooth == True else counts_subsample
                    w = counts_entire_data/counts_subsample
                    hist_subsample_kwargs["label"] = "weigthed subsample"
                    hist_subsample_kwargs["ls"] = "--"
                    weighted_subsample = counts_subsample * w
                    weigthed_subsample = weighted_subsample /np.max(weighted_subsample) if normalize == True else weighted_subsample
                    ax.stairs(weighted_subsample, bins_subsample, **hist_subsample_kwargs)
                ax.legend()
                ax.grid(True)
                if output is not None:
                    fig.savefig(output)
                ret = [] #variables that returns the function :)
                if return_results:
                    ret.append(counts_entire_data)
                    ret.append(bins_entire_data)
                    ret.append(counts_subsample)
                    ret.append(bins_subsample)
                if compute_weights:
                    ret.append(w)
                if compute_diff:
                    diff = counts_entire_data - counts_subsample
                    ret.append(diff)
                if len(ret) > 0:
                    return ret
                else:
                    return None
            else:
                raise TypeError(f"The attr must be an iterable instead of  {type(attr_arr)}.")
        else:
            raise NameError("The attr doesn't exist.")
    def split_optimal_richness(
        self, SNr=10, Nmin=0.1, abs_min=3000, rdistance=20, ratio=True, method = 'mean', split_by_median_redshift = False, estimate_covariance = True,
        width = None, R_profiles = None, N_realizations = 1000, use_bootstrap = False, min_richness = None, redshift_bins = None, estimate_background = False, 
        verbose = True, n_pool = None, use_cov_matrix = False, use_corr_matrix = False, 
        compute_zero_level = False, ymap = None, mask = None, clusters_mask = None,
        initial_richness = None, weighted = True, max_richness = None, richness_bins = None
    ):
        print(f"Spliting data with {n_pool} cores.") if pool is not None else None
        richness = self.richness
        sorted_richness = np.sort(richness)
        sorted_indices = np.argsort(richness)
        unique_richness = np.unique(np.round(sorted_richness))
        rounded_richness = np.round(sorted_richness)
        intervals = [np.round(np.min(richness))]
        SNR_ARR = []
        if method == "mean":
            sorted_profiles = np.array(self.profiles)[sorted_indices]
            sorted_errors = np.array(self.errors)[sorted_indices]
            saved_data = 0
            for i in tqdm(range(1, len(unique_richness))):
                SNr_profiles = []
                current_richness = unique_richness[i]
                richness_cut = np.where( (rounded_richness > intervals[-1]) & (rounded_richness <= current_richness))
                selected_profiles = sorted_profiles[richness_cut]
                selected_errors = sorted_errors[richness_cut]
                for j in range(len(selected_profiles)):
                    current_SNr = np.sqrt(np.sum(selected_profiles[j]**2 / selected_errors[j]**2))
                    if np.isnan(current_SNr):
                        SNr_profiles.append(0)
                    else:
                        SNr_profiles.append(current_SNr)
                total_SNr = np.mean(SNr_profiles)
                if ratio == True:
                    N_min = Nmin * np.abs((len(profiles) - saved_data))
                elif ratio == False:
                    N_min = abs_min
                if (
                    total_SNr >= SNr
                    and len(SNr_profiles) >= N_min
                    and np.abs(np.round(intervals[-1] - current_richness)) >= rdistance
                    and len(SNr_profiles) >= abs_min
                    ):
                    intervals.append(current_richness)
                    saved_data += len(selected_profiles)
                    SNR_ARR.append(total_SNr)
            if intervals[-1] != np.round(np.max(richness)):
                intervals.append(np.round(np.max(richness)))
        elif method == "stacking":
            if richness_bins is not None:
                print("A richness intervals is provided, using it to split the data.")
                SNr = 0 #there no condition of SNr in this case
            min_richness = np.min(self.richness) if min_richness is None else min_richness
            clusters = []
            corrm = self.corr_matrix if hasattr(self, "corr_matrix") else np.eye(len(self.R))
            sorted_richness = np.sort(self.richness)
            sorted_indices = np.argsort(self.richness)
            max_richness = np.round(max(self.richness)) if max_richness is None else max_richness
            richness = np.arange(np.floor(min_richness), max_richness + 1, 1, dtype = int)
            rounded_richness = np.round(self.richness)
            sorted_maps = np.array(self.imap)[sorted_indices]
            sorted_z = np.array(self.z)[sorted_indices]
            interval = [richness[0], richness[1]]
            #iter = tqdm(range(1, len(richness) + int(rdistance), int(rdistance)), desc = "Spliting data using richness.") if pool is None else range(1, len(richness) + int(rdistance), int(rdistance))
            richness_intervals = generate_sequence(min_richness, rdistance, np.max(self.richness) + rdistance).astype(int) if richness_bins is None else np.asarray(richness_bins, dtype = int)
            min_richness = np.min(richness_intervals) if min_richness is None else min_richness
            interval = [min_richness, int(np.min(self.richness))]
            richness_intervals[1] = initial_richness if initial_richness is not None else richness_intervals[1]
            print("Spliting sample in richness:")
            print("*min richness = ", min_richness)
            print("*richness intervals = ", richness_intervals)
            print("*redshift intervals = ", redshift_bins) if redshift_bins is not None else None
            print("*SNr = ",SNr)
            g_last = None
            for i in range(1,len(richness_intervals)):
                interval[1] = richness_intervals[i]
                print("Current richness interval = ", interval)
                g = self.sub_group(richness_interval = interval)
                median_z = np.median(g.z)
                redshift_intervals = [[min(sorted_z), median_z], (median_z, max(sorted_z))]
                if len(g) < 10:
                    continue
                if g_last is not None:
                    if len(g) - len(g_last) < 20:
                        continue
                if split_by_median_redshift == True:
                    g1 = g.sub_group(redshift_interval = redshift_intervals[0], create_path = True)
                    g2 = g.sub_group(redshift_interval = redshift_intervals[1], create_path = True)
                    g1.stacking(R_profiles, plot = True, width = width, N_realizations = N_realizations, n_pool = n_pool,
                                        bootstrap = use_bootstrap, save = True, estimate_background = estimate_background
                                        , use_cov_matrix = use_cov_matrix, use_corr_matrix = use_corr_matrix,  
                                        compute_zero_level = compute_zero_level, clusters_mask = clusters_mask,
                                        estimate_covariance = estimate_covariance, ymap = ymap, mask = mask,
                                        weighted = weighted)
                    prof1 = g1.mean_profile
                    cov1 = g1.cov
                    snr1 = np.sqrt(np.dot(np.dot(prof1, np.linalg.inv(cov1)), prof1.T)) if hasattr(g1, "snr") == False else g1.snr
                    print("\nSNR1 = ", snr1)
                    if snr1 <= SNr or np.isnan(snr1) == True:
                        g1.remove()
                        del g1
                        continue
                    g2.stacking(R_profiles, plot = True, width = width, N_realizations = N_realizations, n_pool = n_pool,
                                        bootstrap = use_bootstrap, save = True, estimate_background = estimate_background
                                        , use_cov_matrix = use_cov_matrix, use_corr_matrix = use_corr_matrix,
                                        compute_zero_level = compute_zero_level, clusters_mask = clusters_mask,
                                        estimate_covariance = estimate_covariance, ymap = ymap, mask = mask,
                                        weighted = weighted, )
                    prof2 = np.array(g2.mean_profile)
                    cov2 = g2.cov
                    snr2 = np.sqrt(np.dot(np.dot(prof2, np.linalg.inv(cov2)), prof2.T)) if hasattr(g2, "snr") == False else g2.snr

                    print("\nSNR2 = ", snr2)
                    if i >= len(richness):
                        g1.plot()
                        g1.save()
                        g1.stacking(only_plot = True)
                        g2.plot()
                        g2.save()
                        g2.stacking(only_plot = True)
                        
                    if snr1 <= SNr or np.isnan(snr1) == True:
                        g1.remove()
                    if snr2 <= SNr or np.isnan(snr2) == True:
                        g2.remove()

                    if snr1 > SNr and snr2 > SNr and i < len(richness):
                        g1.output_path = "/".join(self.output_path.split("/")[0:-2]) + "/" + r"l%.i-%.i_z%.2f-%.2f" % (
                            min(g1.richness), max(g1.richness), min(g1.z), max(g1.z))
                        g2.output_path = "/".join(self.output_path.split("/")[0:-2]) + "/" + r"l%.i-%.i_z%.2f-%.2f" % (
                            min(g2.richness), max(g2.richness), min(g2.z), max(g2.z))
                        #g1.plot()
                        g1.save()
                        #g1.stacking(only_plot = True)
                        #g2.plot()
                        g2.save()
                        #g2.stacking(only_plot = True)
                        clusters.append(g1)
                        clusters.append(g2)
                        interval[0] = richness_intervals[i]
                        g_last = None
                    else:
                        del g1, g2
                elif redshift_bins is not None:
                    gs = []
                    snrs = []
                    for j in range(len(redshift_bins) - 1):
                        z1,z2 = redshift_bins[j], redshift_bins[j + 1]
                        group = g.sub_group(redshift_interval = (z1, z2))
                        group.redshift_bin = [np.round(z1,3), np.round(z2,3)]
                        gs.append(group)
                    if np.any(np.array([len(gsi) for gsi in gs]) < 20) == True:
                        continue
                    for j in range(len(gs)):
                        z1, z2 = gs[j].redshift_bin
                        print(f"current redshift bin = [{z1}, {z2}]")

                        gs[j].output_path = "/".join(self.output_path.split("/")[0:-2]) + "/" + r"l%.i-%.i_z%.2f-%.2f" % (
                                min(gs[j].richness), max(gs[j].richness), min(gs[j].z), max(gs[j].z))
                        gs[j].richness_bin = np.array(interval)
                        gs[j].redshift_bin = np.array([z1, z2])
                        gs[j].stacking(R_profiles, plot = True, width = width, N_realizations = N_realizations, n_pool = n_pool,
                                        bootstrap = use_bootstrap, save = True, estimate_background = False
                                        , use_cov_matrix = use_cov_matrix, use_corr_matrix = use_corr_matrix,
                                        compute_zero_level = compute_zero_level, clusters_mask = clusters_mask,
                                        estimate_covariance = estimate_covariance, ymap = ymap, mask = mask,
                                         weighted = weighted)
                        cov = gs[j].cov
                        prof = np.array(gs[j].mean_profile)
                        snr = np.sqrt(np.dot(np.dot(prof, np.linalg.inv(cov)), prof.T))
                        snrs.append(snr)
                        print(f"SNR {j + 1} = {snr}")
                        if i >= len(richness):
                            print("Saving to ", gs[j].output_path)
                            gs[j].plot()
                            gs[j].save()
                            continue 
                    if np.all(np.array(snrs) >= SNr) and i <= len(richness):
                        for j in range(len(gs)):
                            print("Saving to ", gs[j].output_path)
                            gs[j].plot()
                            gs[j].save()
                        interval[0] = richness[i]
                else:
                    smap = sorted_maps[richness_cut]
                    if len(smap) == 0:
                        continue
                    stack = np.average(smap, axis = 0)
                    R_bins, profile, err, arrs = radial_binning2(stack, R_profiles, patch_size = width)
                    std_ij = [[np.std(arrs[i])*np.std(arrs[j])/(np.sqrt(len(arrs[i]) * len(arrs[j]))) for j in range(len(arrs))] for i in range(len(arrs))]
                    std_ij = np.array(std_ij)
                    cov = corrm*std_ij*(N_realizations / len(smap))
                    snr = np.sqrt(np.dot(np.dot(profile, np.linalg.inv(cov)), np.array(profile).T))
                    if snr > SNr:
                        interval = [interval[1], interval[1]]
                        new_group = self.sub_group(interval)
                        clusters.append(new_group)
                g_last = g
            return clusters
    def compute_weights(self, add_mask = False, use_snr = False):
        errs = self.errors
        profiles = self.profiles
        snr = np.sqrt(np.sum(profiles**2/errs**2, axis = 1))
        weights = 1/np.sum(errs**2, axis = 1)
        weights = weights*np.sum(self.mask, axis = (1,2))*weights if add_mask == True else weights
        weights = wegihts*snr**2 if use_snr == True else weights 
        self.weights = weights
    def split_by_richness(self, richness_bins = None):
        subgroups = []
        if richness_bins is not None:
            for i in range(len(richness_bins) - 1):
                subgroups.append(self.sub_group(richness_interval = [richness_bins[i], richness_bins[i + 1]]))
        return subgroups
    def split_by_redshift(self, redshift_bins = None):
        subgroups = []
        for i in range(len(redshift_bins) - 1):
            subgroups.append(self.sub_group(redshift_interval = [redshift_bins[i],redshift_bins[i+1]]))
        return subgroups
    def sub_group(self, richness_interval=None, redshift_interval=None, create_path = False):
        if richness_interval is not None and redshift_interval is None:
            richness = np.array(self.richness)
            mask = np.where((richness > richness_interval[0]) & (richness <= richness_interval[1]))
        elif richness_interval is None and redshift_interval is not None:
            redshift = np.array(self.z)
            mask = np.where((redshift > redshift_interval[0]) & (redshift <= redshift_interval[1]))  
        else:
            redshift = np.array(self.z)
            richness = np.array(self.richness)
            mask = np.where((redshift > redshift_interval[0]) & (redshift <= redshift_interval[1]) 
                            & richness > richness_interval[0]) & (richness <= richness_interval[1])
        new_group = type(self).empty()
        available_keys = list(self.__dict__.keys())
        for k in available_keys:
            attr = getattr(self, k)
            if np.iterable(attr) == True:
                if len(attr) == len(self.richness):
                    setattr(new_group, k, np.array(attr)[mask])
                elif len(attr) > 0:
                    setattr(new_group, k, np.array(attr))
            else:
                if k != "output_path": 
                    setattr(new_group, k , attr)
        if create_path == True:
            new_group.output_path = str("/".join(str(self.output_path).split("/")[0:-1]) + "/" + r"l%.i-%.i_z%.2f-%.2f" % (
                                min(new_group.richness), max(new_group.richness), min(new_group.z), max(new_group.z)))
        return new_group
    def binning(self, r=None, shape=None, wcs=None, units="arcmin", edges=False, replace=False):
        r = self.R if hasattr(self, "R") else r

        if units != 'rad':
            conversion = (1 + (59 * (units[:3] == 'arc'))) * (1 + (59 * (units == 'arcsec')))
            conversion = np.pi / 180 / conversion
            r = r * conversion
        else:
            conversion = 1.0
        if edges is False:
            r_edges = (r[:-1] + r[1:]) / 2
            r_first = r[0] - (r[1] - r[0]) / 2
            r_last  = r[-1] + (r[-1] - r[-2]) / 2
            r = np.concatenate(([r_first], r_edges, [r_last]))

        r_centers = (r[1:] + r[:-1]) / 2
        nbins = len(r_centers)

        shape = self.imap[0].shape if hasattr(self, "imap") else shape
        wcs = self.wcs if hasattr(self, "wcs") else wcs

        imaps = np.asarray(self.imap)      
        flat_maps = imaps.reshape(imaps.shape[0], -1) 

        modrmap = enmap.zeros(shape, wcs).modrmap()
        digitized_all = np.digitize(modrmap.ravel(), r) - 1 

        valid = (digitized_all >= 0) & (digitized_all < nbins)
        digitized = digitized_all[valid]   

        sums = np.vstack([
            np.bincount(digitized, weights=flat_map_i[valid], minlength=nbins)
            for flat_map_i in flat_maps
        ])  

        sums2 = np.vstack([
            np.bincount(digitized, weights=(flat_map_i[valid]**2), minlength=nbins)
            for flat_map_i in flat_maps
        ])  

        counts = np.bincount(digitized, minlength=nbins)  

        means = sums / np.maximum(counts, 1)
        variances = sums2 / np.maximum(counts, 1) - means**2
        variances = np.maximum(variances, 0.0)
        stds = np.sqrt(variances)

        if replace:
            self.profiles = means
            self.errors = stds

        r_centers = r_centers / conversion
        return r_centers, means, stds

    def discard_by_R(self, rmin = None, rmax = None, replace = False, return_results = False, plot_comparison = False, **kwargs):
        rmin = np.min(self.R) if rmin is None else rmin
        rmax = np.max(self.R) if rmax is None else rmax
        R = self.R
        rmask = np.where((R >= rmin) & (R <= rmax))[0]
        R2 = R[rmask]
        try:
            self.mean(from_path = True)
        except:
            raise Exception("You should load the mean profile before run this function!.")
        profiles = self.profiles[:,rmask]
        errors = self.errors[:,rmask]
        mean_profile = self.mean_profile[rmask]
        mean_err = self.error_in_mean[rmask]
        if hasattr(self, "cov"):
            cov = np.array(self.cov)[:,rmask][rmask,:]
        else:
            cov = None
        if hasattr(self, "random_profiles_cov"):
            self.random_profiles_cov = self.random_profiles_cov[:,:, rmask]
        if hasattr(self, "random_cov_matrices"):
            self.random_cov_matrices = self.random_cov_matrices[:,rmask,:][:,:,rmask]
        if hasattr(self, "random_corr_matrices"):
            self.random_corr_matrices = self.random_corr_matrices[:,rmask,:][:,:,rmask]

        if plot_comparison:
            ri,rf = np.min(self.richness),np.max(self.richness)
            zi,zf = np.min(self.z),np.max(self.z)
            fig, ax = plt.subplots(figsize = (8,5))
            SNR1 = np.sqrt(np.sum(self.mean_profile**2 / self.error_in_mean**2))
            SNR2 = np.sqrt(np.sum(mean_profile**2 / mean_err**2))
            ax.errorbar(R2, mean_profile, yerr = mean_err, color = "black", label = "SNR $= %.3f$" % SNR2)
            ax.plot(R, self.mean_profile, label = "SNR $= %.3f$" % SNR1, ls = '--', color = "black")
            ax.set(ylabel = r"$\kappa$", xlabel = "R (arcmin)", yscale = "log")
            fig.suptitle("$\lambda \in [%.i,%.i]$ $z \in [%.2f, %.2f]$" % (ri,rf,zi,zf))
            ax.legend()
            ax.grid(True)
            fig.savefig(f"{self.output_path}/comparison_profiles.png")
        if replace:
            self.R = self.R[rmask]
            self.profiles = profiles
            self.errors = errors
            self.mean_profile = mean_profile
            self.error_in_mean = mean_err
            self.cov = cov
        if return_results:
            return R[rmask], profiles, errors, mean_profile, mean_err, cov
    def mean(self, method="weighted", from_path=False, search_closest = False):
        if len(self.profiles) > 1 and from_path == False:
            mean = []
            err = []
            SNr = []
            if method == "weighted":
                weights = 1 / (np.array(self.errors) ** 2)
                # mean = np.average(np.array(self.profiles), weights=weights, axis=0)
                # err = 1 / np.sum(weights, axis=0)**0.5
                for i in range(len(self.profiles[0])):
                    w = weights[:, i]
                    x = np.array(self.profiles)[:, i]
                    # wx = np.sum(w[j] * x[j] for j in range(len(x)))
                    # wp = np.sum(w)
                    # current_mean = wx / wp
                    current_mean = np.sum(w * x) / np.sum(w)
                    current_error = np.sqrt(1 / np.sum(w))
                    mean.append(current_mean)
                    err.append(current_error)
            self.mean_profile = np.array(mean)
            self.error_in_mean = np.array(err)
            self.SNr = np.sqrt(np.sum(self.mean_profile**2 / self.error_in_mean**2))
        elif from_path == True:
            if os.path.exists(self.output_path) and hasattr(self, "output_path"):
                profile_data = np.load(self.output_path + "/mean_profile.npy")
                self.R = profile_data[0]
                self.mean_profile = profile_data[1]
                self.error_in_mean = profile_data[2]
            elif os.path.exists(self.output_path) == False:
                if search_closest == True:
                    target_path = self.output_path.split('/')[-1]
                    grouped_clusters_list = [
                        path
                        for path in os.listdir(data_path + grouped_clusters_path)
                        if os.path.isdir(data_path + grouped_clusters_path + path)
                        and
                        path.split('_')[0] == 'GROUPED'
                        ]
                    closest = closest_path(target_path, grouped_clusters_list)
                    path = data_path + grouped_clusters_path + closest
                    profile_data = np.load(path + "/mean_profile.npy")
                    self.mean_profile = profile_data[1]
                    self.error_in_mean = profile_data[2]
            elif hasattr(self, "output_path") == False:
                print("You must define an output path first")
    def compute_covariance_matrices(self, R_profiles, width):
        maps = self.imap
        covs = compute_covariance_per_map(maps, R_profiles, width)
        self.covs = covs
        
    def load_map(self, imap, maptype = "pixell", boxwidth = 1):
        if maptype == "pixell":
            self.imap = []
            dec,ra = self.dec, self.ra
            dec = np.deg2rad(dec)
            ra = np.deg2rad(ra)
            width = np.deg2rad(boxwidth)
            for i in range(len(self)):   
                box = [
                [dec[i] - width / 2.0, ra[i] - width / 2.0],
                [dec[i] + width / 2.0, ra[i] + width / 2.0],
                ]
                smap = imap.submap(box)
                self.imap.append(smap)
    def compute_diffc_matrix(self, use_sz_centroids = True, use_matchs = True):
        diffc_matrices = []
        if hasattr(self, "centroids_sz") == True:
            centroids_sz = np.array(self.centroids_sz)
            if use_matchs == True:
                dict_match = self.match_dict
                available_matchs = list(dict_match.keys())
                match_idx = [dict_match[k]["match_idx2"] for k in available_matchs]
                idx = np.array([])
                for m in match_idx:
                    idx = np.concatenate((idx,m))
                idx = np.unique(idx).astype(int)
                centroidz_sz = centroids_sz[idx]
                rasz = [c["ra_b"][0] if len(c["ra_b"]) > 0 else c["ra_b"] for c in centroids_sz]
                decsz = [c["dec_b"][0] if len(c["dec_b"]) > 0 else c["dec_b"] for c in centroidz_sz]
                ra,dec = self.ra[idx],self.dec[idx]
                ra_match = []
                dec_match = []
                for k in available_matchs:
                    smatch = dict_match[k]
                    ra_match.append(smatch["RA"])
                    dec_match.append(smatch["DEC"])
                centroids_sz = centroids_sz[idx]
                for i in range(len(ra)):
                    rai = ra[i] if hasattr(ra[i],"value") == False else ra[i].value
                    deci = dec[i] if hasattr(dec[i],"value") == False else dec[i].value
                    rai_sz = rasz[i] if hasattr(rasz[i],"value") == False else rasz[i].value
                    deci_sz = decsz[i] if hasattr(decsz[i], "value") == False else decsz[i].value
                    coords = [(rai,deci),(rai_sz, deci_sz)]
                    for n in range(len(ra_match)):
                        rai_match = ra_match[n][i] if hasattr(ra_match[n][i], "value") == False else ra_match[n][i].value
                        deci_match = dec_match[n][i] if hasattr(dec_match[n][i], "value") == False else dec_match[n][i].value
                        coords.append((rai_match,deci_match))
                    current_diff_m = np.zeros((len(coords),len(coords)))
                    for j in range(np.shape(current_diff_m)[0]):
                        c1 = coords[j]
                        for k in range(np.shape(current_diff_m)[1]):
                            c2 = coords[k]
                            diff = np.sqrt((c1[0] - c2[0])**2 - (c1[1] - c2[1])**2)
                            diff = 0. if np.iterable(diff) else diff
                            current_diff_m[j][k] = diff
                    diffc_matrices.append(current_diff_m)
        self.diffc_matrices = diffc_matrices
    def delete_duplicates(self,match):
        match_dicts = self.match_dict
        if match in list(match_dicts.keys()):
            smatch = match_dicts[match]
            idx = smatch["match_idx2"]
            keys = list(smatch.keys())
            new_match_dict = {str(k):[] for k in keys}
            repeated_idx = []
            im = []
            for n,i in enumerate(idx):
                m = np.where(idx == i)[0]
                if len(m) > 1:
                    repeated_idx.append(m)
                    im.append(i)
            s = 0
            for n,i in enumerate(idx):
                if i in im:   
                    ri = repeated_idx[s]
                    ra = smatch["RA"][ri]
                    dec = smatch["DEC"][ri]
                    mra = np.mean(ra)
                    mdec = np.mean(dec)
                    for k in keys:
                        new_match_dict[k].append(smatch[k][n])
                    new_match_dict["RA"][-1] = mra
                    new_match_dict["DEC"][-1] = mdec
                    s+=1
                else:
                    for k in keys:
                        new_match_dict[k].append(smatch[k][n])
    def estimate_centroids_sz(self, inner_r = 5, width = 0.8, compute_diff = False):
        self.centroids_sz = []
        if hasattr(self, "imap"):
            for i in range(len(self.imap)):
                smap = self.imap[i]
                rai,deci = self.ra[i],self.dec[i]
                pixel_width = width / np.shape(smap)[0] #in deg
                x,y = np.indices(np.shape(smap))
                center = np.shape(smap)[0]//2,np.shape(smap)[1]//2
                x,y = (x - center[0])*pixel_width,(y - center[1])*pixel_width
                r = np.sqrt(x **2 + y**2)
                r_arcmin = r * 60
                ra,dec = x + rai, y + deci
                mask = r_arcmin < inner_r
                sra, sdec = ra[mask], dec[mask]
                ssmap = smap[mask]
                bpixel = np.where((ssmap == np.max(ssmap)) & (ssmap != 1))
                rab, decb = sra[bpixel], sdec[bpixel]
                info = dict(ra_b = rab, dec_b = decb, pix_coord = bpixel)
                if compute_diff == True:
                    center_coords = SkyCoord(rab, decb, unit = u.deg)
                    rm_coords = SkyCoord(rai, deci, unit = u.deg)
                    sep = rm_coords.separation(center_coords).to(u.arcmin)
                    info["sep"] = sep
                self.centroids_sz.append(info)
    def remove(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            del self 
    def compute_cov_matrix(self, use_random_profiles = True, rescale = False, replace_sigma = True):
        if use_random_profiles == True and hasattr(self, "random_profiles_cov"):
            profiles = self.random_profiles_cov
            Ncl = len(self.richness)
            Nrand = np.shape(profiles)[0]
            Nsynt = np.shape(profiles)[1]
            Nr = np.shape(profiles)[2]
            cov_matrices = [np.cov(p, rowvar = False) for p in profiles]
            corr_matrices = [np.corr_coef(p, rowvar = False) for p in profiles]
            cov_eff = (Nsynt/Ncl)*np.mean(cov_matrices)
            corr_eff = np.mean(corr_matrices)
            sigma = np.sqrt(np.diag(cov_eff))
            self.cov = cov_eff
            self.corrm = corr_eff
            if replace_sigma == True and rescale == False:
                self.err_in_mean = sigma
            elif rescale == True:
                sigma = self.error_in_mean
    def compute_zero_level(self, rmin = 10, rmax = 15, check_first = False, **kwargs):
        if hasattr(self, "imap") == False:
            raise Exception("You must load the imap first.")
        if hasattr(self, "R") == False:
            raise Exception("You must define the R first.")
        if hasattr(self, "mean_profile") == False:
            self.mean()
        if hasattr(self, "error_in_mean") == False:
            self.mean()
        if hasattr(self, "zero_level") == True:
            print("Zero level already computed, skipping.")
            return
        R = self.R.value if hasattr(self.R, "value") else self.R
        rmask = np.where((R >= rmin) & (R <= rmax))[0]
        R2 = R[rmask]
        zero_level = np.mean(self.stacked_map[rmask])
        zero_level_err = np.std(self.stacked_map)
        self.zero_level = zero_level
        self.zero_level_err = zero_level_err
        self.mean_profile = self.mean_profile - zero_level
    def stacking(self, R_profiles = [100,200,300], plot = True, weighted = False, only_plot = False, estimate_covariance = True,
                 use_shared_memory = False, background_err = False, ymap = None, mask = None, width = 0.8, bootstrap = False, 
                 save = True, clusters_mask = None, wcs = None, corrm2covm = False, reproject_maps = True, N_realizations = 1000,
                 n_pool = 1, verbose = True, compute_zero_level = False, mask_format = "healpy", convert_maps2global = True, 
                 **kwargs):
        print(20*"=")
        print("Running Stacking...")
        print(f"richness = [{np.min(self.richness)}, {np.max(self.richness)}]")
        print(f"redshift = [{np.min(self.z)}, {np.max(self.z)}]")
        print(f"N clusters = {len(self)}")
        print(20*"=")
        default_background_err_kwargs = (
                ("N_total", 100),
                ("N_clusters", None),
                ("ymap", "/data2/javierurrutia/szeffect/data/ilc_SZ_yy.fits"),
                ("mask", "/data2/javierurrutia/szeffect/data/wide_mask_GAL070_apod_1.50_deg_wExtended.fits"),
                ("clusters_mask", "/data2/javierurrutia/szeffect/data/DES_ACT-footprint_unmasked_clusters.fits"),
                ("use_redshift", False),
                ("corr_matrix", True),
                ("min_sep", 1.6)
        )      
        default_bootstrap_kwargs = (
                    ("N_total", 500),
                    ("N_realizations", None),
                    ("compute-cov-matrix", True),
                    ("unbiased-factor", True),
                    ("compute-individual-covs", False),
                    ("weighted", False),
                    ("store_SNR", True),
                    ("check_convergence", True),
                    ("convergence_threshold", 0.05),
                    ("plot_results", True),
        )

        default_covariance_estimation_kwargs = (
            ("N_total", 500),
            ("N_realizations", None),
            ("unbiased-factor", False),
            ("min_sep", None),
            ("covered-area", 4600),
            ("cluster_size", 5),
            ("save_coords", True),
            ("compute-individual-covs", False),
            ("weighted", True),
            ("store_SNR", True),
            ("check_convergence", True),
            ("convergence_threshold", 0.05),
            ("bootstrapping", True),
            ("plot_results", True),
            ("divide_by_Ncl", True)
        )

        default_zero_level_kwargs = (
            ("rmin", 10),
            ("rmax", 15),
            ("check_first",True)
        )

        default_weights_kwargs = (
            ("use_SNr", False),
            ("use_inverse_variance", False),
            ("use_function", False),
            ("func", lambda x,a: x**a),
            ("vars", ["richness"]),
            ("params", [0.38]),
            ("reliability_weights",  False),
            ("use_richness_weights_and_sigma", True)
        )

        bootstrap_kwargs = set_default(kwargs.pop("bootstrap_kwargs", {}), default_bootstrap_kwargs)
        background_err_kwargs = set_default(kwargs.pop("background_err_kwargs", {}), default_background_err_kwargs)   
        zero_level_kwargs = set_default(kwargs.pop("zero_level_kwargs", {}), default_zero_level_kwargs)
        covariance_estimation_kwargs = set_default(kwargs.pop("covariance_estimation_kwargs", {}), default_covariance_estimation_kwargs)
        weights_kwargs = set_default(kwargs.pop("weights_kwargs", {}), default_weights_kwargs)
        
        if os.path.exists(self.output_path) == False and save == True:
            os.mkdir(self.output_path)
        if only_plot == False:
            if ymap is None or mask is None or clusters_mask is None:
                ymap, mask, clusters_mask = self.load_map_and_mask()
            R_bins = np.array([(R_profiles[i] + R_profiles[i + 1])/2 for i in range(len(R_profiles) - 1)])
            print(f"Running stacking algorithm with {n_pool} N_cores") if n_pool > 1 else None
            maps_array = np.array(self.imap)
            wcs = wcs if wcs is not None else ymap.wcs

            if background_err == True and "background.npy" not in os.listdir(self.output_path):
                rmin, rmax, dmin, dmax = np.min(self.ra), np.max(self.ra), np.min(self.dec), np.max(self.dec)
                Nclusters = len(self.richness) if background_err_kwargs["N_clusters"] is None else background_err_kwargs["N_clusters"]
                N_total = int(np.ceil(len(self.richness)/500) * 500) if background_err_kwargs["N_total"] is None else background_err_kwargs["N_total"] 
                coords = SkyCoord(self.ra, self.dec, unit = "deg")
                random_profiles = []

                if n_pool == 1:
                    if verbose:
                        print(f"Running background estimation error using {N_total} realizations with {Nclusters} clusters replicas.")
                        progress_bar = tqdm(desc = "Estimating background error", total = N_total)
                    while len(random_profiles) < N_total:
                        new_maps = []
                        while len(new_maps) < Nclusters:
                            dec2, ra2 = np.random.uniform(dmin, dmax), np.random.uniform(rmin, rmax)
                            box = [
                                [np.deg2rad(dec2) - np.deg2rad(width) / 2.0, np.deg2rad(ra2) - np.deg2rad(width) / 2.0],
                                [np.deg2rad(dec2) + np.deg2rad(width) / 2.0, np.deg2rad(ra2) + np.deg2rad(width) / 2.0],
                            ]
                            smask = mask.submap(box) if reproject_maps == False else reproject.thumbnails(mask, coords = np.deg2rad((dec2, ra2)), r = np.deg2rad(width)/2.) 
                            scmask = extract_patch(ra2, dec2, data = clusters_mask, dtheta = width, dtype = "healpy")

                            scmask[scmask <= 0.75] = 0
                            scmask[scmask >= 0.75] = 1

                            smask[smask <= 0.75] = 0
                            smask[smask >= 0.75] = 1
                        
                            if len(smask[smask == 1])/np.size(smask) >= 0.75 and len(scmask[scmask == 1])/np.size(scmask) >= 0.75:
                                smap = ymap.submap(box) if reproject_maps == False else reproject.thumbnails(ymap, coords = np.deg2rad((dec2, ra2)), r = np.deg2rad(width)/2.) 
                                new_maps.append(smap)
                            else:
                                continue
                        if verbose == True:
                            progress_bar.update(1)
                        random_stack = np.average(new_maps, axis = 0)
                        R_bins, sprofile, _, sdata = radial_binning(random_stack, R_profiles, wcs = wcs)
                        random_profiles.append(sprofile)
                elif n_pool > 1:
                    with Pool(n_pool, initializer=init_random_worker, initargs=(ymap, clusters_mask)) as pool:
                        if verbose == True:
                            print(f"Estimating background error using {background_err_kwargs['N_total']} random realizations each with {Nclusters} replicas.")
                            print(f"Running background estimator with {pool._processes} cores!")
                        manager = Manager()
                        counter = manager.Value("i", 0)
                        N_base = N_total // n_pool
                        N_remainder = N_total % n_pool
                        iter_per_core = [N_base + 1 if i < N_remainder else N_base for i in range(n_pool)]
                        min_sep = background_err_kwargs["min_sep"]
                        #args = [(np.asarray(ymap), np.asarray(mask), np.asarray(clusters_mask), R_profiles, width, wcs, reproject_maps, p, 
                        #            Nclusters, rmin, rmax, dmin, dmax, N_total, coords, None, i) for i,p in enumerate(iter_per_core)]
                        #res = pool.starmap(random_worker, args)
                        res_ = []
                        for i in range(len(iter_per_core)):
                            kwds = dict(ymap = None, mask = None, R_profiles = R_profiles, width = width, wcs = wcs, reproject_maps = reproject_maps, 
                                N_random = iter_per_core[i], Ncl = len(self.richness), N_clusters = len(self.richness), rmin = rmin, rmax = rmax, 
                                dmin = dmin, dmax = dmax, random_coord_size = N_total//3, N_total = N_total, min_sep = min_sep, 
                                worker_id = i, counter = counter, mask_format = mask_format, compute_individual_matrices = False, 
                                save_coords = False, weights = None, return_patches = True, dtype = self.dtype)

                            res_.append(pool.apply_async(random_worker, kwds = kwds))

                        res = [r.get() for r in res_]
                        background_maps_list = [r[0] for r in res]
                        background_profiles_list = [r[1] for r in res]
                        background_profiles = np.concatenate(background_profiles_list, axis = 0).astype(float)
                        background_maps = np.concatenate(background_maps_list, axis = 0).astype(float)
                        background = np.mean(background_maps, axis = (0,1))
                        pool.close()
                        pool.join()
                        np.save(f"{self.output_path}/background.npy", background)
                        del background_maps_list, res, background_maps, background_profiles_list
                        print("\n","="*30,"\n")
                self.background_field = background
                self.background_profiles = background_profiles
                if background_err_kwargs["corr_matrix"] == True:
                    self.corr_matrix_background = np.corrcoef(random_profiles, rowvar = False)
            elif "background.npy" in os.listdir(self.output_path):
                background = np.load(f"{self.output_path}/background.npy")
                self.background_field = background
            if weighted == True:
                weighted_map = maps_array
                w_mask = np.sum(self.mask, axis = (1,2))
                if weights_kwargs["use_SNr"] == True:
                    snrs = np.sum(self.profiles, axis = 1)/np.sqrt(np.sum(self.errors**2, axis = 1)) if hasattr(self, "covs") == False else np.sqrt(
                        np.einsum('ni,nij->n', self.profiles, np.linalg.solve(self.covs, self.profiles[..., None])))
                    weights = snrs**2 if hasattr(self, "weights") == False else self.weights
                    weights = np.nan_to_num(weights, np.nanmin(weights))
                elif weights_kwargs["use_SNr"] == False and weights_kwargs["use_inverse_variance"]:
                    weights = (1/np.sum(self.errors**2, axis = 1)) if hasattr(self, "weights") == False else self.weights
                elif weights_kwargs["use_function"] == True:
                    print("Using a function to define weights!")
                    func = weights_kwargs["func"]
                    data_vec = np.asarray([getattr(self, var) for var in weights_kwargs["vars"]])
                    data_vec = data_vec[0] if len(data_vec) == 1 else data_vec
                    params = weights_kwargs["params"]
                    weights = func(data_vec, *params) if hasattr(self, "weights") == False else self.weights
                elif weights_kwargs["use_richness_weights_and_sigma"] == True:
                    richness_err = self.richness_err if hasattr(self, "richness_err") else None
                    if richness_err is not None:
                        weights = 1/(richness_err**2 + np.sum(self.errors**2, axis = 1)) * w_mask
                        self.weights = weights
                    else:
                        print("Using inverse variance to compute weights. richness_err attribute not found.")
                        weights = 1/np.sum(self.errors**2, axis = 1) * w_mask
                if hasattr(self, "weights") == False:
                    self.weights = weights
                if weights_kwargs["reliability_weights"] == True:
                    V1 = np.sum(self.weights)
                    self.weights = weights/V1

                self_stacked_map_unweighted = np.average(maps_array, axis = 0)
                self.stacked_map = np.average(weighted_map, axis = 0, weights = weights) - self.background_field if hasattr(self, "background_field") else np.average(weighted_map, axis = 0, weights = weights)
                self.stacked_errors = np.std(maps_array, axis = 0)
            else:
                self.stacked_map = np.average(maps_array, axis = 0) - self.background_field if hasattr(self, "background_field") else np.average(maps_array, axis = 0)
                self.stacked_errors = np.std(maps_array, axis = 0)       

            stack = self.stacked_map
            R_bins, profile, err, arrs = radial_binning2(self.stacked_map, R_profiles, width = width, full = True)
            profile = profile - self.mean_random_profiles if (hasattr(self, "mean_random_profiles") and background_err) else profile

            if estimate_covariance == True:
                print()
                if verbose == True:
                    print("Estimating Covariance matrix.")
                save_coords = covariance_estimation_kwargs["save_coords"]
                rmin, rmax, dmin, dmax = np.min(self.ra), np.max(self.ra), np.min(self.dec), np.max(self.dec)
                N_total = covariance_estimation_kwargs["N_total"]
                N_realizations = covariance_estimation_kwargs["N_realizations"]
                min_sep = covariance_estimation_kwargs["min_sep"]
                compute_individual_covs = covariance_estimation_kwargs["compute-individual-covs"]
                coords = SkyCoord(self.ra, self.dec, unit = "deg")
                weights = self.weights if (hasattr(self, "weights") == True and covariance_estimation_kwargs["weighted"] == True) else None
                if min_sep is None:
                    min_sep = min_separation(self.ra, self.dec, deg = True)
                covered_area = covariance_estimation_kwargs["covered-area"]
                clusters_size = covariance_estimation_kwargs["cluster_size"]
                Ncl = len(self.richness)
                new_covered_area = covered_area - 4*np.pi*(clusters_size/60)**2*Ncl
                coords = SkyCoord(self.ra, self.dec, unit = "deg")
                if N_realizations is None:
                    N_realizations = int(len(self.richness) * (new_covered_area / covered_area))
                if n_pool == 1:
                    if verbose == True:
                        print(f"*N realizations = {N_total}")
                        print(f"*N replicas per realization = {N_realizations}")
                        print(f"*New covered area = {new_covered_area} (deg^2)")
                        print(f"*Mean separation between clusters = {round(min_sep * 60, 2)} arcmin")
                        print(f"*Compute individual covs = ", compute_individual_covs)
                        print(f"*Weighted = ", covariance_estimation_kwargs["weighted"])
                        print(f"*Bootstrapping = ", covariance_estimation_kwargs["bootstrapping"])
                        progress_bar = tqdm(total = N_total, desc = "Estimating covariance...")

                    random_profiles = []
                    random_coords = []
                    while random_profiles <= N_total:
                        new_maps = []
                        saved_coords = SkyCoord([],[], unit = "deg")
                        while new_maps <= N_realizations:
                            dec2, ra2 = np.random.uniform(dmin, dmax), np.random.uniform(rmin. rmax)
                            sep1 = (SkyCoord(ra2, dec2, unit = "deg").separation(coords)).deg
                            sep2 = (SkyCoord(ra2, dec2, unit = "deg").separation(saved_coords)).deg
                            if min_sep is not None:
                                if np.any(sep1 <= 2*min_sep) or np.any(sep2 <= 2*min_sep):
                                    continue
                            box = [
                                [np.deg2rad(dec2) - np.deg2rad(width) / 2.0, np.deg2rad(ra2) - np.deg2rad(width) / 2.0],
                                [np.deg2rad(dec2) + np.deg2rad(width) / 2.0, np.deg2rad(ra2) + np.deg2rad(width) / 2.0],
                            ]
                            smask = mask.submap(box) if reproject_maps == False else reproject.thumbnails(mask, coords = np.deg2rad((dec2, ra2)), r = np.deg2rad(width)/2.) 
                            scmask = extract_patch(ra2, dec2, data = clusters_mask, dtheta = width, dtype = "healpy")

                            scmask[scmask <= 0.75] = 0
                            scmask[scmask >= 0.75] = 1

                            smask[smask <= 0.75] = 0
                            smask[smask >= 0.75] = 1
                        
                            if len(smask[smask == 1])/np.size(smask) >= 0.75 and len(scmask[scmask == 1])/np.size(scmask) >= 0.75:
                                smap = ymap.submap(box) if reproject_maps == False else reproject.thumbnails(ymap, coords = np.deg2rad((dec2, ra2)), r = np.deg2rad(width)/2.) 
                                new_maps.append(smap)
                                saved_coords = SkyCoord(ra = np.append(saved_coords.ra.deg, ra2), 
                                                dec = np.append(saved_coords.dec.deg, dec2), unit = "deg")
                                if verbose == True:
                                    progress_bar.update(1)
                            else:
                                continue
                    random_stack = np.average(new_maps, axis = 0)
                    R_bins, sprofile, _, sdata = radial_binning2(random_stack, R_profiles, width = width, full = True)
                    random_profiles.append(sprofile)
                    if covariance_estimation_kwargs["save_coords"] == True:
                        random_coords.append((saved_coords.ra, saved_coord.dec))

                elif n_pool > 1:
                    if verbose == True:
                        print(f"Running covariance estimation algorithm with {n_pool} cores!", flush = True)
                    coords = SkyCoord(self.ra, self.dec, unit = "deg")
                    if verbose == True:
                        print(f"*N realizations = {N_total}", flush = True)
                        print(f"*N replicas per realization = {N_realizations}", flush = True)
                        print(f"*New covered area = {new_covered_area} (deg^2)", flush = True)
                        print(f"*Mean separation between clusters = {round(min_sep * 60, 2)} arcmin", flush = True)
                        print(f"*Compute individual covs = ", compute_individual_covs, flush = True)
                        print(f"*Weighted = ", covariance_estimation_kwargs["weighted"], flush = True)
                        print(f"*Bootstrapping = ", covariance_estimation_kwargs["bootstrapping"], flush = True)
                        print(f"*Plot results = ", covariance_estimation_kwargs["plot_results"], flush = True)
                    manager = Manager()
                    counter = manager.Value("i", 0)
                    N_base = N_total // n_pool
                    N_remainder = N_total % n_pool
                    iter_per_core = [N_base + 1 if i < N_remainder else N_base for i in range(n_pool)]
                    # args = [(np.asarray(ymap), np.asarray(mask), R_profiles, width, wcs, reproject_maps, p, len(self.richness),
                    #             N_realizations, rmin, rmax, dmin, dmax, N_total, coords, min_sep, i, counter, mask_format) for i,p in enumerate(iter_per_core)]
                    # res = pool.starmap(random_worker, args)
                    res_ = []
                    if use_shared_memory == True and convert_maps2global == False:
                        shape_ymap = ymap.shape
                        shape_mask = clusters_mask.shape
                        dtype = ymap.dtype
                        shm_ymap = shared_memory.SharedMemory(create=True, size=np.prod(shape_ymap) * np.dtype(dtype).itemsize)
                        shm_mask = shared_memory.SharedMemory(create=True, size=np.prod(shape_mask) * np.dtype(dtype).itemsize)

                        ymap2 = np.ndarray(shape_ymap, dtype=dtype, buffer=shm_ymap.buf)
                        mask2 = np.ndarray(shape_mask, dtype=dtype, buffer=shm_mask.buf)

                        ymap2[:] = ymap[:]
                        mask2[:] = clusters_mask[:]
                        ymap_input = (shm_ymap.name, shape_ymap)
                        clusters_mask_input = (shm_mask.name, shape_mask)

                        pool = Pool(n_pool)
                        for i in range(len(iter_per_core)):
                            res_.append(pool.apply_async(random_worker, args = (ymap_input, clusters_mask_input, R_profiles, width, wcs, 
                                                    reproject_maps, iter_per_core[i], len(self.richness), N_realizations, rmin, rmax, dmin, dmax, N_total//3,     
                                                    N_total, coords, None, i, counter, mask_format, self.dtype, save_coords, weights)))
                        shm_ymap.close()
                        shm_ymap.unlink()
                        shm_mask.close()
                        shm_mask.unlink()
                    elif use_shared_memory == False and convert_maps2global == True:
                        pool = Pool(n_pool, initializer=init_random_worker, initargs=(ymap, clusters_mask))
                        
                        for i in range(len(iter_per_core)):
                            kwds = dict(ymap = None, mask = None, R_profiles = R_profiles, width = width, wcs = wcs, reproject_maps = reproject_maps, 
                            N_random = iter_per_core[i], Ncl = len(self.richness), N_clusters = N_realizations, rmin = rmin, rmax = rmax, 
                            dmin = dmin, dmax = dmax, random_coord_size = N_total//3, N_total = N_total, min_sep = min_sep, 
                            worker_id = i, counter = counter, mask_format = mask_format, compute_individual_matrices = compute_individual_covs, 
                            save_coords = save_coords, weights = weights, return_patches = False, dtype = self.dtype)
                            
                            res_.append(pool.apply_async(random_worker, kwds = kwds))
                    else:
                        pool = Pool(n_pool)
                        for i in range(len(iter_per_core)):
                            res_.append(pool.apply_async(random_worker, args = (ymap, clusters_mask, R_profiles, width, wcs, 
                                                    reproject_maps, iter_per_core[i], len(self.richness), N_realizations, rmin, rmax, dmin, dmax, N_total//3,     
                                                    N_total, coords, None, i, counter, mask_format, compute_individual_covs, None, save_coords, weights)))
                    res = [r.get() for r in res_]
                    pool.close()
                    pool.join()
                    manager.shutdown() 
                    cov_matrices_list = [r[0] for r in res]
                    mean_profiles_list = [r[1] for r in res]
                    random_profiles_list = [r[2] for r in res]
                    if weights is not None:
                        random_weights_list = [r[3] for r in res]
                    if covariance_estimation_kwargs["save_coords"] == True:
                        coords_list = [r[-1] for r in res]
                        rcoords = np.concatenate(coords_list)
                        self.coords_random_maps = rcoords
                    fig, ax = plt.subplots(figsize = (12,12))
                    ax.scatter(rcoords[0,:,0], rcoords[0,:,1], s = 20, alpha = 0.2, color = "yellow", edgecolor = "black")
                    ax.scatter(rcoords[1,:,0], rcoords[1,:,1], s = 20, alpha = 0.2, color = "darkgreen", edgecolor = "black")
                    fig.savefig(self.output_path + "/coords.png")
                    cov_matrices = np.concatenate(cov_matrices_list, axis = 0).astype(float)
                    corr_matrices = np.zeros(np.shape(cov_matrices))
                    for n,c in enumerate(cov_matrices):
                        sigma = np.sqrt(np.diag(c))
                        corr_matrices[n] = cov_matrices[n]/np.outer(sigma,sigma)
                    
                    mean_profiles = np.concatenate(mean_profiles_list, axis = 0).astype(self.dtype)
                    random_profiles = np.concatenate(random_profiles_list, axis = 0).astype(self.dtype)
                    random_weights = np.concatenate(random_weights_list, axis = 0).astype(self.dtype)
                    
                    if covariance_estimation_kwargs["check_convergence"] == True:
                        if covariance_estimation_kwargs["bootstrapping"] == False:
                            pad = 10
                            snri = np.zeros(len(cov_matrices) - pad, dtype = self.dtype)
                            Nr = np.shape(random_profiles)[-1]
                            cov_realizations = np.zeros((len(cov_matrices), Nr, Nr), dtype = self.dtype)
                            n_realizations = np.arange(pad, len(cov_matrices))
                            diff = []
                            for k,N in enumerate(n_realizations):
                                idx = np.random.choice(np.arange(len(cov_matrices)), size = N).astype(int)
                                random_profiles_i = random_profiles[idx, ...]
                                random_weights_i = random_weights[idx, ...]
                                cov_matrices_i = np.zeros((N, Nr, Nr ), dtype = self.dtype)
                                for n in range(len(random_profiles_i)):
                                    P = random_profiles_i[n]
                                    W = random_weights_i[n]
                                    Wsum = np.sum(W, axis = 0)
                                    mu = (np.sum(P*W , axis = 0) / Wsum)
                                    dev = P - mu[None,:]
                                    for i in range(Nr):
                                        Wmi = W[:,i]
                                        Di = dev[:,i]
                                        for j in range(Nr):
                                            Wnj = W[:,j]
                                            Dj = dev[:,j]
                                            num = np.sum(Wnj * Wmi * Di * Dj)
                                            Wij = Wnj * Wmi
                                            V1  = np.sum(Wij)
                                            V2  = np.sum(Wij * Wij)
                                            denom = V1 - V2 / V1
                                            cov_matrices_i[n, i, j ] = num/denom
                                            if np.isnan(cov_matrices_i[n,i,j]) == True or np.isfinite(cov_matrices_i[n,i,j]) == False:
                                                cov_matrices_i[n, i, j ] = 0
                                cov_realizations[k] = np.nanmedian(cov_matrices_i, axis = 0).astype(self.dtype)
                                if k > 1:
                                    d = np.abs((cov_realizations[k] - cov_realizations[k - 1])/cov_realizations[k - 1])
                                    diff.append(np.mean(d))
                                else:
                                    diff.append(0.5)
                                snr = np.sqrt(np.dot(profile, np.dot(np.linalg.inv(cov_realizations[k]), profile.T)))
                                snri[k] = snr
                            diff = np.array(diff)
                            fig, ax = plt.subplots(figsize = (12,6))
                            snr2 = gaussian_filter1d(snri, sigma = 3)
                            ax.plot(n_realizations ,snr2, lw = 3, color = "darkgreen")
                            ax.set(xlabel = "Number of realizations", ylabel = "SNR", title = "Convergence check")
                            fig.savefig(self.output_path + "/snr_convergence_check.png", dpi = 200)
                            fig, ax = plt.subplots(figsize = (12,6))
                            diff2 = gaussian_filter1d(diff, sigma = 3)
                            ax.plot(n_realizations ,diff2, lw = 3, color = "darkgreen")
                            ax.set(xlabel = "Number of realizations", ylabel = r"$\langle |(C_{i} - C_{i - 1})/C_{i - 1}| \rangle$", title = "Difference between realizations")
                            fig.savefig(self.output_path + "/diff_convergence_check.png", dpi = 200)
                            if np.any(diff < 0.05):
                                conv = np.where(diff < 0.05)[0][0]
                                print("Converged at %i realizations" % (np.argmin(diff) + 1))
                                ax.fill_between(n_realizations[conv:], 0, diff.max(), color = "darkgreen", alpha = 0.2)
                                after_conv_realizations = n_realizations[conv:]
                                after_conv_diff = diff[conv:]
                                saturated = np.where(after_conv_diff > 0.1)[0]
                                if len(saturated) > 0:
                                    saturated_realizations = after_conv_realizations[saturated[0]:]
                                    saturated_diff = after_conv_diff[saturated[0]:]
                                    ax.fill_between(after_conv_realizations[:saturated[0]], 0, diff.max(), color = "darkred", alpha = 0.2)
                            fig.savefig(self.output_path + "/diff_convergence_check.png", dpi = 200)

                            self.diff_realizations = diff   
                            self.snr_realizations = snri
                        else:
                            pass
                if covariance_estimation_kwargs["bootstrapping"] == False:
                    cov = np.nanmean(cov_matrices, axis = 0).astype(self.dtype)
                    corr = np.nanmean(corr_matrices, axis = 0).astype(self.dtype)
                else:

                    print("Computing covariance using bootstrapping.")
                    if covariance_estimation_kwargs["weighted"] == True:
                        mean_profiles = np.average(random_profiles, axis = 1, weights = random_weights).astype(self.dtype)
                    else:
                        mean_profiles = np.nanmean(random_profiles, axis = 1).astype(self.dtype)
                    if covariance_estimation_kwargs["divide_by_Ncl"] == True:
                        cov = np.zeros((len(profile), len(profile)))
                        for i in range(len(profile)):
                            for j in range(len(profile)):
                                cov[i,j] = np.sum(mean_profiles[:,i] - np.mean(mean_profiles, axis = 0)[i] * (mean_profiles[:,j] - np.mean(mean_profiles, axis = 0)[j]))
                            cov /= (len(self.richness) - 1)
                    else:
                        cov = np.cov(mean_profiles, rowvar = False)
                    cond_number = np.linalg.cond(cov)
                    print("Condition number =", cond_number)
                    if cond_number > 1e1:
                        print("Condition number too high! It has a value of %.2e. The covariance matrix is not invertible." % (cond_number))
                        epsilon = 1e-14
                        counter = 0
                        while cond_number > 1e1:
                            cov += epsilon * np.eye(np.shape(cov)[0])
                            cond_number = np.linalg.cond(cov)
                            epsilon *= 10
                            snr = np.dot(profile, np.dot(np.linalg.inv(cov), profile.T))
                            print(f"Condition number = {cond_number}, epsilon = {epsilon}, SNR = {snr}")
                            print("")
                            counter += 1
                    corr = np.corrcoef(mean_profiles, rowvar = False).astype(self.dtype)
                    if covariance_estimation_kwargs["plot_results"] == True:
                        fig, ax = plt.subplots(figsize = (12,6))
                        for i in range(30):
                            ax.plot(self.R, mean_profiles[i], color = "orange", lw = 1, alpha = 0.5)
                        ax.plot(self.R,np.mean(mean_profiles, axis = 0), color = "black", lw = 2, ls = "--")
                        ax.set(xlabel = "R (arcmin)", ylabel = "Profile", title = "Bootstrapped profiles")
                        fig.savefig(f"{self.output_path}/bootstrapped_profiles.png", dpi = 200)
                if covariance_estimation_kwargs["unbiased-factor"] == True:
                    if covariance_estimation_kwargs["bootstrapping"] == False:
                        Nb = np.shape(cov)[1]
                        cov = (N_realizations - 1)/(N_realizations - 2 - Nb)*cov
                    else:
                        Nb = np.shape(cov)[1]
                        cov = (N_total - 1)/(N_total - 2 - Nb)*cov
                else:
                    self.cov = cov
                self.cov = (N_realizations/len(self.richness))*cov
                self.corrm = corr
                self.random_cov_matrices = cov_matrices
                self.random_corr_matrices = corr_matrices
                self.mean_profiles_cov = mean_profiles
                self.random_profiles_cov = random_profiles
                self.random_weights = random_weights
                self.snr = np.sqrt(np.dot(profile, np.dot(np.linalg.inv(self.cov), profile.T)))
                N = bootstrap_kwargs["N_total"]
                if N is None:
                    N = int(np.ceil(len(self.richness)/1000) * 1000)
                N_clusters = bootstrap_kwargs["N_realizations"] if bootstrap_kwargs["N_realizations"] is not None else len(self.richness)
                compute_cov_matrix = bootstrap_kwargs["compute-cov-matrix"]
            if bootstrap == True:
                if verbose == True:
                    print("Bootstrap sampling to error and covariance estimation.")
                    print("N total =", bootstrap_kwargs["N_total"])
                    print("N clusters =", len(self.richness))
                    print("weighted =", bootstrap_kwargs["weighted"])
                    print("Unbiased factor estimator =", bootstrap_kwargs["unbiased-factor"])
                    print("Estimate cov =", bootstrap_kwargs["compute-cov-matrix"])
                    print("Use profiles =", bootstrap_kwargs["use-profiles"])
                N_clusters = len(self.richness)
                N = bootstrap_kwargs["N_total"]
                if n_pool == 1:
                    if verbose == True:
                        progress_bar = tqdm(desc = "Running bootstrap...", total = N)
                    bootstrap_profiles = np.zeros((N, len(profile)))
                    for n in range(N):
                        indx = np.random.choice(np.arange(0, N_clusters, 1), replace = True, size = N_clusters)
                        smaps = maps_array[indx]
                        if bootstrap_kwargs["weighted"] == True:
                            sweights = self.weights[indx] if hasattr(self, "weights") else np.ones(N_clusters)
                        else:
                            sweights = np.ones(N_clusters)
                        sstack = np.average(smaps, axis = 0, weights = sweights)
                        R_bins, sprofile, _, sdata = radial_binning2(sstack, R_profiles, width = width, full = True)
                        bootstrap_profiles[n] = sprofile
                        if verbose == True:
                            progress_bar.update(1)
                elif n_pool > 1:
                    pool = Pool(n_pool)
                    manager = Manager()
                    counter = manager.Value("i", 0)
                    res_ = []
                    if bootstrap_kwargs["weighted"] == True:
                        weights = self.weights if hasattr(self, "weights") else np.ones(N_clusters)
                    else:
                        weights = np.ones(N_clusters)
                    N_base = N // n_pool
                    N_remainder = N % n_pool
                    iter_per_core = [N_base + 1 if i < N_remainder else N_base for i in range(n_pool)]
                    for i in range(len(iter_per_core)):
                        res_.append(pool.apply_async(bootstrap_worker, args = (R_profiles, maps_array, iter_per_core[i], N, counter, width, weights)))
                    res = [r.get() for r in res_]
                    pool.close()
                    bootstrap_profiles = np.concatenate(res, axis = 0)
                fig, ax = plt.subplots()
                for i in range(len(bootstrap_profiles)):
                    prof = bootstrap_profiles[i]
                    ax.plot(self.R, prof, alpha = 0.1, lw = 2)
                fig.savefig(self.output_path + "/bootstrap_profiles.png", dpi = 200)
                bootstrap_profiles = bootstrap_profiles
                self.bootstrap_profiles = bootstrap_profiles
                self.bootstrap_mean = np.mean(bootstrap_profiles, axis=0)
                self.bootstrap_std = np.std(bootstrap_profiles, axis=0)
                self.bootstrap_1sigma_bounds = np.percentile(bootstrap_profiles, [16, 84], axis = 0)
                self.bootstrap_2sigma_bounds = np.percentile(bootstrap_profiles, [2.5, 97.5], axis = 0)
                if bootstrap_kwargs["compute-cov-matrix"]== True:
                    cov = np.cov(bootstrap_profiles, rowvar = False)
                    if bootstrap_kwargs["unbiased-factor"] == True:
                        unbiased_factor = ((N - len(R_bins) - 2) / (N - 1))
                        cov = unbiased_factor * cov
                    sigma = np.sqrt(np.diag(cov))
                    corr = [[cov[i,j]/(sigma[i]*sigma[j]) for i in range(len(sigma))] for j in range(len(sigma))]
                    cond_number = np.linalg.cond(cov)
                    if cond_number > 1e1:
                        print("Condition number too high! It has a value of %.2e. The covariance matrix is not invertible." % (cond_number))
                        epsilon = 1e-14
                        counter = 0
                        while cond_number > 1e1:
                            cov += epsilon * np.eye(np.shape(cov)[0])
                            cond_number = np.linalg.cond(cov)
                            epsilon *= 10
                            snr = np.sqrt(np.dot(profile, np.dot(np.linalg.inv(cov), profile.T)))
                            print(f"Condition number = {cond_number}, epsilon = {epsilon}, SNR = {snr}")
                            print("")
                            counter += 1
                    self.corr_matrix_bootstrap = corr
                    self.cov_matrix_bootstrap = cov
                    self.snr = np.dot(profile, np.dot(np.linalg.inv(cov), profile.T))

            if compute_zero_level == True:
                rmin, rmax = zero_level_kwargs["rmin"], zero_level_kwargs["rmax"]
                check_first = zero_level_kwargs["check_first"]
                while True:
                    if check_first == True:
                        if np.any(profile <= 0):
                            break
                    if verbose == True:
                        print("Estimating zero level signal!.")
                    zero_level_signal = radial_binning2(stack, [rmin, rmax], width = width)
                    profile = profile - zero_level_signal
                    self.zero_level = zero_level_signal
                    self.mean_profile = profile
                    break
            if estimate_covariance == True and corrm2covm == True: 
                corrm = self.corrm
                covm = self.cov
                sigma = err
                scaled_cov_matrix = np.array([[corrm[i,j]*(sigma[i]*sigma[j]) for i in range(len(sigma))] for j in range(len(sigma))])
                self.random_covm = cov
                self.cov = scaled_cov_matrix
                self.snr = np.sqrt(np.dot(profile, np.dot(np.linalg.inv(scaled_cov_matrix), profile.T)))
            elif estimate_covariance == False and bootstrap == True:
                pass
            else:
                pass
            err = np.sqrt(np.diag(self.cov)) if hasattr(self, "cov") else err
            #err = err/np.sqrt(arrs)
            self.R = R_bins
            self.mean_profile = profile
            self.error_in_mean = err
            px,py = np.indices(np.shape(self.stacked_map))
            px,py = (px - np.shape(self.stacked_map)[0]/2), (py - np.shape(self.stacked_map)[1]/2)
            pix_size = width/np.shape(self.stacked_map)[0] * 60
            x,y = px*pix_size, py*pix_size
            self.x = x
            self.y = y
            
            # if use_corr_matrix == True and use_cov_matrix == False:
            #     if hasattr(self,"corr_matrix") == True:
            #         corr = self.corr_matrix
            #         std_ij = np.outer(err, err)
            #         covariance = corr*std_ij*(N_realizations / len(self.richness))
            #         self.cov = covariance
            #         err = np.sqrt(np.diag(covariance))
            #     else:
            #         raise AttributeError("The attribute 'corr_matrix' doesn't exist.")
            # elif use_cov_matrix == True and use_corr_matrix == False:
            #     if hasattr(self, "cov") == True and bootstrap == False:
            #         covariance = self.cov
            #     elif bootstrap == True:
            #         print("Using covariance matrix from bootstrap!")
            #         covariance = self.cov_matrix_bootstrap
            #         std = np.sqrt(np.diag(covariance))
            #         self.error_in_mean = std
            #         self.cov = covariance

        if plot == True or only_plot == True:
            if self.output_path[-2::] == "//":
                output_path = output_path[:-1]
                self.output_path = output_path
            x,y = self.x, self.y
            R_bins = self.R
            if hasattr(self, "background_field"):
                fig, ax = plt.subplots(figsize = (12,12))
                im = ax.imshow(self.background_field, cmap = "turbo", origin = "lower", interpolation = "kaiser",
                     extent = (x[0][0], x[-1][-1], y[0][0], y[-1][-1]))
                ax.set(xlabel = r"$\Delta\theta$ [arcmin]", ylabel = r"$\Delta\theta $ [arcmin]", title = "Mean background field")
                ax.set_aspect("auto")
                fig.savefig(f"{self.output_path}/background_field.png")
            if hasattr(self, "cov") or hasattr(self ,"cov_matrix_bootstrap") :
                covariance = self.cov_matrix_bootstrap if hasattr(self ,"cov_matrix_bootstrap") else self.cov
                self.cov = covariance
                fig,ax = plt.subplots(figsize = (12,12))
                im = ax.imshow(np.log10(np.abs(covariance)), origin = "lower", 
                    extent = (R_bins[0], R_bins[-1] ,R_bins[0] ,R_bins[-1] ), cmap = "seismic")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax = cax)
                cbar.set_label(r"$\log_{10}{|cov|}$", fontsize=12) 
                ax.set(xlabel = "R bin (arcmin)", ylabel = "R bin (arcmin)",
                    title = r"Covariance Matrix $\lambda \in [%.1f,%.1f]$" % (np.min(self.richness), np.max(self.richness)))
                fig.savefig(f"{self.output_path}/covariance.png")
                fig.tight_layout()
        
            fig, ax = plt.subplots(1,2, figsize = (14,8))
            ax[0].errorbar(self.R, self.mean_profile, yerr = self.error_in_mean, label = "stacked profile", color = 'blue', fmt = "o", capsize = 3, alpha = 0.75)
            ax[0].set(xlabel = f"R (arcmin)", ylabel = r"$\langle y\rangle$", yscale = "log", title = r"stacked profile with $\lambda \in [%.1f,%.1f]$" % (np.min(self.richness), np.max(self.richness)))
            ax[0].grid(True)
            if hasattr(self, "zero_level"):
                ax[0].plot(self.R, np.full(len(self.R), self.zero_level), ls = "--", label = "zero level", color=  "grey", lw = 2)
            im = ax[1].imshow(self.stacked_map, cmap = 'turbo', origin = 'lower', interpolation = 'nearest', extent = (x[0][0],x[-1][-1],y[0][0],y[-1][-1]))
            try:
                ax[1].contour(self.stacked_map, color = "white", levels = np.logspace(np.log10(np.min(stack)), np.log10(np.max(stack)), 8)
                        ,extent = (x[0][0],x[-1][-1],y[0][0],y[-1][-1]))
            except:
                print("Could not draw contours!")
            ax[1].set_aspect("auto")
            ax[1].set(xlabel = r"$\Delta\theta$ [arcmin]", ylabel = r"$\Delta\theta $ [arcmin]", title = "Stacked signal")
            if hasattr(self, "bootstrap_1sigma_bounds"):
                one_sigma = self.bootstrap_1sigma_bounds
                two_sigma = self.bootstrap_2sigma_bounds
                ax[0].plot(self.R, self.bootstrap_mean, color = "darkgreen", ls = "--", label = "bootstrap mean profile")
                ax[0].fill_between(
                    self.R,
                    one_sigma[0],
                    one_sigma[1],
                    alpha = 0.1,
                    color = "darkgreen",
                    label = r"$1\sigma$ bootstrap"
                )
                ax[0].fill_between(
                    self.R,
                    two_sigma[0],
                    two_sigma[1],
                    alpha = 0.1,
                    color = "green",
                    label = r"$2\sigma$ bootstrap"
                )
            ax[0].legend()
            plt.colorbar(im, ax = ax[1], label = "$Compton-y$")
            fig.savefig(f"{self.output_path}/stacking.png")
        if save == True:
            self.save()
    def plot(self, plot_histogram = True, plot_profiles = False, plot_scatter = True, plot_mean_profile = True, **kwargs):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        info = r"$\lambda \in [%.i, %.i]\;,\;z\in[%.2f, %.2f]$" % (np.min(self.richness), np.max(self.richness), np.min(self.z), np.max(self.z))
        label =  f"RICHNESS={np.round(np.min(self.richness))}-{np.round(np.max(self.richness))}" + f"REDSHIFT={np.round(np.min(self.z),2)}-{np.round(np.max(self.z),2)}"
        if plot_histogram == True:
            default_hist_kwargs = (
                ("histtype", "barstacked"),
                ("edgecolor", "black"),
                ("alpha", 0.7),
                ("log", True),
                ("color", "green"),
                )
            hist_kwargs = set_default(kwargs.pop("hist_kwargs",{}), default_hist_kwargs)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(self.richness, **hist_kwargs)
            ax.set(
                title="Distribution of richness",
                xlabel="richness $\\lambda$",
                ylabel="N of clusters",
                yscale='log'
            )
            ax.grid(True)
            fig.savefig(f"{self.output_path}/richness_distribution.png")
        if plot_profiles == True:
            default_fig_profiles_kwargs = (
                ("figsize", (12,8)),
            )
            default_cbar_profiles_kwargs = (
                ("cmap", "viridis"),
                ("norm", "linear"),
                ("label", "richness $\\lambda$")
            )
            default_ax_profiles_kwargs = (
                ("xlabel", "R (arcmin)"),
                ("ylabel", "y-compton profile / $E(z)$"),
                ("title", "Individual Profiles " + info)
            )
            default_plot_profiles_kwargs = (
                ("alpha", 0.4),
                ("lw", 0.5),
            )

            fig_profiles_kwargs = set_default(kwargs.pop("fig_profiles_kwargs",{}), default_fig_profiles_kwargs)
            cbar_profiles_kwargs = set_default(kwargs.pop("cbar_profiles_kwargs",{}), default_cbar_profiles_kwargs)
            ax_profiles_kwargs = set_default(kwargs.pop("ax_profiles_kwargs",{}), default_ax_profiles_kwargs)
            plot_profiles_kwargs = set_default(kwargs.pop("plot_profiles_kwargs",{}), default_plot_profiles_kwargs)
            fig, ax = plt.subplots(**fig_profiles_kwargs)
            cmap = getattr(plt.cm, cbar_profiles_kwargs["cmap"])
            if cbar_profiles_kwargs["norm"] == "linear":
                norm = plt.Normalize(np.min(self.richness), np.max(self.richness))
            cbar_profiles_kwargs.pop("cmap")
            cbar_profiles_kwargs.pop("norm")
            for i in range(len(self.profiles)):
                ax.plot(
                    self.R,
                    self.profiles[i] / cosmo.efunc(self.z[i]),
                    color=cmap(norm(self.richness[i])),
                    **plot_profiles_kwargs
                )
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]), **cbar_profiles_kwargs)
            ax.grid(True)
            ax.set(**ax_profiles_kwargs)
            fig.savefig(f"{self.output_path}/profiles.png")

        if plot_scatter == True:
            
            default_fig_scatter_kwargs = (
                ("figsize", (10,10)),
            )
            default_ax_scatter_kwargs = (
                ("xlabel", r"richness $\lambda$"),
                ("ylabel", r"redshift z"),
                ("title", None)
            )
            default_ax_histy_kwargs = (
                ("ylabel", r"richness $\lambda$"),
            )
            default_ax_histx_kwargs = (
                ("xlabel", r"redshift z"),
            )

            fig_scatter_kwargs = set_default(kwargs.pop("fig_scatter_kwargs",{}), default_fig_scatter_kwargs)
            ax_scatter_kwargs = set_default(kwargs.pop("ax_scatter_kwargs",{}), default_ax_scatter_kwargs)
            ax_histx_kwargs = set_default(kwargs.pop("ax_histx_kwargs",{}), default_ax_histx_kwargs)
            ax_histy_kwargs = set_default(kwargs.pop("ax_histy_kwargs",{}), default_ax_histy_kwargs)

            scatter_hist_kwargs = dict(
                fig_kwargs = fig_scatter_kwargs,
                ax_kwargs = ax_scatter_kwargs, 
                ax_histx_kwargs = ax_histx_kwargs,
                ax_histy_kwargs = ax_histy_kwargs
            )

            fig = scatter_hist(self.richness, self.z, fig = None, bins = 25, add_contours = True, **scatter_hist_kwargs) 
            fig.savefig(self.output_path + "/redshift-richness-distribution.png")

        if plot_mean_profile == True:
            default_fig_mean_kwargs = (
                ("figsize", (14,8)),
            )
            default_ax_mean_kwargs = (
                ("xlabel", "R (arcmin)"),
                ("ylabel", "y-compton profile"),
                ("yscale", "log"),
                ("title", "Mean profile " + info)
            )
            default_errorbar_mean_kwargs = (
                ("color", "black"),
                ("fmt", "o"),
                ("alpha", 0.8),
                ("label", "mean radial profile")
            )
            default_plot_mean_kwargs = (
                ("color", "black"),
                ("lw", 2),
                ("alpha", 0.8)
            )
            default_contours_mean_kwargs = (
                ("color", "grey"),
                ("alpha", 0.1),
            )

            fig_mean_kwargs = set_default(kwargs.pop("fig_mean_kwargs",{}), default_fig_mean_kwargs)
            ax_mean_kwargs = set_default(kwargs.pop("ax_mean_kwargs",{}), default_ax_mean_kwargs)
            errorbar_mean_kwargs = set_default(kwargs.pop("errorbar_mean_kwargs",{}), default_errorbar_mean_kwargs)
            plot_mean_kwargs = set_default(kwargs.pop("plot_mean_kwargs",{}), default_plot_mean_kwargs)
            contours_mean_kwargs = set_default(kwargs.pop("contours_mean_kwargs",{}), default_contours_mean_kwargs)

            if hasattr(self, "mean_profile"):
                prof = self.mean_profile
                errs = self.error_in_mean
                R = self.R
                cov = self.cov if hasattr(self, "cov") else np.eye(errs.size) * errs**2
                snr = np.sqrt(np.dot(np.dot(prof, np.linalg.inv(cov)), prof.T))
                fig, ax = plt.subplots(**fig_mean_kwargs)
                ax.plot(R, prof, **plot_mean_kwargs)
                ax.errorbar(R, prof, yerr = errs, **errorbar_mean_kwargs)
                ax.set(**ax_mean_kwargs)
                ax.fill_between(
                        self.R,
                        self.mean_profile - self.error_in_mean,
                        self.mean_profile + self.error_in_mean,
                        **contours_mean_kwargs
                    )
                ax.plot([],[], "None", label = r"$\mathrm{SNR} = %.2f$" % snr)
                if hasattr(self, "bootstrap_mean"):
                    one_sigma = self.bootstrap_1sigma_bounds
                    two_sigma = self.bootstrap_2sigma_bounds
                    ax.fill_between(
                        self.R,
                        one_sigma[0],
                        one_sigma[1],
                        alpha = 0.1,
                        color = "darkgreen",
                        label = r"$1\sigma$ bootstrap"
                    )
                    ax.fill_between(
                        self.R,
                        two_sigma[0],
                        two_sigma[1],
                        alpha = 0.1,
                        color = "green",
                        label = r"$2\sigma$ bootstrap"
                    )
                if hasattr(self, "mean_random_profiles"):
                    ax.plot(self.R, self.mean_random_profiles, label = "background error", color = "black", alpha = 0.6, lw = 2)
                if hasattr(self, "zero_level"):
                    if self.zero_level >= 0:
                        ax.plot(self.R, np.full(len(self.R), self.zero_level), label = "zero level value", color = "darkred", alpha = 0.6, ls = "--", lw = 2)
                ax.legend()
                fig.savefig(self.output_path + "/mean_profile.png")
    def save(self, file_format = "h5", dtype = np.float32):
        if hasattr(self, "mean_profile"):
            output_data = np.zeros((3, len(self.mean_profile)))
            output_data[0] = self.R
            output_data[1] = self.mean_profile
            output_data[2] = self.error_in_mean
            np.save(f"{self.output_path}/mean_profile.npy", output_data)
            if file_format == "h5":
                available_data = list(self.__dict__.keys())
                h5_path = f"{self.output_path}/data.h5"
                if os.path.exists(h5_path):
                    os.remove(h5_path)
                with h5py.File(h5_path, "w") as f:
                    for k in available_data:
                        if k in ("dtype",):
                            continue
                        if k == "wcs":
                            wcs = getattr(self, k)
                            header_str = wcs.to_header().tostring(sep="\n")
                            dt = h5py.string_dtype(encoding="utf-8")
                            f.create_dataset("wcs_header", data=header_str, dtype=dt)
                            continue
                        try:
                            val = getattr(self, k)
                        except Exception as e:
                            continue
                        if val is None:
                            try:
                                f.create_dataset(k, data=np.array([]))
                            except Exception:
                                pass
                            continue
                        if isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)):
                            try:
                                out_dtype = getattr(self, "dtype", None)
                                f.create_dataset(k, data=val, dtype=out_dtype)
                            except Exception as e:
                                print(e)
                                dt = h5py.string_dtype(encoding="utf-8")
                                f.create_dataset(k, data=str(val), dtype=dt)
                            continue
                        if isinstance(val, str):
                            dt = h5py.string_dtype(encoding="utf-8")
                            f.create_dataset(k, data=val, dtype=dt)
                            continue
                        if isinstance(val, np.ndarray):
                            try:
                                f.create_dataset(k, data=val, dtype=val.dtype)
                            except Exception as e:
                                try:
                                    dt = h5py.vlen_dtype(val.dtype)
                                    f.create_dataset(k, data=val, dtype=dt)
                                except Exception as e2:
                                    dt = h5py.string_dtype(encoding="utf-8")
                                    f.create_dataset(k, data=np.array(repr(val), dtype=object), dtype=dt)
                            continue
                        if isinstance(val, (list, tuple)):
                            if all(isinstance(x, np.ndarray) for x in val):
                                shapes = [x.shape for x in val]
                                if all(s == shapes[0] for s in shapes):
                                    try:
                                        stacked = np.stack(val)
                                        f.create_dataset(k, data=stacked, dtype=stacked.dtype)
                                    except Exception:
                                        if k in f: 
                                            del f[k]  
                                        grp = f.create_group(k)

                                        for i, arr in enumerate(val):
                                            try:
                                                grp.create_dataset(str(i), data=arr, dtype=arr.dtype)
                                            except Exception:
                                                grp.create_dataset(str(i), data=np.array(arr, dtype=object))
                                else:
                                    if k in f:
                                        del f[k]    
                                    grp = f.create_group(k)

                                    for i, arr in enumerate(val):
                                        try:
                                            grp.create_dataset(str(i), data=arr, dtype=arr.dtype)
                                        except Exception:
                                            try:
                                                arr = np.asanyarray(arr)
                                                dt = h5py.vlen_dtype(arr.dtype)
                                                grp.create_dataset(str(i), data=arr, dtype=dt)
                                            except Exception:
                                                grp.create_dataset(str(i), data=str(arr))
                                continue

                            try:
                                arr = np.array(val)
                                if arr.dtype.kind in ("U", "S", "O"):
                                    dt = h5py.string_dtype(encoding="utf-8")
                                    f.create_dataset(k, data=arr.astype(str), dtype=dt)
                                else:
                                    f.create_dataset(k, data=arr, dtype=arr.dtype)
                            except Exception:
                                if k in f:
                                    del f[k]
                                grp = f.create_group(k)
                                for i, item in enumerate(val):
                                    try:
                                        grp.create_dataset(str(i), data=item)
                                    except Exception:
                                        grp.create_dataset(str(i), data=str(item),
                                                        dtype=h5py.string_dtype(encoding="utf-8"))
                            continue

                        if isinstance(val, dict):
                            if k in f:
                                grp = f[k]
                            else:
                                grp = f.create_group(k)
                            for subk, subv in val.items():
                                subname = str(subk)
                                try:
                                    if isinstance(subv, str):
                                        dt = h5py.string_dtype(encoding="utf-8")
                                        grp.create_dataset(subname, data=subv, dtype=dt)
                                    else:
                                        grp.create_dataset(subname, data=subv)
                                except Exception:
                                    try:
                                        grp.create_dataset(subname, data=str(subv), dtype=h5py.string_dtype(encoding="utf-8"))
                                    except Exception:
                                        print(f"Could not store dict element {k}/{subk}")
                            continue
                        try:
                            f.create_dataset(k, data=val)
                        except Exception as e:
                            try:
                                dt = h5py.string_dtype(encoding="utf-8")
                                f.create_dataset(k, data=str(val), dtype=dt)
                            except Exception:
                                print(f"Failed to store attribute {k}: {e}")
                                continue
    def mass_richness_func(self, pivot=40, slope=1.29, normalization=10**14.45):
        return lambda l: (normalization * (l / pivot) ** slope)

    def completeness_and_halo_func(self, cosmo = ccl.CosmologyVanillaLCDM(), plot = False, zbins = 6, Mbins = 5, verbose = False, relationship_config = "MASS-RICHNESS RELATIONSHIP",
                                  static = True, use_lambda_obs = None, interpolate = False, cmap = "Purples", interp_imshow = "nearest", smooth = None, 
                                  text_color = "black", use_redshift = True, r_method = "mean", zlambda2zobs = True,
                                  delta = 500, background = "critical", load_from_file = False, **kwargs):
        """
        Computes the completeness and halo mass function for the stacked halo model.
        Parameters
        ----------
        plot : bool, optional
            If True, plots the completeness and halo mass function. Default is False.
        zbins : int, optional
            Number of redshift bins to use for the completeness and halo mass function. Default is 6. If 'use_redshift' is True, this parameter is ignored
            and the redshift bins would be taken from the completeness file.
        Mbins : int, optional
            Number of mass bins to use for the completeness and halo mass function. Default is 5.
        verbose : bool, optional
            If True, prints additional information during the computation. Default is False.
        relationship_config : str, optional
            The relationship configuration to use for the mass-richness relationship. Default is "MASS-RICHNESS RELATIONSHIP".
        static : bool, optional
            If True, uses a static relationship configuration. Default is True.
        use_lambda_obs : bool or list, optional
            If True, uses the observed richness for completeness calculation. If a list, uses the specified range of observed richness. Default is None.
        interpolate : bool, optional
            If True, interpolates the completeness and halo mass function. Default is False.
        interpolation_method : str, optional
            Method to use for interpolation. Options are "griddata", "RectBivariateSpline", or "RegularGridInterpolator". Default is "griddata".
        method : str, optional
            Method to use for interpolation if `interpolation_method` is "RegularGridInterpolator". Default is "cubic".
        use_redshift : bool, optional
            If True, uses redshift bins for the completeness and halo mass function. Default is False.
        r_method : str, optional
            Method to use for redshift reference. Options are "mean", "median", or "weighted_median". Default is "mean".
        zlambda2zobs : bool, optional
            If True, is assumed a relationship between the redshift of the halo z and the observed redshift z_lambda. Default is False.
            It requires pass a P(z_lambda| z) function trough the 'Pzlambda' kwargs.
        **kwargs : dict, optional
            Additional keyword arguments for the completeness calculation.
            completeness_kwargs : dict, optional
                Additional keyword arguments for the completeness calculation. It contains an optional key 'completeness_file' which  is the path to the 
                completeness file, with default value set to "/data2/cristobal/actpol/lensing/cmblensing/des/selection/completeness_des.txt".
            Pzlambda_kwargs : dict, optional
                Additional keyword arguments for the P(z_lambda| z) function. It contains an optional key 'func' which is the function to use for the P(z_lambda| z) 
                calculation and a z_lambda key which is the array of redshifts to use for the calculation. z are contained in the completeness file.
                func could be a string 'dirac' which assumes a Dirac delta function, if this is the case z_lambda is equal to z.
            interpolation_kwargs : dict, optional
                Additional keyword arguments for the interpolation method. It contains an optional key 'interpolation_method' which is the method to use for interpolation
                and a key 'method' which is the method to use for interpolation if `interpolation_method` is "RegularGridInterpolator". Default values are set to 
                "RegularGridInterpolator" and "cubic" respectively.
                methods available are "griddata", "RectBivariateSpline", or "RegularGridInterpolator".
                N_interp is the number of points to use for interpolation, default value is set to 100.
            function_kwargs : dict, optional
                Additional keyword arguments for the direct function evaluation of completeness. It contains the next keys:
                    Nlambda_true : int, optional
                        Number of true richness values to use for completeness calculation. Default is 30.
                    Nlambda_obs : int, optional
                        Number of observed richness values to use for completeness calculation. Default is 50.
                    Nz_lambda : int, optional
                        Number of redshift values to use for completeness calculation. Default is 30.
                    function : str, optional
                        Completeness function in helpers.py file. As default the function is set to 'P_lob_ltr' from Constazi et al 2019.
                        It must have the next syntax: completeness_function(lambda_obs, lambda_true, z, **kwargs)
        Returns
        -------
        None    
        """

        #As default the richness to mass relation is set to the one obtained in McClintock et al 2019, on the other 
        #hand for the mass to richness relation is used the derived in Costanzi et al 2019.
        default_completeness_kwargs = (
            ("completeness_file", "/data2/javierurrutia/szeffect/codes/selection/completeness_des.txt"),
            ("richness2mass_Norm", 10**14.489),
            ("richness2mass_Pivot", 40),
            ("richness2mass_Slope", 1.356),
            ("richness2mass_Slope_redshift", -0.),
            ("richness2mass_Pivot_redshift", 0.35),
            ("mass2richness_Norm", 30),
            ("mass2richness_Pivot", 3e14/(cosmo._params.h)),
            ("mass2richness_Slope", 0.75),
            ("mass2richness_Slope_redshift", 0.),
            ("mass2richness_Pivot_redshift", 0.35),
            ("sigmaRM", 0.25),
            ("pmr_distribution", "log-normal"),
            ("variable", "mass")
        )

        default_function_kwargs = (
            ("Nlambda_true", 50),
            ("max_lambda_true", 350),
            ("min_lambda_true", 10),
            ("Nlambda_obs", 30),
            ("Nz_lambda", zbins),
            ("function", 'P_lob_ltr'),
            ("func_kwargs", {})
        )

        default_halo_mass_function_kwargs = (
            ("halo_mass_functin", "Tinker10"),
            ("Mmin", 13),
            ("Mmax", 15.7),
            ("M_arr", None)
        )

        halo_mass_function_kwargs = set_default(kwargs.pop("halo_mass_function_kwargs",{}), default_halo_mass_function_kwargs)
        completeness_kwargs = set_default(kwargs.pop("completeness_kwargs",{}), default_completeness_kwargs)
        function_kwargs = set_default(kwargs.pop("function_kwargs",{}), default_function_kwargs)
        default_Pzlambda_kwargs = (
            ("func", "dirac"),
            ("z_lambda", None),
        )
        ic(function_kwargs)
        if verbose == True:
            print("Creating completeness and halo-mass function")

        default_interpolation_kwargs = (
            ("interpolation_method", "RegularGridInterpolator"),
            ("method", "cubic"),
            ("N_interp", 100),
            ("save_interpolated", False),
            ("output_file", os.getcwd() + completeness_kwargs["completeness_file"].split("/")[-1].replace(".txt", "_interpolated.txt")) if completeness_kwargs["completeness_file"] is not None else None,
            ("additional_kwargs", {}),
        )

        interpolation_kwargs = set_default(kwargs.pop("interpolation_kwargs",{}), default_interpolation_kwargs)
        
        Pzlambda_kwargs = set_default(kwargs.pop("Pzlambda_kwargs",{}), default_Pzlambda_kwargs)
        #parameters that converts from richness to mass
        richness2mass_Norm = completeness_kwargs["richness2mass_Norm"]
        richness2mass_Pivot = completeness_kwargs["richness2mass_Pivot"]
        richness2mass_Slope = completeness_kwargs["richness2mass_Slope"]
        richness2mass_Slope_redshift = completeness_kwargs["richness2mass_Slope_redshift"]
        richness2mass_Pivot_redshift = completeness_kwargs["richness2mass_Pivot_redshift"]
        #=========
        #and from mass to richness
        mass2richness_Norm = completeness_kwargs["mass2richness_Norm"]
        mass2richness_Pivot = completeness_kwargs["mass2richness_Pivot"]
        mass2richness_Slope = completeness_kwargs["mass2richness_Slope"]
        mass2richness_Slope_redshift = completeness_kwargs["mass2richness_Slope_redshift"]
        mass2richness_Pivot_redshift = completeness_kwargs["mass2richness_Pivot_redshift"]
        #=========

        sigmaRM = completeness_kwargs["sigmaRM"]
        pmr_distribution = completeness_kwargs["pmr_distribution"]
        
        if load_from_file == True:
            completeness_file = completeness_kwargs["completeness_file"]
            if not os.path.exists(completeness_file):
                raise FileNotFoundError(f"The completeness file {completeness_file} does not exist.")
            else:
                if verbose:
                   print(f"Loading completeness from {completeness_file}.")
            df = pd.read_csv(completeness_file, delimiter = "|", usecols = (1,2,3,4))
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            z_arr = np.unique(df["z"]) #z true from completeness file
            if use_lambda_obs is not None:  
                if use_lambda_obs== True:
                    mask1 = df["l_obs"] >= self.richness_bin[0]
                    mask2 = df["l_obs"] <= self.richness_bin[1]
                    if verbose:
                        print("Using lambda obs = [%.i, %.i]" % (self.richness_bin[0], self.richness_bin[1]))
                    df = df[mask1 & mask2].copy()
                elif np.iterable(use_lambda_obs) == True:
                    mask1 = df["l_obs"] >= use_lambda_obs[0]
                    mask2 = df["l_obs"] <= use_lambda_obs[1]
                    df = df[mask1 & mask2].copy() 
            if use_redshift == False:                 
                if r_method == "median":
                    ref_redshift = np.median(self.z)
                elif r_method == "weighted_median":
                    ref_redshift = weighted_median(self.z, self.richness)
                else:
                    ref_redshift = np.mean(self.z)
                if "z" in df.columns:
                    closest_redshift = np.unique(df["z"])[np.argmin(np.abs(np.unique(df["z"]) - ref_redshift))]
                    mask = df["z"] == closest_redshift
                    df2 = df[mask].copy()
                    df2 = df2.drop(columns = "z")
                else:
                    df2 = df.copy()
                probs = df2.pivot(index = "l_true", columns = "l_obs", values = "P(l_obs)")
                lambda_obs = probs.columns.values
                lambda_true = probs.index.values
                prob_distribution = probs.values
                if verbose:
                    ic(lambda_obs.shape)
                    ic(lambda_true.shape)
                    ic(prob_distribution.shape)
            else:
                if zlambda2zobs == True and Pzlambda_kwargs["func"] == "dirac":
                    if verbose:
                        print("Using redshift obs = [%.1f, %.1f]" % (self.redshift_bin[0], self.redshift_bin[1]))
                    mask1 = df["z"] >= self.redshift_bin[0]
                    mask2 = df["z"] <= self.redshift_bin[1]
                    df = df[mask1 & mask2]
                    z_arr = z_arr[np.where((z_arr >= self.redshift_bin[0]) & (z_arr <= self.redshift_bin[1]))]
                prob_distribution = np.zeros((len(np.unique(df["l_true"])), len(np.unique(df["l_obs"])), len(z_arr)))
                lambda_true_vals = np.unique(df["l_true"])
                lambda_obs_vals = np.unique(df["l_obs"])
                for i,z in enumerate(z_arr):
                    mask = df["z"] == z
                    df2 = df[mask].copy()
                    df2 = df2.drop(columns = "z")
                    probs = df2.pivot(index = "l_true", columns = "l_obs", values = "P(l_obs)")
                    probs = probs.reindex(index=lambda_true_vals, columns=lambda_obs_vals, fill_value=0)
                    lambda_obs = probs.columns.values
                    lambda_true = probs.index.values
                    prob_distribution[:,:,i] = probs.values
        else:
            helpers = importlib.import_module("helpers")
            print("Evaluating completeness from function")
            Nlambda_true = function_kwargs["Nlambda_true"]
            Nlambda_obs = function_kwargs["Nlambda_obs"]
            Nz_lambda = function_kwargs["Nz_lambda"]
            f = getattr(helpers, function_kwargs["function"])
            kf = function_kwargs["func_kwargs"]
            lambda_true = np.linspace(function_kwargs["min_lambda_true"], function_kwargs["max_lambda_true"], Nlambda_true)
            if use_lambda_obs is None:
                lambda_obs = np.linspace(function_kwargs["min_lambda_obs"], function_kwargs["max_lambda_obs"], Nlambda_obs)
            elif use_lambda_obs == True:
                lambda_obs = np.linspace(self.richness_bin[0], self.richness_bin[1], Nlambda_obs)
            elif np.iterable(use_lambda_obs):
                lambda_obs = np.linspace(use_lambda_obs[0], use_lambda_obs[1], Nlambda_obs)
            if use_redshift == False:
                z_arr = np.linspace(function_kwargs["min_z"], function_kwargs["max_z"], Nz_lambda)
            else:
                z_arr = np.linspace(self.redshift_bin[0], self.redshift_bin[1], Nz_lambda)
            ic(lambda_obs.min(), lambda_obs.max(), np.shape(lambda_obs))
            ic(lambda_true.min(), lambda_true.max(), np.shape(lambda_true))
            ic(z_arr.min(), z_arr.max(), np.shape(z_arr))
            ic(function_kwargs["function"])
            ic(kf)
            lambda_true_grid, lambda_obs_grid, z_arr_grid = np.meshgrid(lambda_true, lambda_obs, z_arr, indexing = "ij")
            prob_distribution = f(lambda_obs_grid, lambda_true_grid, z_arr_grid, **kf)

        if verbose:
            ic(prob_distribution.shape)
            ic(lambda_obs.shape)
            ic(lambda_true.shape)
            ic(z_arr.shape)
        if interpolate == True:
            interpolation_method = interpolation_kwargs["interpolation_method"]
            method = interpolation_kwargs["method"]
            N_interp = interpolation_kwargs["N_interp"]
            additional_kwargs = kwargs.pop("additional_kwargs", {})
            lambda_true_interp = np.linspace(np.min(lambda_true), np.max(lambda_true), N_interp)
            lambda_obs_interp = lambda_obs
            z_arr_interp = np.linspace(np.min(z_arr), np.max(z_arr), N_interp)
            if interpolation_method == "RegularGridInterpolator":
                if use_redshift == False:
                    interp_func = RegularGridInterpolator(
                        (lambda_true, lambda_obs), 
                        np.log10(prob_distribution), 
                        method=method,
                        **additional_kwargs)
                    lambda_true_grid, lambda_obs_grid = np.meshgrid(lambda_true_interp, lambda_obs_interp, indexing='ij')
                    points = np.array([lambda_true_grid.flatten(), lambda_obs_grid.flatten()]).T
                    prob_distribution = 10**interp_func(points).reshape((N_interp, N_interp))
                else:
                    interp_func = RegularGridInterpolator(
                        (lambda_true, lambda_obs, z_arr), 
                        np.log10(prob_distribution), 
                        method=method,
                        **additional_kwargs)
                    lambda_true_grid, lambda_obs_grid, z_arr_grid = np.meshgrid(lambda_true_interp, lambda_obs_interp, z_arr_interp, indexing='ij')
                    points = np.array([lambda_true_grid.flatten(), lambda_obs_grid.flatten(), z_arr_grid.flatten()]).T
                    prob_distribution = 10**interp_func(points).reshape((N_interp, len(lambda_obs), N_interp))
            self.interpolator = interp_func
            lambda_true = lambda_true_interp
            z_arr = z_arr_interp
            self._lambda_true = lambda_true_interp
            self._z_arr = z_arr_interp
            if interpolation_kwargs["save_interpolated"] == True:
                output_file = interpolation_kwargs["output_file"]
                if output_file is not None:
                    if use_redshift == False:
                        new_df = pd.DataFrame({
                            "l_true": lambda_true,
                            "l_obs": lambda_obs,
                            "P(l_obs)": prob_distribution.flatten()
                        })
                        new_df.to_csv(output_file, sep="|", index=False)
                    else:
                        new_df = pd.DataFrame({
                            "l_true": np.repeat(lambda_true, len(lambda_obs) * len(z_arr)),
                            "l_obs": np.tile(np.repeat(lambda_obs, len(z_arr)), len(lambda_true)),
                            "z": np.tile(z_arr, len(lambda_true) * len(lambda_obs)),
                            "P(l_obs)": prob_distribution.flatten()
                        })
                        new_df.to_csv(output_file, sep="|", index=False)
                    print(f"Interpolated data saved to {output_file}")
        if zlambda2zobs == True:    
            if Pzlambda_kwargs["func"] == "dirac":
                if verbose:
                    print("Assuming a Dirac delta function for P(z_lambda| z)!")
                z_lambda = self.z #observed redshift (i.e z_lambda)
                z_arr = z_arr[np.where((z_arr >= self.redshift_bin[0]) & (z_arr <= self.redshift_bin[1]))]
                prob_distribution = prob_distribution[:, :, np.where((z_arr >= self.redshift_bin[0]) & (z_arr <= self.redshift_bin[1]))[0]]
            else:
                z_lambda = Pzlambda_kwargs["z_lambda"] if Pzlambda_kwargs["z_lambda"] is not None else np.arange(self.z.min(), self.z.max() + 0.05, 0.05)
                self.z_lambda = z_lambda
                Pzlambda_func = Pzlambda_kwargs["func"]
                z_lambda_grid, z_arr_grid = np.meshgrid(z_lambda, z_arr, indexing = "ij")
                Pzlambda_z = Pzlambda_func(z_lambda_grid, z_arr_grid) #P(z_lambda| z)
        if smooth is not None:
            prob_distribution = gaussian_filter(prob_distribution, smooth)

        print(20*"=") if verbose else None
        Plambda_true = np.array(prob_distribution) # P(lambda_obs | lambda_true)
        self.Plambda_true = Plambda_true #shape (len(lambda_true), len(lambda_obs))
        self.lambda_obs = lambda_obs #observed richness ==> [richness_min, richness_max]
        self.lambda_true = lambda_true #true richness ==> [20, 300] from Costazi et al 2019
        M = np.logspace(13, 16, Mbins) #mass interval ==> [13, 16]
        self.M = M #halo mass bins
        self.z_arr = z_arr #redshift bins

        if verbose:
            print("Final shapes")
            ic(Plambda_true.shape)
            ic(M.shape)
            ic(z_arr.shape)
            ic(lambda_obs.shape)
            ic(lambda_true.shape)

        self.completeness_kwargs = completeness_kwargs
        M200ctoM200m = generate_M200c2M200mInterpolator()
        if use_redshift == True:
            lambda_true_grid, M_grid, z_grid = np.meshgrid(lambda_true, M, z_arr, indexing = "ij")
            if background == "critical":
                Mm_grid = 10**M200ctoM200m((np.log10(M_grid),z_grid))
                lambda_model = mass2richness_Norm * (Mm_grid / (mass2richness_Pivot))**mass2richness_Slope * \
                            ((1 + z_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift  
            else:
                lambda_model = mass2richness_Norm * (M_grid / (mass2richness_Pivot))**mass2richness_Slope * \
                            ((1 + z_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift                   
        else:
            lambda_true_grid, M_grid = np.meshgrid(lambda_true, M)
            if background == "critical":
                Mm_grid = 10**M200ctoM200m((np.log10(M_grid), 0.35))
                lambda_model = mass2richness_Norm * (Mm_grid / (mass2richness_Pivot))**mass2richness_Slope 
            else:
                lambda_model = mass2richness_Norm * (M_grid / (mass2richness_Pivot))**mass2richness_Slope
        sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + sigmaRM**2)

        if pmr_distribution == "log-normal":
            Plambda_true_Mass = 1/(np.sqrt(2 * np.pi * sigma_model**2) * lambda_true_grid) * np.exp(
                - (np.log(lambda_true_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2))
        elif pmr_distribution == "normal":
            Plambda_true_Mass = 1/(np.sqrt(2*np.pi*sigmaRM**2))*np.exp(
                -(np.log1(lambda_true_grid) - np.log(lambda_model))**2/ (2*sigmaRM**2))

        if zlambda2zobs == False or Pzlambda_kwargs["func"] == "dirac":
            PllM = Plambda_true[:,:,None,:]* Plambda_true_Mass[:,None,:,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) ==> P(lambda_obs | lambda_true, M)
        else:
            PllM = (Plambda_true[:,:,None,:]* Plambda_true_Mass[:,None,:,:])[:,:,:,:,None] * Pzlambda_z.T[None, None, None, :,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) * P(z_lambda | z) ==> P(lambda_obs | lambda_true, M, z, z_lambda)
            PllM = trapz(PllM, axis = -1, x = z_lambda) #integrate over z_lambda to get P(lambda_obs | lambda_true, M, z)
        Plambda_obs_M = trapz(PllM, axis = 0, x = lambda_true) # P(lambda_obs | M)
        Plambda_obs_M = gaussian_filter(Plambda_obs_M, smooth) if smooth is not None else Plambda_obs_M
        P_Mass = trapz(Plambda_obs_M, axis = 0, x = lambda_obs) # P(M)
        self.sigmaRM = sigmaRM
        self.Plambda_true_Mass = Plambda_true_Mass
        self.pmr_distribution = pmr_distribution
        self.PllM = PllM
        self.P_Mass = P_Mass
        self.Plambda_obs_M = Plambda_obs_M
        self.lambda_model = lambda_model
        

        mdef = f"{int(delta)}c" if background == "critical" else f"{int(delta)}m"
        a = 1 / (1 + z_arr)

        mfunc = ccl.halos.MassFuncTinker10(mass_def = mdef) #mass function from Tinker et al 2010
        dndM = np.array([[mfunc(cosmo, mi, ai) for mi in M ] for ai in a]) #dN/dlog10M
        dndM = dndM/(np.log(10)*M)
        s = 0.037
        q = 1.008
        dndM = dndM * (s*np.log(M/10**(13.8)/cosmo._params.h) + q)
        self.dndM = dndM
        print(20*"==")
        if verbose:
            ic(Plambda_obs_M.shape)
            ic(PllM.shape)
            ic(P_Mass.shape)
            ic(Plambda_true_Mass.shape)
            ic(pmr_distribution)
            ic(dndM.shape)
        if plot == True:
            from plottery.plotutils import update_rcParams
            update_rcParams()
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            fig, ax = plt.subplots(figsize = (12,12))
            if use_redshift == True:
                print("Plotting P(lambda_true|lambda_obs,z)")
                robs_bin = input("min and max richness obs to plot = ").split(",")
                if len(robs_bin) == 1:
                    robs_min, robs_max = lambda_obs.min(), lambda_obs.max()
                else:
                    robs_min, robs_max = np.array(robs_bin, dtype = float)
                rtrue_bin = input("min and max richness true to plot = ").split(",")
                if len(rtrue_bin) == 1:
                    rtrue_min, rtrue_max = lambda_true.min(), lambda_true.max()
                else:
                    rtrue_min, rtrue_max = np.array(rtrue_bin, dtype = float)

                idx_obs = np.logical_and(lambda_obs >= robs_min, lambda_obs <= robs_max)
                idx_true = np.logical_and(lambda_true >= rtrue_min, lambda_true <= rtrue_max)

                lambda_true = lambda_true[idx_true]
                lambda_obs = lambda_obs[idx_obs]

                z_ref = float(input("z ref to plot = "))
                idx_z = np.argmin(np.abs(z_arr - z_ref))
                z_ref = z_arr[idx_z]
                idx = np.array(idx_obs[None,:]*idx_true[:,None], dtype = bool)
                Plambda_true_z = Plambda_true[:,:, idx_z][idx].reshape((len(lambda_true), len(lambda_obs)))
                im = ax.imshow(Plambda_true_z, norm = LogNorm(vmin = 1e5*np.min(Plambda_true_z)), origin = "lower", aspect = "auto", cmap = "Purples",
                    extent = (lambda_true.min(), lambda_true.max(), lambda_obs.min(), lambda_obs.max()))
                lambda_true_ref = 25
                idx_r = np.argmin(np.abs(lambda_true - lambda_true_ref))
                axins = inset_axes(ax, loc = "lower right", borderpad = 5, width="30%", height="30%")
                axins.plot(lambda_obs, np.log10(Plambda_true_z[lambda_true_ref,:]+1e-20), label = r"$\lambda_{\text{true}} = %.2f$" %lambda_true_ref)
                axins.set_xlabel(r"richness observed $\lambda_{\text{obs}}$", fontsize = 12) 
                axins.set_ylabel(r"$\log_{10}{P}$", fontsize = 12)
                axins.set_title(r"$P(\lambda_{\text{obs}} | \lambda_{\text{true}} = %.i, z = %.2f)$" %(lambda_true_ref, z_ref), fontweight = "bold", fontsize = 14)
                ax.set(ylabel = r"richness true $\lambda_{\text{true}}$", xlabel = r"richness observed $\lambda_{\text{obs}}$")
                fig.suptitle(r"$P(\lambda_{\text{obs}} | \lambda_{\text{true}}, M, z = %.2f)$" %z_ref, fontweight = "bold", fontsize = 30)
                ax.axvline(lambda_true_ref, lw = 3, alpha = 0.5, color = 'purple', ls = "--")
                cbar = plt.colorbar(im)
                cbar.set_label(r"$P(\lambda_{\text{true}} | \lambda_{\text{obs}}, z = %.2f)$" %z_ref, fontsize = 12)
                print("Saving to " + self.output_path + "/PllM_z%.2f.png" %z_ref)
                fig.savefig(self.output_path + "/PllM_z%.2f.png" %z_ref)
                print(20*"==")
                print("Plotting P(M|z)")
                z_ref = 0.22
                idx_z = np.argmin(np.abs(z_arr - z_ref))
                z_ref = z_arr[idx_z]
                fig, ax = plt.subplots(figsize = (14,10))
                cmap = getattr(plt.cm, "Reds")
                norm = plt.Normalize(np.min(z_arr), np.max(z_arr))
                for i in range(len(z_arr)):
                    ax.plot(M, np.cumsum(P_Mass[:,i])/np.cumsum(P_Mass[:,i])[-1], alpha = np.clip(z_arr[i]/np.median(z_arr), 0.2, 1)
                        ,color = cmap(norm(z_arr[i])), lw = 1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                cbar = plt.colorbar(sm, cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]))
                ax.set(xlabel = r"Mass $[\mathrm{M}_{\odot}]$", ylabel = r"$P(M | z)$", xscale = "log", yscale = "linear")
                cbar.set_label("z")
                fig.suptitle(r"$P(M | z)$", fontweight = "bold", fontsize = 30)
                print("Saving to " + self.output_path + "/P_Mass.png")
                fig.savefig(self.output_path + "/P_Mass.png")
            else:
                ax.imshow(Plambda_true, norm = LogNorm(vmin = 1e5*np.min(Plambda_true)), origin = "lower", aspect = "auto", cmap = "Purples",
                    extent = (lambda_true.min(), lambda_true.max(), lambda_obs.min(), lambda_obs.max()))
                ax.set(ylabel = r"richness true $\lambda_{\text{true}}$", xlabel = r"richness observed $\lambda_{\text{obs}}$")
                fig.suptitle(r"$P(\lambda_{\text{obs}} | \lambda_{\text{true}}, M)$", fontweight = "bold", fontsize = 30)
                print("Saving to " + self.output_path + "/PllM.png")
                fig.savefig(self.output_path + "/PllM.png")


            pass

    def compute_pivots(self, weights = None, verbose = False):
        weights = self.richness if weights is None else weights
        richness_pivot = weighted_median(self.richness, weights)
        redshift_pivot = weighted_median(self.z, weights)
        if verbose:
            print("recommended pivots:")
            print("richness:", richness_pivot)
            print("redshift", redshift_pivot) 
    def load_map_and_mask(self, use_pixell = True, use_healpix = False, clusters_mask_format = 'healpy'):
        if use_pixell:
            m = enmap.read_map(self.map_path) if hasattr(self, "map_path") else enmap.zeros()
            mask = enmap.read_map(self.mask_path) if hasattr(self, "mask_path") else enmap.zeros(m.shape, m.wcs)
        elif use_healpix:
            m = hp.read_map(self.map_path)
            mask = hp.read_map(self.mask_path)
        

        clusters_mask = hp.fitsfunc.read_map(self.clusters_mask_path) if clusters_mask_format == 'healpy' else enmap.read_map(self.clusters_mask_path)
        
        return m, mask, clusters_mask
    def stacked_halo_model_func(self, one_halo_profile, units = "arcmin", cosmo = ccl.CosmologyVanillaLCDM(), 
                                pix_size = 0.5, rbins = 25, zbins = 11, Mbins = 10,
                                filters = None, use_filters = False, use_two_halo_term = False, fixed_RM_relationship = True,
                                rebinning = False , mis_centering = False,  interpolate_2halo = False, eval_lambda = False,
                                two_halo_profile = None, redshift_weight_function = False, richness_weight_function = False,
                                mis_centering_func = lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2)), verbose = True,
                                delta = 500, background = "critical", pyccl_cosmo = None, eval_mass = False, 
                                apply_filter_per_profile = False, return_1h2h = False, infere_mass = False,
                                redshift_pivot = 0.4737, richness_pivot = 32.68, weighted = False, 
                                subr_grid = True, compute_completeness = False, numba = False, 
                                physical = False, **kwargs):   

        print()
        print("rbins:", rbins)
        print("zbins:", zbins)
        print("Mbins:", Mbins)

        print("use_mis_centering:", mis_centering)
        print("use_rebinning:", rebinning)
        print("use_two_halo_term:", use_two_halo_term)
        print("use_filters:", use_filters)
        print("use_redshift_weight_function:", redshift_weight_function)
        print("use_richness_weight_function:", richness_weight_function)
        print("compute_completeness")
        print("numba:", numba)
        verbose = True

        float_dtype = self.dtype
        if verbose:
            print(10*"=")
            print("Creating a new stacked halo model func to grouped clusters:\n")
            self.stats()

        default_mass_inf_kwargs = (
            ("is_kappa", True),
            ("use_therm_EQ", False),
            ("use_function", False),
            ("func", None),
            ("interpolate", True),
            ("Nr", 50),
        )

        default_compl_kwargs = (
            ("zbins", zbins),
            ("Mbins", Mbins),
            ("interpolate", False),
            ("use_lambda_obs", True),
            ("use_redshift", True),
            ("background", background),
            ("delta", delta),
            ("verbose", verbose)
            )

        default_rebinning_kwargs = (
            ("nbins", 50),
            ("method", 'interp1d'),
            ("pixel_size", 0.01),
            ("interpolation_kwargs", dict(kind='cubic', bounds_error=False, fill_value=0))
        )

        default_mis_centering_kwargs = (
            ("Roff", np.linspace(0, 2, 30)),
            ("distribution", lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2))),
            ("params", [0.245, 0.354]),
            ('func', None),
            ("theta", np.linspace(0, 2*np.pi, 30))
        )

        default_two_halo_kwargs = (
            ("background", background),
            ("R", np.logspace(-1, 1.7, 10)),
            ("k", np.logspace(-3, 4, 40)),
            ("M", np.logspace(13,16, 25)),
            ("z", np.linspace(1e-3,1, 15)),
            ("cosmo", cosmo),
            ("delta", delta),
            ("two_halo_power_func", lambda z, p: p),
            ("eval_only_mass", False),
            ("eval_only_richness", False),
            ("eval_only_redshift", True),    
            ("N", 500),
            ("h", 0.001)
        )

        default_weights_function_kwargs = (
            ("zmin", 0.05),
            ("zmax", 1),
            ("deltaz", 0.005),
            ("interpolation", "linear"),
            ("file", str("/".join(str(self.output_path).split("/")[0:-1]) + "/" + "weights_z.txt")),
            ("overwrite", False)
        )

        default_richness_weights_function_kwargs = (
            ("func", lambda x,a: 1),
            ("params", [1]),

        )

        default_subr_grid_kwargs = (
            ("rmin", 0.1),
            ("rmax", 15),
            ("n" , 20)
        )

        M200mtoM200c = generate_M200m2M200cInterpolator()

        two_halo_kwargs = set_default(kwargs.pop("two_halo_kwargs", {}), default_two_halo_kwargs)
        compl_kwargs = set_default(kwargs.pop("completeness_kwargs", {}), default_compl_kwargs)
        mis_centering_kwargs = set_default(kwargs.pop("mis_centering_kwargs", {}), default_mis_centering_kwargs)
        rebinning_kwargs = set_default(kwargs.pop("rebinning_kwargs", {}), default_rebinning_kwargs)
        weights_function_kwargs = set_default(kwargs.pop("weights_function_kwargs", {}),default_weights_function_kwargs)
        richness_weights_function_kwargs = set_default(kwargs.pop("richness_weights_function_kwargs",{}), default_richness_weights_function_kwargs)
        two_halo_profile = one_halo_profile if two_halo_profile is None else two_halo_model
        mass_inf_kwargs = set_default(kwargs.pop("mass_inf_kwargs", {}), default_mass_inf_kwargs)
        subr_grid_kwargs = set_default(kwargs.pop("subr_grid_kwargs", {}), default_subr_grid_kwargs)
        
        ic(two_halo_kwargs)
        ic(compl_kwargs)
        ic(mis_centering_kwargs)
        ic(rebinning_kwargs)
        ic(weights_function_kwargs)
        ic(richness_weights_function_kwargs)
        ic(two_halo_profile)
        ic(mass_inf_kwargs)
        ic(subr_grid_kwargs)

        #pre-compute completeness and halo mass function
        use_redshift = compl_kwargs["use_redshift"]
        if hasattr(self, "completeness_kwargs") == False or compute_completeness == True:
            self.completeness_and_halo_func(**compl_kwargs)
        else:
            cond = [compl_kwargs[k] == v for k,v in self.completeness_kwargs.items() if k in list(compl_kwargs.keys())]
            if np.any(cond == False):
                self.completeness_and_halo_func(**compl_kwargs)

        PllM = self.PllM.astype(float_dtype) # P(lambda_true | lambda_obs, M, z)
        P_Mass = self.P_Mass.astype(float_dtype) # P(M) or P(M|z)
        dndM = self.dndM.astype(float_dtype) # dN/dM(M,z)
        Plambda_obs_M = self.Plambda_obs_M.astype(float_dtype) # P(lambda_obs | M)
        M = self.M.astype(float_dtype) # halo mass bins
        z_arr = self.z_arr.astype(float_dtype) # redshift bins
        lambda_obs = self.lambda_obs.astype(float_dtype)
        lambda_true = self.lambda_true.astype(float_dtype) # true richness bins

        #Mass to richness params (power law)
        mass2richness_Norm = self.completeness_kwargs["mass2richness_Norm"]
        mass2richness_Pivot = self.completeness_kwargs["mass2richness_Pivot"]
        mass2richness_Slope = self.completeness_kwargs["mass2richness_Slope"]
        mass2richness_Slope_redshift = self.completeness_kwargs["mass2richness_Slope_redshift"]
        mass2richness_Pivot_redshift = self.completeness_kwargs["mass2richness_Pivot_redshift"]
        
        sigmaRM = self.sigmaRM if hasattr(self, "sigmaRM") else 0.25
        pmr_distribution = self.pmr_distribution if hasattr(self, "pmr_distribution") else "log-normal"
        if use_redshift == False:
            lambda_true_grid, M_grid = np.meshgrid(lambda_true, M)
            lambda_true_grid = lambda_true_grid.astype(float_dtype)
            self.lambda_true_grid = lambda_true_grid
            M_grid = M_grid.astype(float_dtype)
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope 
            self.D_ang =  (ccl.angular_diameter_distance(cosmo, 1/(z1 + z_arr)) * (1 + z_arr))[None,None,:] + 0*M_grid[None,...]
        else:
            lambda_true_grid, M_grid, z_grid = np.meshgrid(lambda_true, M, z_arr, indexing="ij")
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope * \
                            (z_grid / mass2richness_Pivot_redshift)**mass2richness_Slope_redshift
            self.M_grid = M_grid
            self.z_grid = z_grid
            self.lambda_true_grid = lambda_true_grid
            self.lambda_model = lambda_model
            self.D_ang = (ccl.angular_diameter_distance(cosmo, 1/(1 + z_arr)) * (1 + z_arr))[None,None,None,:] + 0*M_grid[None,...]
        if redshift_weight_function == False:
            Wz = lambda x: 1
        else:
            import scipy
            if weights_function_kwargs["overwrite"] == False:
                w, b = np.loadtxt(weights_function_kwargs["file"]).T
                Wz = scipy.interpolate.interp1d(cbins, w)
            else:
                zmin, zmax = weights_function_kwargs["zmin"], weights_function_kwargs["zmax"]
                deltaz = weights_function_kwargs["deltaz"]
                bins = np.arange(zmin, zmax, deltaz)
                interpolation_mode = weights_function_kwargs["interpolation"]
                Z = self.z 
                w, b = np.histogram(Z, bins = bins, density = True)
                cbins = np.array([b[i] + b[i + 1] for i in range(len(b) - 1)])/2
                np.savetxt(weights_function_kwargs["file"], np.stack((cbins, w)))
                Wz = scipy.interpolate.interp1d(cbins, w)
    
        Wr = richness_weights_function_kwargs["func"]
        params = richness_weights_function_kwargs["params"]

        dV = ccl.background.comoving_volume_element(cosmo, 1/(1 + z_arr))
        self.dV = dV

        if (use_two_halo_term is not None) and type(use_two_halo_term) in (str, bool):
            if use_two_halo_term == True or use_two_halo_term == "only":
                if verbose:
                    print("Creating\033[92m 1+2-halo function.\033[0m") if use_two_halo_term == True else print("Creating\033[92m 2-halo function.\033[0m")
                R2halo = two_halo_kwargs["R"]
                delta = two_halo_kwargs["delta"]
                M_arr2halo = two_halo_kwargs["M"]  
                cosmo2halo = two_halo_kwargs["cosmo"]
                k2halo = two_halo_kwargs["k"]    
                N = two_halo_kwargs["N"]
                h = two_halo_kwargs["h"]
                mdef = f"{int(delta)}c" if two_halo_kwargs["background"] == "critical" else f"{int(delta)}m"
                mfunc = ccl.halos.MassFuncTinker10(mass_def = mdef)#mass function from Tinker et al 2010
                dndM2halo = np.array([[mfunc(cosmo2halo, Mi, 1/(zi + 1)) for Mi in M_arr2halo] for zi in z_arr]).astype(float_dtype)
                dndM2halo = dndM2halo/(np.log(10)*M_arr2halo)
                s = 0.037
                q = 1.008
                dndM2halo = dndM2halo * (s*np.log(M_arr2halo/10**(13.8)/cosmo._params.h) + q)
                bias = ccl.halos.HaloBiasTinker10(mass_def=mdef) 
                bh = np.array([bias(cosmo2halo, M, 1/(1 + zi)) for zi in z_arr]).astype(float_dtype)
                bM = np.array([[bias(cosmo2halo, Mi, 1/(1 + zi)) for Mi in M_arr2halo] for zi in z_arr]).astype(float_dtype)
                Pk = np.array([ccl.linear_matter_power(cosmo2halo, k2halo, 1/(1+zi)) for zi in z_arr]).astype(float_dtype)
                Rgrid, z2halo_grid, M2halo_grid = np.meshgrid(R2halo, z_arr, M_arr2halo, indexing = "ij")
                ki_r = R2halo[None,:]*k2halo[:,None]
                sin_term = np.sin(ki_r) / np.where(ki_r != 0, ki_r, 1)
                self.h = HankelSphericalTransform(N=N, h=h)
                self.dndM2halo = dndM2halo
                self.bh = bh
                self.bM = bM
                self.Pk = Pk
                self.sin_term = sin_term
                self.Rgrid = Rgrid
                self.z2halo_grid = z2halo_grid
                self.M2halo_grid = M2halo_grid
                self.k2halo = k2halo
                self.R2halo = R2halo
                self.M_arr2halo = M_arr2halo
                self.D_ang2halo = (ccl.angular_diameter_distance(cosmo2halo, 1/(1 + z_arr))*(1 + z_arr))[None,:]
                lambda2halo_grid = mass2richness_Norm * (M2halo_grid / mass2richness_Pivot)**mass2richness_Slope * \
                                    ((1 + z2halo_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift  
                two_halo_power_func = two_halo_kwargs["two_halo_power_func"]
                self.labmda2halo_grid = lambda2halo_grid
                self.two_halo_func = two_halo_power_func
                self.two_halo_func_evals = (two_halo_kwargs["eval_only_mass"], two_halo_kwargs["eval_only_richness"], two_halo_kwargs["eval_only_redshift"])
            else:
                if verbose:
                    print("Creating\033[92m 1-halo function.\033[0m")

        if verbose:
            print("Using fixed\033[92m halo model\033[0m") if fixed_RM_relationship == True else print("Using free\033[92m halo model\033[0m") 

        if mis_centering == True:
            Roff = np.array(mis_centering_kwargs["Roff"])[:, None] #Roff of mis-centering
            rho_Roff = mis_centering_kwargs["distribution"] #p(Roff)
            mis_centering_func = mis_centering_kwargs["func"]
            theta = mis_centering_kwargs["theta"]
            print("Adding\033[92m mis-centering\033[0m") if verbose else None

        if use_filters and filters is not None:
            print("Adding filters:")
            [print(f"\033[92m{k}\033[0m: {v}") for k,v in filters.items()]
            func_names = list(filters.keys())
            func_args = list(filters.values())
            self_output_path = self.output_path
            self_output_path = self_output_path + "/" if self_output_path[-1] != "/" else self_output_path
            func_filters = [load_function_from_file(self_output_path + "filters.py", n) for n in func_names]

        if rebinning == True:
            print(f"Using \033[92mrebinning\033[0m")
            method_rebinning = rebinning_kwargs["method"]
            interpolation_kwargs = rebinning_kwargs["interpolation_kwargs"]
            nbins_rebinning = rebinning_kwargs["nbins"]
            pixel_size_rebinning = rebinning_kwargs["pixel_size"]
        if subr_grid == True:
            rmin, rmax = subr_grid_kwargs["rmin"], subr_grid_kwargs["rmax"]
            n = subr_grid_kwargs["n"]
            subR_grid = np.linspace(rmin, rmax, n)
        if weighted == True:
            print(f"Using \033[92mweights\033[0m")
            if hasattr(self, "weights"):
                W = self.weights
                redshift = self.z
                richness = self.richness
                lambda_obs = self.lambda_obs

                z_edges = (z_arr[:-1] + z_arr[1:]) / 2
                z_first = z_arr[0] - (z_arr[1] - z_arr[0]) / 2
                z_last  = z_arr[-1] + (z_arr[-1] - z_arr[-2]) / 2
                edges = np.concatenate(([z_first], z_edges, [z_last]))

                lambda_edges = (lambda_obs[:-1] + lambda_obs[1:]) / 2
                lambda_first = lambda_obs[0] - (lambda_obs[1] - lambda_obs[0]) / 2
                lambda_last  = lambda_obs[-1] + (lambda_obs[-1] - lambda_obs[-2]) / 2
                lambda_edges = np.concatenate(([lambda_first], lambda_edges, [lambda_last]))
                
                weights = np.zeros((len(z_arr), len(lambda_obs)))
                for i in range(len(z_edges)-1):
                    for j in range(len(lambda_edges)-1):
                        mask = np.where((redshift > edges[i]) & (redshift < edges[i+1]) & (richness > lambda_edges[j]) & (richness < lambda_edges[j+1]))[0]
                        if len(mask) > 0:
                            weights[i][j] = np.sum(W[mask])
                        else:
                            continue
                weights = weights.T
            else:
                print("Weights not available!")
                weights = np.ones((len(z_arr), len(lambda_obs))).T
        else:
            weights = np.ones((len(z_arr), len(lambda_obs))).T
        if numba == True:
            print("Using \033[92mnumba\033[0m")
        self.W = weights

        norm = np.trapz( 
                dV * np.trapz( dndM.T * 
                    np.trapz(
                        np.trapz(PllM, axis = 0, x = lambda_true), axis = 0, x = lambda_obs
                        ), axis = 0, x = M)
                    , axis = 0, x = z_arr)
        self.norm = norm
        global func
        if numba == False:
            def func(r, params, model1h = None, model2h = None, RM_params = None, new_PllM = None, new_sigmaRM = None, rbins = 35, new_Plambda_true = None, 
                    smooth = None, eval_lambda = True, mis_centering_params = None, Roff = np.logspace(-1, 1, 10), return_2halo_term = False,
                    theta = np.linspace(0,2*np.pi,60), mass2richness_Pivot = 3e14/0.7, mass2richness_Pivot_redshift = 0.35
                    , sigmaRM = 0.25, two_halo_power = None, return_profile_grid = False, R_intp = None):
                if subr_grid == True:
                    R = subR_grid
                else:
                    R = r
                h = self.h
                M,z_arr = self.M, self.z_arr
                lambda_true = self.lambda_true
                lambda_obs = self.lambda_obs
                lambda_true_grid = self.lambda_true_grid
                M_grid = self.M_grid
                z_grid = self.z_grid
                lambda_true_grid = self.lambda_true_grid
                Plambda_true = self.Plambda_true
                lambda_model = self.lambda_model
                PllM = self.PllM
                Plambda_obs_M = self.Plambda_obs_M
                P_Mass = self.P_Mass
                norm = self.norm
                D_ang = self.D_ang
                weights = self.W
                dV = self.dV
                Mgrid_mis, zgrid_mis = np.meshgrid(M, z_arr)
                R_Mpc_mis = ((R * 180/np.pi / 60)[:,None,None] * ((ccl.angular_diameter_distance(cosmo, 1/(1 + z_arr))))[None, None,:])
                theta = np.array(theta, dtype = float_dtype)
                Roff = np.array(Roff, dtype = float_dtype)
                Roff2 = Roff[:, None, None,None]

                f2halo = self.two_halo_func if hasattr(self, "two_halo_func") else None

                xmis = (Roff2**2 + R_Mpc_mis[None, :,:,:]**2 + 2 * (R_Mpc_mis[None,:,:,:] * Roff2)[None,...] * np.cos(theta[:, None, None, None, None])) ** 0.5

                if new_PllM is not None:
                    PllM = new_PllM
                if new_Plambda_true is not None:
                    Plambda_true = new_Plambda_true
                if new_sigmaRM is not None:
                    sigmaRM = new_sigmaRM

                if fixed_RM_relationship == False and RM_params is not None:
                    mass2richnes_Norm, mass2richness_Slope, mass2richness_Slope_redshift = RM_params
                    if use_redshift == True:
                        lambda_true_grid, M_grid2, z_grid2 = np.meshgrid(lambda_true, M, z_arr, indexing = "ij")
                        lambda_true_grid = lambda_true_grid.astype(float_dtype)
                        M_grid2 = M_grid2.astype(float_dtype)
                        z_grid2 = z_grid2.astype(float_dtype)

                        lambda_model = mass2richness_Norm * (M_grid2 / mass2richness_Pivot)**mass2richness_Slope * \
                                    ((1 + z_grid2)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift     
                    else:
                        lambda_true_grid2, M_grid2 = np.meshgrid(lambda_true, M)
                        lambda_model = mass2richness_Norm * (M_grid2 / mass2richness_Pivot)**mass2richness_Slope

                    sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + sigmaRM**2)

                    if pmr_distribution == "log-normal":
                        Plambda_true_Mass = 1/(np.sqrt(2 * np.pi**2 * sigma_model**2) * lambda_true_grid) * np.exp(
                            - (np.log(lambda_true_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2))
                    elif pmr_distribution == "normal":
                        Plambda_true_Mass = 1/(np.sqrt(2*np.pi*sigmaRM**2))*np.exp(
                            -(np.log(lambda_true_grid) - np.log(lambda_model))**2/ (2*sigmaRM**2))

                    PllM = Plambda_true [:,:,None,:]* Plambda_true_Mass[:,None,:,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) ==> P(lambda_obs | lambda_true, M)
                    Plambda_obs_M = np.trapz(PllM, axis = 0, x = lambda_true) # P(lambda_obs | M)
                    Plambda_obs_M = gaussian_filter(Plambda_obs_M, smooth) if smooth is not None else Plambda_obs_M
                    P_Mass = np.trapz(Plambda_obs_M, axis = 0, x = lambda_obs) # P(M)
                    norm = np.trapz( 
                            dV * np.trapz( dndM.T * 
                                np.trapz(
                                    np.trapz(PllM*weights[None,:,None,:], axis = 0, x = lambda_true), axis = 0, x = lambda_obs
                                    ), axis = 0, x = M)
                                , axis = 0, x = z_arr) 

                if use_two_halo_term != "only" and return_2halo_term == False:
                    x_grid = lambda_model if eval_lambda == True else M_grid
                    R_Mpc = ((R * np.pi/180 / 60)[:,None,None,None] * self.D_ang)
                    if model1h is None:
                        one_halo_term = one_halo_profile(R_Mpc, x_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else one_halo_profile(R_Mpc, x_grid, M_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) #result is the profile model evaluated at R, M/lambda, z            
                    else:
                        one_halo_term = model1h(R_Mpc, x_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else model1h(R_Mpc, x_grid, M_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) #result is the profile model evaluated at R, M/lambda, z
                    weighted_one_halo_term = one_halo_term[:,:,None,:,:]*PllM[None,...]#weights[None,None,:,None,:]
                if use_two_halo_term == True or use_two_halo_term == "only":
                    
                    dndM2halo = self.dndM2halo
                    bh = self.bh
                    bM = self.bM
                    Pk = self.Pk
                    sin_term = self.sin_term
                    Rgrid = self.Rgrid
                    z2halo_grid = self.z2halo_grid
                    M2halo_grid = self.M2halo_grid
                    k2halo = self.k2halo
                    R2halo = self.R2halo
                    M_arr2halo = self.M_arr2halo

                    if model1h is None and model2h is None:
                        PRMz = two_halo_profile(Rgrid, 0, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == True else two_halo_profile(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot)             
                    elif model1h is not None and model2h is None:
                        PRMz = model1h(Rgrid, 0, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == True else model1h(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot)             
                    elif model2h is not None:
                        PRMz = model2h(Rgrid, 0, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == True else model2h(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot)           

                    uP = 4*np.pi* Rgrid[:,None,:,:]**2 * (sin_term[:,:, None, None]) * PRMz[:,None,:,:]
                    uPw = (dndM2halo*bM).T[None,None,:,:] * uRMz
                    PhP = bh.T[:,None,None,None,:] *(Pk.T[None,:,None,:] * uPk)[None,...]
                    xhi_P = np.trapz(((sin_term2 * k2halo[None,:,None]**2)[:, None, None,:,None,:] * bhuPk[None,...])/(2*np.pi**2), axis = 3, x = k2halo)
                    
                    two_halo_term = np.trapz(np.trapz(xhi_P, x = R2halo, axis = 1), x = M_arr2halo, axis = 2)/(2*np.pi**2)
                    
                    weighted_two_halo_term = PllM[None,:,:,:,:] * xhi_P[:,:,None,:,:]
                    if two_halo_power is not None:
                        if hasattr(self, "two_halo_func_evals"):
                            if np.all(self.two_halo_func_evals == True):
                                weighted_two_halo_term = f2halo(lambda_model, M_grid, z_grid, two_halo_power)[:,None,None,:,:]*weighted_two_halo_term
                        else:
                            pass
                    P2halo = np.trapz(np.trapz(weighted_two_halo_term, axis = 1, x = lambda_true), axis = 1, x = lambda_obs)
                    if return_2halo_term == True:
                        return P2halo
                P1halo = np.trapz(np.trapz(weighted_one_halo_term, axis = 1, x = lambda_true), axis = 1, x = lambda_obs) #integrate over the observed richness
                if hasattr(self, "two_halo_func_evals") and two_halo_power is not None and use_two_halo_term == True:
                    if self.two_halo_func_evals[0] == False and self.two_halo_func_evals[1] == False and self.two_halo_func_evals[2] == True:
                        z_grid2, M_grid2 = np.meshgrid(z_arr, M)
                        z_unique, z_index = np.unique(z_grid2, return_inverse = True)
                        P2halo = self.two_halo_func(z_unique, two_halo_power)[z_index].reshape(z_grid2.shape)[None,...]*P2halo
                if use_two_halo_term == False:
                    PRMz = [P1halo]
                elif use_two_halo_term == True:
                    PRMz = [P1halo, P2halo]
                    if return_1h2h == False:
                        PRMz = [P1halo + P2halo]
                elif use_two_halo_term == "only":
                    PRMz = [P2halo]
                if return_profile_grid == True:
                    return PRMz

                output = np.zeros((2, len(R))) if return_1h2h == True else np.zeros((1, len(R)))
                infered_Mass = 0
                for k, P in enumerate(PRMz):
                    if mis_centering == True and mis_centering_params is not None:
                        if len(mis_centering_params) == 2:
                            fmis, mis_centering_params_func = mis_centering_params[0], mis_centering_params[1::] if mis_centering_params is not None else [0.246, 0.385]
                        elif len(mis_centering_params) > 2 and mis_centering_func is not None:
                            M2, z2 = np.meshgrid(M, z_arr)
                            p = mis_centering_func(M2, z2, mis_centering_params)
                            fmis, mis_centering_params_func = p[0], p[1::]
                        weights_mc = np.array(rho_Roff(Roff, *mis_centering_params_func), dtype = float_dtype)
                        funcs = [
                        [UnivariateSpline(R_Mpc_mis[:,i,j], pj, k=1, s=0) for j,pj in enumerate(pi)] for i,pi in enumerate(P.T)
                        ]
                        off = np.array(
                            [[np.trapz(fj(xmis[:,:,:,i,j]), theta, axis=0) for j,fj in enumerate(fi)] for i,fi in enumerate(funcs)]
                        )
                        woff = np.array(
                            [[np.trapz(weights_mc[...,None] * off[i,j,:,:], Roff, axis = 0)/np.trapz(weights_mc, Roff) for j,pj in enumerate(pi)] 
                            for i,pi in enumerate(off)]
                        )
                        P = (1 - fmis)*P + fmis*woff.T/(2*np.pi)

                    if infere_mass == True:
                        if mass_inf_kwargs["is_kappa"] == True:
                            sigma_crit = sigma_crit_cmb(z_arr)
                            rho_RMz = (sigma_crit[None, None, :] * P).value
                            infered_Masses = np.zeros((len(z_arr), len(M)))
                            for i in range(len(z_arr)):
                                rho_i = np.array(rho_RMz[:,:,i], dtype = object)
                                r_mpc = np.array(R_Mpc[:,0,:,i], dtype = object)
                                if mass_inf_kwargs["interpolate"] == True:
                                    new_R = np.linspace(r.min(), r.max(), mass_inf_kwargs["Nr"])
                                    new_rho_i = np.zeros((len(new_R), len(M)))
                                    for j in range(len(M)):
                                        new_rho_i[:,j] = UnivariateSpline(r_mpc[:,j], rho_i[:,j], k=1, s=0)(new_R)
                                    rho_i = new_rho_i
                                    r_mpc = np.repeat(new_R, len(M)).reshape((len(new_R), len(M)))
                                infered_Masses[i] = trapz(2*np.pi*r_mpc * rho_i, x = r_mpc, axis = 0)
                            infered_Mass += np.trapz(dV * np.trapz((dndM * infered_Masses).T, axis = 0, x = M), axis = 0, x = z_arr)/norm

                    if apply_filter_per_profile == True:
                        Pz = np.trapz(dndM.T*P, axis = 1, x = M)
                        new_P = np.zeros_like(Pz)
                        for zi in range(len(z_arr)):
                            for i in range(len(func_filters)):
                                if i == 0:
                                    new_P[:,zi] = func_filters[i](R, Pz[:,zi], **func_args[i])
                                else:
                                    new_P[:,zi] = func_filters[i](R, new_P[:,zi], **func_args[i])
                        stacked_P = np.trapz(dV[None,...]*new_P, x = z_arr, axis = 1)/norm
                    else:
                        stacked_P = np.trapz(dV[None,...] * np.trapz(dndM.T * P, axis = 1, x = M), x = z_arr, axis = 1)/norm #integrate over the mass and redshift 
                    if apply_filter_per_profile == False:
                        if use_filters == True and filters is not None:
                            for i in range(len(func_filters)):
                                stacked_P = func_filters[i](R, stacked_P, **func_args[i])
                    if rebinning == True:
                        if method_rebinning == 'interp1d':
                            intp = interp1d(R, stacked_P, **interpolation_kwargs)
                        elif method_rebinning == 'spline':
                            intp = UnivariateSpline(R, stacked_P, **interpolation_kwargs)
                        pix_size = pixel_size_rebinning

                        R_edges = np.zeros(len(r) + 1)
                        R_edges[1:-1] = 0.5 * (r[1:] + r[:-1])
                        R_edges[0]  = r[0] - 0.5 * (r[1] - r[0])
                        R_edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

                        x = np.arange(-R_edges.max(), R_edges.max(), pix_size)
                        y = np.arange(-R_edges.max(), R_edges.max(), pix_size)
                        x,y = np.meshgrid(x,y)
                        r_intp = np.sqrt(x**2 + y**2)
                        P_r = intp(r_intp)

                        stacked_P = np.zeros(len(R_edges)-1, dtype = float_dtype)
                        for i in range(len(R_edges) - 1):
                            ri,rf = R_edges[i], R_edges[i+1]
                            mask = np.where((r_intp >= ri) & (r_intp < rf))
                            stacked_P[i] = np.nanmean(P_r[mask]) if len(P_r[mask]) > 0 else 0
                    output[k] = stacked_P
                Ptotal = np.sum(output, axis = 0).astype(float_dtype) if len(output) > 1 else output[0]
                if return_1h2h == True and use_two_halo_term == True:
                    P1h, P2h = output
                    if infere_mass == True:
                        return Ptotal, P1h, P2h, infered_Mass
                    return Ptotal, P1h, P2h
                elif return_1h2h == False and use_two_halo_term == 'only':
                    if infere_mass ==  True:
                        return P2h, infered_Mass
                    return P2h
                else:
                    if infere_mass == True:
                        return Ptotal, infered_Mass
                    return Ptotal
            return func
        else:
            global func_numba
            def func_numba(r, params, model1h = None, model2h = None, RM_params = None, new_PllM = None, new_sigmaRM = None, rbins = 35, new_Plambda_true = None, 
                    smooth = None, eval_lambda = True, mis_centering_params = None, Roff = np.logspace(-1, 1, 10), return_2halo_term = False,
                    theta = np.linspace(0,2*np.pi,60), mass2richness_Pivot = 3e14/0.7, mass2richness_Pivot_redshift = 0.35
                    , sigmaRM = 0.25, two_halo_power = None, return_profile_grid = False, R_intp = None):
                if subr_grid == True:
                    R = subR_grid
                else:
                    R = r
                h = self.h
                M,z_arr = self.M, self.z_arr
                lambda_true = self.lambda_true
                lambda_obs = self.lambda_obs
                lambda_true_grid = self.lambda_true_grid
                M_grid = self.M_grid
                z_grid = self.z_grid
                lambda_true_grid = self.lambda_true_grid
                Plambda_true = self.Plambda_true
                lambda_model = self.lambda_model
                PllM = self.PllM
                Plambda_obs_M = self.Plambda_obs_M
                P_Mass = self.P_Mass
                norm = self.norm
                D_ang = self.D_ang
                weights = self.W
                dV = self.dV
                Mgrid_mis, zgrid_mis = np.meshgrid(M, z_arr)
                if physical == False:
                    R_Mpc_mis = ((R * np.pi / (180*60))[:,None,None] * ((z2dA(zgrid_mis ) * (1 + zgrid_mis)))[None,:,:])
                else:
                    M_ = M[None, :, None]
                    z_ = z_arr[None, None, :]
                    R_Mpc_mis = R[:,None,None] + M_ + z_
                theta = np.array(theta, dtype = float_dtype)
                Roff = np.array(Roff, dtype = float_dtype)
                Roff2 = Roff[:, None, None,None]

                f2halo = self.two_halo_func if hasattr(self, "two_halo_func") else None

                xmis = compute_xmis(Roff, R_Mpc_mis, theta)

                if new_PllM is not None:
                    PllM = new_PllM
                if new_Plambda_true is not None:
                    Plambda_true = new_Plambda_true
                if new_sigmaRM is not None:
                    sigmaRM = new_sigmaRM

                if fixed_RM_relationship == False and RM_params is not None:
                    mass2richnes_Norm, mass2richness_Slope, mass2richness_Slope_redshift = RM_params
                    if use_redshift == True:
                        lambda_true_grid, M_grid2, z_grid2 = np.meshgrid(lambda_true, M, z_arr, indexing = "ij")
                        lambda_true_grid = lambda_true_grid.astype(float_dtype)
                        M_grid2 = M_grid2.astype(float_dtype)
                        z_grid2 = z_grid2.astype(float_dtype)

                        lambda_model = mass2richness_Norm * (M_grid2 / mass2richness_Pivot)**mass2richness_Slope * \
                                    ((1 + z_grid2)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift     
                    else:
                        lambda_true_grid2, M_grid2 = np.meshgrid(lambda_true, M)
                        lambda_model = mass2richness_Norm * (M_grid2 / mass2richness_Pivot)**mass2richness_Slope

                    sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + sigmaRM**2)

                    if pmr_distribution == "log-normal":
                        Plambda_true_Mass = 1/(np.sqrt(2 * np.pi**2 * sigma_model**2) * lambda_true_grid) * np.exp(
                            - (np.log(lambda_true_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2))
                    elif pmr_distribution == "normal":
                        Plambda_true_Mass = 1/(np.sqrt(2*np.pi*sigmaRM**2))*np.exp(
                            -(np.log(lambda_true_grid) - np.log(lambda_model))**2/ (2*sigmaRM**2))

                    PllM = Plambda_true [:,:,None,:]* Plambda_true_Mass[:,None,:,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) ==> P(lambda_obs | lambda_true, M)
                    Plambda_obs_M = trapz_axis0(PllM, lambda_true)           # np.trapz(PllM, axis=0, x=lambda_true)
                    Plambda_obs_M = gaussian_filter(Plambda_obs_M, smooth) if smooth is not None else Plambda_obs_M
                    P_Mass = trapz_axis0(Plambda_obs_M, lambda_obs)           # np.trapz(Plambda_obs_M, axis=0, x=lambda_obs)
                    norm = trapz_1d(
                            dV * trapz_axis0(dndM.T *
                                trapz_axis0(
                                    trapz_axis0(PllM*weights[None,:,None,:], lambda_true), lambda_obs
                                    ), M)
                                , z_arr)                                       # np.trapz( dV * np.trapz( dndM.T * np.trapz( np.trapz(PllM*weights[None,:,None,:], axis=0, x=lambda_true), axis=0, x=lambda_obs ), axis=0, x=M ), axis=0, x=z_arr )

                if use_two_halo_term != "only" and return_2halo_term == False:

                    x_grid = lambda_model if eval_lambda == True else M_grid
                    if physical == False:
                        R_Mpc = ((R * np.pi/(180 * 60))[:,None,None,None] * self.D_ang)
                    else:
                        lambda_ = lambda_model[None, :, None, None]
                        M_ = M[None, None, :, None]
                        z_ = z_arr[None, None, None, :]
                        R_Mpc = R[:, None, None, None] + lambda_ + M_ + z_
                    if model1h is None:
                        one_halo_term = one_halo_profile(R_Mpc, x_grid, z_grid, params, rbins = rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else one_halo_profile(R_Mpc, x_grid, M_grid, z_grid, params, rbins = rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) #result is the profile model evaluated at R, M/lambda, z            
                    else:
                        one_halo_term = model1h(R_Mpc, x_grid, z_grid, params, rbins = rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else model1h(R_Mpc, x_grid, M_grid, z_grid, params, rbins = rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) #result is the profile model evaluated at R, M/lambda, z         
                    weighted_one_halo_term = weight_one_halo(
                        weights,
                        one_halo_term,
                        PllM * dV[None,None,None,:] * dndM.T[None,None,:,:]
                    )

                if use_two_halo_term == True or use_two_halo_term == "only":
                    
                    dndM2halo = self.dndM2halo
                    bh = self.bh
                    bM = self.bM
                    Pk = self.Pk
                    sin_term = self.sin_term
                    Rgrid = self.Rgrid
                    z2halo_grid = self.z2halo_grid
                    M2halo_grid = self.M2halo_grid
                    k2halo = self.k2halo
                    R2halo = self.R2halo
                    M_arr2halo = self.M_arr2halo

                    if model1h is None and model2h is None:
                        PRMz = two_halo_profile(Rgrid, lambda2halo_grid, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False) if eval_mass == True else two_halo_profile(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False)             
                    elif model1h is not None and model2h is None:
                        PRMz = model1h(Rgrid, lambda2halo_grid, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False) if eval_mass == True else model1h(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False)             
                    elif model2h is not None:
                        PRMz = model2h(Rgrid, lambda2halo_grid, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False) if eval_mass == True else model2h(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot, projected = False)           

                    two_halo_profiles = p2h(k2halo, M_arr2halo, z_arr, M, R_Mpc, R2halo, PRMz, dndM2halo, Pk, bM, bh, h)
                    weighted_two_halo_term = PllM[None, ...] * two_halo_profiles[:,None,None,:,:] * dndM.T[None,None,None,:,:] * dV[None,None,None,None,:]
                    if two_halo_power is not None:
                        if hasattr(self, "two_halo_func_evals"):
                            if np.all(self.two_halo_func_evals == True):
                                weighted_two_halo_term = f2halo(lambda_model, M_grid, z_grid, two_halo_power)[:,None,None,:,:]*weighted_two_halo_term
                        else:
                            pass  

                    if two_halo_power is not None:
                        if hasattr(self, "two_halo_func_evals"):
                            if np.all(self.two_halo_func_evals == True):
                                weighted_two_halo_term = f2halo(lambda_model, M_grid, z_grid, two_halo_power)[:,None,None,:,:]*weighted_two_halo_term
                        else:
                            pass
                    P2halo = trapz_axis1(trapz_axis1(weighted_two_halo_term, lambda_true), lambda_obs)  # np.trapz(np.trapz(..., axis=1, x=lambda_true), axis=1, x=lambda_obs)
                    if return_2halo_term == True:
                        return P2halo
                P1halo = trapz_axis1(trapz_axis1(weighted_one_halo_term, lambda_true), lambda_obs)  # np.trapz(np.trapz(..., axis=1, x=lambda_true), axis=1, x=lambda_obs)
                if hasattr(self, "two_halo_func_evals") and two_halo_power is not None and use_two_halo_term == True:
                    if self.two_halo_func_evals[0] == False and self.two_halo_func_evals[1] == False and self.two_halo_func_evals[2] == True:
                        z_grid2, M_grid2 = np.meshgrid(z_arr, M)
                        z_unique, z_index = np.unique(z_grid2, return_inverse = True)
                        P2halo = self.two_halo_func(z_unique, two_halo_power)[z_index].reshape(z_grid2.shape)[None,...]*P2halo
                if use_two_halo_term == False:
                    PRMz = [P1halo]
                elif use_two_halo_term == True:
                    PRMz = [P1halo, P2halo]
                    if return_1h2h == False:
                        PRMz = [P1halo + P2halo]
                elif use_two_halo_term == "only":
                    PRMz = [P2halo]
                if return_profile_grid == True:
                    return PRMz

                output = np.zeros((2, len(R))) if return_1h2h == True else np.zeros((1, len(R)))
                infered_Mass = 0
                
                for k, P in enumerate(PRMz):
                    if mis_centering == True and mis_centering_params is not None:
                        t3 = time()
                        if len(mis_centering_params) == 2:
                            fmis, mis_centering_params_func = mis_centering_params[0], mis_centering_params[1::] if mis_centering_params is not None else [0.246, 0.385]
                        elif len(mis_centering_params) > 2 and mis_centering_func is not None:
                            M2, z2 = np.meshgrid(M, z_arr)
                            p = mis_centering_func(M2, z2, mis_centering_params)
                            fmis, mis_centering_params_func = p[0], p[1::]
                        weights_mc = np.array(rho_Roff(Roff, *mis_centering_params_func), dtype = float_dtype)
                        P = miscenter_core(
                            P,
                            R_Mpc_mis,
                            xmis,
                            theta,
                            Roff,
                            weights_mc,
                            fmis
                        )
                    if infere_mass == True:
                        if mass_inf_kwargs["is_kappa"] == True:
                            sigma_crit = sigma_crit_cmb(z_arr)
                            rho_RMz = (sigma_crit[None, None, :] * P).value
                            infered_Masses = np.zeros((len(z_arr), len(M)))
                            for i in range(len(z_arr)):
                                rho_i = np.array(rho_RMz[:,:,i], dtype = object)
                                r_mpc = np.array(R_Mpc[:,0,:,i], dtype = object)
                                if mass_inf_kwargs["interpolate"] == True:
                                    new_R = np.linspace(r.min(), r.max(), mass_inf_kwargs["Nr"])
                                    new_rho_i = np.zeros((len(new_R), len(M)))
                                    for j in range(len(M)):
                                        new_rho_i[:,j] = UnivariateSpline(r_mpc[:,j], rho_i[:,j], k=1, s=0)(new_R)
                                    rho_i = new_rho_i
                                    r_mpc = np.repeat(new_R, len(M)).reshape((len(new_R), len(M)))
                                infered_Masses[i] = trapz_axis0(2*np.pi*r_mpc * rho_i, r_mpc)  # trapz(2*np.pi*r_mpc * rho_i, x=r_mpc, axis=0)
                            infered_Mass += trapz_1d(trapz_axis0((dndM * infered_Masses).T, M), z_arr)  # np.trapz( dV * np.trapz(..., axis=0, x=M), axis=0, x=z_arr ) / norm

                    if apply_filter_per_profile == True:
                        Pz = trapz_axis1(P, M)                         # np.trapz(dndM.T*P, axis=1, x=M)
                        new_P = np.zeros_like(Pz)
                        for zi in range(len(z_arr)):
                            for i in range(len(func_filters)):
                                if i == 0:
                                    new_P[:,zi] = func_filters[i](R, Pz[:,zi], **func_args[i])
                                else:
                                    new_P[:,zi] = func_filters[i](R, new_P[:,zi], **func_args[i])
                        stacked_P = trapz_1d(new_P, z_arr)/norm   # np.trapz(..., x=z_arr, axis=1) / norm
                    else:
                        stacked_P = trapz_axis1(trapz_axis1(P, M), z_arr) / norm
                    if apply_filter_per_profile == False:
                        if use_filters == True and filters is not None:
                            for i in range(len(func_filters)):
                                stacked_P = func_filters[i](R, stacked_P, **func_args[i])
                    if rebinning == True:
                        if method_rebinning == 'interp1d':
                            intp = interp1d(R, stacked_P, **interpolation_kwargs)
                        elif method_rebinning == 'spline':
                            intp = UnivariateSpline(R, stacked_P, **interpolation_kwargs)
                        pix_size = pixel_size_rebinning

                        R_edges = np.zeros(len(r) + 1)
                        R_edges[1:-1] = 0.5 * (r[1:] + r[:-1])
                        R_edges[0]  = r[0] - 0.5 * (r[1] - r[0])
                        R_edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

                        x = np.arange(-R_edges.max(), R_edges.max(), pix_size)
                        y = np.arange(-R_edges.max(), R_edges.max(), pix_size)
                        x,y = np.meshgrid(x,y)
                        r_intp = np.sqrt(x**2 + y**2)
                        P_r = intp(r_intp)

                        stacked_P = np.zeros(len(R_edges)-1, dtype = float_dtype)
                        for i in range(len(R_edges) - 1):
                            ri,rf = R_edges[i], R_edges[i+1]
                            mask = np.where((r_intp >= ri) & (r_intp < rf))
                            stacked_P[i] = np.nanmean(P_r[mask]) if len(P_r[mask]) > 0 else 0
                    output[k] = stacked_P
                Ptotal = np.sum(output, axis = 0).astype(float_dtype) if len(output) > 1 else output[0]
                if return_1h2h == True and use_two_halo_term == True:
                    P1h, P2h = output
                    if infere_mass == True:
                        return Ptotal, P1h, P2h, infered_Mass
                    return Ptotal, P1h, P2h
                elif return_1h2h == False and use_two_halo_term == 'only':
                    if infere_mass ==  True:
                        return P2h, infered_Mass
                    return P2h
                else:
                    if infere_mass == True:
                        return Ptotal, infered_Mass
                    return Ptotal
            return func_numba
            
    def create_beam_filter(self, mode = "w", beam_size = 1.6, gaussian = False, 
        beam_file = "/data2/javierurrutia/szeffect/data/act-beams/f90_beam.npy",
        theta_file = "/data2/javierurrutia/szeffect/data/act-beams/theta_arcmin_f90_beam.npy"):

        output_path = self.output_path
        if gaussian == True:
            content = f"""
from scipy.ndimage import gaussian_filter1d
import numpy as np
def apply_beam(R, data, fwhm = {beam_size}):
    fwhm = float(fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dr = (R[-1] - R[0]) / (len(R) - 1)
    sigma_pix = sigma / dr
    return gaussian_filter1d(data, sigma=sigma_pix, mode='constant', cval=0.0)
        """
        else:

            content = f"""
from scipy.interpolation import interp1d
import numpy as np
from scipy.signal import fftconvolve

beam = np.load('{beam_file}')
theta = np.load('{theta_file}')

beam = beam / np.sum(beam)

interp_beam = interp1d(theta, beam, bounds_error=False, fill_value=0.0)

def apply_beam(R, data, nr = 100):
    r = np.linspace(-R.max(), R.max(), nr)
    xr, yr = np.meshgrid(r, r)
    r2d = np.sqrt(xr**2 + yr**2)
    
    interp_profile = interp1d(R, data, bounds_error=False, fill_value=0.0)

    profile2d = interp_profile(r2d)
    beam2d = interp_beam(r2d)

    convolved_profile = fftconvolve(profile2d, beam2d, mode="same")

    R_edges = (R[:-1] + R[1:]) / 2
    R_first = R[0] - (R[1] - R[0]) / 2
    R_last  = R[-1] + (R[-1] - R[-2]) / 2
    R_edges = np.concatenate(([R_first], R_edges, [R_last]))
    
    digitized = np.digitize(r2d, R_edges)

    digitized = digitized.ravel()
    convolved_profile = convolved_profile.ravel()

    count = np.bincount(digitized)[1:-1]
    profile = np.bincount(digitized, weights=convolved_profile)[1:-1] / count

    return profile
"""
        with open(output_path + "/filters.py", mode) as f:
            f.write(content)
        
    def stats(self):
        print("Number of clusters:", len(self))
        print("Richness:", self.richness.min(), "-", self.richness.max())
        print("Redshift:", self.z.min(), "-", self.z.max())
        print("Mean richness:", np.mean(self.richness))
        print("Mean redshift:", np.mean(self.z))
        if hasattr(self, "snr"):
            print("SNR^2 (stack):", self.snr**2)
        else:
            cov = self.cov
            prof = self.mean_profile
            snr = np.sqrt(np.dot(prof, np.dot(np.linalg.inv(cov), prof.T)))
            self.snr = snr
            print("SNR^2 (stack):", self.snr**2)
        try:
            print("SNR^2 (median):", np.mean(np.sum(self.profiles**2/self.errors**2, axis = 1)))
        except:
            pass
    @classmethod
    def compute_joint_cov(self, paths = None, off_diag = False, groups = None, corr = False,
                         bootstrap = True, Nbootstrap = 500, shrinkage = False):
        if paths is not None and np.iterable(paths) and groups is None:
            groups = []
            covs = []
            if paths is not None and np.iterable(paths):
                for i in range(len(paths)):
                    sub_group = grouped_clusters.load_from_path(paths[i])
                    covs.append(sub_group.cov)
                    groups.append(sub_group)
        covs = []
        for g in groups:
            if hasattr(g, "background") == False:
                cov = g.cov
                covs.append(cov)
            else:
                cov = g.cov + np.diag(g.background**2)
                covs.append(cov)
        
        full_covariance_matrix = block_diag(*covs)
        if off_diag == True:
            if bootstrap == False:
                N = 0
                off_diag_matrix = np.zeros(np.shape(full_covariance_matrix))
                for i in range(len(groups)):
                    for j in range(i):
                        N+=1
                        if i!=j:
                            g1, g2 = groups[i], groups[j]
                            prof1 = g1.random_profiles_cov if hasattr(g1, "random_profiles_cov") else g1.profiles
                            prof2 = g2.random_profiles_cov if hasattr(g2, "random_profiles_cov") else g2.profiles

                            if np.ndim(prof1) == 3 or np.ndim(prof2) == 3:
                                if np.ndim(prof1) > np.ndim(prof2):
                                    Nrand = np.shape(prof1)[0]
                                    N2 = len(prof2)
                                    idx = np.random.choice(Nrand, size=(Nrand, N2), replace=True)
                                    prof2 = prof2[idx]
                                if np.ndim(prof2) > np.ndim(prof1):
                                    Nrand = np.shape(prof2)[0]
                                    N1 = len(prof1)
                                    idx = np.random.choice(Nrand, size=(Nrand, N1), replace=True)
                                    prof1 = prof1[idx]
                                Nrand1, Nrand2 = len(prof1), len(prof2)
                                if Nrand1 != Nrand2:
                                    Nnew = min((Nrand1, Nrand2))
                                    prof1 = prof1[:Nnew]
                                    prof2 = prof2[:Nnew]
                                Nr = np.shape(prof1)[-1]
                                N1, N2 = np.shape(prof1)[1], np.shape(prof2)[1]
                                mean1 = np.mean(prof1, axis=0)
                                mean2 = np.mean(prof2, axis=0)
                                resid1 = prof1 - mean1 
                                resid2 = prof2 - mean2  
                                off = np.zeros((Nr,Nr))
                                for k in range(Nr):
                                    for l in range(Nr):
                                        off[k,l] = np.mean(np.sum(resid1[:,None,k] * resid2[:,:,None,l], axis = (1,2)), axis = 0)/(N1*N2)
                                if hasattr(g1, "background") and hasattr(g2, "background"):
                                    off = np.abs(g1.background_std * g2.background_std) * off
                                off_diag_matrix[int(i*Nr):int((i+1)*Nr), int(j*Nr):int((j+1)*Nr)] = off
                            else:
                                Nbase = 100
                                indx1 = np.random.choice(np.arange(len(prof1)), size = (Nbase, len(prof1)), replace = True)
                                indx2 = np.random.choice(np.arange(len(prof2)), size = (Nbase, len(prof2)), replace = True)
                                prof1 = prof1[indx1]
                                prof2 = prof2[indx2]
                                mean1 = np.mean(prof1, axis=0)
                                mean2 = np.mean(prof2, axis=0)

                                resid1 = prof1 - mean1 
                                resid2 = prof2 - mean2  
                                N1, N2 = np.shape(prof1)[1], np.shape(prof2)[1]
                                Nr = np.shape(prof1)[-1]
                                resid1 = prof1 - mean1 
                                resid2 = prof2 - mean2  
                                off = np.zeros((Nr,Nr))
                                for k in range(Nr):
                                    for l in range(Nr):
                                        off[k,l] = np.mean(np.sum(resid1[:,None,k] * resid2[:,:,None,l], axis = (1,2)), axis = 0)/(N1*N2)
                                off_diag_matrix[int(i*Nr):int((i+1)*Nr), int(j*Nr):int((j+1)*Nr)] = off

                full_covariance_matrix = full_covariance_matrix + off_diag_matrix + off_diag_matrix.T
            else:
                off_diag_matrix = np.zeros(np.shape(full_covariance_matrix))
                blocks = np.zeros((len(groups), len(groups), len(groups[0].R), len(groups[0].R)))
                if shrinkage == True:
                    alphas = np.zeros((len(groups), len(groups)))
                    for i in range(len(groups)):
                        for j in range(len(groups)):
                            if i == j:
                                mean_profiles = groups[i].bootstrap_profiles
                                lw = LedoitWolf()
                                lw.fit(mean_profiles)
                                alpha = lw.shrinkage_
                                cov = lw.covariance_
                                alphas[i,j] = alpha
                for i in range(len(groups)):
                    for j in range(i): 
                        g1, g2 = groups[i], groups[j]
                        if hasattr(g1, "bootstrap_profiles") == False or hasattr(g2, "bootstrap_profiles") == False:
                            idx1 = np.random.choice(np.arange(0, len(g1.richness)), size = (Nbootstrap, len(g1.richness)), replace = True)
                            idx2 = np.random.choice(np.arange(0, len(g2.richness)), size = (Nbootstrap, len(g2.richness)), replace = True)
                            bootstrap_profiles1 = g1.profiles[idx1]
                            bootstrap_profiles2 = g2.profiles[idx2]
                            weights1 = g1.weights[idx1] if hasattr(self,"weights") else np.ones(np.shape(bootstrap_profiles1))
                            weights2 = g2.weights[idx2] if hasattr(self,"weights") else np.ones(np.shape(bootstrap_profiles2))
                            mean_profiles1 = np.average(bootstrap_profiles1, axis = 1, weights = weights1)
                            mean_profiles2 = np.average(bootstrap_profiles2, axis = 1, weights = weights2)
                        else:
                            mean_profiles1 = g1.bootstrap_profiles
                            mean_profiles2 = g2.bootstrap_profiles
                        Nr = np.shape(mean_profiles1)[1]
                        off = np.zeros((Nr,Nr))
                        for k in range(Nr):
                            for l in range(Nr):
                                off[k,l] = np.mean((mean_profiles1[:,k] - np.mean(mean_profiles1[:,k], axis = 0)) * 
                                                 (mean_profiles2[:,l] - np.mean(mean_profiles2[:,l], axis = 0)))
                                if hasattr(g1, "background") and hasattr(g2, "background"):
                                    off[k,l] = off[k,l] + g1.background_std[0] * g2.background_std[0]
                        if shrinkage == True:
                            if i != j:
                                alpha_i = cross_covariance_shrinkage(mean_profiles1, mean_profiles2)
                                alphas[i,j] = alpha_i
                                alphas[j,i] = alpha_i
                        blocks[i,j] = off
                        blocks[j,i] = off
                        off_diag_matrix[int(i*Nr):int((i+1)*Nr), int(j*Nr):int((j+1)*Nr)] = off
                full_covariance_matrix = full_covariance_matrix + off_diag_matrix + off_diag_matrix.T
        if corr == False:
            return groups, full_covariance_matrix
        else:
            sigma = np.sqrt(np.diag(full_covariance_matrix))
            full_correlation_matrix = full_covariance_matrix/np.outer(sigma, sigma)
            return groups, full_covariance_matrix, full_correlation_matrix

    @classmethod
    def stacked_halo_model_func_by_paths(self, profile_model, units = "arcmin", 
                                        full = False, Rbins = 25, Mbins = 10, Zbins = 15,
                                        paths = None, verbose_pivots = False,
                                        rotate_cov = False, use_filters = False,
                                        filters = None, off_diag = False, verbose = True,
                                        recompute_cov = False, use_two_halo_term = False,
                                        fixed_RM_relationship = True, use_mis_centering = False,
                                        delta = 500, background = "critical", eval_mass = False,
                                        return_cov = False,  apply_filter_per_profile = False,
                                        rebinning = False, dtype = np.float32, return_1h2h = False,
                                        infere_mass = False, sort = True, subr_grid = False, 
                                        numba = False, weighted = False, cosmo = ccl.CosmologyVanillaLCDM(),
                                        **kwargs):

        print("Rbins:",Rbins)
        print("Mbins:",Mbins)
        print("Zbins:",Zbins)
        print("paths:",paths)
        print("verbose_pivots:",verbose_pivots)
        print("rotate_cov:",rotate_cov)
        print("use_filters:",use_filters)
        print("filters:",filters)
        print("off_diag:",off_diag)
        print("recompute_cov:",recompute_cov)
        print("use_two_halo_term:",use_two_halo_term)
        print("fixed_RM_relationship:",fixed_RM_relationship)
        print("use_mis_centering:",use_mis_centering)
        print("delta:",delta)
        print("background:",background)
        print("eval_mass:",eval_mass)
        print("return_cov:",return_cov)
        print("apply_filter_per_profile:",apply_filter_per_profile)
        print("rebinning:",rebinning)
        print("return_1h2h:",return_1h2h)
        print("verbose:",verbose)

        default_completeness_kwargs = (
            ("zbins", Zbins),
            ("Mbins", Mbins),
            ("interpolate", False),
            ("use_lambda_obs", True),
            ("use_redshift", True),
            ("background", background),
            ("delta", delta),
            ("verbose", verbose)
        )

        default_mass_inf_kwargs = (
            ("is_kappa", True),
            ("use_therm_EQ", False),
            ("use_function", False),
            ("func", None)
        )
        default_rebinning_kwargs = (
            ("nbins", 50),
            ("method", 'interp1d'),
            ("pixel_size", 0.5),
            ("interpolation_kwargs", dict(kind='cubic', bounds_error=False, fill_value=0))
        )

        default_subr_grid_kwargs = (
            ("rmin", 0.1),
            ("rmax", 15),
            ("n", 30)
        )
        default_two_halo_kwargs = (
            ("background", background),
            ("R", np.logspace(-1, 1.7, 20)),
            ("k", np.logspace(-15,15, 50)),
            ("M_arr", np.logspace(13,16, Mbins)),
            ("z_arr", np.linspace(1e-3,1, Zbins)),
            ("cosmo", cosmo),
            ("delta", delta)    
        )

        default_mis_centering_kwargs = (
            ("Roff", np.linspace(0, 2, 30)),
            ("distribution", lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2))),
            ("params", [0.245, 0.354]),
            ("theta", np.linspace(0, 2*np.pi, 30))
        )
        
        mis_centering_kwargs = set_default(kwargs.pop("mis_centering_kwargs", {}), default_mis_centering_kwargs)
        rebinning_kwargs = set_default(kwargs.pop("rebinning_kwargs", {}), default_rebinning_kwargs)
        completeness_kwargs = set_default(kwargs.pop("completeness_kwargs", {}), default_completeness_kwargs)
        two_halo_kwargs = set_default(kwargs.pop("two_halo_kwargs",{}), default_two_halo_kwargs)
        mass_inf_kwargs = set_default(kwargs.pop("mass_inf_kwargs",{}), default_mass_inf_kwargs)
        subr_grid_kwargs = set_default(kwargs.pop("subr_grid_kwargs",{}), default_subr_grid_kwargs)
        
        
        groups = []
        covs = []
        profiles = np.array([])
        about_clusters = []
        funcs = []
        bins = []
        if paths is not None and np.iterable(paths):
            for i in range(len(paths)):
                sub_group = grouped_clusters.load_from_path(paths[i])
                if rotate_cov:
                    print("rotating covariance matrix")
                    sub_group.rotate_cov_matrix()
                profiles = np.concatenate((profiles, sub_group.mean_profile))
                covs.append(sub_group.cov)
                groups.append(sub_group)
                redshift_bin = sub_group.redshift_bin
                richness_bin = sub_group.richness_bin
                bins.append([*richness_bin,*redshift_bin])
                about_clusters.append(
                    dict(
                        richness = (np.min(sub_group.richness),np.max(sub_group.richness)),
                        redshift = (np.min(sub_group.z),np.max(sub_group.z)),
                        N = len(sub_group),
                        path = paths[i]
                    )
                )
        bins = np.array(bins)
        if sort == True:
            
            sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
            bins = bins[sorted_idx]
            groups = [groups[i] for i in sorted_idx]
            covs = [covs[i] for i in sorted_idx]
            profiles = profiles[sorted_idx]
            about_clusters = [about_clusters[i] for i in sorted_idx]
        for i in range(len(groups)):
            sub_group = groups[i]

            funcs.append(sub_group.stacked_halo_model_func(profile_model, units, rbins = Rbins, zbins = Zbins, Mbins = Mbins,
                                use_filters = use_filters, filters = filters, use_two_halo_term = use_two_halo_term, 
                                fixed_RM_relationship = fixed_RM_relationship, two_halo_kwargs = two_halo_kwargs,
                                mis_centering = use_mis_centering, mis_centering_kwargs = mis_centering_kwargs,
                                background = background, delta = delta, eval_mass = eval_mass,
                                apply_filter_per_profile = apply_filter_per_profile, rebinning = rebinning,
                                rebinning_kwargs = rebinning_kwargs, return_1h2h = return_1h2h, verbose = verbose,
                                infere_mass = infere_mass, mass_inf_kwargs = mass_inf_kwargs, subr_grid = subr_grid,
                                subr_grid_kwargs = subr_grid_kwargs, numba = numba, weighted = weighted, cosmo = cosmo))
        
        _,full_covariance_matrix = grouped_clusters.compute_joint_cov(groups = groups, off_diag = off_diag)

        global func_gen
        def func_gen(R, params, model1h = None, model2h = None, RM_params = None, new_PllMs = None, smooth = None, eval_lambda = True, 
                    mis_centering_params = None, theta = np.linspace(0, 2*np.pi, 30), Roff = np.linspace(0, 2, 30),
                    two_halo_power = 1, cbin = None):
            if cbin is None:
                results = np.zeros(len(R) * len(funcs), dtype = dtype)
                if return_1h2h == True:
                    p1halo = np.zeros(len(R) * len(funcs), dtype = dtype)
                    p2halo = np.zeros(len(R) * len(funcs), dtype = dtype)
                if infere_mass == True:
                    M = np.zeros(len(funcs), dtype = dtype)
                for n,f in enumerate(funcs):
                    new_PllM = new_PllMs[n] if new_PllMs is not None else None
                    current_results = f(R,params, model1h = model1h, model2h = model2h, RM_params = RM_params, new_PllM = new_PllM, smooth = smooth, 
                                        eval_lambda = eval_lambda, mis_centering_params = mis_centering_params,
                                        theta = theta, Roff = Roff, two_halo_power = two_halo_power)
                    if return_1h2h == False:
                        if infere_mass == True:
                            M[n] = current_results[-1]
                            results[n*len(R):(n+1)*len(R)] = current_results[0]
                        else:
                            results[n*len(R):(n+1)*len(R)] = current_results
                    else:
                        if infere_mass == False:
                            results[n*len(R):(n+1)*len(R)], p1halo[n*len(R):(n+1)*len(R)], p2halo[n*len(R):(n+1)*len(R)] = current_results
                        else:
                            results[n*len(R):(n+1)*len(R)], p1halo[n*len(R):(n+1)*len(R)], p2halo[n*len(R):(n+1)*len(R)], M[n] = current_results
                if return_1h2h == True:
                    if infere_mass == True:
                        return results, p1halo, p2halo, M
                    return results, p1halo, p2halo
                else:
                    if infere_mass == True:
                        return results, M
                    else:
                        return results
            elif cbin is not None and len(cbin) == 4 and np.ndim(cbin) == 1:
                idx = np.where((bins == c).all(axis=1))[0][0]  
                return funcs[idx](R, params, model1h = model1h, model2h = model2h, RM_params = RM_params, new_PllM = new_PllMs, smooth = smooth, 
                                    eval_lambda = eval_lambda, mis_centering_params = mis_centering_params,
                                    theta = theta, Roff = Roff, two_halo_power = two_halo_power)
            elif cbin is not None and np.ndim(cbin) > 1:
                results = np.zeros(len(R) * len(cbin), dtype = dtype)
                if return_1h2h == True:
                    p1halo = np.zeros(len(R) * len(cbin), dtype = dtype)
                    p2halo = np.zeros(len(R) * len(cbin), dtype = dtype)
                if infere_mass == True:
                    M = np.zeros(len(cbin), dtype = dtype)
                for n,c in enumerate(cbin):
                    diff = np.sum(np.array(c) - bins, axis = 1)
                    idx = np.where(diff == 0)[0][0]
                    current_results = funcs[idx](R, params, model1h = model1h, model2h = model2h, RM_params = RM_params, new_PllM = new_PllMs, smooth = smooth, 
                                        eval_lambda = eval_lambda, mis_centering_params = mis_centering_params,
                                        theta = theta, Roff = Roff, two_halo_power = two_halo_power)
                    if return_1h2h == False:
                        if infere_mass == True:
                            M[n] = results[-1]
                            results[n*len(R):(n+1)*len(R)] = current_results[0]
                        else:
                            results[n*len(R):(n+1)*len(R)] = current_results
                    else:
                        if infere_mass == False:
                            results[n*len(R):(n+1)*len(R)], p1halo[n*len(R):(n+1)*len(R)], p2halo[n*len(R):(n+1)*len(R)] = current_results
                        else:
                            results[n*len(R):(n+1)*len(R)], p1halo[n*len(R):(n+1)*len(R)], p2halo[n*len(R):(n+1)*len(R)], M[n] = current_results
                if return_1h2h == True:
                    if infere_mass == True:
                        return results, p1halo, p2halo, M
                    return results, p1halo, p2halo
                else:
                    if infere_mass == True:
                        return results, M
                    else:
                        return results
        if full == True:
            return func_gen, full_covariance_matrix, about_clusters, groups, profiles, funcs
        else:
            return func_gen


    def stacked_halo_model_func_by_bins(self, profile_model, units = "arcmin", 
                                        full = False, rb = None, zb = None,
                                        Rbins = 25, Mbins = 10, Zbins = 11,
                                        paths = None, verbose_pivots = False,
                                        rotate_cov = False, use_filters = False,
                                        filters = None, off_diag = False,
                                        recompute_cov = False, use_two_halo_term = False,
                                        fixed_RM_relationship = True, use_mis_centering = False,
                                        delta = 500, background = "critical", eval_mass = False,
                                        return_cov = False,  apply_filter_per_profile = False,
                                        rebinning = False, return_1h2h = False, verbose = False,
                                        **kwargs):
        print("rb:",rb)
        print("zb:",zb)
        print("Rbins:",Rbins)
        print("Mbins:",Mbins)
        print("Zbins:",Zbins)
        print("paths:",paths)
        print("verbose_pivots:",verbose_pivots)
        print("rotate_cov:",rotate_cov)
        print("use_filters:",use_filters)
        print("filters:",filters)
        print("off_diag:",off_diag)
        print("recompute_cov:",recompute_cov)
        print("use_two_halo_term:",use_two_halo_term)
        print("fixed_RM_relationship:",fixed_RM_relationship)
        print("use_mis_centering:",use_mis_centering)
        print("delta:",delta)
        print("background:",background)
        print("eval_mass:",eval_mass)
        print("return_cov:",return_cov)
        print("apply_filter_per_profile:",apply_filter_per_profile)
        print("rebinning:",rebinning)
        print("return_1h2h:",return_1h2h)
        print("verbose:",verbose)
        
        default_completeness_kwargs = (
            ("zbins", Zbins),
            ("Mbins", Mbins),
            ("interpolate", False),
            ("use_lambda_obs", True),
            ("use_redshift", True),
            ("background", background),
            ("delta", delta),
            ("verbose", verbose)
        )
        default_rebinning_kwargs = (
            ("nbins", 50),
            ("method", 'interp1d'),
            ("pixel_size", 0.5),
            ("interpolation_kwargs", dict(kind='cubic', bounds_error=False, fill_value=0))
        )

        default_two_halo_kwargs = (
            ("background", background),
            ("R", np.logspace(-1, 1.7, 20)),
            ("k", np.logspace(-15,15, 50)),
            ("M_arr", np.logspace(13,16, 20)),
            ("z_arr", np.linspace(1e-3,1, 20)),
            ("cosmo", ccl.CosmologyVanillaLCDM()),
            ("delta", delta)    
        )

        default_mis_centering_kwargs = (
            ("Roff", np.linspace(0, 2, 30)),
            ("distribution", lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2))),
            ("params", [0.245, 0.354]),
            ("theta", np.linspace(0, 2*np.pi, 30))
        )

        mis_centering_kwargs = set_default(kwargs.pop("mis_centering_kwargs", {}), default_mis_centering_kwargs)
        rebinning_kwargs = set_default(kwargs.pop("rebinning_kwargs", {}), default_rebinning_kwargs)
        completeness_kwargs = set_default(kwargs.pop("completeness_kwargs", {}), default_completeness_kwargs)
        two_halo_kwargs = set_default(kwargs.pop("two_halo_kwargs",{}), default_two_halo_kwargs)
        self.completeness_and_halo_func(**completeness_kwargs)

        if zb is None:
            raise Exception("You must specify at least one redshift bin.")
        if rb is None:
            grouped_by_richness = self.split_optimal_richness(method = "stacking", width = width)
        elif rb is not None:
            grouped_by_richness = self.split_by_richness(richness_bins = rb)
        groups = []
        covs = []
        profiles = np.array([])
        about_clusters = []
        funcs = []
        k = 0
        if verbose_pivots == True:
            self.compute_pivots()
        if paths is None:
            for i,group in enumerate(grouped_by_richness):
                sub_group = group.split_by_redshift(zb[i])
                for j,s in enumerate(sub_group):
                    s.load_from_h5()
                    s.mean(from_path = True)
                    if recompute_cov == True:
                        s.compute_cov_matrix()
                    profiles = np.concatenate((profiles,s.mean_profile))
                    if rotate_cov:
                        s.rotate_cov_matrix()
                    covs.append(s.cov)
                    about_clusters.append(
                        dict(
                            richness = (np.min(s.richness),np.max(s.richness)),
                            redshift = (np.min(s.z),np.max(s.z)),
                            N = len(s),
                            coords = (i,j)
                        )
                    )
                    funcs.append(s.stacked_halo_model_func(profile_model, units, rbins = Rbins, zbins = Zbins, Mbins = Mbins,
                                 use_filters = use_filters, filters = filters, use_two_halo_term = use_two_halo_term, 
                                 fixed_RM_relationship = fixed_RM_relationship, two_halo_kwargs = two_halo_kwargs,
                                 mis_centering = use_mis_centering, mis_centering_kwargs = mis_centering_kwargs,
                                 background = background, delta = delta, eval_mass = eval_mass, 
                                 apply_filter_per_profile = apply_filter_per_profile, rebinning = rebinning,
                                 rebinning_kwargs = rebinning_kwargs, return_1h2h = return_1h2h, verbose = verbose))
                    groups.append(s)
        elif paths is not None and np.iterable(paths):
            for i in range(len(paths)):
                sub_group = grouped_clusters.load_from_path(paths[i])
                if rotate_cov:
                    print("rotating covariance matrix")
                    sub_group.rotate_cov_matrix()
                profiles = np.concatenate((profiles, sub_group.mean_profile))
                covs.append(sub_group.cov)
                groups.append(sub_group)
                funcs.append(sub_group.stacked_halo_model_func(profile_model, units, rbins = Rbins, zbins = Zbins, Mbins = Mbins,
                                    use_filters = use_filters, filters = filters, use_two_halo_term = use_two_halo_term, 
                                    fixed_RM_relationship = fixed_RM_relationship, two_halo_kwargs = two_halo_kwargs,
                                    mis_centering = use_mis_centering, mis_centering_kwargs = mis_centering_kwargs,
                                    background = background, delta = delta, eval_mass = eval_mass, verbose = verbose,
                                    apply_filter_per_profile = apply_filter_per_profile, rebinning = rebinning,
                                    rebinning_kwargs = rebinning_kwargs, return_1h2h = return_1h2h))
                about_clusters.append(
                    dict(
                        richness = (np.min(sub_group.richness),np.max(sub_group.richness)),
                        redshift = (np.min(sub_group.z),np.max(sub_group.z)),
                        N = len(sub_group),
                        path = paths[i]
                    )
                )
        full_covariance_matrix = block_diag(*covs)
        if off_diag == True:
            print("Computing off diagional elements")
            N = 0 
            for i in range(len(groups)):
                for j in range(i):
                    N+=1
                    if i!=j:
                        g1, g2 = groups[i], groups[j]
                        prof1 = g1.random_profiles_cov if hasattr(g1, "random_profiles_cov") else g1.profiles
                        prof2 = g2.random_profiles_cov if hasattr(g2, "random_profiles_cov") else g2.profiles
                        if np.ndim(prof1) == 3 and np.ndim(prof2) == 3:
                            Nrand1, Nrand2 = len(prof1), len(prof2)
                            if Nrand1 != Nrand2:
                                Nnew = min((Nrand1, Nrand2))
                                prof1 = prof1[:Nnew]
                                prof2 = prof2[:Nnew]
                            Nr = np.shape(prof1)[-1]
                            N1, N2 = np.shape(prof1)[1], np.shape(prof2)[1]
                            mean1 = np.mean(prof1, axis=0)
                            mean2 = np.mean(prof2, axis=0)
                            resid1 = prof1 - mean1 
                            resid2 = prof2 - mean2  
                            off = np.zeros((Nr,Nr))
                            for k in range(Nr):
                                for l in range(Nr):
                                    off[k,l] = np.mean(np.sum(resid1[:,None,k] * resid2[:,:,None,l], axis = (1,2)), axis = 0)/(N1*N2)
                            full_covariance_matrix[int(i*Nr):int((i+1)*Nr), int(j*Nr):int((j+1)*Nr)] = off
                        else:
                            mean1 = np.mean(prof1, axis = 0)
                            mean2 = np.mean(prof2, axis = 0)
                            resid1 = prof1 - mean1 
                            resid2 = prof2 - mean2  
                            Nr = np.shape(prof1)[-1]
                            off = np.zeros((Nr,Nr))
                            N1, N2 = np.shape(prof1)[0], np.shape(prof2)[0]
                            for k in range(Nr):
                                for l in range(Nr):
                                    off[k,l] = np.sum(resid1[:,k] * resid2[:, None,l], axis = (0,1))/(N1*N2)
                            full_covariance_matrix[int(i*Nr):int((i+1)*Nr), int(j*Nr):int((j+1)*Nr)] = off

            full_covariance_matrix = full_covariance_matrix + full_covariance_matrix.T - np.diag(np.diag(full_covariance_matrix))
        if return_cov == True:
            return full_covariance_matrix
        global func_gen
        def func_gen(R, params, RM_params = None, new_PllMs = None, smooth = None, eval_lambda = True, 
                    mis_centering_params = None, theta = np.linspace(0, 2*np.pi, 30), Roff = np.linspace(0, 2, 30),
                    two_halo_power = 1):
            results = np.zeros(len(R) * len(funcs), dtype = self.dtype)
            for n,f in enumerate(funcs):
                new_PllM = new_PllMs[n] if new_PllMs is not None else None
                current_results = f(R,params, RM_params = RM_params, new_PllM = new_PllM, smooth = smooth, 
                                    eval_lambda = eval_lambda, mis_centering_params = mis_centering_params,
                                    theta = theta, Roff = Roff, two_halo_power = two_halo_power)
                results[n*len(R):(n+1)*len(R)] = current_results
            return results
        if full == True:
            return func_gen, full_covariance_matrix, about_clusters, groups, profiles, funcs
        else:
            return func_gen

    def load_cov_matrix(self, path = None):
        path = path if path is not None else self.output_path
        h5file = f"{path}/data.h5"
        try:
            with h5py.File(h5file,"r") as f:
                self.cov = f["cov"][:]
        except KeyError:
            self.cov = np.sqrt(np.diag(self.error_in_mean))
    def rotate_cov_matrix(self):
        self.cov= np.rot90(self.cov)
    def load_from_h5(self, search_closest = False):
        if hasattr(self, "output_path"):
            if search_closest == False:
                file = f"{self.output_path}/data.h5"
            else:
                try:
                    target_path = self.output_path.split('/')[-1]
                    grouped_clusters_list = [
                        path
                        for path in os.listdir(data_path + grouped_clusters_path)
                        if os.path.isdir(data_path + grouped_clusters_path + path)
                        and
                        path.split('_')[0] == 'GROUPED'
                        ]
                    closest = closest_path(target_path, grouped_clusters_list)
                    file = data_path + grouped_clusters_path + closest + "/data.h5"
                    self.output_path = data_path + grouped_clusters_path + closest
                except:
                    file = f"{self.output_path}/data.h5"
            with h5py.File(file, "r") as f:
                available_keys = list(f.keys())
                for k in available_keys:
                    try:
                        if k == "wcs":
                            header_str = f[k][()].decode("utf-8")
                            header = Header.fromstring(header_str, sep = "\n")
                            wcs = WCS(header)
                            setattr(self, "wcs", wcs)
                            continue

                        setattr(self, k, f[k][:])
                    except:
                        continue

def init_random_worker(ymap, mask):
    global shared_ymap, shared_mask
    shared_ymap = ymap
    shared_mask = mask


from time import time
import gc
global random_worker
def random_worker(ymap = None, mask = None, R_profiles = None, width = None, wcs = None, reproject_maps = None, 
                  N_random = None, Ncl = None, N_clusters = None, rmin = None, rmax = None, dmin = None, dmax = None, 
                  random_coord_size = 500, N_total = None, min_sep = None, worker_id = None, counter = None, 
                  mask_format = "healpy", compute_individual_matrices = True, save_coords = True,
                  weights = None, return_patches = False, dtype = np.float32,
                  ):    
    sys.stdout.write(f"\rStarting worker {worker_id} with {N_random} realizations each with {N_clusters} simulated clusters.\n")
    sys.stdout.flush()
    mean_profiles = np.zeros((N_random, len(R_profiles) - 1), dtype = dtype)
    random_profiles = np.zeros((N_random, N_clusters, len(R_profiles) - 1), dtype = dtype)
    cov_matrices = np.zeros((N_random, len(R_profiles) - 1, len(R_profiles) - 1), dtype = dtype)
    stored_weights = np.zeros((N_random, N_clusters, len(R_profiles) - 1), dtype = dtype)
    if ymap is None and mask is None and "shared_ymap" in globals() and "shared_mask" in globals():
        #print("Loading maps from globals()!")
        ymap = shared_ymap
        mask = shared_mask
    if len(ymap) == 2:
        shared_ymap_name = ymap[0]
        shape_ymap = ymap[1]
        shm_ymap = shared_memory.SharedMemory(name=shared_ymap_name)
        ymap = np.ndarray(shape_ymap, dtype=dtype, buffer=shm_ymap.buf)

    if len(mask) == 2:
        shared_mask_name = mask[0]
        shape_mask = mask[1]
        shm_mask = shared_memory.SharedMemory(name=shared_mask_name)
        clusters_mask = np.ndarray(shape_mask, dtype=dtype, buffer=shm_mask.buf)

    if weights is not None:
        hist, bins = np.histogram(np.nan_to_num(weights, np.nanmin(weights)), bins = 200, density = True)
        probs = hist * np.diff(bins)
        probs /= probs.sum()
        inds = np.random.choice(np.arange(len(probs)), size = (N_random, N_clusters), p = probs)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        new_weights = bin_centers[inds]
        
    rng = np.random.default_rng()

    dec2, ra2 = np.zeros((2, N_random, N_clusters)).astype(dtype)
    dec2, ra2 = rng.uniform(dmin, dmax, (N_random, N_clusters)), rng.uniform(rmin, rmax, (N_random, N_clusters))
    for i in range(N_random):
        accepted_coords = 0
        while accepted_coords < N_clusters:
            new_dec, new_ra = rng.uniform(dmin, dmax, random_coord_size), rng.uniform(rmin, rmax, random_coord_size)
            if mask_format == "pixell":
                #pixell uses a CAR projection
                ypix, xpix = (enmap.sky2pix(mask.shape, mask.wcs, np.deg2rad(np.stack((new_dec, new_ra))))).astype(int)
                mask_values = mask[ypix, xpix]
            elif mask_format == "healpy":
                #healpy uses a teselation of the sky 
                theta = np.deg2rad(90.0 - new_dec)
                phi = np.deg2rad(new_ra)
                pixels = hp.ang2pix(hp.get_nside(mask), theta, phi)
                mask_values =  mask[pixels]
                mask_values = np.nan_to_num(mask_values)
            if np.all(mask_values == 0):
                continue
            p = mask_values / np.sum(mask_values)
            p = np.nan_to_num(p)
            dec_in_mask, ra_in_mask = rng.choice(np.stack((new_dec, new_ra)).T, size = random_coord_size, p = p).T
            coords_pairs = np.unique(np.stack((dec_in_mask, ra_in_mask)).T, axis=0)
            dec_in_mask, ra_in_mask = coords_pairs[:, 0], coords_pairs[:, 1]
            if min_sep is not None:
                min_sep = 5 if min_sep < 5 else min_sep
                coords_batch = SkyCoord(
                    ra=ra_in_mask * u.deg,
                    dec=dec_in_mask * u.deg,
                    frame="icrs",
                )
                idx1, idx2, sep2d, _ = coords_batch.search_around_sky(
                    coords_batch,
                    seplimit=min_sep * u.arcmin,
                )
                mask_ij = idx1 < idx2
                bad_i = idx1[mask_ij]
                bad_j = idx2[mask_ij]
                to_remove = set(bad_j.tolist())

                if to_remove:
                    keep_indices = np.setdiff1d(
                        np.arange(len(dec_in_mask)), np.array(list(to_remove))
                    )
                    dec_in_mask = dec_in_mask[keep_indices]
                    ra_in_mask  = ra_in_mask[keep_indices]

            end_idx = min(accepted_coords + len(dec_in_mask), N_clusters)

            available_space = N_clusters - accepted_coords
            num_to_store = min(len(dec_in_mask), available_space)

            if num_to_store > 0:
                dec2[i,accepted_coords:end_idx] = dec_in_mask[:num_to_store]
                ra2[i,accepted_coords:end_idx] = ra_in_mask[:num_to_store]
                accepted_coords = end_idx
            else:
                continue
    ra2, dec2 = ra2.flatten(), dec2.flatten()
    coords = np.deg2rad(np.stack((dec2, ra2))).T.astype(dtype)
    
    t1 = time()
    new_maps = reproject.thumbnails(ymap, coords = coords, r = np.deg2rad(width)/2., oversample = 2, order = 1) # if isinstance(ymap, np.ndarray) else reproject.thumbnails(enmap.ndmap(ymap, wcs = wcs), coords = np.deg2rad((dec2, ra2)), r = np.deg2rad(width)/2.)
    t2 = time()
    Rbins, new_profiles, sigma, _,  = radial_binning2(new_maps, R_profiles, width = width, full = True)
    random_profiles = np.reshape(new_profiles, (N_random, N_clusters, -1))
    random_sigma = np.reshape(sigma, (N_random, N_clusters, -1))
    random_maps = np.reshape(new_maps, (N_random, N_clusters, *np.shape(new_maps[0]))).astype(np.float64)
    stacks = np.average(random_maps, axis = 1)
    mean_profiles = radial_binning2(stacks, R_profiles, width = width)
    if return_patches == True:
        return random_maps, random_profiles
    #mean_profiles = np.reshape(mean_profiles, (N_random, len(R_profiles) - 1))
    cov_matrices = np.zeros((N_random, len(R_profiles) - 1, len(R_profiles) - 1), dtype = np.float64)
    Nr = len(R_profiles) - 1
    if compute_individual_matrices == False:
        for n in range(N_random):
            dev = random_profiles[n] - np.mean(random_profiles[n], axis = 0)[None, :]
            if weights is None:
                cov = np.cov(random_profiles[n], rowvar = False, ddof = 1)
                cov_matrices[n] = cov
            else:
                w = new_weights[n]
                s = sigma[n]
                P = random_profiles[n]
                W = w[:, None] / s**2
                stored_weights[n,...] = W
                Wsum = np.sum(W, axis = 0)
                mu = np.sum(P*W , axis = 0) / Wsum
                dev = P - mu[None,:]
                for i in range(Nr):
                    Wmi = W[:,i]
                    Di = dev[:,i]
                    for j in range(Nr):
                        Wnj = W[:,j]
                        Dj = dev[:,j]
                        num = np.sum(Wnj * Wmi * Di * Dj)
                        Wij = Wnj * Wmi
                        V1  = np.sum(Wij)
                        V2  = np.sum(Wij * Wij)
                        denom = V1 - (V2 / V1)
                        cov_matrices[n, i, j ] = num/denom
                        if np.isnan(cov_matrices[n,i,j]) == True or np.isfinite(cov_matrices[n,i,j]) == False:
                            cov_matrices[n, i, j ] = 0
    else:
        for n in range(N_random):
            individual_cov_matrices = compute_covariance_per_map(np.asarray(random_maps[n]), R_profiles, width = width)
            cov_matrices[n,:,:] = np.median(individual_cov_matrices, axis = 0) if weights is None else np.average(individual_cov_matrices, axis = 0, weights = new_weights[n])
            del individual_cov_matrices
    del ra2, dec2, new_maps, new_profiles, random_maps, stacks
    gc.collect()
    if worker_id is not None:
        sys.stdout.write(f"\rWorker {worker_id} has already finished in {t2 - t1} second!\n")
        sys.stdout.flush()
    output = [cov_matrices, mean_profiles, random_profiles]
    output.append(stored_weights) if weights is not None else None
    coords = np.reshape(coords, (N_random, N_clusters, 2))
    output.append(np.rad2deg(coords)) if save_coords == True else None
    return output

global bootstrap_worker
def bootstrap_worker(R_profiles, maps, N_bootstrap, N_total, counter, width, weights):
    mean_profiles = np.zeros((N_bootstrap, len(R_profiles) - 1))
    for i in range(N_bootstrap):
        indx = np.random.randint(0, len(maps), size = len(maps))
        maps_i = maps[indx]
        weights_i = weights[indx]
        sstack = np.average(maps_i, axis = 0, weights = weights_i)
        mean_profiles[i] = radial_binning2(np.average(maps_i, axis = 0, weights = weights_i), R_profiles, width = width)
        counter.value += 1
        sys.stdout.write(f"\rBootstrap progress: ({counter.value} / {N_total})")
        sys.stdout.flush()
        del maps_i, weights_i
    return mean_profiles


from matplotlib.patches import Circle

def plot_profiles(clusters, sort = True, figsize = (8,8)):
    if sort == True:
        bins = np.array([[*c.richness_bin, *c.redshift_bin] for c in clusters])
        sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
        clusters = [clusters[i] for i in sorted_idx]
    nrows = 2
    ncols = 4
    fig = plt.figure(figsize=(10 + 5*nrows, 5 * nrows))
    gs = GridSpec(2, 4, wspace = 0, hspace = 0)

    axes = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            sharex = axes[0][j] if i > 0 else None
            sharey = row[0] if j > 0 else None
            ax = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
            row.append(ax)
        axes.append(row)
    axes = np.array(axes)
    FWHM = 1.6
    sigma = FWHM / 2.355
    for i in range(len(clusters)):
        zmin, zmax = clusters[i].redshift_bin
        lambda_min, lambda_max = clusters[i].richness_bin
        row_indx = 0 if zmin < 0.3 else 1
        profile = clusters[i].mean_profile - clusters[i].background if hasattr(clusters[i], "background") else clusters[i].mean_profile
        cov = clusters[i].cov + clusters[i].background_std[0]**2 if hasattr(clusters[i], "background_std") else clusters[i].cov

        errs = np.sqrt(clusters[i].error_in_mean**2 + clusters[i].background_std**2) if hasattr(clusters[i], "background_std") else clusters[i].error_in_mean
        snr = np.dot(profile, np.dot(np.linalg.inv(cov), profile.T))
        R = clusters[i].R
        r = np.arange(np.min(R), np.max(R), 0.1)
        beam = np.max(profile)*np.exp(-r**2/(2*sigma**2))
        axes[row_indx, i//2].plot(r, beam, ls = "--", color = "darkblue", lw = 3, alpha = 0.4, label = "ACT beam")
        axes[row_indx, i//2].errorbar(R, profile, yerr = errs, color = "black", capsize = 3, fmt = "-o", markersize = 10, linewidth = 3, label = "stacked profiles")
        # if hasattr(clusters[i], "background"):
        #     axes[row_indx, i//2].axhline(np.abs(clusters[i].background[0]), color = "darkred", ls = "--", lw = 3, label = "background")
        #     axes[row_indx, i//2].fill_between(R, np.abs(clusters[i].background) - clusters[i].background_std, np.abs(clusters[i].background) + clusters[i].background_std, color = "darkred", alpha = 0.2)

        if row_indx == 1:
            axes[row_indx, i//2].set(xlabel = "R (arcmin)")
        if i//2 == 0 :
            axes[row_indx, i//2].set(ylabel = "compton-y profile")
        axes[row_indx, i//2].set(yscale = "log")
        label = r"$\mathbf{\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]}$" % (lambda_min, lambda_max, zmin, zmax)
        axes[row_indx, i//2].text(0.95, 0.95, label, transform=axes[row_indx, i//2].transAxes, fontsize=14, ha='right', va='top')
        axes[row_indx, i//2].set_ylim(1e-7, 3e-4)
        if hasattr(clusters[i], "tng_profile"):
            tng_profile = clusters[i].tng_profile
            tng_errors = clusters[i].tng_errors
            axes[row_indx, i//2].errorbar(R+0.5, tng_profile, yerr = tng_errors, color = "purple", capsize = 3, fmt = "-o", markersize = 10, linewidth = 3, label = "TNG300-3")
        if row_indx == 1:
            yticks = axes[row_indx, i//2].get_yticks()
            yticks = yticks[np.where((yticks >= 1e-7) & (yticks <= 1e-3))]
            yticks = yticks[:-1]
            axes[row_indx, i//2].set_yticks(yticks)
            axes[row_indx, i//2].set_yticklabels([r"$10^{%.i}$" % int(np.log10(y)) for y in yticks])
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i][j]
                if j == 0:
                    continue
                else:
                    ax.tick_params(labelleft=False)
                if i == nrows - 1:
                    continue
                else:
                    ax.tick_params(labelbottom=False)
    axes[0,0].legend(frameon = False, fontsize = 14, 
                    loc = "center right", bbox_to_anchor = (0.9, 0.6),
                    bbox_transform = axes[0,0].transAxes)
    return fig
def plot_correlation_matrices(clusters, sort = True, figsize = (16, 4)):
    if sort == True:
        bins = np.array([[*c.richness_bin, *c.redshift_bin] for c in clusters])
        sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
        clusters = [clusters[i] for i in sorted_idx]
    nrows = 2
    ncols = 4
    fig = plt.figure(figsize=(10 + 5*nrows, 5 * nrows))
    gs = gs = GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 0.1, 0.15],
              wspace=0.0, hspace=0)
    axes = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            sharex = axes[0][j] if i > 0 else None
            sharey = row[0] if j > 0 else None
            ax = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
            row.append(ax)
        axes.append(row)
    axes = np.array(axes)
    corrs = []
    for i in range(len(clusters)):
        cov = clusters[i].cov
        sigma = np.sqrt(np.diag(cov))
        corr = np.array([[cov[i,j]/(sigma[i]*sigma[j]) for i in range(len(cov))] for j in range(len(cov))])
        corrs.append(corr)
    vmin = np.min(corrs)
    vmax = np.max(corrs)
    vmin = -1
    vmax = 1
    for i in range(len(clusters)):
        cov = clusters[i].cov
        R = clusters[i].R
        sigma = np.sqrt(np.diag(cov))
        corr = np.array([[cov[i,j]/(sigma[i]*sigma[j]) for i in range(len(cov))] for j in range(len(cov))])
        zmin, zmax = clusters[i].redshift_bin
        rmin, rmax = clusters[i].richness_bin
        row_indx = 0 if zmin < 0.3 else 1
        im = axes[row_indx, i//2].imshow(corr, cmap = "coolwarm", vmin = vmin, vmax = vmax, origin = "lower",
                                        extent = (R.min(), R.max(), R.min(), R.max()))
        if i//2 == 0:
            axes[row_indx, i//2].set(ylabel = "R (arcmin)")
        if row_indx == 1:
            axes[row_indx, i//2].set(xlabel = "R (arcmin)")
        if row_indx == 0:
            axes[row_indx, i//2].set_title("$\mathbf{\lambda \in [%i, %i]}$" % (rmin, rmax),
                                fontsize = 24, fontweight = "bold")
        if i//2 == 0:
            axes[row_indx, i//2].text(-0.3,0.5,"$\mathbf{z\in[%.2f, %.2f]}$" % (zmin, zmax), ha = "center", va = "center", 
                                    transform = axes[row_indx, i//2].transAxes, fontsize = 24, fontweight = "bold",
                                    rotation = 90)
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if j == 0:
                continue
            else:
                ax.tick_params(labelleft=False)
            if i == nrows - 1:
                continue
            else:
                ax.tick_params(labelbottom=False)
    axes[0,0].tick_params(labelbottom=False)
    cax = fig.add_subplot(gs[:, -1]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Correlation value", fontweight = "bold")
    return fig

from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm
def plot_signal(clusters, sort = True, figsize = (20, 10), plot_corr = False, patch_size = 0.6, cmap = "coolwarm", 
                share_colorbar = False, vmin = None, vmax = None,
                lw = 3, color = "black", xlim = None, ylim = None, 
                log = False, symlog = False):
    
    if sort == True:
        bins = np.array([[*c.richness_bin, *c.redshift_bin] for c in clusters])
        sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
        clusters = [clusters[i] for i in sorted_idx]
    
    richness_bins = np.unique([c.richness_bin for c in clusters], axis = 0)
    redshift_bins = np.unique([c.redshift_bin for c in clusters], axis = 0)
    fig = plt.figure(figsize=figsize)
    ncols = len(richness_bins)
    nrows = len(redshift_bins)
    gs = gs = GridSpec(nrows, ncols, wspace=0.1, hspace=0.425)
    axes = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            sharex = axes[0][j] if i > 0 else None
            sharey = row[0] if j > 0 else None
            ax = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
            row.append(ax)
        axes.append(row)
    axes = np.array(axes)
    if share_colorbar == True and vmin is None and vmax is None:
        vmin = np.min([c.stacked_map for c in clusters])
        vmax = np.max([c.stacked_map for c in clusters])
    elif vmax is None and vmin is None:
        vmin = None
        vmax = None
    for i in range(len(clusters)):
        c = clusters[i]
        rmin, rmax = c.richness_bin
        zmin, zmax = c.redshift_bin
        idx = 0 if zmin < 0.3 else 1
        ax = axes[idx, i//2]
        extent = np.array([-patch_size, patch_size, -patch_size, patch_size])/2 * 60
        if log == False and symlog == False:
            im = ax.imshow(c.stacked_map, extent = extent, cmap = cmap, 
                            vmin = vmin, vmax = vmax, origin = "lower", 
                            aspect = "auto")
        elif log == True and symlog == False:
            im = ax.imshow(c.stacked_map, extent = extent, cmap = cmap, 
                            vmin = vmin, vmax = vmax, origin = "lower", 
                            aspect = "auto", norm = LogNorm())
        else:
            im = ax.imshow(c.stacked_map, extent = extent, cmap = cmap, 
                            vmin = vmin, vmax = vmax, origin = "lower", 
                            aspect = "auto", 
                            norm = SymLogNorm(linscale= 1, 
                            linthresh = 1e-7))
        cax = fig.add_axes([ax.get_position().x0,
                ax.get_position().y1 + 0.005,
                ax.get_position().width,
                0.025])
        cbar = plt.colorbar(im, cax = cax, orientation = "horizontal")
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=12, top=True, bottom=False)
        cbar.ax.text(
            0.5, 0.5,
            "Compton-y",
            ha="center",
            va="center",
            fontsize = 12,
            fontweight = "bold",
            transform=cbar.ax.transAxes
        )   
        if symlog == True:
                        
            vmin_image = im.norm.vmin
            vmax_image = im.norm.vmax
            logvmin = np.round(np.log10(np.abs(vmin_image)))
            logvmax = np.round(np.log10(np.abs(vmax_image)))
            positive_ticks = np.logspace(logvmax, 0, 3, endpoint = False)
            negantive_ticks = -np.logspace(logvmin, 0, 3, endpoint = False)
            ticks = np.concatenate((negantive_ticks, positive_ticks))
            
            tick_labels = [r"$-10^{%.i}$" % np.log10(np.abs(ticks[j])) if ticks[j] < 0 
                         else r"$10^{%.i}$" % np.log10(ticks[j]) 
                         for j in range(len(ticks))]
            ticks = np.append(ticks, 0)
            tick_labels.append("0")
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        if i//2 == 0:
            axes[idx, i//2].set(ylabel = r"$\Delta$ RA (arcmin)")
        if idx == 1:
            axes[idx, i//2].set(xlabel = r"$\Delta$ DEC (arcmin)")
        ax.axvline(0, color = color, linewidth = lw, alpha = 0.7, ls = "--")
        ax.axhline(0, color = color, linewidth = lw, alpha = 0.7, ls = "--")
        circle = Circle((0,0), radius=1.6, color=color, fill=False, lw=lw, alpha = 0.7, ls = "--")
        ax.add_patch(circle)   
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if j == 0:
                continue
            else:
                ax.tick_params(labelleft=False)
            if i == nrows - 1:
                continue
            else:
                ax.tick_params(labelbottom=False)
    for i, richness in enumerate(richness_bins):
        axes[0,i].text(0.0, 1.23, r"$\mathbf{\lambda \in [%.i, %.i]}$" % tuple(richness),
                        va='center', ha='left',
                        fontsize=22, transform=axes[0,i].transAxes)

    for j, z in enumerate(redshift_bins):
        pos = axes[j,0].get_position()
        fig.text(-0.45, 0.5, r"$\mathbf{z \in [%.2f, %.2f]}$" % tuple(z), rotation = 90,
                ha='center', va='center', fontsize=22, transform=axes[j, 0].transAxes)
    split_col = 1  
    ax_left  = axes[0, split_col]
    ax_right = axes[0, split_col + 1]

    x_mid = 0.5 * (
        ax_left.get_position().x1 +
        ax_right.get_position().x0
    )

    y_bottom = axes[-1, 0].get_position().y0
    y_top    = axes[0, 0].get_position().y1
    divider = Line2D(
        [x_mid, x_mid],
        [y_bottom, y_top],
        transform=fig.transFigure,
        color="black",
        linewidth=2.5,
        alpha=0.8
    )

    fig.add_artist(divider)
    return fig


def plot_cib_comparison(R_profiles, width = 0.6):
    from plottery.plotutils import update_rcParams
    update_rcParams()
    
    paths = [
        "/data2/javierurrutia/szeffect/data/ycompton-no-CIB-deproj/",
        "/data2/javierurrutia/szeffect/data/ycompton-deproj-cib_1.0_10.7/",
        "/data2/javierurrutia/szeffect/data/ycompton-deproj-cib_1.2_10.7/",
        "/data2/javierurrutia/szeffect/data/ycompton-deproj-cib_1.7_10.7/",
        "/data2/javierurrutia/szeffect/data/ycompton-deproj-cib_2.0_10.7/"
    ]
    labels = [
        "no CIB deprojection",
        r"$\beta = 1.0$", 
        r"$\beta = 1.2$", 
        r"$\beta = 1.7$", 
        r"$\beta = 2.0$"
    ]

    data = [grouped_clusters.load_from_path(path + "entire_sample") for path in paths]
    richness_bins = [[20, 40], [40, 60], [60, 100], [100, 350]]
    redshift_bins = [[0.1, 0.4], [0.4, 1]]
    fig, axes = plt.subplots(2,4, figsize = (16, 8), sharex = "col", sharey = "row")
    profiles = np.zeros((len(redshift_bins), len(richness_bins), len(data), len(R_profiles)-1))
    sigma = np.zeros((len(redshift_bins), len(richness_bins), len(data), len(R_profiles)-1))
    colors = ["black", "darkgreen", "darkblue", "darkred", "darkorange"]
    for i in range(len(richness_bins)):
        for j in range(len(data)):
            dj = data[j]
            ax1, ax2 = axes[0, i], axes[1, i]
            sub_group = dj.sub_group(richness_interval = richness_bins[i])
            dj1 = sub_group.sub_group(redshift_interval = redshift_bins[0])
            dj2 = sub_group.sub_group(redshift_interval = redshift_bins[1])
            szmap1 = np.average(dj1.szmap, axis = 0)
            szmap2 = np.average(dj2.szmap, axis = 0)
            R_bins, prof1, err1, arrs = radial_binning2(szmap1, R_profiles, width = width, full = True)
            R_bins, prof2, err2, arrs = radial_binning2(szmap2, R_profiles, width = width, full = True)
            if labels[j] == "no CIB deprojection":
                ax1.errorbar(R_bins + j*0.25, prof1, yerr = err1, label = labels[j], capsize = 2, 
                alpha = 0.8, ls = "solid", color = colors[j], lw = 3)
                ax2.errorbar(R_bins + j*0.25, prof2, yerr = err2, label = labels[j], capsize = 2, 
                alpha = 0.8, ls = "solid", color = colors[j], lw = 3)
            else:
                ax1.errorbar(R_bins + j*0.25, prof1, yerr = err1, label = labels[j], capsize = 2, 
                alpha = 0.8, ls = "--", color = colors[j])
                ax2.errorbar(R_bins + j*0.25, prof2, yerr = err2, label = labels[j], capsize = 2, 
                alpha = 0.8, ls = "--", color = colors[j])
            ax1.set_title(r"$\lambda \in [%.i, %.i]$" % tuple(richness_bins[i]))
            ax2.set(xlabel = "R (arcmin)")
            if i == 0:
                ax1.set(ylabel = "y profile")
                ax2.set(ylabel = "y profile")  
            ax1.set_yscale("log")
            ax2.set_yscale("log")

            profiles[0, i, j, : ] = prof1
            profiles[1, i, j, : ] = prof2
            sigma[0, i, j, : ] = err1
            sigma[1, i, j, : ] = err2
    axes[0,0].legend(loc = "best", fontsize = 8, frameon = False)
    for j, z in enumerate(redshift_bins):
        pos = axes[j,0].get_position()
        fig.text(0.04, (pos.y0 + pos.y1) / 2, r"$z \in [%.2f, %.2f]$" % tuple(z), ha='center', 
        va='center', fontsize=14, rotation = 90)
    
    ref = profiles[:,:,0,:]
    sigma_ref = sigma[:,:,0,:]
    fig2, axes2 = plt.subplots(2,4, figsize = (16, 8), sharex = "col", sharey = "row")
    for i in range(len(richness_bins)):
        for j in range(1, len(data)):
            ax1, ax2 = axes2[0, i], axes2[1, i]
            delta_sigma1 = (profiles[0,i,j] - ref[0,i]) / np.sqrt(sigma_ref[0, i]**2 + sigma[0,i,j]**2)
            delta_sigma2 = (profiles[1,i,j] - ref[1,i]) / np.sqrt(sigma_ref[1, i]**2 + sigma[1,i,j]**2)
            ax1.plot(R_bins, delta_sigma1, label = labels[j], alpha = 0.8, lw = 3, color = colors[j])
            ax2.plot(R_bins, delta_sigma2, label = labels[j], alpha = 0.8, lw = 3, color = colors[j])
            if i == 0:
                ax1.set_ylabel(r"difference in $\sigma$")
                ax2.set_ylabel(r"difference in $\sigma$")
            ax1.set_title(r"$\lambda \in [%.i, %.i]$" % tuple(richness_bins[i]))
            ax2.set(xlabel = "R (arcmin)")
    axes2[-1,-1].legend(loc = "best", fontsize = 8, frameon = False)
    for j, z in enumerate(redshift_bins):
        pos = axes[j,0].get_position()
        fig2.text(0.04, (pos.y0 + pos.y1) / 2, r"$z \in [%.2f, %.2f]$" % tuple(z), ha='center', 
        va='center', fontsize=14, rotation = 90)
              
    return fig, fig2



def test_profiles(c, model, params):

    M200 = c.M
    lambda_true = c.lambda_true

    R = c.R #np.linspace(c.R.min(), c.R.max(), 25)

    cosmo = ccl.CosmologyVanillaLCDM()
    mdef = "200m"
    mfunc = ccl.halos.MassFuncTinker10(mass_def=mdef)
    helpers = importlib.import_module("helpers")
    func = getattr(helpers, "P_lob_ltr")

    lambda_min, lambda_max = c.richness_bin
    zmin, zmax = c.redshift_bin

    Mmin, Mmax = 10**(14.45)*(lambda_min/40)**(1.29),10**(14.45)*(lambda_max/40)**(1.29)
    Mobs = np.logspace(np.log10(Mmin), np.log10(Mmax), 30)

    zmean = (zmax + zmin)/2
    lambda_obs = c.lambda_obs
    zobs = c.z_arr
    
    dndlog10M = np.array([[mfunc(cosmo, mi, 1/(1 + zi)) for mi in M200] for zi in zobs])
    dndM = (dndlog10M/(np.log(10)*M200)).T
    s = 0.037
    q = 1.008
    dndM = dndM #* (s*np.log(M200/10**(13.8)/0.67) + q)[:,None]

    dV = planck18.differential_comoving_volume(zobs).value
    ltrue_grid, lobs_grid, zobs_grid = np.meshgrid(lambda_true, lambda_obs, zobs, indexing = "ij")
    P_lobs_ltrue_z = func(lobs_grid, ltrue_grid, zobs_grid)

    P_ltrue_lobs = np.trapz(P_lobs_ltrue_z, x = zobs, axis = 2)

    ltrue_M_grid, M_grid, zobs_M_grid = np.meshgrid(lambda_true, M200, zobs, indexing = "ij")

    lambda_model = 30 * (M_grid / (3e14/0.67))**0.75 * ((1 + zobs_M_grid)/(1 + 0.35))**(-0.0)
    sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
    P_ltrue_M_z = (1 / (ltrue_M_grid * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log(ltrue_M_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2)))

    P_joint = P_lobs_ltrue_z[:, :, None, :] * P_ltrue_M_z[:, None, :, :]

    weights = P_joint * dndM[None,None,:,:] * dV[None,None,None,:]

    norm = np.trapz(np.trapz(np.trapz(np.trapz(weights, x = lambda_true, axis = 0), x = lambda_obs, axis = 0), x = M200, axis = 0), x = zobs, axis = 0)

    R_Mpc = R[:,None]*(np.pi/180) / 60 * (planck18.angular_diameter_distance(zobs).value * (1 + zobs)[None,:])
    profiles = np.array([[model(R_Mpc[:,i], 10, Mi, zi, params) for Mi in M200] for i,zi in enumerate(zobs)]).T
    observed_profiles = np.array([[model(R_Mpc[:,i], 10, Mi, zi, params) for Mi in Mobs] for i,zi in enumerate(zobs)]).T

    theta = np.linspace(0, 2*np.pi, 20)
    Roff = np.linspace(0, 1, 25)

    xmis = np.sqrt(Roff[None, None, :, None]**2 + R_Mpc[:,:,None,None]**2 + 2 * Roff[None,None,:,None] * R_Mpc[:,:,None,None] * np.cos(theta)[None,None,None,:])


    M2, z2 = np.meshgrid(M200, zobs)

    #fmis, sigma_mis = mis_centering_model(M200, zobs, [0.246, 0.385])
    fmis = 0.6
    sigma_mis = 1
    rho_Roff = lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2))

    weights_mc = rho_Roff(Roff, sigma_mis)

    funcs = [
    [UnivariateSpline(R_Mpc[:,i], pj, k=1, s=0) for j,pj in enumerate(pi)] for i,pi in enumerate(profiles.T)
    ]
    off = np.array(
        [[np.trapz(fj(xmis[:,i,:,:]), theta, axis=-1) for j,fj in enumerate(fi)] for i,fi in enumerate(funcs)]
    )
    woff = np.array(
        [[np.trapz(weights_mc[None,:] * off[i,j,:,:], Roff, axis = 1)/np.trapz(weights_mc, Roff) for j,pj in enumerate(pi)] 
        for i,pi in enumerate(off)]
    )

    profiles = (1 - fmis)*profiles + fmis*woff.T/(2*np.pi)

    dndM2halo = c.dndM2halo
    bh = c.bh
    bM = c.bM
    Pk = c.Pk
    sin_term = c.sin_term
    Rgrid = c.Rgrid
    z2halo_grid = c.z2halo_grid
    M2halo_grid = c.M2halo_grid
    k2halo = c.k2halo
    R2halo = c.R2halo
    M_arr2halo = c.M_arr2halo

    PRMz = model(Rgrid, 10, M2halo_grid, z2halo_grid, params)             
    uRM = np.trapz(4*np.pi * Rgrid[None,:,:,:]**2 * sin_term[:,:,None,None] * PRMz, x = R2halo, axis = 1)
    PhP = bh[:,:,None] * Pk[:,None] * np.trapz((dndM2halo*bM)*uRM, axis = 2, x = M_arr2halo).T[:,None,:]
    sin_term2 = np.sin(R_Mpc[None,...] * k2halo[:,None,None,None,None]) / np.where(
        R_Mpc[None,...] * k2halo[:,None,None,None,None] != 0, R_Mpc[None,...] * k2halo[:,None,None,None,None], 1)

    xhi_P = np.trapz(PhP.T[:,:,:,None,None] * sin_term2 * k2halo[:,None,None,None,None]**2, axis = 0, x = k2halo)/(2*np.pi**2)

    weighted_two_halo_term = weights[:,:,:,None,None,:]*xhi_P[None,:,None,:,:,:]

    weighted_P2halo = np.trapz(np.trapz(weighted_two_halo_term, axis = 0, x = lambda_true), axis = 0, x = lambda_obs)
    weighted_P2halo = weighted_P2halo.transpose(1,0,2)

    P2halo = np.trapz(np.trapz(weighted_P2halo, x = M200, axis = 1), axis = 1, x = zobs)/norm
    observed_mean_profile = np.average(observed_profiles, axis = (1,2))


    weighted_profiles = profiles[:, None, None, :, :] * weights[None, :, :, :]
    
    P1halo = np.trapz(np.trapz(np.trapz(np.trapz(weighted_profiles, x = lambda_true, axis = 1), x = lambda_obs, axis = 1), x = M200, axis = 1), x = zobs, axis = 1)/norm
    
    Ptotal = P1halo
    
    Ptotal = P1halo + P2halo
    fwhm = 1.6
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dr = (R[-1] - R[0]) / (len(R) - 1)
    sigma_pix = np.float64(sigma / dr)
    Ptotal = gaussian_filter1d(np.float64(Ptotal), sigma=np.float64(sigma_pix))

    fig, ax = plt.subplots(figsize = (8, 6))
    ax.semilogy(R, Ptotal, color = "black", ls = "--", lw = 4, label = "Arnaud10 (halo model)")
    ax.semilogy(R, observed_mean_profile, color = "black", lw = 4, label = "Arnaud10 (observed)")
    ax.errorbar(c.R, c.mean_profile, yerr = c.error_in_mean, fmt = "-o", color = "darkorange", lw = 4, label = "data")
    ax.legend(frameon = False)
    ax.set(xlabel = "R (Mpc)", ylabel = "y profile")
    fig.savefig("comparison_stacked_profiles.png")


from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

def compute_masses(clusters):
    M200 = np.logspace(13, 16, 50)
    lambda_true = np.linspace(10, 350, 100)
    cosmo = ccl.CosmologyVanillaLCDM()
    mdef = "200m"
    mfunc = ccl.halos.MassFuncTinker10(mass_def=mdef)
    helpers = importlib.import_module("helpers")
    func = getattr(helpers, "P_lob_ltr")
    
    fig_P_ltrue = plt.figure(figsize = (16, 16))
    gs_P_ltrue = GridSpec(4, 4)

    axs_P_ltrue_lobs = np.empty((2, 4), dtype = "object")
    sharex_row = [None]*4
    for i in range(2):
        for j in range(4):
            ax = fig_P_ltrue.add_subplot(gs_P_ltrue[i, j])
            axs_P_ltrue_lobs[i, j] = ax

    axs_P_ltrue = fig_P_ltrue.add_subplot(gs_P_ltrue[2:, :])

    fig_P_M = plt.figure(figsize = (16,16))
    gs_P_M = GridSpec(4, 4)

    axs_P_M_ltrue = np.empty((2, 4), dtype = "object")
    sharex_row = [None]*4
    for i in range(2):
        for j in range(4):
            ax = fig_P_M.add_subplot(gs_P_M[i, j]
                                        , sharex = sharex_row[i])
            if sharex_row[i] is None:
                sharex_row[i] = ax
            axs_P_M_ltrue[i, j] = ax
    
    axs_P_M = fig_P_M.add_subplot(gs_P_M[2:, :])
    
    fig_posterior = plt.figure(figsize = (16, 16))
    gs_posterior = GridSpec(4, 4)
    axs_posterior = np.empty((2, 4), dtype = "object")
    sharex_row = [None]*4
    for i in range(2):
        for j in range(4):
            ax = fig_posterior.add_subplot(gs_posterior[i, j]
                                        , sharex = sharex_row[i])
            if sharex_row[i] is None:
                sharex_row[i] = ax
            axs_posterior[i, j] = ax
    ax_marginal = fig_posterior.add_subplot(gs_posterior[2:, :])

    lambda_errors = []
    lambda_centers = []
    masses_no_prior = []
    stds_no_prior = []

    masses_posterior = []
    stds_posterior = []
    colors = ["purple", "darkblue", "darkgreen", "darkred"]
    lambda_bins = []
    redshift_bins = []

    z_centers = []
    z_errors = []

    M200ctoM200m = generate_M200c2M200mInterpolator()

    for i, c in enumerate(clusters):
        lambda_min, lambda_max = c.richness_bin
        lambda_bins.append(c.richness_bin)
        zmin, zmax = c.redshift_bin
        redshift_bins.append(c.redshift_bin)
        zmean = (zmax + zmin)/2
        lambda_obs = np.linspace(lambda_min, lambda_max, 50)
        zobs = np.linspace(zmin, zmax, 30)
        dndlog10M = np.array([[mfunc(cosmo, mi, 1/(1 + zi)) for mi in M200] for zi in zobs])
        dndM = dndlog10M/(np.log(10)*M200)
        s = 0.037
        q = 1.008
        #dndM = dndM/(np.log0) #* (s * np.log(M200/(10**(13.8)/0.67)) + q)
        ltrue_grid, lobs_grid, zobs_grid = np.meshgrid(lambda_true, lambda_obs, zobs, indexing = "ij")
        P_lobs_ltrue_z = func(lobs_grid, ltrue_grid, zobs_grid)
        #P_lobs_ltrue_z = 1/np.sqrt(0.25**2 * 2 * np.pi * lobs_grid**2) * np.exp(-(np.log(lobs_grid) - np.log(ltrue_grid))**2/0.25**2)
        P_ltrue_lobs = np.trapz(P_lobs_ltrue_z, x = zobs, axis = 2)
        ax_index = 0 if zmean < 0.4 else 1
        ls = "solid" if zmean < 0.4 else "--"
        P_ltrue_lobs_norm = P_ltrue_lobs/np.trapz(np.trapz(P_ltrue_lobs, x = lambda_obs, axis = 1), x = lambda_true)
        im_P_ltrue_lobs = axs_P_ltrue_lobs[ax_index, i//2].imshow(P_ltrue_lobs_norm.T, aspect = "auto", 
            extent = [lambda_true[0], lambda_true[-1], lambda_obs[0], lambda_obs[-1]]
            , cmap = "Purples", origin = "lower", interpolation = "bilinear", 
            norm = LogNorm())
        if ax_index == 1:
            axs_P_ltrue_lobs[ax_index, i//2].set_xlabel(r"$\lambda_{true}$")
        if i//2 == 0:
            axs_P_ltrue_lobs[ax_index, i//2].set_ylabel(r"$\lambda_{obs}$")
        axs_P_ltrue_lobs[ax_index, i//2].plot(lambda_obs, lambda_obs, alpha = 0.8, lw = 3, color = "darkred")
        axs_P_ltrue_lobs[ax_index, i//2].set_title(r"$\lambda \in [%.i, %.i], z \in [%.2f, %.2f]$" % (lambda_min, lambda_max, zmin, zmax), fontsize = 12)
        P_ltrue = np.trapz(P_ltrue_lobs, x = lambda_obs, axis = 1)
        norm_P_ltrue = np.trapz(P_ltrue, x = lambda_true)
        P_ltrue_norm = P_ltrue/norm_P_ltrue
        axs_P_ltrue.plot(lambda_true, P_ltrue_norm/np.max(P_ltrue_norm), alpha = 0.8, lw = 3, ls = ls, color = colors[i//2])
        axs_P_ltrue.text((lambda_obs[0] + lambda_obs[-1])/2, 1.25, 
            r"$\lambda \in [%.2f, %.2f]$" % (lambda_min, lambda_max), fontsize = 12, va = "center", ha = "center", rotation = 75)
        if zmean < 0.4:
            axs_P_ltrue.fill_between(lambda_obs, 0, 10, alpha = 0.1, color = colors[i//2])
        axs_P_ltrue.set_title("$P(\lambda_{\mathrm{true}}) = \int d\lambda_{\mathrm{obs}} \int dzP(\lambda_{\mathrm{true}}|\lambda_{\mathrm{obs}}, z)$"
                             , fontsize = 20)
        axs_P_ltrue.set(xlabel = "$\lambda_{\mathrm{true}}$", ylabel = "$P(\lambda_{\mathrm{true}})/\max(P(\lambda_{\mathrm{true}}))$") 
        axs_P_ltrue.set_ylim(0, 1.5)
        
        ltrue_M_grid, M_grid, zobs_M_grid = np.meshgrid(lambda_true, M200, zobs, indexing = "ij")
        M200m_grid = M_grid #10**M200ctoM200m((np.log10(M_grid), zobs_M_grid))
        lambda_model = 30 * (M200m_grid / (3e14/0.67))**0.75 * ((1 + zobs_M_grid)/(1 + 0.35))**(-0.0)
        sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
        
        P_ltrue_M_z = (1 / (ltrue_M_grid * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log10(ltrue_M_grid) - np.log10(lambda_model))**2 / (2 * sigma_model**2)))

        #ln_lambda_model = np.log(lambda_model)
        #x_min = (np.log(np.min(lambda_true)) - ln_lambda_model) / (np.sqrt(2) * sigma_model)
        #x_max = (np.log(np.max(lambda_true)) - ln_lambda_model) / (np.sqrt(2) * sigma_model)
        #P_ltrue_M_z = 0.5 * (erf(x_max) - erf(x_min))
        P_lobs_ltrue_z_norm = P_lobs_ltrue_z/np.trapz(np.trapz(np.trapz(P_lobs_ltrue_z, x = zobs, axis = 2), x = lambda_obs, axis = 1), x = lambda_true)
        P_ltrue_M_z_norm = P_ltrue_M_z/np.trapz(np.trapz(np.trapz(P_ltrue_M_z, x = zobs, axis = 2), x = lambda_true, axis = 0), x = M200)
        P_joint = P_lobs_ltrue_z[:, :, None, :] * P_ltrue_M_z[:, None, :, :]
        P_ltrue_M = np.trapz(np.trapz(P_joint, x = zobs, axis = 3), axis = 1, x = lambda_obs)
        norm_P_ltrue_M = np.trapz(np.trapz(P_ltrue_M, axis = 0, x = lambda_true), x = M200)
        P_ltrue_M_norm = P_ltrue_M/norm_P_ltrue_M
        
        im_P_ltrue_M = axs_P_M_ltrue[ax_index,i//2].imshow(P_ltrue_M_norm, aspect = "auto",
            extent = [M200[0], M200[-1], lambda_true[0], lambda_true[-1]]
            , cmap = "Reds", origin = "lower", interpolation = "bilinear", 
            norm = LogNorm())
        if ax_index == 1:
            axs_P_M_ltrue[ax_index, i//2].set_xlabel(r"$M_{200}$")
        if i//2 == 0:
            axs_P_M_ltrue[ax_index, i//2].set_ylabel(r"$\lambda_{true}$")
        axs_P_M_ltrue[ax_index, i//2].set_xscale("log")
        axs_P_M_ltrue[ax_index, i//2].plot(M200, 30*(M200/(3e14))**0.75, alpha = 0.8, lw = 3, color = "darkblue")
        axs_P_M_ltrue[ax_index, i//2].set_title(r"$\lambda \in [%.i, %.i], z \in [%.2f, %.2f]$" % (lambda_min, lambda_max, zmin, zmax), fontsize = 12)
        P_M = np.trapz(P_ltrue_M_norm, x = lambda_true, axis = 0)
        axs_P_M.plot(M200, P_M/np.max(P_M), alpha = 0.8, lw = 3, ls = ls, color = colors[i//2])

        mass_no_prior = np.trapz(P_M * M200, x = M200)/np.trapz(P_M, x = M200)
        var_no_prior = np.trapz(P_M * (M200 - mass_no_prior)**2, x = M200)/np.trapz(P_M, x = M200)
        std_no_prior = np.sqrt(var_no_prior)

        masses_no_prior.append(mass_no_prior)
        stds_no_prior.append(std_no_prior)

        axs_P_M.text(mass_no_prior, 1.25, r"$M_{200} = %.2f \pm %.2f$" % (np.log10(mass_no_prior), std_no_prior/(np.log(10)*mass_no_prior)),
                     fontsize = 14, ha = "center", va = "center", color = "white", rotation = 75,
                     bbox = dict(facecolor = colors[i//2], alpha = 0.5, edgecolor = "black", linestyle = ls))
        axs_P_M.set_ylim(0, 1.5)
        axs_P_M.set_xscale("log")

        dV = planck18.differential_comoving_volume(zobs)
        prior_Mz = dndM.T * dV[None,:]

        posterior = P_joint * prior_Mz[None,None,:,:]

        P_ltrue_given_M = np.trapz(np.trapz(posterior, x = zobs, axis = 3), axis = 1, x = lambda_obs)
        norm_P_ltrue_given_M = np.trapz(np.trapz(P_ltrue_given_M, axis = 0, x = lambda_true), x = M200)
        P_ltrue_given_M_norm = P_ltrue_given_M/norm_P_ltrue_given_M

        im_posterior = axs_posterior[ax_index, i//2].imshow(P_ltrue_given_M_norm, aspect = "auto",
            extent = [M200[0], M200[-1], lambda_true[0], lambda_true[-1]]
            , cmap = "Blues", origin = "lower", interpolation = "bilinear", 
            norm = LogNorm())
        if ax_index == 1:
            axs_posterior[ax_index, i//2].set_xlabel(r"$M_{200}$")
        if i//2 == 0:
            axs_posterior[ax_index, i//2].set_ylabel(r"$\lambda_{true}$")
        axs_posterior[ax_index, i//2].set_xscale("log")
        axs_posterior[ax_index, i//2].set_title(r"$\lambda \in [%.i, %.i], z \in [%.2f, %.2f]$" % (lambda_min, lambda_max, zmin, zmax), fontsize = 12)

        marginal = np.trapz(P_ltrue_given_M_norm, x = lambda_true, axis = 0)
        ax_marginal.plot(M200, marginal/np.max(marginal), alpha = 0.8, lw = 3, ls = ls, color = colors[i//2])
        mass_posterior = np.trapz(marginal * M200, x = M200)/np.trapz(marginal, x = M200)
        var_posterior = np.trapz(marginal * (M200 - mass_posterior)**2, x = M200)/np.trapz(marginal, x = M200)
        std_posterior = np.sqrt(var_posterior)

        ax_marginal.text(mass_posterior, 1.25, r"$M_{200} = %.2f \pm %.2f$" % (np.log10(mass_posterior), std_posterior/(np.log(10)*mass_posterior)),
                     fontsize = 14, ha = "center", va = "center", color = "white", rotation = 75,
                     bbox = dict(facecolor = colors[i//2], alpha = 0.5, edgecolor = "black", linestyle = ls))
        ax_marginal.set_xscale("log")
        ax_marginal.set_ylim(0, 1.5)

        masses_posterior.append(mass_posterior)
        stds_posterior.append(std_posterior)

        lambda_center = np.trapz(np.trapz(P_ltrue_given_M_norm * lambda_true[:,None], axis = 0, x = lambda_true), x = M200)/np.trapz(np.trapz(P_ltrue_given_M_norm, axis = 0, x = lambda_true), x = M200)
        lambda_err = np.trapz(np.trapz(P_ltrue_given_M_norm * (lambda_true - lambda_center)[:,None]**2, axis = 0, x = lambda_true), x = M200)/np.trapz(np.trapz(P_ltrue_given_M_norm, axis = 0, x = lambda_true), x = M200)
        lambda_centers.append(lambda_center)
        lambda_errors.append(np.sqrt(lambda_err))

        z_center = np.trapz(np.trapz(np.trapz(np.trapz(P_joint*zobs[None,None,None,:], axis = 3, x = zobs), axis = 2, x = M200), axis = 1, x = lambda_obs), x = lambda_true)/np.trapz(np.trapz(np.trapz(np.trapz(P_joint, axis = 3, x = zobs), axis = 2, x = M200), axis = 1, x = lambda_obs), x = lambda_true)
        z_std = np.trapz(np.trapz(np.trapz(np.trapz(P_joint*(zobs - z_center)[None,None,None,:]**2, axis = 3, x = zobs), axis = 2, x = M200), axis = 1, x = lambda_obs), x = lambda_true)

        z_centers.append(z_center)
        z_errors.append(z_std)
    fig_P_ltrue.tight_layout()

    fig_P_ltrue.savefig("P_ltrue.png")

    fig_P_M.tight_layout()
    fig_P_M.savefig("P_M.png")

    fig_posterior.tight_layout()
    fig_posterior.savefig("posterior.png")

    fig, ax = plt.subplots(figsize = (12, 8))

    ltrue_M_grid, M_grid = np.meshgrid(lambda_true, M200)
    lambda_model = 30 * (M_grid / (3e14/0.67))**0.75
    sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
    P_ltrue_M = (1 / (ltrue_M_grid * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log(ltrue_M_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2)))
    norm_P_ltrue_M = P_ltrue_M/np.trapz(np.trapz(P_ltrue_M, axis = 1, x = lambda_true), x = M200)
    log10norm_P_ltrue_M = np.log10(norm_P_ltrue_M)
    levels = [np.std(P_ltrue_M)/(np.log(10)*np.mean(P_ltrue_M)), 2*np.std(P_ltrue_M)/(np.log(10)*np.mean(P_ltrue_M)), 5*np.std(P_ltrue_M)/(np.log(10)*np.mean(P_ltrue_M))]
    cs = ax.contour(ltrue_M_grid, M_grid, log10norm_P_ltrue_M, levels = levels, colors = "black")
    fmt = {
        levels[0]: r'$5\sigma$',
        levels[1]: r'$2\sigma$',
        levels[2]: r'$1\sigma$',
    }
    ax.clabel(
        cs,
        levels=levels,
        fmt=fmt,
        inline=True,
        fontsize=10
    )
    for i in range(len(lambda_centers)):
        if redshift_bins[i][0] < 0.4:
            ax.errorbar(lambda_centers[i], masses_no_prior[i], yerr = stds_no_prior[i], 
            xerr = np.array(lambda_errors[i]), fmt = "o", color = colors[i//2],
            markersize = 15, alpha = 0.5, capsize = 2)
            ax.errorbar(lambda_centers[i], masses_posterior[i], yerr = stds_posterior[i], 
            xerr = np.array(lambda_errors[i]), fmt = "o", color = colors[i//2],
            markersize = 15, alpha = 0.5, capsize = 2)
        elif redshift_bins[i][0] > 0.4:
            ax.errorbar(lambda_centers[i], masses_posterior[i], yerr = stds_posterior[i], 
            xerr = np.array(lambda_errors[i]), fmt = "s", color = colors[i//2],
            markersize = 15, alpha = 0.5, capsize = 2)
            ax.errorbar(lambda_centers[i], masses_no_prior[i], yerr = stds_no_prior[i], 
            xerr = np.array(lambda_errors[i]), fmt = "s", color = colors[i//2],
            markersize = 15, alpha = 0.5, capsize = 2)
            ax.fill_between(lambda_bins[i], 0, 1e16, color = colors[i//2], alpha = 0.1)
    for i,l in enumerate(np.unique(lambda_bins, axis = 0)):
        lmin, lmax = l
        center = np.sqrt(lmin*lmax)
        text = ax.text(center, 3e13, r"$\lambda \in [%.i, %.i]$" % (lmin, lmax), ha = "center", va = "center", 
                fontsize = 10, rotation = 25, color = colors[i])
        text.set_path_effects([
            pe.Stroke(linewidth=2, foreground='black'), 
            pe.Normal()
        ])
    ax.scatter([],[], marker = "o", color = "black", label = r"$z\in[0.1,0.4]$", s = 30)
    ax.scatter([],[], marker = "s", color = "black", label = r"$z\in[0.4,1]$", s = 30)
    ax.plot(lambda_centers, masses_no_prior, ls = "--", lw = 3, color = "orange")
    ax.plot(lambda_centers, masses_posterior, ls = "solid", lw = 3, color = "brown")
    ax.text(5e1, 2.7*1e14, r"$dndM + dV$",
            ha = "center", va = "center", fontsize = 15, rotation = 25, color = "brown")
    ax.text(4e1, 3e15, r"no mass/redshift prior",
            ha = "center", va = "center", fontsize = 15, rotation = 25, color = "orange")
    M200mtoM200c = generate_M200m2M200cInterpolator()
    prediction_M = 10**(14.489)*(np.array(lambda_true)/40)**(1.356)
    prediction_M = prediction_M/0.67
    prediction_M = 10**M200mtoM200c((np.log10(prediction_M), 0.35))
    ax.plot(lambda_true, prediction_M, color = "black", ls = "--", label = "McClintock et al 2019")
    ax.plot([], [], color = "black", label = "$P(\lambda_{true} | M_{200})$ (Costanzi et al 2019)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$M_{200},c$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(frameon = False, fontsize = 10)
    ax.grid(True)
    fig.savefig("richness_vs_mass.png")
    return lambda_centers, lambda_errors, masses_posterior, stds_posterior, z_centers, z_errors

def compute_Y500(clusters, M200, M200_err, z):

    Y500, Y500_err = [], []
    M500_list, M500_err_list = [], []

    fM200toM500, _ = create_mass_interpolator()

    for i, ci in enumerate(clusters):

        signal = ci.stacked_map
        zi = z[i]

        m200 = M200[i].value if hasattr(M200[i], "value") else M200[i]
        m200err = M200_err[i].value if hasattr(M200_err[i], "value") else M200_err[i]

        # ---- M500 ----
        m500 = 10**fM200toM500((np.log10(m200), zi))

        rho_c = planck18.critical_density(zi).to(u.Msun/u.Mpc**3).value
        R500 = (m500 / (4*np.pi/3 * 500 * rho_c))**(1/3)

        delta = 0.01
        m500_plus = 10**fM200toM500((np.log10(m200*(1+delta)), zi))
        dm500_dm200 = (m500_plus - m500) / (m200 * delta)
        m500_err = np.abs(dm500_dm200 * m200err)

        M500_list.append(m500)
        M500_err_list.append(m500_err)

        R500_err = (1/3) * (R500/m500) * m500_err

        dA = planck18.angular_diameter_distance(zi).value

        theta500 = (R500 / dA) * (180/np.pi) * 60

        patch_size = 0.6 * 60
        pix_size = patch_size / signal.shape[0]
        pix_size_rad = pix_size * np.pi / (180 * 60)
        x, y = np.indices(signal.shape)
        x -= signal.shape[0]//2
        y -= signal.shape[1]//2
        r = np.sqrt((x*pix_size)**2 + (y*pix_size)**2)

        mask = r < theta500
        y500 = np.sum(signal[mask]) * pix_size_rad**2

        Y500.append(y500 * dA**2)
        sigma_y = np.std(signal[~mask])
        Y500_err.append(sigma_y * np.sqrt(mask.sum()) * pix_size_rad**2 * dA**2)

    return np.array(Y500), np.array(Y500_err), np.array(M500_list), np.array(M500_err_list)

def loadtng():
    folders = os.listdir("/data2/javierurrutia/szeffect/data/TNG300-3")
    folders = [f for f in folders if f[0] != "."]
    folders = [f for f in folders if f.split(".")[-1] == "h5" and f.split("_")[0] == "compton"]
    data = [h5py.File(f"/data2/javierurrutia/szeffect/data/TNG300-3/{f}") for f in folders]

    redshifts = [float(re.search(r"z=([0-9]*\.?[0-9]+)", f).group(1)) for f in folders]
    M500 = np.concatenate([data[i]["R500_measurements"]["M500"] for i in range(len(data))])
    M200 = np.concatenate([data[i]["R200_measurements"]["M200"] for i in range(len(data))])
    Y500 = np.concatenate([data[i]["R500_measurements"]["Y500"] for i in range(len(data))])*1e-6
    z = np.concatenate([np.full(len(data[i]["R500_measurements"]["Y500"]), redshifts[i]) for i in range(len(data))])
    ymaps = [data[i]["compton_y_maps"] for i in range(len(data))]
    flatten_ymaps = []
    for i in range(len(ymaps)):
        for j in range(len(ymaps[i])):
            ymap = ymaps[i][j]
            nx, ny = ymap.shape
            if nx < 50 or ny < 50:
                continue
            ymap_cut = ymap[:50, :50]
            flatten_ymaps.append(ymap_cut)
    return M500, Y500, M200, z, flatten_ymaps

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

def ymockTNG(ymap, R, redshift, pix_size_kpc = 100, convolve = True, pix_size_arcmin = 0.5, order = 1, fwhm = 1.6): 
    Da = planck18.angular_diameter_distance(redshift).to(u.kpc).value
    theta_pix_arcmin = (pix_size_kpc / Da) * (180 / np.pi) * 60
    zoom_factor = theta_pix_arcmin / pix_size_arcmin
    y_act = zoom(ymap, zoom_factor, order=order)
    if convolve == True:
        sigma_arcmin = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_pix = sigma_arcmin / pix_size_arcmin
        y_act = gaussian_filter(y_act, sigma_pix)
    shape = np.shape(y_act)
    x,y = np.indices(shape)
    x,y = (x - shape[0]//2), (y - shape[1]//2)
    x,y = x * pix_size_arcmin, y * pix_size_arcmin
    r = np.sqrt(x**2 + y**2)
    profile = []
    std = []
    for i in range(len(R) - 1):
        mask = (r > R[i]) & (r < R[i+1])
        profile.append(np.nanmean(y_act[mask]))
        std.append(np.nanstd(y_act[mask]))
    return profile, std, y_act


def generate_stacked_tng_profiles(R, clusters, pix_size_kpc = 100, pix_size_arcmin = 0.5, beam_size = 1.6):
    if os.path.exists("/data2/javierurrutia/szeffect/codes/tng_richness.txt"):
        data_TNG = np.loadtxt("tng_richness.txt")
        m200tng, z, lambda_tng = data_TNG.T
        M500, Y500, M200, _, flatten_ymaps = loadtng()
        mask = np.where(M200 > 1e14)
        M500 = M500[mask]
        Y500 = Y500[mask]
        M200 = M200[mask]
        flatten_ymaps = np.array(flatten_ymaps)[mask]
    else:
        lambda_tng, M500, M200, Y500, z, flatten_ymaps = compute_richness_tng(clusters)
    profiles = np.zeros((len(M500), len(R) - 1))
    stds = np.zeros((len(M500), len(R) - 1))
    ymaps_act = []
    for i,y in enumerate(flatten_ymaps):
        profile, std, y_act = ymockTNG(y, R, z[i], pix_size_kpc = pix_size_kpc, convolve = True, pix_size_arcmin = pix_size_arcmin, order = 1, fwhm = beam_size)
        profiles[i] = profile
        stds[i] = std
        ymaps_act.append(y_act)
        if i == 0:
            signal = ymaps_act[i]
            shape = np.shape(signal)
            xmin, xmax = -shape[0]//2, shape[0]//2
            ymin, ymax = -shape[1]//2, shape[1]//2
            xmin_arcmin = xmin*pix_size_arcmin
            xmax_arcmin = xmax*pix_size_arcmin
            ymin_arcmin = ymin*pix_size_arcmin
            ymax_arcmin = ymax*pix_size_arcmin
            R500 = (M500[i] / (4/3 * np.pi * 500 * planck18.critical_density(z[i]).to(u.Msun / u.Mpc**3).value))**(1/3)
            R500_arcmin = R500 / planck18.angular_diameter_distance(z[i]).to(u.Mpc).value * (180 / np.pi) * 60
            fig, ax = plt.subplots(figsize = (10,10))
            im = ax.imshow(signal, cmap = "coolwarm", norm = LogNorm(), aspect = "equal"
            , extent = [xmin_arcmin, xmax_arcmin, ymin_arcmin, ymax_arcmin])
            circle = plt.Circle((0, 0), R500_arcmin, color = "white", fill = False, linewidth = 3, ls = "--")
            beam = plt.Circle((0,0), 1.6, color = "darkblue", fill = False, linewidth = 3)
            ax.text(0, 16, "$R_{500}$", color = "white", ha = "center", va = "center", fontsize = 16)
            ax.text(0, 3, "ACT-beam", color = "darkblue", ha = "center", va = "center", fontsize = 16)
            text = ax.text(-20, 20, "TNG300-3 halo ID = 0", color = "white", ha = "left", va = "top", fontsize = 16)
            text.set_path_effects([
            pe.Stroke(linewidth=2, foreground='black'), 
            pe.Normal()
            ])
            ax.add_patch(circle)
            ax.add_patch(beam)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, ax = ax, cax = cax)
            cbar.set_label("Compton-y", fontsize = 16, rotation = 0, labelpad = 15, fontweight = "bold")
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.yaxis.set_label_coords(0.5, 1.05)
            ax.set(xlabel = f"$\Delta x $ arcmin", ylabel = f"$\Delta y$ arcmin")
            fig.savefig("szmap_tng_ID=0.png")
    fig, ax = plt.subplots(2,4, figsize = (16, 8), sharex = "col", sharey = "row")
    for i, ci in enumerate(clusters):
        profile = ci.mean_profile
        errors = ci.error_in_mean
        lambda_min, lambda_max = ci.richness_bin
        zmin, zmax = ci.redshift_bin
        if zmin == np.float32(0.1):
            zmin = 0
        mask = np.where((lambda_tng >= lambda_min) & (lambda_tng <= lambda_max) & (z >= zmin) & (z <= zmax))
        profiles_tng = profiles[mask]
        for j in range(len(profiles_tng)):
            pi = profiles_tng[j].copy()
            pi = np.nan_to_num(pi, 0)

            if np.any(pi != 0):
                pi[pi == 0] = pi[pi != 0][-1]

            profiles_tng[j] = pi
        profile_tng = np.nanmean(profiles_tng, axis=0)
        N_bootstrap = 500 
        bootstrap_profiles = []
        for j in range(N_bootstrap):
            idx = np.random.choice(len(mask[0]), len(mask[0]), replace = True)
            bootstrap_profiles.append(np.nanmean(profiles_tng[idx], axis = 0))
        
        cov = np.cov(bootstrap_profiles, rowvar = False)
        err = np.sqrt(np.diag(cov))
        mass = M200[mask]
        
        snr = np.sqrt(np.sum(profile_tng**2/err**2))
        ci.tng_profile = profile_tng
        ci.tng_errors = err

        R = ci.R
        zmin, zmax = ci.redshift_bin
        lambda_min, lambda_max = ci.richness_bin
        row_indx = 0 if zmin < 0.3 else 1
        ax[row_indx][i//2].errorbar(R, profile, yerr = errors, color = "black", capsize = 3)
        ax[row_indx][i//2].errorbar(R+0.5, profile_tng, yerr = err, color = "purple", capsize = 3, label = "TNG300-3")
        if row_indx == 1:
            ax[row_indx][i//2].set(xlabel = "R (arcmin)")
        if i//2 == 0:
            ax[row_indx][i//2].set(ylabel = "compton-y profile", yscale = "log")
        ax[row_indx][i//2].set(title = r"$\lambda \in [%.i, %.i]\;z \in [%.2f, %.2f]$" % (lambda_min, lambda_max, zmin, zmax))
        ci_tng = grouped_clusters.empty()
        
        ci_tng.output_path = "/data2/javierurrutia/szeffect/data/TNG300-3/l%.i-%i_z%.2f-%.2f" % (lambda_min, lambda_max, zmin, zmax)
        
        corr = cov / np.outer(err, err)

        fig2, ax2 = plt.subplots(figsize=(8,8))
        im = ax2.imshow(
            corr,
            origin="lower",
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        fig2.colorbar(im)
        fig2.savefig(ci_tng.output_path + "/correlation_matrix.png")
        profile_tng = np.nan_to_num(profile_tng)
        ci.tng_profile = profile_tng
        ci.tng_errors = err
        ci.save()
        ci_tng.richness = lambda_tng[mask]
        ci_tng.z = z[mask]

        print(lambda_min, lambda_max, zmin, zmax, np.log10(np.mean(mass)), np.std(M200)/((np.log(10)*np.mean(mass))), snr**2, len(mass))
        
        ci_tng.profiles = profiles[mask]
        ci_tng.bootstrap_profiles = bootstrap_profiles
        ci_tng.richness = lambda_tng[mask]
        ci_tng.imap = flatten_ymaps[mask]
        ci_tng.stacked_map = np.average(flatten_ymaps[mask], axis = 0)
        ci_tng.z = z[mask]
        ci_tng.R = R
        ci_tng.richness_bin = ci.richness_bin
        ci_tng.redshift_bin = ci.redshift_bin
        ci_tng.mean_profile = profile_tng
        ci_tng.error_in_mean = err
        ci_tng.M200 = mass
        ci_tng.cov = cov
        ci_tng.create_beam_filter()
        ci_tng.plot()
        ci_tng.save()
    ax[0][0].legend(frameon = False)
    fig.savefig("comparison_stacked_profiles.png")   

def load():
    arnaud10 = pd.read_csv('arnaud10.txt', sep=r'\s+', skiprows=2,
                    names=['Cluster', 'R_500', 'Y_X',
                    'Y_sph_R2500','Y_sph_R500',
                    'P_500', 'P_0', 'c_500', 'alpha', 'gamma', 'chi2_dof'])
    ade11 = pd.read_csv('Ade11.txt', delim_whitespace = True, engine="python",
                    names=['Cluster', 'RA', 'Dec', 'z', 'R500', 'TX','Mg500',
                    'YX500', 'D2_Y500' ,'M500', 'LX500', 'CC'])
    y500arnaud10 = arnaud10["Y_sph_R500"]
    Y500arnaud10, Y500arnaud10e = [],[]
    for y in y500arnaud10:
        m, e = y.split("±")
        Y500arnaud10.append(float(m)*1e-5)
        Y500arnaud10e.append(float(e)*1e-5)
    R_500arnaud10 = arnaud10["R_500"]
    redshift_data = {
        'RXC_J0003.8+0203': 0.0924,'RXC_J0006.0-3443': 0.1147,'RXC_J0020.7-2542': 0.1410,
        'RXC_J0049.4-2931': 0.1084,'RXC_J0145.0-5300': 0.1168,'RXC_J0211.4-4017': 0.1008,
        'RXC_J0225.1-2928': 0.0604,'RXC_J0345.7-4112': 0.0603,'RXC_J0547.6-3152': 0.1483,
        'RXC_J0605.8-3518': 0.1392,'RXC_J0616.8-4748': 0.1164,'RXC_J0645.4-5413': 0.1644,
        'RXC_J0821.8+0112': 0.0822,'RXC_J0958.3-1103': 0.1669,'RXC_J1044.5-0704': 0.1342,
        'RXC_J1141.4-1216': 0.1195,'RXC_J1236.7-3354': 0.0796,'RXC_J1302.8-0230': 0.0847,
        'RXC_J1311.4-0120': 0.1832,'RXC_J1516+0005': 0.1181,'RXC_J1516.5-0056': 0.1198,
        'RXC_J2014.8-2430': 0.1612,'RXC_J2023.0-2056': 0.0564,
        'RXC_J2048.1-1750': 0.1475,'RXC_J2129.8-5048': 0.0796,
        'RXC_J2149.1-3041': 0.1184,'RXC_J2157.4-0747': 0.0579,
        'RXC_J2217.7-3543': 0.1486,'RXC_J2218.6-3853': 0.1411,
        'RXC_J2234.5-3744': 0.1510,'RXC_J2319.6-7313': 0.0984,
    }
    z = list(redshift_data.values())
    M500arnaud10 = R_500arnaud10**3 * 4/3 * np.pi * 500 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value
    M500ade11 = np.array([float(m.split("±")[0]) for m in ade11["M500"]])*1e14
    y500ade11 = ade11["D2_Y500"]
    Y500ade11, Y500ade11e = [], []
    for y in y500ade11:
        m, e = y.split("±")
        Y500ade11.append(float(m)*1e-4)
        Y500ade11e.append(float(e)*1e-4)
    return M500arnaud10, Y500arnaud10, Y500arnaud10e, M500ade11, Y500ade11, Y500ade11e

def compare_with_arnaud():
    arnaud10 = pd.read_csv('arnaud10.txt', sep=r'\s+', skiprows=2,
                    names=['Cluster', 'R_500', 'Y_X',
                    'Y_sph_R2500','Y_sph_R500',
                    'P_500', 'P_0', 'c_500', 'alpha', 'gamma', 'chi2_dof'])
    R_500arnaud10 = arnaud10["R_500"]  # en Mpc
    P_0 = arnaud10["P_0"]
    c_500 = arnaud10["c_500"]
    alpha = arnaud10["alpha"]
    gamma = arnaud10["gamma"]
    beta = 5.49

    redshift_data = {
        'RXC_J0003.8+0203': 0.0924,'RXC_J0006.0-3443': 0.1147,'RXC_J0020.7-2542': 0.1410,
        'RXC_J0049.4-2931': 0.1084,'RXC_J0145.0-5300': 0.1168,'RXC_J0211.4-4017': 0.1008,
        'RXC_J0225.1-2928': 0.0604,'RXC_J0345.7-4112': 0.0603,'RXC_J0547.6-3152': 0.1483,
        'RXC_J0605.8-3518': 0.1392,'RXC_J0616.8-4748': 0.1164,'RXC_J0645.4-5413': 0.1644,
        'RXC_J0821.8+0112': 0.0822,'RXC_J0958.3-1103': 0.1669,'RXC_J1044.5-0704': 0.1342,
        'RXC_J1141.4-1216': 0.1195,'RXC_J1236.7-3354': 0.0796,'RXC_J1302.8-0230': 0.0847,
        'RXC_J1311.4-0120': 0.1832,'RXC_J1516+0005': 0.1181,'RXC_J1516.5-0056': 0.1198,
        'RXC_J2014.8-2430': 0.1612,'RXC_J2023.0-2056': 0.0564,
        'RXC_J2048.1-1750': 0.1475,'RXC_J2129.8-5048': 0.0796,
        'RXC_J2149.1-3041': 0.1184,'RXC_J2157.4-0747': 0.0579,
        'RXC_J2217.7-3543': 0.1486,'RXC_J2218.6-3853': 0.1411,
        'RXC_J2234.5-3744': 0.1510,'RXC_J2319.6-7313': 0.0984,
    }
    z = list(redshift_data.values())
    
    # Calcular M500
    M500arnaud10 = R_500arnaud10**3 * 4/3 * np.pi * 500 * \
                   planck18.critical_density(z).to(u.Msun / u.Mpc**3).value
    
    modeled_profiles = []
    
    for i in range(len(M500arnaud10)):
        c500 = c_500[i]
        P0 = P_0[i]  # adimensional
        a = alpha[i]
        g = gamma[i]
        R500 = R_500arnaud10[i]  # Mpc
        M500 = M500arnaud10[i]  # Msun
        zi = z[i]
        
        # Radio proyectado (perpendicular a la línea de visión)
        R = np.linspace(0.03*R500, 3*R500, 100)  # Mpc
        
        # P500 en keV/cm³
        P500_val = P500(M500, zi, planck18)
        
        # Coordenada a lo largo de la línea de visión
        R_los = np.linspace(-3*R500, 3*R500, 500)  # Mpc, suficientes puntos para buena integración
        
        # Radio 3D: r = sqrt(R² + R_los²)
        R_los_grid = R_los[:, np.newaxis]  # shape (500, 1)
        R_grid = R[np.newaxis, :]  # shape (1, 100)
        r_3d = np.sqrt(R_los_grid**2 + R_grid**2)  # shape (500, 100)
        
        # Perfil 3D de presión (adimensional, normalizado por P500)
        x = c500 * r_3d / R500
        P_3d = P0 / (x**g * (1 + x**a)**((beta - g)/a))
        
        # Integrar a lo largo de la línea de visión
        # Resultado en unidades adimensionales
        P_projected = 2 * np.trapz(P_3d, R_los, axis=0)
        
        # Multiplicar por P500 para obtener presión en keV/cm²
        pressure_profile = P_projected * P500_val * u.keV / u.cm**2
        
        modeled_profiles.append(pressure_profile)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(14, 10))
    R = np.linspace(0.03, 3, 100)  # en unidades de R500
    
    for p in modeled_profiles:
        ax.plot(R, p.value)
    
    ax.set_xlabel(r'$R/R_{500}$', fontsize=14)
    ax.set_ylabel(r'Presión proyectada [keV/cm$^2$]', fontsize=14)
    ax.loglog()
    ax.grid(True, alpha=0.3)
    fig.savefig("arnaud_profiles.png", dpi=150, bbox_inches='tight')
    plt.show()
def loadtng():
    folders = os.listdir("/data2/javierurrutia/szeffect/data/TNG300-3")
    folders = [f for f in folders if f[0] != "." and f.split(".")[-1] == "h5"]
    data = [h5py.File(f"/data2/javierurrutia/szeffect/data/TNG300-3/{f}") for f in folders]
    redshifts = [float(re.search(r"z=([0-9]*\.?[0-9]+)", f).group(1)) for f in folders]
    M500 = np.concatenate([data[i]["R500_measurements"]["M500"] for i in range(len(data))])
    M200 = np.concatenate([data[i]["R200_measurements"]["M200"] for i in range(len(data))])
    Y500 = np.concatenate([data[i]["R500_measurements"]["Y500"] for i in range(len(data))])*1e-6
    z = np.concatenate([np.full(len(data[i]["R500_measurements"]["Y500"]), redshifts[i]) for i in range(len(data))])
    ymaps = [data[i]["compton_y_maps"] for i in range(len(data))]
    flatten_ymaps = []
    for i in range(len(ymaps)):
        for j in range(len(ymaps[i])):
            ymap = ymaps[i][j]
            nx, ny = ymap.shape
            if nx < 50 or ny < 50:
                continue
            ymap_cut = ymap[:50, :50]
            flatten_ymaps.append(ymap_cut)
    return M500, Y500, M200, z, flatten_ymaps

def compute_richness_tng(clusters):
    M200c2M200m = generate_M200c2M200mInterpolator()
    c = clusters[0]
    szmaps = []
    for i in range(1, len(clusters)):
        c += clusters[i]
    szmaps = np.concatenate([ci.imap for ci in clusters])
    richness = c.richness
    M500, Y500, M200, z, ymaps = loadtng()
    mask = np.where(M200 > 1e14)
    M200 = M200[mask]
    z = z[mask]
    ymaps = np.array(ymaps)[mask]
    M500 = M500[mask]
    Y500 = Y500[mask]
    lambda_true = np.linspace(10, 350, 50)
    helpers = importlib.import_module("helpers")
    func = getattr(helpers, "P_lob_ltr")
    lambda_obs = np.linspace(10, 350, 500)
    lambda_tng = []
    for i in tqdm(range(len(M200))):
        mi = M200[i]
        zi = z[i]
        mi200m = 10**M200c2M200m((np.log10(mi), zi))
        lobs_grid, ltrue_grid = np.meshgrid(lambda_obs, lambda_true)
        P_lob_ltr = func(lobs_grid, ltrue_grid, zi)
        lambda_model = 30 * (mi200m / (3e14))**0.75 * ((1 + zi)/(1 + 0.35))**(-0.3)
        sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
        P_ltrue_M_z = (1 / (lambda_true * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log(lambda_true) - np.log(lambda_model))**2 / (2 * sigma_model**2))) 
        P_joint = P_lob_ltr * P_ltrue_M_z[:,None]
        weights = np.trapz(P_joint, axis = 0, x = lambda_true)
        weights = np.array(weights, dtype = np.float64)
        weights = weights/np.sum(weights)
        lobs = np.random.choice(lambda_obs, p = weights)
        lambda_tng.append(lobs)
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.hist(lambda_tng, log = True, bins = 20, color = "purple", alpha = 0.8, density = True, histtype = "step", label = "TNG300-3", lw = 3)
    ax.hist(richness, log = True, bins = 20, density = True, label = "DES-Y3 RedMaPPer", color = "blue", alpha = 0.8, histtype = "step", lw = 3)
    ax.set(xlabel = "richness $\lambda$", ylabel = "PDF")
    ax.legend(frameon = False, fontsize = 12)
    fig.savefig("richness_distribution.png")
    output = np.column_stack((M200, z, lambda_tng))
    np.savetxt("tng_richness.txt", output)
    return lambda_tng, M500, M200, Y500, z, ymaps

from scipy.interpolate import RegularGridInterpolator

def compute_individual_masses(clusters):
    M200c2M200m = generate_M200c2M200mInterpolator()
    _, _, M200tng, ztng, _ = loadtng()
    c = clusters[0]
    szmaps = []
    for i in range(1, len(clusters)):
        c += clusters[i]
    szmaps = np.concatenate([ci.imap for ci in clusters])
    richness = c.richness
    redshift = c.z
    masses = np.logspace(14, 16, 500)
    lambda_true = np.linspace(10, 350, 50)
    ltrue_grid, M_grid = np.meshgrid(lambda_true, masses)
    
    helpers = importlib.import_module("helpers")
    func = getattr(helpers, "P_lob_ltr")
    M200 = []

    cosmo = ccl.CosmologyVanillaLCDM()
    mdef = "200c"
    mfunc = ccl.halos.MassFuncTinker10(mass_def=mdef)
    redshift_true = np.arange(0.05, 1.2, 0.01)
    dndlog10M = np.array([[mfunc(cosmo, mass, 1/(1 + z)) for mass in masses] for z in redshift_true])
    dndM = dndlog10M/(np.log(10)*masses)
    dndM_func = RegularGridInterpolator((np.log10(masses), redshift_true), np.log10(dndM.T))
    for i in tqdm(range(len(richness))):
        lambda_obs = richness[i]
        z_obs = redshift[i]
        P_lob_ltr = func(lambda_obs, lambda_true, z_obs)
        M200m_grid = 10**M200c2M200m((np.log10(M_grid), z_obs))
        lambda_model = 30 * (M200m_grid / (3e14/0.67))**0.75 * ((1 + z_obs)/(1 + 0.35))**(-0.3)
        sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
        P_ltrue_M_z = (1 / (ltrue_grid * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log(ltrue_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2))) 
        P_joint = P_lob_ltr[None,:] * P_ltrue_M_z 
        dndM = 10**dndM_func((np.log10(M_grid), z_obs))
        dV = planck18.differential_comoving_volume(z_obs).value
        weights = np.trapz(P_joint*dndM*dV, lambda_true, axis = 1)
        weights = np.array(weights, dtype = np.float64)
        weights = weights/np.sum(weights)
        M = np.random.choice(masses, p = weights)
        M200.append(M)
    
    fig, ax =  plt.subplots(figsize = (8, 5))
    ax.hist(M200, log = True, bins = np.logspace(13, 16, 50), density = True, label = "DES-Y3 RedMaPPer", color = "blue", alpha = 0.8, histtype = "step")
    ax.hist(M200tng, log = True, bins = np.logspace(13, 16, 50), density = True, label = "TNG300-3", color = "purple", alpha = 0.8, histtype = "step")
    ax.set(xscale = "log", xlabel = r"$M_{200} [M_{\odot}]$", ylabel = "PDF")
    ax.legend(fontsize = 12, frameon = False)
    fig.savefig("mass_distribution.png")

    data_out = np.column_stack([richness, redshift, M200])
    header = (
        "# richness  redshift  M200[M_sun]\n"
        "# M200 inferred from P(M|lambda_obs,z) with flat log-mass prior"
    )

    np.savetxt(
        "cluster_masses.txt",
        data_out,
        header=header,
        fmt=["%.2f", "%.4f", "%.5e"]
    )

def background_worker(N_realizations, Ncl, rmin, rmax, dmin, dmax, R_profiles, width, use_pixels = True, N_total = None, counter = None):
    ymap = globals()["ymap"]
    mask = globals()["mask"]
    profiles = np.zeros((N_realizations, Ncl, len(R_profiles) - 1))
    shape = ymap.shape
    wcs = ymap.wcs
    values = np.zeros(N_realizations)
    Ncl = Ncl if use_pixels == True else 1000
    for n in range(N_realizations):
        if use_pixels == False:
            i = 0
            while i < Ncl:
                rai, deci = np.random.uniform(rmin, rmax), np.random.uniform(dmin, dmax)
                theta = np.deg2rad(90.0 - deci)
                phi = np.deg2rad(rai)
                pixels = hp.ang2pix(hp.get_nside(mask), theta, phi)
                mask_values =  mask[pixels]
                mask_values = np.nan_to_num(mask_values)
                if mask_values == 1:
                    coords = np.deg2rad(np.array((deci,rai)))
                    mapi = reproject.thumbnails(ymap, coords = coords, r = np.deg2rad(width)/2., oversample = 2, order = 1) 
                    Rbins, profile, sigma, _,  = radial_binning2(mapi, R_profiles, width = width, full = True)
                    profiles[n,i,:] = profile  
                    i+=1 
                    counter.value += 1
                    sys.stdout.write(f"\rProcessed pixels: ({counter.value} / {N_total})")
                    sys.stdout.flush()
        else:
            while True:
                rai, deci = np.random.uniform(rmin, rmax), np.random.uniform(dmin, dmax)
                theta = np.deg2rad(90.0 - deci)
                phi = np.deg2rad(rai)
                pixels = hp.ang2pix(hp.get_nside(mask), theta, phi)
                mask_values =  mask[pixels]
                mask_values = np.nan_to_num(mask_values)
                if mask_values == 1:
                    coords = np.deg2rad(np.array((deci,rai)))
                    ypix,xpix = enmap.sky2pix(ymap.shape, ymap.wcs, coords)
                    values[n] = ymap[int(ypix),int(xpix)]
                    
                    counter.value += 1
                    sys.stdout.write(f"\rProcessed pixels: ({counter.value} / {N_total})")
                    sys.stdout.flush()
                    
                    break
    if use_pixels == False:
        return profiles
    else:
        return values

def init_background_worker(ymap_path, mask_path):
    global ymap
    global mask
    ymap = enmap.read_map(ymap_path)
    mask = hp.fitsfunc.read_map(mask_path)
def compute_background(ymap_path, mask_path, R_profiles, clusters = None, N_total = 100, Ncl = None, width = 0.6, 
                       n_pool = 1, ncores = 1, compute_per_cluster = False, use_pixels = True, N_realizations = None):
    ras_min, ras_max = [], []
    decs_min, decs_max = [], []
    if clusters is not None:
        for i in range(len(clusters)):
            c0 = clusters[i]
            rmin, rmax = np.min(c0.ra), np.max(c0.ra)
            dmin, dmax = np.min(c0.dec), np.max(c0.dec)
            ras_min.append(rmin)
            ras_max.append(rmax)
            decs_min.append(dmin)
            decs_max.append(dmax)
    rmin, rmax, dmin, dmax = np.min(ras_min), np.max(ras_max), np.min(decs_min), np.max(decs_max)
    if compute_per_cluster == True:
        for i in range(len(clusters)):
            ci = clusters[i]
            Ncl = len(ci.richness)
            N_realizations = N_total*Ncl
            print(ci.stats())
            print("Number of random realizations:", N_realizations)
            if ncores == 1:
                profiles = background_worker(N_total, Ncl, rmin, rmax, dmin, dmax, R_profiles, width, use_pixels)

            else:
                manager = Manager()
                counter = manager.Value("i", 0)
                N_base = N_realization // ncores
                N_remainder = N_realizations % ncores
                iter_per_core = [N_base + 1 if i < N_remainder else N_base for i in range(ncores)]
                print(iter_per_core)
                pool = Pool(ncores, initializer = init_background_worker, initargs = (ymap_path, mask_path))
                res_ = []
                for i in range(len(iter_per_core)):
                    res_.append(pool.apply_async(background_worker, 
                                args = (iter_per_core[i], Ncl, rmin, rmax, dmin, dmax, 
                                        R_profiles, width, use_pixels, N_total * Ncl, counter)))
                res = [r.get() for r in res_]
                profiles = [r for r in res]
                profiles = np.concatenate(profiles, axis = 0)
            background = np.mean(profiles, axis = 0)
            ci.background = background
            ci.random_profiles = profiles
            ci.save()
    else:
        Rbins, profile, sigma, counts  = radial_binning2(clusters[0].imap[0], R_profiles, width = width, full = True)
        if N_realizations is None:
            N_total = np.max(counts)
            if Ncl is None:
                Ncl = np.max([len(ci) for ci in clusters])
            N_realizations = N_total*Ncl
        print("Number of random realizations:", N_realizations)
        if ncores == 1:
            profiles = background_worker(N_total, Ncl, rmin, rmax, dmin, dmax, R_profiles, width, use_pixels)
        else:
            manager = Manager()
            counter = manager.Value("i", 0)
            N_base = N_realizations // ncores
            N_remainder = N_realizations % ncores
            iter_per_core = [N_base + 1 if i < N_remainder else N_base for i in range(ncores)]
            print(iter_per_core)
            pool = Pool(ncores, initializer = init_background_worker, initargs = (ymap_path, mask_path))
            res_ = []
            for i in range(len(iter_per_core)):
                res_.append(pool.apply_async(background_worker, 
                            args = (iter_per_core[i], Ncl, rmin, rmax, dmin, dmax, 
                                    R_profiles, width, use_pixels, N_realizations, counter)))
            res = [r.get() for r in res_]
            pool.close()
            pool.join()
            if use_pixels == False:
                profiles = [r for r in res]
                profiles = np.concatenate(profiles, axis = 0)
                output = profiles
            else:
                values = [r for r in res]
                values = np.concatenate(values, axis = 0)
                output = values
        if use_pixels == False:
            background = np.mean(profiles, axis = 0)
            background_std = np.std(profiles, axis = 0)
        else:
            background = np.mean(values)
            background_std = np.std(values)
        return background, background_std, output
def plot_mass_vs_richness():
    M200ctoM200m = generate_M200c2M200mInterpolator()
    M200mtoM200c = generate_M200m2M200cInterpolator()
    data_DES = np.loadtxt("cluster_masses.txt", skiprows = 2)
    richness_des, redshift_des, m200_des = data_DES.T
    data_TNG = np.loadtxt("tng_richness.txt")
    m200tng, redshift_tng, richness_tng = data_TNG.T
    m200 = np.logspace(13, 16, 150)
    lambda_true = np.linspace(10, 350, 120)
    zobs = np.mean(redshift_des)
    lambda_obs = np.linspace(20, 350, 100)
    lobs_grid, ltrue_grid = np.meshgrid(lambda_obs, lambda_true)
    ltrue_M_grid, M_grid = np.meshgrid(lambda_true, m200)
    M200m_grid = 10**M200ctoM200m((np.log10(M_grid), zobs))
    helpers = importlib.import_module("helpers")
    func = getattr(helpers, "P_lob_ltr")
    P_lobs_ltrue = func(lobs_grid, ltrue_grid, zobs)

    lambda_model = 30 * (M200m_grid / (3e14/0.67))**0.75 * ((1 + zobs)/(1 + 0.35))**(-0.3)
    sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + 0.25**2)
    P_ltrue_M_z = (1 / (ltrue_M_grid * sigma_model * np.sqrt(2 * np.pi)) * np.exp(-(np.log(ltrue_M_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2)))
    
    mdef = "200c"
    cosmo = ccl.CosmologyVanillaLCDM()
    hmf = ccl.halos.MassFuncTinker08(mass_def=mdef)
    dndlog10 = np.array([hmf(cosmo, mi, 1/(1 + zobs) ) for mi in m200])
    dndM = dndlog10/(np.log(10)*m200)
    P_joint = P_lobs_ltrue[None,:,:] * P_ltrue_M_z[:,:, None] 
    P_lobs_M = np.trapz(P_joint, x = lambda_true, axis = 1)

    P_lobs_M_tng = P_lobs_M
    P_lobs_M_des = P_lobs_M * dndM[:,None]


    fig = plt.figure(figsize = (20,10))
    gs = fig.add_gridspec(3,6, hspace = 0, wspace = 0)
    scatter_tng = fig.add_subplot(gs[1:3, 1:3])
    scatter_tng.tick_params(axis = "x", labelbottom = True, labeltop = False)
    scatter_tng.tick_params(axis = "y", labelleft = False, labelright = False)
    scatter_tng.set(xscale = "log", xlabel = r"$M_{200}[M_{\odot}]$")
    scatter_tng.set_ylim(20, 350)
    scatter_tng.set_xlim(10**(13.8), 1e16)
    M200_tng = np.logspace(13.8, 16, 100)
    Mcclintock19 = 10**(14.489)*(lambda_obs/40)**(1.356)*((1 + zobs)/(1 + 0.35))**(-0.3)
    Mcclintock19 = 10**(M200mtoM200c((np.log10(Mcclintock19), zobs)))
    scatter_tng.plot(Mcclintock19, lambda_obs, lw = 3, color = "darkblue", label = "Mcclintock et al 2019")
    
    M200m_tng = 10**M200ctoM200m((np.log10(M200_tng), zobs))
    lambda_Costanzi_tng = 1/0.67 * 30 *  (M200m_tng / (3e14/0.67))**0.75 * ((1 + zobs)/(1 + 0.35))**(-0.3)
    scatter_tng.plot(M200_tng, lambda_Costanzi_tng, color = "darkgreen", lw = 3)

    scatter_tng.contour(m200, lambda_obs, np.log10(P_lobs_M_tng.T), colors = "black", levels = 10)

    hist_richness_tng = fig.add_subplot(gs[1:3,0], sharey = scatter_tng)
    hist_richness_tng.set(xlabel = "N clusters", ylabel = r"$\lambda_{\mathrm{obs}}$")
    hist_mass_tng = fig.add_subplot(gs[0, 1:3], sharex = scatter_tng)
    hist_mass_tng.set(ylabel = "N clusters")
    hist_mass_tng.tick_params(axis = "x", labelbottom = False, labeltop = False)
    scatter_des = fig.add_subplot(gs[1:3, 3:5], sharey = scatter_tng)
    scatter_des.set(xscale = "log", xlabel = r"$M_{200}[M_{\odot}]$")
    scatter_des.set_ylim(20, 350)
    M200_des = np.logspace(14, 15.7, 100)
    scatter_des.plot(Mcclintock19, lambda_obs, lw = 3, color = "darkblue", label = "Mcclintock et al 2019")

    M200m_des = 10**M200ctoM200m((np.log10(M200_des), zobs))
    fig.savefig("M200cvsM200m.png")
    lambda_Costanzi_des = 1/0.67 * 30 * (M200m_des / (3e14/0.67))**0.75 * ((1 + zobs)/(1 + 0.35))**(-0.3)
    scatter_des.plot(M200_des, lambda_Costanzi_des, color = "darkgreen", lw = 3)

    scatter_des.contour(m200, lambda_obs, np.log10(P_lobs_M_des.T), colors = "black", levels = 10)

    hist_richness_des = fig.add_subplot(gs[1:3,5], sharey = scatter_des)
    hist_richness_des.set(xlabel = "N clusters")
    hist_mass_des = fig.add_subplot(gs[0, 3:5], sharex = scatter_des)
    max_lambda = 200

    scatter_tng.scatter(m200tng, richness_tng, s = 10, color = "purple", alpha = 0.7)
    hist_mass_tng.hist(m200tng, bins = np.logspace(13.8, 15.7, 50), color = "purple", alpha = 0.7, log = True, histtype = "step", lw = 3)
    hist_richness_tng.hist(richness_tng, bins = np.linspace(10, max_lambda, 50), color = "purple", alpha = 0.7, orientation = "horizontal", log = True, histtype = "step", lw = 3)
    scatter_des.scatter(m200_des, richness_des, s = 10, color = "red", alpha = 0.7)
    scatter_des.tick_params(axis = "x", labelbottom = True, labeltop = False)
    scatter_des.tick_params(axis = "y", labelleft = False, labelright = False)
    hist_mass_des.hist(m200_des, bins = np.logspace(14, 15.7, 50), color = "red", alpha = 0.7, log = True, histtype = "step", lw = 3)
    hist_mass_des.tick_params(axis = "x", labelbottom = False, labeltop = False)
    hist_mass_des.tick_params(axis = "y", labelleft = False, labelright = True)
    hist_richness_des.hist(richness_des, bins = np.linspace(10, max_lambda, 50), color = "red", alpha = 0.7, orientation = "horizontal", log = True, histtype = "step", lw = 3)
    hist_richness_des.tick_params(axis = "y", labelleft = False, labelright = False)
    scatter_des.set_xlim(left = 1e14)
    scatter_tng.text(0.05, 0.95, "TNG300-3", color = "purple", fontsize = 30, ha = "left", va = "top", transform = scatter_tng.transAxes)
    scatter_des.text(0.05, 0.95, "DES-Y3 RedMaPPer", color = "red", fontsize = 30, ha = "left", va = "top", transform = scatter_des.transAxes)

    scatter_tng.plot([],[], lw = 3, ls = "solid", label = "Costanzi et al 2019", color = "darkgreen")
    scatter_tng.plot([],[], lw = 3, ls = "--", label = "Probability distribution", color = "black")
    scatter_tng.legend(frameon = True, fontsize = 10, loc = "upper right")
    return fig 


def nfw_mass(r, rs, rho_s):
    x = r / rs
    return 4*np.pi*rho_s*rs**3 * (np.log(1+x) - x/(1+x))


from scipy.optimize import brentq

def generate_M200m2M200cInterpolator():
    M200m = np.logspace(13, 16, 100)
    z = np.linspace(1e-3, 1.5, 100)
    M200m_grid, z_grid = np.meshgrid(M200m, z)
    rho_c = planck18.critical_density(z_grid).to(u.Msun/ (u.Mpc**3)).value
    rho_m = planck18.Om(z_grid)*rho_c

    R200m = (M200m_grid / (4*np.pi/3 * 200*rho_m))**(1/3)
    c200m = np.asarray([concentration.concentration(
        M200m, '200m', zi, model = 'bhattacharya13') for zi in z])
    rs = R200m / c200m
    f = np.log(1+c200m) - c200m/(1+c200m)
    rho_s = M200m / (4*np.pi*rs**3 * f)
    M200c = np.zeros(M200m_grid.shape)
    for i in range(len(z)):
        for j in range(len(M200m)):
            def eq(R):
                return nfw_mass(R, rs[i,j], rho_s[i,j])/(4*np.pi*R**3/3) - 200*rho_c[i,j]
            R200c = brentq(eq, 0.05*R200m[i,j], R200m[i,j])
            M200c[i,j] = nfw_mass(R200c, rs[i,j], rho_s[i,j])
    fM200mtoM200c = RegularGridInterpolator((np.log10(M200m), z), np.log10(M200c.T), bounds_error = False)
    return fM200mtoM200c

def generate_M200c2M200mInterpolator():
    M200c = np.logspace(13, 16, 150)
    z = np.linspace(1e-3, 1.5, 100)
    M200c_grid, z_grid = np.meshgrid(M200c, z)
    rho_c = planck18.critical_density(z_grid).to(u.Msun/ (u.Mpc**3)).value
    rho_m = planck18.Om(z_grid)*rho_c
    R200c = (M200c_grid / (4*np.pi/3 * 200*rho_c))**(1/3)
    c200c = np.asarray([concentration.concentration(
        M200c, '200c', zi, model = 'bhattacharya13') for zi in z])
    rs = R200c / c200c
    f = np.log(1+c200c) - c200c/(1+c200c)
    rho_s = M200c / (4*np.pi*rs**3 * f)
    M200m = np.zeros(M200c_grid.shape)
    for i in range(len(z)):
        for j in range(len(M200c)):
            def eq(R):
                return nfw_mass(R, rs[i,j], rho_s[i,j])/(4*np.pi*R**3/3) - 200*rho_m[i,j]
            R200m = brentq(eq, 0.05*R200c[i,j], 5*R200c[i,j])
            M200m[i,j] = nfw_mass(R200m, rs[i,j], rho_s[i,j])
    fM200ctoM200m = RegularGridInterpolator((np.log10(M200c), z), np.log10(M200m.T), bounds_error = True)
    return fM200ctoM200m

from scipy.optimize import curve_fit

def Y500_model(M500, a, b):
    return a*(M500 / 1e14)**b
def smooth_broken_power_law(x, A, xb, alpha1, alpha2, delta):
    x = np.asarray(x)
    return A * (x/xb)**alpha1 * (1 + (x/xb)**(1/delta))**((alpha2-alpha1)*delta)
def Y500vsM500(clusters):
    M500arnaud10, Y500arnaud10, Y500arnaud10e, M500ade11, Y500ade11, Y500ade11e = load()
    M500tng, Y500tng, M200tng, ztng, _ = loadtng()
    
    data_DES = np.loadtxt("cluster_masses.txt", skiprows = 2)
    richness_des, redshift_des, m200_des = data_DES.T

    fM200toM500, _ = create_mass_interpolator()

    M500_des = 10**fM200toM500((np.log10(m200_des), redshift_des))
    Y500_des = []
    Y500_err = []
    try:
        data_y = np.loadtxt("Y500_des.txt")
        Y500_des, Y500_err = data_y.T
    except:
        szmaps = np.concatenate([ci.imap for ci in clusters])
        for i in tqdm(range(len(M500_des))):
            m500 = M500_des[i]
            zi = redshift_des[i]
            dA = planck18.angular_diameter_distance(zi).value
            rho_c = planck18.critical_density(zi).to(u.Msun/ (u.Mpc**3)).value
            R500 = (m500 / (4/3 * np.pi * 500 * rho_c))**(1/3)
            theta500 = (R500/dA)*(180/np.pi)*60
            signal = szmaps[i]
            patch_size = 0.6 * 60
            pix_size = patch_size / signal.shape[0]
            pix_size_rad = pix_size * np.pi / (180 * 60)
            x, y = np.indices(signal.shape)
            x -= signal.shape[0]//2
            y -= signal.shape[1]//2
            r = np.sqrt((x*pix_size)**2 + (y*pix_size)**2)

            mask = r < theta500
            y500 = np.sum(signal[mask]) * pix_size_rad**2
            Y500_des.append(y500 * dA**2)
            sigma_y = np.std(signal[~mask])
            Y500_err.append(sigma_y * np.sqrt(mask.sum()) * pix_size_rad**2 * dA**2)

        data = np.column_stack((Y500_des, Y500_err))
        np.savetxt("Y500_des.txt",data)
    m500_arr = np.logspace(13, 15.6, 100)

    fig, ax = plt.subplots(figsize = (14,14))
    ax.scatter(M500_des, Y500_des*planck18.efunc(redshift_des)**(-3/2), color = "red", alpha = 0.3, s = 3, label = "DES-Y3 RedMaPPer")
    pars, cov = curve_fit(Y500_model, M500_des, Y500_des*planck18.efunc(redshift_des)**(-3/2), p0 = (1e-5, 5/3))
    pars_errs = np.sqrt(np.diag(cov))
    upper_bound = pars+pars_errs
    lower_bound = pars-pars_errs
    Y500_model_des_upper_bound = Y500_model(m500_arr, *upper_bound)
    Y500_model_des_lower_bound = Y500_model(m500_arr, *lower_bound)

    ax.fill_between(m500_arr, Y500_model_des_lower_bound, Y500_model_des_upper_bound, color = "red", alpha = 0.2)

    txt = ax.text(1e15, 6e-6, r"slope = $%.2f\pm %.2f$" % (pars[1], pars_errs[1]), color = "red", rotation = 25, fontsize = 20, ha = "center", va = "center")
    txt.set_path_effects([
        pe.Stroke(linewidth=2, foreground="black"),
        pe.Normal()
    ])

    line, = ax.plot(m500_arr, Y500_model(m500_arr, *pars), color = "red", alpha = 0.8, linewidth = 2)
    line.set_path_effects([
        pe.Stroke(linewidth=4, foreground="black"),
        pe.Normal()                                  
    ])

    ax.errorbar(M500ade11, Y500ade11, yerr = Y500ade11e, color = "darkblue", markersize = 5, capsize = 3, fmt = "o", label = "Ade et al 2011")
    ax.errorbar(M500arnaud10, Y500arnaud10, yerr = Y500arnaud10e, color = "darkgreen", markersize = 5, capsize = 3, fmt = "o", label = "Arnaud et al 2010")
    
    M500aa, Y500aa = np.concatenate((M500arnaud10, M500ade11)), np.concatenate((Y500arnaud10, Y500ade11))
    Y500aa_errs = np.concatenate((Y500arnaud10e, Y500ade11e))
    pars, cov = curve_fit(Y500_model, M500aa, Y500aa, sigma = Y500aa_errs)
    pars_errs = np.sqrt(np.diag(cov))
    upper_bound_pars = pars+pars_errs
    lower_bound_pars = pars-pars_errs
    Y500_model_arnaud_ade = Y500_model(m500_arr, *pars)
    Y500_model_arnaud_ade_upper_bound = Y500_model(m500_arr, *upper_bound_pars)
    Y500_model_arnaud_ade_lower_bound = Y500_model(m500_arr, *lower_bound_pars)


    line, = ax.plot(m500_arr, Y500_model(m500_arr, *pars), color = "darkgreen", lw = 3)
    line.set_path_effects([
        pe.Stroke(linewidth=4, foreground="black"), 
        pe.Normal()
    ])
    ax.fill_between(m500_arr, Y500_model_arnaud_ade_lower_bound, Y500_model_arnaud_ade_upper_bound, color = "darkgreen", alpha = 0.2)
    ax.scatter(M500tng, Y500tng*planck18.efunc(ztng)**(-3/2)/(0.67**2), color = "purple", s = 5, alpha = 0.6, edgecolor = "black", label = "TNG300-3")
    ax.loglog()
    ax.set_xlim(left = 1e14)
    ax.set_ylim(bottom = 1e-8)
    txt = ax.text(1e15, 4e-4, r"slope = $%.2f\pm %.2f$" % (pars[1], pars_errs[1]), color = "darkgreen", rotation = 25, fontsize = 20, ha = "center", va = "center")
    txt.set_path_effects([
        pe.Stroke(linewidth=2, foreground="black"),
        pe.Normal()
    ])
    ax.set(xlabel = r"$M_{200} [M_{\odot}]$", ylabel = "$Y500 \times E(z)^{-3/2}[Mpc^2]$")
    ax.legend(fontsize = 20, frameon = False, loc = "lower right")
    fig.savefig("M500vsY500.png")

def plot_rs(clusters, c200_0, c200_M, c200_z, M0, z0, c200c2c200m):
    R = clusters[0].R
    profiles = np.array([ci.mean_profile for ci in clusters])
    errs = np.array([np.sqrt(np.diag(ci.cov)) for ci in clusters])
    masses = np.array([np.mean(10**(14.489) * ((ci.richness)/40)**(1.356)*((1 + ci.z)/(1 + 0.35))**(-0.3)) for ci in clusters])
    redshifts = np.array([np.mean(ci.z) for ci in clusters])
    if c200c2c200m:
        c200_0 = c200_0 * planck18.Om(redshifts)**(-1/3)
    rho_c = planck18.critical_density(redshifts).to(u.Msun / (u.Mpc**3))
    rho_m = planck18.Om(redshifts) * rho_c
    masses = np.array(masses)*u.Msun
    R200m = ((3 * masses / (4 * np.pi * 200 * rho_m))**(1/3)).to(u.Mpc).value

    c200m = c200_0 * (masses/M0)**(c200_M)*((1 + redshifts)/(1 + z0))**(c200_z)
    rs = (R200m/c200m).value
    Da = planck18.angular_diameter_distance(redshifts).to(u.Mpc).value
    rs_arcmin = (rs * planck18.arcsec_per_kpc_comoving(redshifts).to(u.arcmin/u.Mpc)).value
    R200m_arcmin = (R200m * planck18.arcsec_per_kpc_comoving(redshifts).to(u.arcmin/u.Mpc)).value
    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    for i in range(len(clusters)):
        ci = clusters[i]
        rmin, rmax = ci.richness_bin
        zmin, zmax = ci.redshift_bin
        row_indx = 0 if zmin < 0.3 else 1
        profile = profiles[i] - ci.background if hasattr(ci, "background") else profiles[i]
        errors = np.sqrt(errs[i]**2 + np.abs(ci.background)**2) if hasattr(ci, "background") else errs[i]
        ax[row_indx, i//2].errorbar(R, profile, yerr = errors, fmt = "-o", color = "black", linewidth = 2, markersize = 5)
        ax[row_indx, i//2].axvline(rs_arcmin[i], ls = "--", lw = 3, color = "darkred", label = r"$r_s$ [arcmin]")
        ax[row_indx, i//2].axvline(R200m_arcmin[i], ls = "--", lw = 3, color = "darkgreen", label = r"$R_{200m}$ [arcmin]")
        if i//2 == 0:
            ax[row_indx, i//2].set(ylabel = "R (arcmin)")
        if row_indx == 1:
            ax[row_indx, i//2].set(xlabel = "R (arcmin)")
        if row_indx == 0:
            ax[row_indx, i//2].set_title("$\mathbf{\lambda \in [%i, %i]}$" % (rmin, rmax),
                                fontsize = 24, fontweight = "bold")
        if i//2 == 0:
            ax[row_indx, i//2].text(-0.3,0.5,"$\mathbf{z\in[%.2f, %.2f]}$" % (zmin, zmax), ha = "center", va = "center", 
                                    transform = ax[row_indx, i//2].transAxes, fontsize = 24, fontweight = "bold",
                                    rotation = 90)
    ax[0,0].legend(fontsize = 12)
    return fig




from colossus.halo import mass_defs

def plot_concentration():
    M500c = 10**(14.35)
    z = np.linspace(0.1, 0.7, 100)
    c500c_arnaud = 1.177
    c500c_lim = 0.887 * (M500c/1e14)**(0.735)

    r_arnaud = np.array([mass_defs.changeMassDefinition(M500c, c500c_arnaud , zi, '500c', '200m') for zi in z])
    M200m, R200m, c200m_arnaud = r_arnaud.T
    r_lim = np.array([mass_defs.changeMassDefinition(M500c, c500c_lim, zi, '500c', '200m') for zi in z])

    M200m, R200m, c200m_lim = r_lim.T
    
    c_tng = 10**(0.41)*(M500c/10**(14.35))**(-0.0)*((1 + z)/(1 + 0.47))**(-0.74)
    c200m_u = 10**(0.40)*(M500c/10**(14.35))**(-0.08)*((1 + z)/(1 + 0.47))**(-2.26)
    c200m_u_lower = 10**(0.40)*(M500c/10**(14.35))**(-0.19 - 0.04)*((1 + z)/(1 + 0.47))**(-2.26 - 0.08)
    c200m_u_upper = 10**(0.40)*(M500c/10**(14.35))**(-0.19 + 0.04)*((1 + z)/(1 + 0.47))**(-2.26 + 0.08)


    c_icm = np.vstack([c200m_arnaud, c200m_lim, c200m_u, c200m_u_lower, c200m_u_upper]).T
    contour = np.max(c_icm, axis = 1)
    c_duffy08 = np.array([concentration.concentration(M500c, '200m', zi, model = "duffy08") for zi in z])
    c_bhattacharya13 = np.array([concentration.concentration(M500c, '200m', zi, model = "bhattacharya13") for zi in z])

    fig = plt.figure(figsize = (10, 8))
    ax = plt.axes()
    ax.fill_between(z, contour + 0.1, 7, color = "black", alpha = 0.1)
    ax.fill_between(z, 0, contour + 0.1, color = "purple", alpha = 0.1)
    ax.plot(z, c200m_arnaud, color = "darkgreen", alpha = 0.5, lw = 4, ls = "--")
    ax.plot(z, c200m_lim, color = "darkblue", lw = 4, alpha = 0.5, ls = "-.")
    ax.plot(z, c_duffy08, color = "black", lw = 4, ls = "--")
    ax.plot(z, c_bhattacharya13, color = "black", lw = 4, ls = ":")
    ax.plot(z, c_tng, color = "darkorange", lw = 4, ls = (0, (3, 1, 1, 1, 1, 1)))
    ax.plot(z, c200m_u, label = r"This work", color = "black", lw = 4)
    ax.fill_between(z, c200m_u_lower, c200m_u_upper, alpha = 0.5, color = "grey", edgecolor = "black")
    ax.set(xlabel = "redshift", ylabel = r"$c_{200,m}$")
    ax.set_xlim(0.1, 0.7)
    ax.set_ylim(0.5, 7)

    ax.text(0.05, 0.05, "ICM", fontsize = 20, transform = ax.transAxes, ha = "left", va = "bottom", fontweight = "bold", color = "purple")
    ax.text(0.95, 0.95, "Dark Matter", fontsize = 20, transform = ax.transAxes, ha = "right", va = "top", fontweight = "bold")
    ax.text(0.6, 2.95, "Lim et al 2021", fontsize = 14, color = "darkblue", ha = "center", va = "center", fontweight = "bold", rotation = -3)
    ax.text(0.2, 2.6, "Arnaud et al 2010", fontsize = 14, color = "darkgreen", ha = "center", va = "center", fontweight = "bold", rotation = -5)
    ax.text(0.3, 3, "TNG300-3", fontsize = 14, color = "darkorange", ha = "center", va = "center", fontweight = "bold", rotation = -5)
    ax.text(0.2, 5.5, "Duffy et al 2008", fontsize = 14, color = "black", ha = "center", va = "center", fontweight = "bold", rotation = -13.5)
    ax.text(0.4, 5.5, "Bhattacharya et al 2013", fontsize = 14, color = "black", ha = "center", va = "center", fontweight = "bold", rotation = -14)
    ax.legend(frameon = False, loc = "lower right")
    fig.savefig("concentration_evolution.png")


def test_single_profile(Mobs, zobs, R, params, model, cosmo, mdef = "200m", rbins = 50, projected = True):
    
    redshift_pivot = 0.47
    richness_pivot = 32

    mfunc = ccl.halos.MassFuncTinker10(mass_def = mdef)
    bias = ccl.halos.HaloBiasTinker10(mass_def = mdef)

    M2halo = np.logspace(13, 16, 150)
    r2halo = np.logspace(-2, 2, 150)
    k2halo = np.logspace(-4, 3, 150)

    zgrid, mgrid = np.meshgrid(zobs, Mobs)

    R_Mpc = ((R*(np.pi/180 / 60))[:,None] * (ccl.angular_diameter_distance(cosmo, 1/(zobs + 1)) * (1 + zobs)[None,:]))[:,None,:] + 0*mgrid[None,:,:]

    bM2halo = np.array([bias(cosmo, M2halo, 1/(1 + zi)) for zi in zobs])
    bMobs = np.array([bias(cosmo, Mobs, 1/(1 + zi)) for zi in zobs])
    Pk = np.array([ccl.linear_matter_power(cosmo, k2halo, 1/(1 + zi)) for zi in zobs])
    
    dndM2halo = np.array([mfunc(cosmo, M2halo, 1/(1 + zi)) for zi in zobs])
    dndM2halo = dndM2halo / (np.log(10) * M2halo)

    Rgrid, M2halo_grid, z2halo_grid = np.meshgrid(r2halo, M2halo, zobs, indexing = "ij")
    PRMz = model(Rgrid, 10, M2halo_grid, z2halo_grid, params, rbins = rbins, projected = projected)
    if projected == True:                
        weighted_two_halo_profiles = compute_two_halo_term(
            PRMz, Rgrid, r2halo, dndM2halo, bM2halo,
            bMobs, Pk, R_Mpc, k2halo, M2halo, np.ones((1,1, len(Mobs), len(zobs))), 0
        )
    else:
        weighted_two_halo_profiles = compute_two_halo_term_3d(
                        PRMz, Rgrid, r2halo, dndM2halo, bM2halo,
            bMobs, Pk, R_Mpc, k2halo, M2halo, np.ones((1,1, len(Mobs), len(zobs))), 0)
        Rlos = np.logspace(-2, 2, rbins)
        weighted_two_halo_profiles = 2*trapz_axis0(weighted_two_halo_profiles, Rlos)

    p2halo = weighted_two_halo_profiles
    
    p1halo = model(R_Mpc, 10, mgrid, zgrid, params, rbins = rbins)

    return p1halo, p2halo

from scipy.integrate import simpson as simp

def test_two_halo_term(c, two_halo_profile, params, cosmo, rbins = 100, eval_mass = True, 
                        delta = 200, background = "matter", float_dtype = np.float32,
                        nm2h = 50, nk = 70, nr = 60, log = False):

    h = HankelSphericalTransform(N=1000, h=0.001)

    R = c.R
    R = np.linspace(0, 100, 100)
    PllM = c.PllM
    lambda_obs = c.lambda_obs
    lambda_true = c.lambda_true

    richness_pivot = 32
    redshift_pivot = 0.47

    zmin, zmax = np.min(c.z), np.max(c.z)

    Nm, Nz = np.shape(PllM)[2], np.shape(PllM)[3]

    Mobs = c.M
    zobs = c.z_arr

    lambda_grid, mgrid, zgrid = np.meshgrid(lambda_true, Mobs, zobs, indexing = "ij")
    R_Mpc = (R[:,None,None,None]*np.pi/180/60)*(ccl.angular_diameter_distance(cosmo, 1/(1 + zobs)) * (1 + zobs))[None,None,None,:] + 0*mgrid[None,:,:,:]

    mdef = f"{int(delta)}c" if background == "critical" else f"{int(delta)}m"
    a = 1 / (1 + zobs)

    mfunc = ccl.halos.MassFuncTinker10(mass_def = mdef) #mass function from Tinker et al 2010

    a = 1 / (1 + zobs)

    dndM = c.dndM
    dV = c.dV

    dndM2halo = c.dndM2halo
    bh = c.bh
    bM = c.bM
    Pk = c.Pk
    sin_term = c.sin_term
    Rgrid = c.Rgrid
    z2halo_grid = c.z2halo_grid
    M2halo_grid = c.M2halo_grid
    k2halo = c.k2halo
    R2halo = c.R2halo
    M_arr2halo = c.M_arr2halo

    PRMz = two_halo_profile(Rgrid, 10, M2halo_grid, z2halo_grid, params, rbins = rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot,
                            projected = False)
    p2halo = p2h(k2halo, M_arr2halo, zobs, Mobs, R_Mpc, R2halo, PRMz, dndM2halo, Pk, bM, bh, h)
    weights = PllM * dndM.T[None,None,:,:] * dV[None,None,None,:]

    weighted_two_halo_profiles = weights[None,:,:,:,:] * p2halo[:,None,None,:,:]
    norm = simp(simp(simp(simp(weights, axis = 0, x = c.lambda_true), axis = 0, x = c.lambda_obs), axis = 0, x = Mobs), axis = 0, x = zobs)

    P2halo = trapz_axis1(trapz_axis1(weighted_two_halo_profiles, c.lambda_true), c.lambda_obs)

    mean_p2halo_profile = trapz_axis1(trapz_axis1(P2halo, Mobs), zobs) / norm
    one_halo_profiles = two_halo_profile(R_Mpc, 10, mgrid, zgrid, params, rbins = rbins)

    weighted_one_halo_profiles = weights[None,:,:,:,:] * one_halo_profiles[:,:,None,:,:]

    mean_p1halo_profile = simp(simp(simp(simp(weighted_one_halo_profiles, x = c.lambda_true, axis = 1), x = c.lambda_obs, axis = 1), x = Mobs, axis = 1), x = zobs, axis = 1)/norm    

    fwhm = float(1.6)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dr = (R[-1] - R[0]) / (len(R) - 1)
    sigma_pix = sigma / dr
    mean_p1halo_profile = gaussian_filter1d(np.float64(mean_p1halo_profile), sigma=np.float64(sigma_pix))
    mean_p2halo_profile = gaussian_filter1d(np.float64(mean_p2halo_profile), sigma=np.float64(sigma_pix))

    fig, ax = plt.subplots(figsize = (6, 8))
    ax.loglog(R, mean_p1halo_profile, label = "1h", ls = "--", lw = 3, color = "black")
    ax.loglog(R, mean_p2halo_profile, label = "2h", ls = "dotted", lw = 3, color = "black")
    ax.loglog(R, mean_p1halo_profile + mean_p2halo_profile, label = "total", lw = 3, color = "black")
    
    #ax.errorbar(R, c.mean_profile - c.background, yerr = np.sqrt(c.error_in_mean**2 + np.mean(c.background_std)**2) , fmt = "-o", color = "purple")
    ax.set_ylim(1e-9, 1e-4)
    ax.grid(True)
    #ax.set(xscale = "linear")
    fig.savefig("two_halo_term_test.png")

    return R, mean_p1halo_profile, mean_p2halo_profile








def compare_parameters():

    M = 3e14
    M0 = 10**(14.35)
    z = 0.47
    z0 = 0.40


    om    = z2Om(z)
    rho   = z2rho(z)
    E     = z2E(z)

    fig = plt.figure(figsize = (18, 7))
    gs = fig.add_gridspec(1,3, wspace = 0, hspace = 0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex = ax1, sharey = ax1)
    ax3 = fig.add_subplot(gs[2], sharex = ax1, sharey = ax1)

    R = np.logspace(-1, np.log10(3), 100)   
    R200 = (M / (4.0 * np.pi / 3.0 * 200.0 * om * rho)) ** (1.0 / 3.0)
    gammas = np.linspace(-0.27 - 3*0.21, -0.27 + 3*0.14, 100)
    alphas = np.linspace(2.59 - 3*0.36, 2.59 + 3*0.36, 100)
    cs = np.linspace(0.37 - 3*0.04, 0.37 + 3*0.05, 100)

    profiles_gamma = np.zeros((len(gammas), len(R)))
    profiles_alpha = np.zeros((len(alphas), len(R)))
    profiles_c = np.zeros((len(cs), len(R)))

    P0 = 10**(9.08)*((M/M0))**(0.79)*E**(2.67)
    gamma0 = 10**(-0.27)*(M/M0)**(0.33)*((1 + z)/(1 + z0))**(1.55)
    c0 = 10**(0.37)*(M/M0)**(-0.07)*((1 + z)/(1 + z0))**(-1.73)

    beta = 4.13

    for i in range(len(gammas)):
        gamma = 10**(gammas[i])*(M/M0)**(0.33)*((1 + z)/(1 + z0))**(1.55)
        alpha = 2.59
        c = 10**(0.37)*(M/M0)**(-0.07)*((1 + z)/(1 + z0))**(-1.73)
        rs = R200 / c
        x = R / rs
        p = ycompton_factor * P0/(x**gamma * (1 + x**alpha)**((beta - gamma)/alpha))
        profiles_gamma[i] = p
    cm = plt.cm.Reds
    norm = plt.Normalize(vmin = np.min(gammas), vmax = np.max(gammas))
    for i in range(len(gammas)):
        ax1.loglog(R, profiles_gamma[i], color = cm(norm(gammas[i])))

    sm = plt.cm.ScalarMappable(cmap = cm, norm = norm)
    cbar = plt.colorbar(sm, cax = fig.add_axes([0.125, 0.8785, 0.2585, 0.02]), orientation = "horizontal")
    cbar.set_label(r"$\mathbf{\log_{10}{\gamma_0}}$", fontsize = 20, fontweight = "bold")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    for i in range(len(alphas)):
        gamma = 10**(-0.27)*(M/M0)**(0.33)*((1 + z)/(1 + z0))**(1.55)
        alpha = alphas[i]
        c = 10**(0.37)*(M/M0)**(-0.07)*((1 + z)/(1 + z0))**(-1.73)
        rs = R200 / c
        x = R / rs
        p = ycompton_factor * P0/(x**gamma * (1 + x**alpha)**((beta - gamma)/alpha))
        profiles_alpha[i] = p
    cm = plt.cm.Blues
    norm = plt.Normalize(vmin = np.min(alphas), vmax = np.max(alphas))
    for i in range(len(gammas)):
        ax2.loglog(R, profiles_alpha[i], color = cm(norm(alphas[i])))
    sm = plt.cm.ScalarMappable(cmap = cm, norm = norm)
    cbar = plt.colorbar(sm, cax = fig.add_axes([0.3835, 0.8785, 0.2585, 0.02]), orientation = "horizontal")
    cbar.set_label(r'$\mathbf{\alpha}$', fontsize = 20, fontweight = "bold")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    for i in range(len(cs)):
        gamma = 10**(-0.27)*(M/M0)**(0.33)*((1 + z)/(1 + z0))**(1.55)
        alpha = 2.59
        c = 10**(cs[i])*(M/M0)**(-0.07)*((1 + z)/(1 + z0))**(-1.73)
        R200 = (M / (4.0 * np.pi / 3.0 * 200.0 * om * rho)) ** (1.0 / 3.0)
        rs = R200 / c
        x = R / rs
        p = ycompton_factor * P0/(x**gamma * (1 + x**alpha)**((beta - gamma)/alpha))
        profiles_c[i] = p
    cm = plt.cm.Greens
    norm = plt.Normalize(vmin = np.min(cs), vmax = np.max(cs))
    for i in range(len(gammas)):
        ax3.loglog(R, profiles_c[i], color = cm(norm(cs[i])))
    sm = plt.cm.ScalarMappable(cmap = cm, norm = norm)
    cbar = plt.colorbar(sm, cax = fig.add_axes([0.64225, 0.878, 0.2585, 0.02]), orientation = "horizontal")
    cbar.set_label("$\mathbf{\log_{10}{c_{200,0}}}$", fontsize = 20)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    gamma0 = 10**(-0.27)*(M/M0)**(0.33)*((1 + z)/(1 + z0))**(1.55)
    c0 = 10**(0.37)*(M/M0)**(-0.07)*((1 + z)/(1 + z0))**(-1.73)

    R200 = (M / (4.0 * np.pi / 3.0 * 200.0 * om * rho)) ** (1.0 / 3.0)
    rs = R200 / c0
    x = R / rs
    alpha0 = 2.59
    p0 = ycompton_factor * P0/(x**gamma0 * (1 + x**alpha0)**((beta - gamma0)/alpha0))



    ax1.loglog(R, p0, color = "black", lw = 3, ls = "--")
    ax2.loglog(R, p0, color = "black", lw = 3, ls = "--")
    ax3.loglog(R, p0, color = "black", lw = 3, ls = "--", label = "best-fitting")

    ax1.tick_params(right = False)
    ax2.tick_params(left=False, labelleft=False, right = False)
    ax3.tick_params(left=False, labelleft=False)
    ax1.tick_params(top=False)
    ax2.tick_params(top=False)
    ax3.tick_params(top=False)


    ax1.set(xlabel = "R (Mpc)", ylabel = r'ycompton profile')
    ax2.set(xlabel = "R (Mpc)")
    ax3.set(xlabel = "R (Mpc)")
    fig.savefig("parameters_evolution.png")


from collections import defaultdict

def generate_radio_map(
        c,
        vlass,
        f90map,
        ivar,
        uK2mJy=0.016869897065456527,
        radius_arcmin=5,
        patch_radius_arcmin=6,
        pixsize_arcmin=0.5,
        fwhm_arcmin=1.6,
        outfile="/data2/javierurrutia/szeffect/data/VLASS/radio_mapf90.fits",
    ):


    ra_clusters = c.ra
    dec_clusters = c.dec

    rmin, rmax = ra_clusters.min(), ra_clusters.max()
    dmin, dmax = dec_clusters.min(), dec_clusters.max()

    ra_radio = vlass["RA"].to_numpy()
    dec_radio = vlass["DEC"].to_numpy()

    ra_radio[ra_radio > 180] -= 360

    mask = (
        (ra_radio >= rmin)
        &
        (ra_radio <= rmax)
        &
        (dec_radio >= dmin)
        &
        (dec_radio <= dmax)
    )

    ra_radio = ra_radio[mask]
    dec_radio = dec_radio[mask]

    coords_clusters = SkyCoord(
        ra_clusters*u.deg,
        dec_clusters*u.deg
    )

    coords_radio = SkyCoord(
        ra_radio*u.deg,
        dec_radio*u.deg
    )

    idx_radio, idx_cluster, _, _ = coords_clusters.search_around_sky(
        coords_radio,
        radius_arcmin*u.arcmin
    )

    cluster_sources = defaultdict(list)

    for iradio, icluster in zip(idx_radio, idx_cluster):
        cluster_sources[icluster].append(iradio)


    radio_map = enmap.zeros(f90map.shape, f90map.wcs)

    radius = np.deg2rad(patch_radius_arcmin/60.)

    for icluster, src in tqdm(cluster_sources.items()):

        pos = np.deg2rad([
            dec_clusters[icluster],
            ra_clusters[icluster]
        ])

        patch = reproject.thumbnails(
            f90map,
            pos,
            r=radius
        )

        ivar_patch = reproject.thumbnails(
            ivar,
            pos,
            r=radius
        )

        ny, nx = patch.shape

        xpix = np.empty(len(src))
        ypix = np.empty(len(src))

        for i, s in enumerate(src):

            coords = np.deg2rad([
                dec_radio[s],
                ra_radio[s]
            ])

            yp, xp = patch.sky2pix(coords)

            xpix[i] = xp
            ypix[i] = yp

        fluxes, amps, cov, model, residual = fit_radio_sources(
            np.asarray(patch),
            np.asarray(ivar_patch),
            xpix,
            ypix,
            pixsize_arcmin=pixsize_arcmin,
            fwhm_arcmin=fwhm_arcmin,
            uK2mJy=uK2mJy,
        )

        for flux, s in zip(fluxes, src):

            coords = np.deg2rad([
                dec_radio[s],
                ra_radio[s]
            ])

            yp, xp = enmap.sky2pix(
                f90map.shape,
                f90map.wcs,
                coords
            )

            yp = int(np.round(yp))
            xp = int(np.round(xp))

            if (
                0 <= yp < radio_map.shape[-2]
                and
                0 <= xp < radio_map.shape[-1]
            ):
                radio_map[yp, xp] += flux

    sigma = np.deg2rad(
        (fwhm_arcmin/60.)
        /
        (2*np.sqrt(2*np.log(2)))
    )

    radio_map = enmap.smooth_gauss(
        radio_map,
        sigma
    )

    enmap.write_map(
        outfile,
        radio_map,
        allow_modify=True
    )

    return radio_map

def fit_radio_sources(
    patch_f90,
    ivar_patch,
    xpix,
    ypix,
    pixsize_arcmin=0.5,
    fwhm_arcmin=1.6,
    uK2mJy=0.016869897065456527,
    fit_background=True,
    ):

    ny, nx = patch_f90.shape

    yy, xx = np.indices((ny, nx))

    sigma_pix = (
        fwhm_arcmin /
        (2*np.sqrt(2*np.log(2))) /
        pixsize_arcmin
    )

    nsrc = len(xpix)

    cols = []

    for xc, yc in zip(xpix, ypix):

        beam = np.exp(
            -((xx-xc)**2 + (yy-yc)**2) /
            (2*sigma_pix**2)
            )

        beam = np.nan_to_num(beam)
        beam /= beam.max()

        beam = np.ones_like(beam)
        cols.append(beam.ravel())

    if fit_background:
        cols.append(np.ones(nx*ny))

    M = np.column_stack(cols)

    d = patch_f90.ravel()

    w = ivar_patch.ravel()

    good = (
        np.isfinite(d)
        &
        np.isfinite(w)
        &
        (w > 0)
    )

    M = M[good]

    d = d[good]

    w = w[good]

    sw = np.sqrt(w)

    Mw = M * sw[:,None]

    dw = d * sw

    pars, residuals, rank, s = np.linalg.lstsq(
        Mw,
        dw,
        rcond=None
    )


    A = Mw.T @ Mw

    cov = np.linalg.pinv(A)

    amplitudes = pars[:nsrc]

    fluxes = amplitudes * uK2mJy

    model = (M @ pars)

    residual = d - model

    print(fluxes)
    return (
        fluxes,
        amplitudes,
        cov,
        model.reshape(-1),
        residual.reshape(-1)
    )

def cross_with_VLASS(c, vlass, szmaps, n_cores = 30, N_bootstrap = 100):

    ra_clusters, dec_clusters = c.ra, c.dec
    rmin, rmax = ra_clusters.min(), ra_clusters.max()
    dmin, dmax = dec_clusters.min(), dec_clusters.max()
    ra_radio, dec_radio = vlass["RA"].to_numpy(), vlass["DEC"].to_numpy()
    radio_flux = vlass["Total_flux"].to_numpy()

    ra_radio[ra_radio > 180] = ra_radio[ra_radio > 180] - 360

    mask = np.where((ra_radio >= rmin) & (ra_radio <= rmax) &
                                (dec_radio >= dmin) & (dec_radio <= dmax))
    ra_radio = ra_radio[mask]
    dec_radio = dec_radio[mask]
    radio_flux = radio_flux[mask]

    coords_vlass = SkyCoord(ra = ra_radio * u.deg, dec = dec_radio * u.deg, unit = "deg")

    coords_clusters = SkyCoord(ra = ra_clusters * u.deg, dec = dec_clusters * u.deg, unit = "deg")

    idx_radio, idx_clusters, d2d, d3d = coords_clusters.search_around_sky(coords_vlass, 10 * u.arcmin)

    radio_clusters = np.unique(idx_clusters)

    matched_sources_ra = ra_radio[idx_radio]
    matched_sources_dec = dec_radio[idx_radio]
    matched_sources_flux = radio_flux[idx_radio]

    N_sources = len(matched_sources_ra)
    N_clusters_with_soures = len(np.unique(idx_clusters))

    theta = np.arange(0, 10, 1)

    richess_bins = [20, 40, 80, 100, 300]
    redshift_bins = [0.1, 0.4, 1]

    fig, axes = plt.subplots(2, 4, figsize = (20, 10), sharex = True)

    for i in range(len(richess_bins) - 1):
        for j in range(len(redshift_bins) - 1):
            mask = np.where((c.richness > richess_bins[i]) & (c.richness <= richess_bins[i + 1]) & (c.z > redshift_bins[j]) & (c.z <= redshift_bins[j + 1]))[0]
            
            has_radio = np.isin(mask, radio_clusters)

            szmaps_ij = szmaps[mask]
            szmaps_ij_radio = szmaps_ij[has_radio]
            szmaps_ij_no_radio = szmaps_ij[~has_radio]


            R, total_profile, total_std, total_N = radial_binning2(np.average(szmaps_ij, axis = 0), theta, width = 0.6, full = True)
            R, contaminated_profile, contaminated_std, contaminated_total_N = radial_binning2(np.average(szmaps_ij_radio, axis = 0), theta, width = 0.6, full = True)
            R, no_contaminated_profile, no_contaminated_std, no_contaminated_total_N = radial_binning2(np.average(szmaps_ij_no_radio, axis = 0), theta, width = 0.6, full = True)

            pool = Pool(processes = n_cores)

            N_bootstrap_per_core = N_bootstrap // n_cores
            N_per_core = np.full(n_cores, N_bootstrap_per_core)
            N_per_core[-1] = N_per_core[-1] + N_bootstrap % n_cores

            manager = Manager()
            counter = manager.Value("i", 0)

            pool = Pool(n_cores)

            res_ = []

            for k in range(len(N_per_core)):
                res_.append(pool.apply_async(bootstrap_worker, args = (theta, szmaps_ij, N_per_core[k], N_bootstrap, counter, 0.6, np.ones(len(szmaps_ij)))))

            res = [r.get() for r in res_]

            bprofiles = np.array([r[0] for r in res])
            pool.close()
            print("")
            manager = Manager()
            counter = manager.Value("i", 0)

            pool = Pool(n_cores)

            res_ = []

            for k in range(len(N_per_core)):
                res_.append(pool.apply_async(bootstrap_worker, args = (theta, szmaps_ij_radio, N_per_core[k], N_bootstrap, counter, 0.6, np.ones(len(szmaps_ij_radio)))))

            res = [r.get() for r in res_]

            bcontaminated_profiles = np.array([r[0] for r in res])
            pool.close()
            print("")
            manager = Manager()
            counter = manager.Value("i", 0)

            pool = Pool(n_cores)

            res_ = []

            for k in range(len(N_per_core)):
                res_.append(pool.apply_async(bootstrap_worker, args = (theta, szmaps_ij_no_radio, N_per_core[k], N_bootstrap, counter, 0.6, np.ones(len(szmaps_ij_no_radio)))))

            res = [r.get() for r in res_]

            bno_contaminated_profiles = np.array([r[0] for r in res])
            pool.close()

            cov_total = np.cov(bprofiles, rowvar = False)
            cov_contaminated = np.cov(bcontaminated_profiles, rowvar = False)
            cov_no_contaminated = np.cov(bno_contaminated_profiles, rowvar = False)

            std_total = np.sqrt(np.diag(cov_total))
            std_contaminated = np.sqrt(np.diag(cov_contaminated))
            std_no_contaminated = np.sqrt(np.diag(cov_no_contaminated))

            ax = axes[j, i]
            ax.errorbar(R, total_profile, yerr = std_total, color = "black", fmt = "-o", alpha = 0.8, lw = 4)
            ax.errorbar(R + 0.1, contaminated_profile, yerr = std_contaminated, color = "green", fmt = "-o", alpha = 0.8, lw = 4)
            ax.errorbar(R + 0.2, no_contaminated_profile, yerr = std_no_contaminated, color = "cyan", fmt = "-o", alpha = 0.8, lw = 4)

            ax.set(yscale = "log", xlabel = "R (arcmin)", ylabel = "y profile")

    fig.tight_layout()
    fig.savefig("/data2/javierurrutia/szeffect/data/VLASS/stacked_profiles.png")
