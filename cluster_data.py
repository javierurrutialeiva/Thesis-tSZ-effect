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

# Astropy
import astropy
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
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

@njit(cache = True, fastmath=True)
def compute_two_halo(Rgrid, lambda2halo_grid, z2halo_grid, params, Pk, bh, 
                    dndM2halo, bM, R2halo, M_arr2halo, k2halo, lambda_true, 
                    lambda_obs, PRMzk, sin_term, R, M, z_arr, R_Mpc2halo,
                    k2halo_grid, PllM):
    uRMz = 4*np.pi *((dndM2halo*bM).T[None, :, :, None] * Rgrid**2 * sin_term * PRMzk)
    uPk = Pk[None, None, :,:] * uRMz
    bhuPk = bh[None, None, :, :, None] * uPk[:,:,:,None,:]
    ki_R2 = R_Mpc2halo[:,:,None] * k2halo[None,None,:]
    sin_term2 = np.sin(ki_R2) / np.where(ki_R2 != 0, ki_R2, 1)
    P2halo = ((sin_term2[None, None, :,:,:]* k2halo_grid[:,:,None,:,:]**2)[:,:,:,:,None,:] * bhuPk[:,:,None,:,:,:])/(2*np.pi**2)
    return P2halo

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
                                         weighted = weighted)
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
                        gs.append(g.sub_group(redshift_interval = (z1, z2)))
                    if np.any(np.array([len(gsi) for gsi in gs]) < 20) == True:
                        continue
                    for j in range(len(gs)):
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
                    ("N_total", 300),
                    ("N_realizations", None),
                    ("compute-cov-matrix", True),
                    ("unbiased-factor", True),
                    ("compute-individual-covs", False)
        )

        default_covariance_estimation_kwargs = (
            ("N_total", 100),
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
            ("convergence_threshold", 0.05)
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
                self.load_map_and_mask()
                ymap = self.map
                mask = self.mask
                clusters_mask = self.clusters_mask

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

                    cov_matrices = np.concatenate(cov_matrices_list, axis = 0).astype(float)
                    corr_matrices = np.zeros(np.shape(cov_matrices))
                    for n,c in enumerate(cov_matrices):
                        sigma = np.sqrt(np.diag(c))
                        corr_matrices[n] = cov_matrices[n]/np.outer(sigma,sigma)
                    
                    mean_profiles = np.concatenate(mean_profiles_list, axis = 0).astype(self.dtype)
                    random_profiles = np.concatenate(random_profiles_list, axis = 0).astype(self.dtype)
                    random_weights = np.concatenate(random_weights_list, axis = 0).astype(self.dtype)
                                        
                    if covariance_estimation_kwargs["check_convergence"] == True:
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

                    cov = np.median(cov_matrices, axis = 0).astype(self.dtype)
                    corr = np.median(corr_matrices, axis = 0).astype(self.dtype)

                if covariance_estimation_kwargs["unbiased-factor"] == True:
                    Nb = np.shape(cov)[0]
                    self.cov = (N_realizations - 1)/(N_realizations - 2 - Nb)*cov
                else:
                    self.cov = cov
                self.cov = (N_total/len(self.richness))*cov
                self.corrm = corr
                self.random_cov_matrices = cov_matrices
                self.random_corr_matrices = corr_matrices
                self.mean_profiles_cov = mean_profiles
                self.random_profiles_cov = random_profiles
                self.snr = np.sqrt(np.dot(profile, np.dot(np.linalg.inv(self.cov), profile.T)))
            elif bootstrap == True:
                print()
                N = bootstrap_kwargs["N_total"]
                if N is None:
                    N = int(np.ceil(len(self.richness)/1000) * 1000)
                N_clusters = bootstrap_kwargs["N_realizations"] if bootstrap_kwargs["N_realizations"] is not None else len(self.richness)
                compute_cov_matrix = bootstrap_kwargs["compute-cov-matrix"]

                if verbose == True:
                    print("Bootstrap sampling to error and covariance estimation.")
                    print("N total =", N)
                    print("N clusters =", N_clusters)
                    print("Unbiased factor estimator =", bootstrap_kwargs["unbiased-factor"])
                    print("Estimate cov =", compute_cov_matrix)
                if n_pool == 1:
                    indx = np.random.choice(np.arange(0, len(maps_array), 1), replace = True, size = (N, N_clusters))
                    stacks = np.average(maps_array[indx], axis = 1)
                    bootstrap_profiles = radial_binning2(stacks, R_profiles, width = width)

                    # if verbose == True:
                    #     progress_bar = tqdm(desc = "Running bootstrap...", total = N)
                    # bootstrap_profiles = np.zeros((N, len(R_profiles) - 1))
                    # for n in range(N):
                    #     indx = np.random.choice(np.arange(0, len(maps_array), 1), replace = True, size = N)
                    #     smaps = maps_array[indx]
                    #     sstack = np.average(smaps, axis = 0)
                    #     R_bins, sprofile, _, sdata = radial_binning2(sstack, R_profiles, width = width)
                    #     bootstrap_profiles[n] = sprofile
                    #     if verbose == True:
                    #         progress_bar.update(1)
                elif n_pool > 1:
                    indx = np.random.choice(np.arange(0, len(maps_array), 1), replace = True, size = (N, N_clusters))
                    maps = self.imap[indx]
                    pool = Pool(n_pool)
                    counter = manager.Value("i", 0)
                    maps_per_core = np.array_split(maps, n_pool, axis = 0)
                    res_ = []
                    for m in maps_per_core:
                        res_.append(pool.apply_async(bootstrap_worker, args = (R_profiles, m, N, counter, width)))
                    res = [r.get() for r in res_]
                    bootstrap_profiles = np.concatenate(res, axis = 0)

                bootstrap_profiles = bootstrap_profiles - self.random_mean_profiles if hasattr(self, "random_mean_profiles") else bootstrap_profiles
                self.bootstrap_profiles = bootstrap_profiles
                self.bootstrap_mean = np.mean(bootstrap_profiles, axis=0)
                self.bootstrap_std = np.std(bootstrap_profiles, axis=0)
                self.bootstrap_1sigma_bounds = np.percentile(bootstrap_profiles, [16, 84], axis = 0)
                self.bootstrap_2sigma_bounds = np.percentile(bootstrap_profiles, [2.5, 97.5], axis = 0)
                if compute_cov_matrix == True:
                    print("Computing covariance matrix...")
                    y = np.array(self.bootstrap_profiles)
                    bar_y = np.mean(y, axis = 0)
                    Nrand = len(y)
                    cov = np.zeros((len(R_bins), len(R_bins)))
                    for i in range(len(R_bins)):
                        for j in range(len(R_bins)):
                            cov[i,j] = 1/Nrand * np.sum((y[:,i] - bar_y[i]) * (y[:,j] - bar_y[j]))
                    if bootstrap_kwargs["unbiased-factor"] == True:
                        unbiased_factor = ((N - len(R_bins) - 2) / (N - 1))
                        cov = unbiased_factor * cov
                    sigma = np.sqrt(np.diag(cov))
                    corr = [[cov[i,j]/(sigma[i]*sigma[j]) for i in range(len(sigma))] for j in range(len(sigma))]
                    self.corr_matrix_bootstrap = corr
                    self.cov_matrix_bootstrap = cov

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
                    extent = (R_bins[0], R_bins[-1] ,R_bins[-1] ,R_bins[0] ), cmap = "seismic")
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
            with h5py.File(f"{self.output_path}/data.h5", "w") as f:
                for k in available_data:
                    if k != "dtype" and k != "wcs":
                        try:
                            if type(getattr(self, k)) is float:
                                f.create_dataset(k, data = getattr(self, k), dtype = dtype)
                            elif type(getattr(self, k)) is str:
                                f.create_dataset(k, data = str(getattr(self, k)), dtype = str)
                            else:
                                if getattr(self, k) is not None:
                                    f.create_dataset(k, data = getattr(self, k), dtype = getattr(self, k).dtype)
                        except ValueError:
                            dt = h5py.special_dtype(vlen=float) if type(k) == float else h5py.special_dtype(vlen=str)
                            f.create_dataset(k, data = getattr(self, k), dtype = dt)
                        except AttributeError:
                            f.create_dataset(k, data = getattr(self, k))
                            try:
                                f.create_dataset(k, data = getattr(self, np.array(k, dtype = str)), dtype = str)
                            except:
                                print(f"An exception has ocurred trying to store {k} attribute!")
                        except TypeError:
                            print(f"An exception has ocurred trying to store {k} attribute!")
                            continue
                        pass
                    elif k == "wcs":
                        wcs = getattr(self, k)
                        header_str = wcs.to_header().tostring(sep="\n")
                        dt = h5py.string_dtype(encoding="utf-8")       # <--- important
                        f.create_dataset("wcs_header", data=header_str, dtype=dt)
    def mass_richness_func(self, pivot=40, slope=1.29, normalization=10**14.45):
        return lambda l: (normalization * (l / pivot) ** slope)

    def completeness_and_halo_func(self, plot = False, zbins = 6, Mbins = 5, verbose = False, relationship_config = "MASS-RICHNESS RELATIONSHIP",
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
            ("richness2mass_Slope_redshift", -0.3),
            ("richness2mass_Pivot_redshift", 0.35),
            ("mass2richness_Norm", 30),
            ("mass2richness_Pivot", 3e14/0.7),
            ("mass2richness_Slope", 0.75),
            ("mass2richness_Slope_redshift", 0.0),
            ("mass2richness_Pivot_redshift", 0.35),
            ("sigmaRM", 0.25),
            ("pmr_distribution", "log-normal"),
        )

        default_function_kwargs = (
            ("Nlambda_true", 50),
            ("max_lambda_true", 350),
            ("min_lambda_true", 10),
            ("Nlambda_obs", 30),
            ("Nz_lambda", zbins),
            ("function", 'P_lob_ltr'),
            ("func_kwags", {})
        )

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
            kf = function_kwargs["func_kwags"]
            lambda_true = np.linspace(function_kwargs["min_lambda_true"], function_kwargs["max_lambda_true"], Nlambda_true).astype(int)
            if use_lambda_obs is None:
                lambda_obs = np.linspace(function_kwargs["min_lambda_obs"], function_kwargs["max_lambda_obs"], Nlambda_obs).astype(int)
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
            prob_distribution = f(lambda_true_grid, lambda_obs_grid, z_arr_grid, **kf)

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
                z_arr = z_arr[np.where((z_arr >= np.min(z_lambda)) & (z_arr <= np.max(z_lambda)))]
                prob_distribution = prob_distribution[:, :, np.where((z_arr >= np.min(z_lambda)) & (z_arr <= np.max(z_lambda)))[0]]
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
        M = np.logspace(13, 15.75, Mbins) #mass interval ==> [13, 15.75]
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
        if use_redshift == True:
            lambda_true_grid, M_grid, z_grid = np.meshgrid(lambda_true, M, z_arr, indexing = "ij")
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope * \
                        ((1 + z_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift     
        else:
            lambda_true_grid, M_grid = np.meshgrid(lambda_true, M)
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope

        sigma_model = np.sqrt(((lambda_model - 1) / lambda_model**2) + sigmaRM**2)

        if pmr_distribution == "log-normal":
            Plambda_true_Mass = 1/(np.sqrt(2 * np.pi**2 * sigma_model**2) * lambda_true_grid) * np.exp(
                - (np.log(lambda_true_grid) - np.log(lambda_model))**2 / (2 * sigma_model**2))
        elif pmr_distribution == "normal":
            Plambda_true_Mass = 1/(np.sqrt(2*np.pi*sigmaRM**2))*np.exp(
                -(np.log(lambda_true_grid) - np.log(lambda_model))**2/ (2*sigmaRM**2))

        if zlambda2zobs == False or Pzlambda_kwargs["func"] == "dirac":
            PllM = Plambda_true [:,:,None,:]* Plambda_true_Mass[:,None,:,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) ==> P(lambda_obs | lambda_true, M)
        else:
            PllM = (Plambda_true [:,:,None,:]* Plambda_true_Mass[:,None,:,:])[:,:,:,:,None] * Pzlambda_z.T[None, None, None, :,:] # P(lambda_true | lambda_obs) * P(M | lambda_true) * P(z_lambda | z) ==> P(lambda_obs | lambda_true, M, z, z_lambda)
            PllM = trapz(PllM, axis = -1, x = z_lambda) #integrate over z_arr to get P(lambda_obs | lambda_true, M, z)
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
        #Compute halo mass function 
        cosm = ccl.Cosmology(**cosmological_model) #cosmological model
        mdef = ccl.halos.massdef.MassDef(delta, background)
        a = 1 / (1 + z_arr)
        mfunc = ccl.halos.mass_function_from_name("Tinker10") #mass function from Tinker et al 2010
        mfunc = mfunc(cosm, mdef)
        dndM = np.array([[mfunc(cosm, mi, ai) for mi in M ] for ai in a]) #dN/dM
        dndM = dndM * 1/(M * np.log(10)) #convert from log10
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
        

        clusters_mask = hp.fitsfunc.read(self.clusters_mask_path) if clusters_mask_format == 'healpy' else enmap.read_map(self.clusters_mask_path)
        
        self.map = m.astype(self.dtype)
        self.mask = mask.astype(self.dtype)
        self.clusters_mask = clusters_mask.astype(self.dtype)
        return m, mask, clusters_mask
    def stacked_halo_model_func(self, one_halo_profile,units = "arcmin", pix_size = 0.5, rbins = 25, zbins = 11, Mbins = 10,
                                filters = None, use_filters = False, use_two_halo_term = False, fixed_RM_relationship = True,
                                rebinning = False , mis_centering = False,  interpolate_2halo = False, eval_lambda = False,
                                two_halo_profile = None, redshift_weight_function = False, richness_weight_function = False,
                                mis_centering_func = lambda x,sigma: x/sigma**2*np.exp(-x**2/(2*sigma**2)), verbose = True,
                                delta = 500, background = "critical", pyccl_cosmo = None, eval_mass = False, 
                                apply_filter_per_profile = False, return_1h2h = False, infere_mass = False,
                                redshift_pivot = 0.4737, richness_pivot = 32.68, weighted = True, 
                                subr_grid = True, **kwargs):   

        from astropy.cosmology import Planck18 as planck18

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
            ("theta", np.linspace(0, 2*np.pi, 30))
        )

        default_two_halo_kwargs = (
            ("background", background),
            ("R", np.logspace(-1, 1.7, 20)),
            ("k", np.logspace(-15,15, 50)),
            ("M_arr", np.logspace(13,16, 20)),
            ("z_arr", np.linspace(1e-3,1, 20)),
            ("cosmo", ccl.CosmologyVanillaLCDM()),
            ("delta", delta),
            ("two_halo_power_func", lambda z, params: np.full(z.shape,params[0])),
            ("eval_only_mass", False),
            ("eval_only_richness", False),
            ("eval_only_redshift", True),    
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

        two_halo_kwargs = set_default(kwargs.pop("two_halo_kwargs", {}), default_two_halo_kwargs)
        compl_kwargs = set_default(kwargs.pop("completeness_kwargs", {}), default_compl_kwargs)
        mis_centering_kwargs = set_default(kwargs.pop("mis_centering_kwargs", {}), default_mis_centering_kwargs)
        rebinning_kwargs = set_default(kwargs.pop("rebinning_kwargs", {}), default_rebinning_kwargs)
        weights_function_kwargs = set_default(kwargs.pop("weights_function_kwargs", {}),default_weights_function_kwargs)
        richness_weights_function_kwargs = set_default(kwargs.pop("richness_weights_function_kwargs",{}), default_richness_weights_function_kwargs)
        two_halo_profile = one_halo_profile if two_halo_profile is None else two_halo_model
        mass_inf_kwargs = set_default(kwargs.pop("mass_inf_kwargs", {}), default_mass_inf_kwargs)
        subr_grid_kwargs = set_default(kwargs.pop("subr_grid_kwargs", {}), default_subr_grid_kwargs)
        #pre-compute completeness and halo mass function
        use_redshift = compl_kwargs["use_redshift"]
        if hasattr(self, "completeness_kwargs") == False:
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
            M_grid = M_grid.astype(float_dtype)
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope 
            self.D_ang =  ((planck18.angular_diameter_distance(z_grid) * (1 + z_grid)))[None,:,:]
        else:
            lambda_true_grid, M_grid, z_grid = np.meshgrid(lambda_true, M, z_arr, indexing="ij")
            lambda_model = mass2richness_Norm * (M_grid / mass2richness_Pivot)**mass2richness_Slope * \
                                    ((1 + z_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift 
            self.M_grid = M_grid
            self.z_grid = z_grid
            self.lambda_model = lambda_model
            self.D_ang = ((planck18.angular_diameter_distance(z_grid) * (1 + z_grid)))[None,:,:,:]
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

        dV = Wz(z_arr)*cosmo.differential_comoving_volume(z_arr).to(u.kpc**3 / u.sr)
        if (use_two_halo_term is not None) and type(use_two_halo_term) in (str, bool):
            if use_two_halo_term == True or use_two_halo_term == "only":
                if verbose:
                    print("Creating\033[92m 1+2-halo function.\033[0m") if use_two_halo_term == True else print("Creating\033[92m 2-halo function.\033[0m")
                R2halo = two_halo_kwargs["R"]
                delta = two_halo_kwargs["delta"]
                M_arr2halo = two_halo_kwargs["M_arr"]  
                cosmo2halo = two_halo_kwargs["cosmo"]
                k2halo = two_halo_kwargs["k"]    
                mdef = ccl.halos.MassDef(delta, two_halo_kwargs["background"])
                mfunc = ccl.halos.mass_function_from_name("Tinker10")
                mfunc = mfunc(cosmo2halo, mdef)
                dndM2halo = np.array([[mfunc(cosmo2halo, Mi, 1/(zi + 1)) for Mi in M_arr2halo] for zi in z_arr]).astype(float_dtype)
                dndM2halo = np.array(dndM2halo * 1/(M_arr2halo * np.log(10))).astype(float_dtype)
                bias = ccl.halos.HaloBiasTinker10(cosmo2halo, mass_def=mdef) 
                bh = np.array([bias.get_halo_bias(cosmo2halo, M, 1/(1 + zi)) for zi in z_arr]).astype(float_dtype)
                bM = np.array([[bias.get_halo_bias(cosmo2halo, Mi, 1/(1 + zi)) for Mi in M_arr2halo] for zi in z_arr]).astype(float_dtype)
                Pk = np.array([ccl.linear_matter_power(cosmo2halo, k2halo, 1/(1+zi)) for zi in z_arr]).astype(float_dtype)
                Rgrid, M2halo_grid, z2halo_grid, k2halo_grid = np.meshgrid(R2halo, M_arr2halo, z_arr, k2halo, indexing = "ij")
                ki_r = Rgrid*k2halo_grid
                sin_term = np.sin(ki_r) / np.where(ki_r != 0, ki_r, 1)
                self.D_ang2halo = (planck18.angular_diameter_distance(z_arr) * (1 + z_arr))[None,:]
                lambda2halo_grid = mass2richness_Norm * (M2halo_grid / mass2richness_Pivot)**mass2richness_Slope * \
                                    ((1 + z2halo_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift  
                two_halo_power_func = two_halo_kwargs["two_halo_power_func"]
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
            print(f"Using \033[92mweight\033[0m")
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
                
                weights = np.ones((len(z_arr), len(lambda_obs)))
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

        self.W = weights

        norm = np.trapz( 
                dV * np.trapz( dndM.T * 
                    np.trapz(
                        np.trapz(PllM*weights[None,:,None,:], axis = 0, x = lambda_true), axis = 0, x = lambda_obs
                        ), axis = 0, x = M)
                    , axis = 0, x = z_arr) 

        self.norm = norm
        global func
        def func(r, params, RM_params = None, new_PllM = None, new_sigmaRM = None, rbins = 35, new_Plambda_true = None, 
                smooth = None, eval_lambda = True, mis_centering_params = None, Roff = np.logspace(-1, 1, 10), return_2halo_term = False,
                theta = np.linspace(0,2*np.pi,60), mass2richness_Pivot = 3e14/0.7, mass2richness_Pivot_redshift = 0.35
                , sigmaRM = 0.25, two_halo_power = None, return_profile_grid = False, R_intp = None):
            if subr_grid == True:
                R = subR_grid
            else:
                R = r
            M,z_arr = self.M, self.z_arr
            lambda_true = self.lambda_true
            lambda_obs = self.lambda_obs
            M_grid = self.M_grid
            z_grid = self.z_grid
            Plambda_true = self.Plambda_true
            lambda_model = self.lambda_model
            PllM = self.PllM
            Plambda_obs_M = self.Plambda_obs_M
            P_Mass = self.P_Mass
            norm = self.norm
            D_ang = self.D_ang
            weights = self.W
            Mgrid_mis, zgrid_mis = np.meshgrid(M, z_arr)
            R_Mpc_mis = ((R * u.arcmin).to(u.rad)[:,None,None] * ((planck18.angular_diameter_distance(zgrid_mis) * (1 + zgrid_mis)))[None,:,:]).value
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
                R_Mpc = ((R * u.arcmin).to(u.rad)[:,None,None,None] * self.D_ang).value
                one_halo_term = one_halo_profile(R_Mpc, x_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else one_halo_profile(R_Mpc, x_grid, M_grid, z_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) #result is the profile model evaluated at R, M/lambda, z
                weighted_one_halo_term = weights[None,None,:,None,:]*one_halo_term[:,:,None,:,:] * PllM[None,...]

            if use_two_halo_term == True or use_two_halo_term == "only":
                PRMzk = two_halo_profile(Rgrid, lambda2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot) if eval_mass == False else two_halo_profile(Rgrid, lambda2halo_grid, M2halo_grid, z2halo_grid, params, rbins, richness_pivot = richness_pivot, redshift_pivot = redshift_pivot)
                R_Mpc2halo = ((R * u.arcmin).to(u.rad)[:,None] * self.D_ang2halo).value       
                P2halo = compute_two_halo(Rgrid, lambda2halo_grid, z2halo_grid, np.array(params, dtype = np.float64), Pk, bh, 
                    dndM2halo, bM, R2halo, M_arr2halo, k2halo, lambda_true, lambda_obs, PRMzk, 
                    sin_term, R, M, z_arr, R_Mpc2halo, k2halo_grid, PllM)
                two_halo_term = np.reshape(np.trapz(np.trapz(np.trapz(P2halo, axis = 0, x = R2halo), axis = 0, x = M_arr2halo)
                                            , axis = -1, x = k2halo), (len(R), len(M), len(z_arr)))
                if return_2halo_term == True:
                    return two_halo_term
                weighted_two_halo_term = PllM[:,None, ...] * two_halo_term[None,:,None,:,:]*weights[None,None,:,None,:]
                if two_halo_power is not None:
                    if hasattr(self, "two_halo_func_evals"):
                        if np.all(self.two_halo_func_evals == True):
                            weighted_two_halo_term = f2halo(lambda_model, M_grid, z_grid, two_halo_power)[:,None,None,:,:]*weighted_two_halo_term
                    else:
                        pass
                P2halo = np.trapz(np.trapz(weighted_two_halo_term, axis = 0, x = lambda_true), axis = 1, x = lambda_obs)

            P1halo = trapz(trapz(weighted_one_halo_term, axis = 1, x = lambda_true), axis = 1, x = lambda_obs) #integrate over the observed richness
            if hasattr(self, "two_halo_func_evals") and two_halo_power is not None:
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

            output = np.zeros((len(PRMz), len(r)), dtype = float_dtype)
            infered_Mass = 0
            for k, P in enumerate(PRMz):
                if mis_centering == True and mis_centering_params is not None:
                    fmis, mis_centering_params_func = mis_centering_params[0], mis_centering_params[1::] if mis_centering_params is not None else [0.246, 0.385]
                    weights = np.array(rho_Roff(Roff, *mis_centering_params_func), dtype = float_dtype)

                    funcs = [
                    [UnivariateSpline(R_Mpc_mis[:,i,j], pj, k=1, s=0) for j,pj in enumerate(pi)] for i,pi in enumerate(P.T)
                    ]
                    off = np.array(
                        [[np.trapz(fj(xmis[:,:,:,i,j]), theta, axis=0) for j,fj in enumerate(fi)] for i,fi in enumerate(funcs)]
                    )
                    woff = np.array(
                        [[np.trapz(weights[:,None] * off[i,j,:,:], Roff, axis = 0)/np.trapz(weights, Roff) for j,pj in enumerate(pi)] 
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

    def create_beam_filter(self, mode = "a"):
        output_path = self.output_path
        content = f"""
from scipy.ndimage import gaussian_filter1d
import numpy as np
def apply_beam(R, data, fwhm = 1.6):
    fwhm = float(fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dr = (R[-1] - R[0]) / (len(R) - 1)
    sigma_pix = sigma / dr
    return gaussian_filter1d(data, sigma=sigma_pix, mode='constant', cval=0.0)
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
    def compute_joint_cov(self, paths = None, off_diag = False, groups = None, corr = False):
        if paths is not None and np.iterable(paths) and groups is None:
            groups = []
            covs = []
            if paths is not None and np.iterable(paths):
                for i in range(len(paths)):
                    sub_group = grouped_clusters.load_from_path(paths[i])
                    covs.append(sub_group.cov)
                    groups.append(sub_group)
        else:
            covs = [g.cov for g in groups]

        full_covariance_matrix = block_diag(*covs)

        if off_diag == True:
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
        if corr == False:
            return groups, full_covariance_matrix
        else:
            sigma = np.sqrt(np.diag(full_covariance_matrix))
            full_correlation_matrix = full_covariance_matrix/np.outer(sigma, sigma)
            return groups, full_covariance_matrix, full_correlation_matrix

    @classmethod
    def stacked_halo_model_func_by_paths(self, profile_model, units = "arcmin", 
                                        full = False, Rbins = 25, Mbins = 10, Zbins = 11,
                                        paths = None, verbose_pivots = False,
                                        rotate_cov = False, use_filters = False,
                                        filters = None, off_diag = False, verbose = False,
                                        recompute_cov = False, use_two_halo_term = False,
                                        fixed_RM_relationship = True, use_mis_centering = False,
                                        delta = 500, background = "critical", eval_mass = False,
                                        return_cov = False,  apply_filter_per_profile = False,
                                        rebinning = False, dtype = np.float32, return_1h2h = False,
                                        infere_mass = False, sort = True, subr_grid = False,
                                        **kwargs):
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
                funcs.append(sub_group.stacked_halo_model_func(profile_model, units, rbins = Rbins, zbins = Zbins, Mbins = Mbins,
                                    use_filters = use_filters, filters = filters, use_two_halo_term = use_two_halo_term, 
                                    fixed_RM_relationship = fixed_RM_relationship, two_halo_kwargs = two_halo_kwargs,
                                    mis_centering = use_mis_centering, mis_centering_kwargs = mis_centering_kwargs,
                                    background = background, delta = delta, eval_mass = eval_mass,
                                    apply_filter_per_profile = apply_filter_per_profile, rebinning = rebinning,
                                    rebinning_kwargs = rebinning_kwargs, return_1h2h = return_1h2h, verbose = verbose,
                                    infere_mass = infere_mass, mass_inf_kwargs = mass_inf_kwargs, subr_grid = subr_grid,
                                    subr_grid_kwargs = subr_grid_kwargs))
                about_clusters.append(
                    dict(
                        richness = (np.min(sub_group.richness),np.max(sub_group.richness)),
                        redshift = (np.min(sub_group.z),np.max(sub_group.z)),
                        N = len(sub_group),
                        path = paths[i]
                    )
                )
        
        _,full_covariance_matrix = grouped_clusters.compute_joint_cov(groups = groups, off_diag = off_diag)
        bins = np.array(bins)
        if sort == True:
            sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
            bins = bins[sorted_idx]
            groups = [groups[i] for i in sorted_idx]
            covs = [covs[i] for i in sorted_idx]
            funcs = [funcs[i] for i in sorted_idx]
            profiles = profiles[sorted_idx]
            about_clusters = [about_clusters[i] for i in sorted_idx]
        global func_gen
        def func_gen(R, params, RM_params = None, new_PllMs = None, smooth = None, eval_lambda = True, 
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
                    current_results = f(R,params, RM_params = RM_params, new_PllM = new_PllM, smooth = smooth, 
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
                idx = np.argmin(np.sum(np.array(cbin) - bins, axis = 0), axis = 0)   
                return funcs[idx](R, params, RM_params = RM_params, new_PllM = new_PllMs, smooth = smooth, 
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
                    current_results = funcs[idx](R, params, RM_params = RM_params, new_PllM = new_PllMs, smooth = smooth, 
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
                                        rebinnng = False, return_1h2h = False, verbose = False,
                                        **kwargs):
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
                  weights = None, return_patches = False, dtype = np.float32
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
    # if hasattr(ymap, "wcs") == False:
    #     ymap = enmap.ndmap(ymap, wcs=wcs)
    #     mask = enmap.ndmap(mask, wcs=wcs) if mask_format == "pixell" else mask

    rng = np.random.default_rng()

    dec2, ra2 = np.zeros((2, N_random, N_clusters)).astype(dtype)
    dec2, ra2 = rng.uniform(dmin, dmax, (N_random, N_clusters)), rng.uniform(rmin, rmax, (N_random, N_clusters))
    for i in range(N_random):
        accepted_coords = 0
        while accepted_coords < N_clusters - 1:
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
            dec_in_mask, ra_in_mask = np.unique(dec_in_mask), np.unique(ra_in_mask)

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
                W = w[:, None] / s
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
                        denom = V1 - V2 / V1
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
    output.append(np.rad2deg(coords)) if save_coords == True else None
    return output

global bootstrap_worker
def bootstrap_worker(R_profiles, maps, N_total, counter, width):
    mean_profiles = np.zeros((len(maps), len(R_profiles) - 1))
    for i in range(len(maps)):
        mean_profiles[i]= radial_binning2(np.mean(maps[i], axis = 0), R_profiles, width = width)
        counter.value += 1

    sys.stdout.write(f"\rBootstrap progress: ({counter.value} / {N_total})")
    sys.stdout.flush()
    return mean_profiles


# path = "/data2/javierurrutia/szeffect/data/ycompton-no-CIB-deproj/entire_sample"
# c = grouped_clusters.load_from_path(path)
# ymap = enmap.read_map("/data2/javierurrutia/szeffect/data/ilc_SZ_yy.fits")
# mask = enmap.read_map("/data2/javierurrutia/szeffect/data/wide_mask_GAL070_apod_1.50_deg_wExtended.fits")
# new_mask = hp.read_map("/data2/javierurrutia/szeffect/data/DES_ACT-footprint_unmasked_clusters.fits")
# wcs = ymap.wcs
# backmidea@entel.cl