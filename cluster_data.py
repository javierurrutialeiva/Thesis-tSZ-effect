import h5py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import emcee
from PIL import Image
from matplotlib.cm import ScalarMappable
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
import os
from matplotlib.colors import Normalize
import corner
from astropy.cosmology import Planck18 as cosmo
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy import constants as const
from tqdm import tqdm
from scipy.optimize import curve_fit
import pandas as pd
from scipy.spatial import cKDTree as KD
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps, trapz
import pyccl as ccl
from scipy.signal import convolve2d as conv2d
from configparser import ConfigParser
from pixell import enmap, utils, reproject
from helpers import *
import profiles
import importlib
import warnings
import emcee
from time import time
from scipy.stats.kde import gaussian_kde
from lmfit import Model, Parameters
import matplotlib.patches as patches
from astropy.table import Table
from config import *

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
prior_parameters = dict(config["STACKED_HALO_MODEL"])
prior_parameters = {
    key: list(prop2arr(prior_parameters[key], dtype=str))
    for key in list(prior_parameters.keys())
}
match_agn = str2bool(config["EXTRACT"]["MATCH_AGN"])
if match_agn:
    r_agn_match = float((config["EXTRACT"]["R_AGN_MATCH"])) * u.arcmin

prior_parameters = list(prior_parameters.values())
nwalkers = int(config["STACKED_HALO_MODEL"]["nwalkers"])
nsteps = int(config["STACKED_HALO_MODEL"]["nsteps"])

if not os.path.exists(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"])

if not os.path.exists(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"]):
    os.mkdir(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"])
if not os.path.exists(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + '/profiles'):
    os.mkdir(data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + '/profiles')

data_mask_ratio = float(config["EXTRACT"]["MASK_RATIO"])

width, w_units = prop2arr(config["EXTRACT"]["width"], dtype=str)
width = np.deg2rad(float(width)) if w_units == "deg" else float(width)

print(f"Loading DATA from \033[92m{data_path}\033[0m.")
szmap = enmap.read_map(data_path + DR6)
sz_clusters = fits.open(data_path + DR5)[1].data
cluster_catalog = fits.open(data_path + DES_Y3)[1].data
agn_catalog = Table(fits.open(data_path + MILLIQUAS)[1].data)

def extract_cluster_data(richness_range=None):
    """
    Extract the cluster data from the files specified in config.ini

    - data_mask_ratio: change which must be the ratio between the number of pixells equal to one with the total pixells
      in the mask.

    """
    ufrom, uto = prop2arr(config["EXTRACT"]["CHANGE_UNIT"], dtype=str)
    zmin, zmax = prop2arr(config["EXTRACT"]["redshift"], dtype=np.float64)
    rewrite = bool(config["EXTRACT"]["REWRITE"])
    #I should change the name of the variable but its so boring
    redmapper = cluster_catalog
    milliquas = agn_catalog
    dr5 = sz_clusters
    redmapper = redmapper[(redmapper["Z"] >= zmin) & (redmapper["Z"] < zmax)]
    if richness_range is not None:
        if np.iterable(richness_range):
            redmapper = redmapper[
                (redmapper["LAMBDA_CHISQ"] >= richness_range[0])
                & (redmapper["LAMBDA_CHISQ"] < richness_range[1])
            ]
            iter = range(0, len(redmapper))
            t1 = time()
            warnings.filterwarnings("ignore", category=RuntimeWarning)
    else:
        iter = tqdm(
            range(0, len(redmapper)),
            desc="Extracting Clusters",
            bar_format="{l_bar}{bar}{r_bar}",
            dynamic_ncols=True,
        )
    redmapper_RA, redmapper_dec = redmapper["RA"], redmapper["DEC"]
    mask = enmap.read_map(data_path + MASK_DR6)
    R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
    R_units = config["CLUSTER PROPERTIES"]["r_units"]
    FWHM, FWHM_units = prop2arr(config["CLUSTER PROPERTIES"]["FWHM"], dtype=str)
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
    ra, dec = redmapper_RA, redmapper_dec
    clusters = []
    R_min = []
    if match == True:
        dr5 = dr5[(dr5["redshift"] >= zmin) & (dr5["redshift"] < zmax)]
        dr5ra, dr5dec = dr5["RADeg"], dr5["decDeg"]
        dr5ra[dr5ra > 180] = dr5ra[dr5ra > 180] - 360
        dr5ra, dr5dec = np.deg2rad(dr5ra), np.deg2rad(dr5dec)
        rm_coords = SkyCoord(ra=np.rad2deg(dr5ra)*u.degree, dec=np.rad2deg(dr5dec)*u.degree)
        dr5_coords = SkyCoord(ra=np.rad2deg(ra)*u.degree, dec=np.rad2deg(dec)*u.degree)
        rm_indices, dr5_indices, catalog2_indices, separation = rm_coords.search_around_sky(dr5_coords, r_match)
        print(f"N matched clusters in DR5 = {len(rm_indices)}")
    if match_agn == True:
        mill = milliquas[(milliquas["Z"] >= zmin) & (milliquas["Z"] < zmax)]
        ra_mill, dec_mill = mill["RA"], mill["DEC"]
        mill_coords = SkyCoord(ra = ra_mill, dec = dec_mill, unit = (u.hourangle, u.deg))
        rm_coords = SkyCoord(ra=np.rad2deg(ra)*u.degree, dec=np.rad2deg(dec)*u.degree)
        rm_mill_indices, mill_indices, cat_mill_indices, separation_mill = rm_coords.search_around_sky(mill_coords, r_agn_match)
        print(f"AGN matched with RM = {len(rm_mill_indices)}")
    for i in iter:
        box = [
            [dec[i] - width / 2.0, ra[i] - width / 2.0],
            [dec[i] + width / 2.0, ra[i] + width / 2.0],
        ]
        smap = szmap.submap(box)
        smask = mask.submap(box)
        shape = np.shape(smap)
        center = shape[0]//2, shape[1]//2
        pixel_width = np.rad2deg(width) / np.shape(smap)[0] * 60
        x,y = np.indices(np.shape(smap))
        theta = np.sqrt(((x - center[0])*pixel_width)**2 + (((y - center[1]))*pixel_width)**2) * u.arcmin
        R = (theta.to(u.radian) * cosmo.angular_diameter_distance(redmapper[i]["Z"])).value * u.kpc * 1000
        x,y = (x - center[0])*pixel_width, (y - center[1])*pixel_width
        cluster = sz_cluster(
            np.rad2deg(ra[i]),
            np.rad2deg(dec[i]),
            redmapper[i]["LAMBDA_CHISQ"],
            redmapper[i]["LAMBDA_CHISQ_E"],
            R,
            smap,
            smask,
            redmapper[i]["Z"],
            box,
            redmapper[i]["MEM_MATCH_ID"]
        )
        if cluster.MASK_FLAG == True:
            continue
        cluster.theta = theta
        cluster.x = x * u.arcmin
        cluster.y = y * u.arcmin
        cluster.generate_profile(r=R_profiles)
        cluster.output_path = (
            data_path
            + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]
            + "redmapper_ID="
            + str(redmapper[i]["MEM_MATCH_ID"])
        )
        cluster.ID = redmapper[i]["MEM_MATCH_ID"]
        cluster.save_and_plot(plot=True, force=True)
        if i in rm_indices:
            match_indice = np.where(rm_indices == i)
            cluster.match(dr5[dr5_indices[match_indice]])
            cluster.save_match()
        else:
            cluster.matched = False
        if match_agn:
            if i in rm_mill_indices:
                match_indices = np.where(rm_mill_indices == i)
                cluster.agn(mill[mill_indices[match_indices]])
                #cluster.save_agn()
                cluster.save_and_plot(plot=True, force=True)
        clusters.append(cluster)
    if richness_range is not None:
        warnings.filterwarnings("default", category=RuntimeWarning)
        t2 = time()
        print(f"richness interval {richness_range} was finish in {t2 - t1} seconds.")
        return clusters
    grouped = np.sum(clusters)
    return grouped

@check_none
class sz_cluster:
    def __init__(
        self,
        RA,
        DEC,
        richness,
        richness_error,
        r,
        szmap,
        mask,
        redshift,
        box,
        redmapper_ID="NO - ID",
        data_mask_ratio=0.2,
    ):
        if RA is not None:
            self.RA = RA
            self.DEC = DEC
            self.richness = richness
            self.cluster_radius = r
            self.szmap = szmap
            self.mask = mask
            self.box = box
            self.z = float(redshift)
            self.MASK_FLAG = (
                np.size(mask[mask == 1]) / np.size(mask) <= 1 - data_mask_ratio
            )
            self.y_c = np.max(szmap)
            self.match()
            self.ID = 0 if redmapper_ID == None else redmapper_ID
            self.output_path = (
                data_path
                + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]
                + "redmapper_ID="
                + str(self.ID)
            )
            self.total_SNr_map = np.mean(self.szmap) ** 2 / np.std(self.szmap) ** 2
            if os.path.exists(self.output_path) == False:
                os.mkdir(self.output_path)
            self.match_with_agn = False
        else:
            pass
    def agn(self, agn = None):
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
    def save_and_plot(
        self, plot=False, scale="log", xlabel="r (kpc)", ylabel="$y$", force=False
    ):
        if len(os.listdir(self.output_path)) == 0 or force == True:
            about_cluster = {
                "RA": self.RA,
                "DEC": self.DEC,
                "richness": self.richness,
                "y_c": self.y_c,
                "y_c_err": self.y_c_err,
                "redshift": self.z,
                "MASK_FLAG": self.MASK_FLAG,
                "signal/noise ratio map": self.total_SNr_map,
            }
            if hasattr(self, "total_SNr"):
                about_cluster["total_SNr"] = self.total_SNr
            pd.DataFrame(about_cluster, index=[0]).to_csv(
                f"{self.output_path}/about_redmapper_ID={self.ID}.csv"
            )
            np.save(f"{self.output_path}/szmap.npy", self.szmap)
            np.save(f"{self.output_path}/mask.npy", self.mask)
            if hasattr(self, "profile") == True:
                data = np.zeros((5, len(self.profile)))
                data[0] = self.R
                data[1] = self.profile
                data[2] = self.errors
                data[4] = self.SNr
                np.save(f"{self.output_path}/profile.npy", data)

            if plot == True:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                im = ax1.imshow(
                    self.szmap
                )  # ,extent=(self.box[0][0],self.box[1][0],self.box[0][1],self.box[1][1]),origin='lower')
                cbar = plt.colorbar(im, ax=ax1, label="$y_{sz}$")
                ax1.plot([], [], " ", label=f"$\\lambda = $ {np.round(self.richness)}")
                ax1.plot([], [], " ", label=f"$z = $ {np.round(self.z,2)}")
                ax1.plot(
                    [],
                    [],
                    " ",
                    label=f"$y_c \\times 10^{4} = $ {np.round(self.y_c*1e4,3)}",
                )
                ax1.plot([], [], " ", label=f"$S/N =${np.round(self.total_SNr_map,2)}")
                ax2.imshow(
                    self.mask
                )  # ,extent=(self.box[0][0],self.box[1][0],self.box[0][1],self.box[1][1]),origin='lower')
                ax1.legend()
                fig.suptitle(f"Cluster DES-Y3 REDMAPPER ID={self.ID} and its mask")
                fig.tight_layout()
                fig.savefig(
                    f"{self.output_path}/redmapper_ID={self.ID}_y_compton_map-DR6.png"
                )
                plt.close()
                if hasattr(self, "profile") == True:
                    self.plot_profile()
        elif len(os.listdir(self.output_path)) > 0:
            pass

    # 		r = input(f"the folder {self.output_path} have already information. Do you want replace it? Y/N\n").strip().upper()
    # 		if "Y" in r:
    # 			self.save_and_plot(force=True)
    def generate_profile(
        self,
        from_path=False,
        r=[0, 300, 700],
        method_func=np.mean,
        center="DES",
        full_data=None,
        t_error = 'area',
    ):
        if from_path == False:
            if len(self.__dict__) == 0:
                raise Empty_Data("The cluster_data is totally empty!")
            else:
                if r.unit == u.kpc:
                    radius = self.cluster_radius
                elif r.unit == u.arcmin:
                    radius = self.theta
                theta = self.theta
                data = self.szmap
                R_bins, profile, err = radial_binning(data, r, patch_size = np.rad2deg(width))
                SNr = np.array([profile[i]/err[i] for i in range(len(err))])
                circles = []  # circles in the plot
                z = self.z
                alpha0 = 0.1
                brighter_pixel = np.mean(data[theta <= 2 * u.arcmin])
                brighter_pixel_err = np.std(data[theta <= 2 * u.arcmin]) / np.sqrt(np.size(data[theta <= 2 * u.arcmin]))
                indx_bp = np.where(data == brighter_pixel)
                self.ACT_center = indx_bp
                self.y_c = brighter_pixel
                self.y_c_err = brighter_pixel_err
                limits = self.x[0][0].value,self.x[-1][-1].value,self.y[0][0].value,self.y[-1][-1].value
                self.limits = limits
                if center == "ACT":
                    pass
                for i in range(len(r) - 1):
                    if r.unit == u.kpc:
                        r_circle = (
                            np.rad2deg(
                                (
                                 r[i + 1] / cosmo.angular_diameter_distance(z).to(u.kpc)
                                ).value
                            )
                            * 60
                        )
                    elif r.unit == u.arcmin:
                         r_circle = ((r[i + 1]).value)
                    circles.append(
                        plt.Circle(
                            (0, 0),
                            r_circle,
                            alpha=(alpha0 - i * (alpha0 / len(r))),
                            facecolor="red",
                            edgecolor="black",
                        )
                    )
                    circles.append(
                        plt.Circle((0, 0), r_circle, edgecolor="red", fill=False)
                    )

                self.profile = profile
                self.errors = err
                self.circles = circles
                self.R = R_bins * r.unit
                self.SNr = SNr
                self.total_SNr = np.sqrt(np.sum(SNr**2))

        elif from_path == True:
            if os.path.exists(self.output_path):
                profile_data = np.load(f"{self.output_path}/profile.npy")
                self.R = profile_data[0]
                self.profile = profile_data[1]
                self.errors = profile_data[2]
                self.SNr = profile_data[5]
                if full_data == True:
                    other_data = pd.read_csv(
                        f"{self.output_path}/about_redmapper_ID={self.ID}.csv"
                    )
                    self.RA = other_data.loc[0]["RA"]
                    self.DEC = other_data.loc[0]["DEC"]
                    self.richness = other_data.loc[0]["richness"]
                    self.y_c = other_data.loc[0]["y_c"]
                    #self.y_c_err = other_data.loc[0]["y_c_err"]
                    self.MASK_FLAG = other_data.loc[0]["MASK_FLAG"]
                    self.z = other_data.loc[0]["redshift"]
                    self.total_SNr = other_data.loc[0]["total_SNr"]
                    self.szmap = np.load(self.output_path+"/szmap.npy")
            if "match.csv" in os.listdir(self.output_path):
                match_csv = pd.read_csv(self.output_path + "/match.csv")
                self.match(match_csv)

    def plot_profile(self):
        if len(self.__dict__) == 0:
            raise Empty_Data("You can't plot a empty data!")
        elif hasattr(self, "profile") == False:
            raise Attr_error("The attribute profile is not defined yet.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
            ax2.imshow(self.szmap, origin="lower", interpolation = 'nearest', extent=self.limits)
            ax2.set(xlabel="(arcmin)", ylabel="(arcmin)")
            ax2.set_aspect("auto")
            ax2.scatter(
                0,
                0,
                marker="o",
                color="None",
                edgecolors="snow",
                s=50,
                label="DES centroid",
            )

            cord_x = np.interp(
                self.ACT_center[1],
                (0, self.szmap.shape[1]),
                (self.limits[0], self.limits[1]),
            )
            cord_y = np.interp(
                self.ACT_center[0],
                (0, self.szmap.shape[0]),
                (self.limits[2], self.limits[3]),
            )
            ax2.scatter(
                cord_x,
                cord_y,
                marker="<",
                color="None",
                edgecolors="black",
                s=50,
                label="brighter central pixell",
            )
            if self.match_with_agn == True:
                position = self.RA_agn, self.DEC_agn
                posx, posy = position[0] - self.RA, position[1] - self.DEC
                ax2.scatter(posx,posy, marker = "s", color = "cyan", edgecolors = "black", s = 50, label = "AGN")
            ax2.legend()
            [plt.gca().add_artist(c) for c in self.circles]
            ax1.plot([], [], " ", label=f"$\\lambda = $ {np.round(self.richness)}")
            ax1.plot([], [], " ", label=f"$z = $ {np.round(self.z,2)}")
            ax1.plot(
                [], [], " ", label=f"$y_c \\times 10^{4} = $ {np.round(self.y_c*1e4,3)}"
            )
            ax1.plot([], [], " ", label=f"$S/N$ ratio = {np.round(self.total_SNr,2)}")
            ax1.errorbar(
                self.R,
                self.profile,
                yerr=self.errors,
                ecolor="black",
                color="red",
                capsize=2,
            )
            ax1.legend()
            ax1.grid(True)
            ax1.set(yscale="log", xlabel=f"radio ({str(self.R.unit)})", ylabel="$y_{sz}$")

            fig.suptitle(f"Cluster DES-Y3 REDMAPPER ID={self.ID} and its profile")
            fig.tight_layout()
            fig.savefig(
                f"{self.output_path}/redmapper_ID={self.ID}_y_compton_profile-DR6.png"
            )
            plt.close()

    def __str__(self):
        dr5 = False if self.matched == False else f"matched with {self.name_match}"
        if hasattr(self, "profile") == False:
            return f"""Cluster data:
* (RA,DEC): \033[92m({self.RA},{self.DEC}\033[0m)
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* szmap shape: \033[92m{np.shape(self.szmap)}\033[0m
* mask flag value: \033[92m{self.MASK_FLAG}\033[0m
* output path: \033[92m{self.output_path}\033[0m
* DR5 : \033[92m{dr5}\033[0m
"""

        else:
            return f"""Cluster data:
* (RA,DEC): \033[92m({self.RA},{self.DEC}\033[0m)
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* szmap shape: \033[92m{np.shape(self.szmap)}\033[0m
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
        return grouped_clusters(
            self.R,
            self.theta,
            self.x,
            self.y,
            profiles,
            [self.errors, other.errors],
            [self.richness, other.richness],
            [self.z, other.z],
            [self.MASK_FLAG, other.MASK_FLAG],
            [self.szmap, other.szmap],
            [self.y_c, other.y_c],
            [self.y_c_err, other.y_c_err],
            [self.RA, other.RA],
            [self.DEC, other.DEC]
        )
    def calculate_Y(self, R, plot = False):
        if hasattr(R,'unit'):
            if hasattr(self,"theta") and R.unit == u.arcmin:
                r = self.theta
            elif hasattr(self,"cluster_radius") and R.unit == u.kpc:
                r = self.cluster_radius
            else:
                return 0
            dr = np.where(r < R)
            ydr = self.szmap[dr]
            Y = cosmo.angular_diameter_distance(self.z)**(-2) * simps(ydr)
            dr = r[dr].flatten()
            h = dr[0] - dr[1]
            d2y = np.gradient(np.gradient(dr))
            err = np.abs(h**4 / 180 * (dr[-1]) * (d2y[-1]/2))
            return Y.value
        else:
            return 0

        return Y
@check_none
class grouped_clusters:
    def __init__(
        self,
        R,
        theta,
        x,
        y,
        profiles,
        errors,
        richness,
        redshift,
        MASK_FLAG,
        szmap = [],
        y_c = [],
        y_c_err = [],
        ra = [],
        dec = []
    ):
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
        self.MASK_FLAG = (
            MASK_FLAG.tolist() if isinstance(MASK_FLAG, np.ndarray) else MASK_FLAG or []
        )
        self.z = (
            redshift.tolist() if isinstance(redshift, np.ndarray) else redshift or []
        )
        self.szmap = (
            szmap.tolist() if isinstance(szmap, np.ndarray) else szmap or []
        )
        self.y_c = (
            y_c.tolist() if isinstance(y_c, np.ndarray) else y_c or []
        )
        self.y_c_err = (
            y_c_err.tolist() if isinstance(y_c_err, np.ndarray) else y_c_err or []
        )
        self.ra = (
            ra.tolist() if isinstance(ra, np.ndarray) else ra or []
        )
        self.dec = (
            dec.tolist() if isinstance(dec, np.ndarray) else dec or []
        )
        if len(self.richness) > 0 and len(self.z) > 0:
            self.output_path = (
                data_path
                + config["FILES"]["GROUPED_CLUSTERS_PATH"]
                + f"GROUPED_CLUSTER_RICHNESS={np.round(np.min(self.richness))}-{np.round(np.max(self.richness))}"
                + f"REDSHIFT={np.round(np.min(self.z),2)}-{np.round(np.max(self.z),2)}"
            )
            self.N = len(self.richness)
        else:
            self.output_path = (
                data_path
                + config["FILES"]["GROUPED_CLUSTERS_PATH"]
                + f"GROUPED_CLUSTER_RICHNESS={np.round(self.richness)}"
                + f"REDSHIFT={np.round(self.z,2)}"
            )
            self.N = 1
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
                [other.MASK_FLAG],
                [other.szmap],
                [other.y_c],
                [other.y_c_err],
                [other.RA],
                [other.DEC]
            )

        new_profiles = list(self.profiles) + [
            profile for profile in other.profiles if profile is not None
        ]
        new_errors = list(self.errors) + list(other.errors)
        new_richness = list(self.richness) + list(other.richness)
        new_MASK_FLAG = list(self.MASK_FLAG) + list(other.MASK_FLAG)
        new_redshift = list(self.z) + list(other.z)
        new_szmap = list(self.szmap) + list(other.szmap)
        new_y_c = list(self.y_c) + list(other.y_c)
        new_y_c_err = list(self.y_c_err) + list(other.y_c_err)
        new_ra = list(self.ra) + list(other.ra)
        new_dec = list(self.dec) + list(other.dec)
        return grouped_clusters(
            self.R,
            self.theta,
            self.x,
            self.y,
            new_profiles,
            new_errors,
            new_richness,
            new_redshift,
            new_MASK_FLAG,
            new_szmap,
            new_y_c,
            new_y_c_err,
            new_ra,
            new_dec
        )

    def __str__(self):
        if len(self.profiles) < 4:
            return f"""Grouped cluster data:
* richness: \033[92m{self.richness}\033[0m
* redshift: \033[92m{self.z}\033[0m
* mask flag value: \033[92m{self.MASK_FLAG}\033[0m
* R: \033[92m{self.R}\033[0m
* profile_shape: \033[92m{np.shape(self.profiles)}\033[0m
"""
        else:
            mt = np.count_nonzero(self.MASK_FLAG)
            mf = len(self.MASK_FLAG) - mt
            return f"""Grouped cluster data:
* richness: [\033[92m{np.min(np.round(self.richness))},{np.max(np.round(self.richness))}\033[0m]
* redshift: [\033[92m{np.min(np.round(self.z,2))},{np.max(np.round(self.z,2))}\033[0m]
* mask flag value: \033[95mFalse\033[0m: {mf} , \033[95mTrue\033[0m: {mt}
* R: \033[92m{self.R}\033[0m
* profile_shape: \033[92m{np.shape(self.profiles)}\033[0m
"""

    def split_optimal_richness(
        self, SNr=10, Nmin=0.1, abs_min=4000, rdistance=2, ratio=True, method = 'mean', return_only_intervals = False
    ):
        richness = self.richness
        sorted_richness = np.sort(richness)
        profiles = np.array(self.profiles)
        errors = np.array(self.errors)
        intervals = [np.round(np.min(richness))]
        sorted_indices = np.argsort(richness)
        sorted_profiles = profiles[sorted_indices]
        sorted_errors = profiles[sorted_indices]
        unique_richness = np.unique(np.round(sorted_richness))
        rounded_richness = np.round(sorted_richness)
        saved_data = 0
        SNR_ARR = []
        for i in tqdm(range(1, len(unique_richness))):
            SNr_profiles = []
            current_richness = unique_richness[i]
            richness_cut = np.where(
                (rounded_richness > intervals[-1])
                & (rounded_richness <= current_richness)
            )
            selected_profiles = profiles[richness_cut]
            selected_errors = errors[richness_cut]
            SNr_profiles = []
            for j in range(len(selected_profiles)):
                current_SNr = np.sqrt(
                    np.sum(
                        [
                            selected_profiles[j][k] ** 2 / selected_errors[j][k] ** 2
                            for k in range(len(selected_profiles[j]))
                        ]
                    )
                )
                if np.isnan(current_SNr):
                    SNr_profiles.append(0)
                else:
                    SNr_profiles.append(current_SNr)
            total_SNr = np.median(SNr_profiles)
            print(total_SNr)
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
        output_subdata = []
        fig, ax = plt.subplots(figsize=(8, 6))
        hist, bins, _ = ax.hist(
            self.richness, bins=np.array(intervals)[1::], color="black", log = True
        )

        for i, value in enumerate(hist):
            plt.text(
                bins[i] + 0.5 * (bins[i + 1] - bins[i]),
                value,
                str(np.round(SNR_ARR[i], 2)),
                ha="center",
                va="bottom",
                transform=plt.gca().transData,
                fontsize=6,
                horizontalalignment='center', 
                verticalalignment='center'
            )
        ax.set(
            title=f"bins of richness $\\lambda$ with $S/N$ ratio $\\geq$ {SNr}",
            xlabel="richness $\\lambda$",
            ylabel="N of clusters"
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(
            f'{data_path}{config["FILES"]["GROUPED_CLUSTERS_PATH"]}/optimal_richness_bins.png'
        )
        plt.close()
        np.save(
            f"{data_path}{config['FILES']['GROUPED_CLUSTERS_PATH']}/intervals.npy",
            np.array(intervals),
        )
        for i in range(len(intervals) - 1):
            output_subdata.append(self.sub_group([intervals[i], intervals[i + 1]]))
        if return_only_intervals == False:
            return output_subdata
        else:
            return intervals
    def split_by_redshift(self, redshift_bins = None):
        subgroups = []
        for i in range(len(redshift_bins) - 1):
            subgroups.append(self.sub_group(redshift_interval = [redshift_bins[i],redshift_bins[i+1]]))
        return subgroups
    def sub_group(self, richness_interval=None, redshift_interval=None):
        if richness_interval is not None:
            richness = np.array(self.richness)
            richness_cut = np.where(
                (richness >= richness_interval[0]) & (richness < richness_interval[1])
            )
            subgroup = {}
            subgroup["profile"] = np.array(self.profiles)[richness_cut]
            subgroup["errors"] = np.array(self.errors)[richness_cut]
            subgroup["richness"] = richness[richness_cut]
            subgroup["z"] = np.array(self.z)[richness_cut]
            subgroup["MASK_FLAG"] = np.array(self.MASK_FLAG)[richness_cut]
            if hasattr(self, "szmap") and len(self.szmap) > 0:
                subgroup["szmap"] = np.array(self.szmap)[richness_cut]
            if hasattr(self, "y_c") and len(self.y_c) > 0:
                subgroup["y_c"] = np.array(self.y_c)[richness_cut]
            if hasattr(self,"y_c_err") and len(self.y_c_err) > 0:
                subgroup["y_c_err"] = np.array(self.y_c_err)[richness_cut]
            subgroup["ra"] = np.array(self.ra)[richness_cut]
            subgroup["dec"] = np.array(self.dec)[richness_cut]
            return type(self)(self.R, self.theta, self.x , self.y, *list(subgroup.values()))
        if redshift_interval is not None:
            redshift = np.array(self.z)
            redshift_cut = np.where(
                (redshift >= redshift_interval[0]) & (redshift < redshift_interval[1])
            )
            subgroup = {}
            subgroup["profile"] = np.array(self.profiles)[redshift_cut]
            subgroup["errors"] = np.array(self.errors)[redshift_cut]
            subgroup["richness"] = np.array(self.richness)[redshift_cut]
            subgroup["z"] = np.array(self.z)[redshift_cut]
            subgroup["MASK_FLAG"] = np.array(self.MASK_FLAG)[redshift_cut]
            if hasattr(self, "szmap") and len(self.szmap) > 0:
                subgroup["szmap"] = np.array(self.szmap)[redshift_cut]
            if hasattr(self, "y_c") and len(self.y_c) > 0:
                subgroup["y_c"] = np.array(self.y_c)[redshift_cut]
            if hasattr(self,"y_c_err") and len(self.y_c) > 0:
                subgroup["y_c_err"] = np.array(self.y_c_err)[redshift_cut]
            subgroup["ra"] = np.array(self.ra)[redshift_cut]
            subgroup["dec"] = np.array(self.dec)[redshift_cut]
            return type(self)(self.R, self.theta, self.x, self.y, *list(subgroup.values()))

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
            elif hastattr(self, "output_path"):
                print("You must define an output path first")
    def stacking(self, R_profiles = [100,200,300], plot = True, replace_mean = True, weighted = False, background_err = False, szmap = None, return_stack = False):
        if os.path.exists(self.output_path) == False:
            os.mkdir(self.output_path)
        maps_array = []
        z_mean = np.mean(self.z)
        szmap = enmap.read_map(data_path + DR6) if szmap is None else szmap
        for i in range(len(self.ra)):
            ra,dec = np.deg2rad([self.ra[i],self.dec[i]])
            stamp = reproject.thumbnails(szmap, coords = (dec,ra), r = width/2.)
            maps_array.append(stamp)
        with h5py.File(self.output_path + "szmaps.h5","w") as f:
            data = np.stack(maps_array, axis=0)
            f.create_dataset("maps", data = data, compression='gzip')
        stack = np.average(maps_array, axis = 0)
        self.stacking_map = np.average(maps_array, axis = 0)
        self.stacking_errors = np.std(maps_array, axis = 0)
        background_err = np.std(maps_array) / np.sqrt(np.size(maps_array))
        if return_stack == True:
            return self.stacking_map, self.stacking_errors, background_err
        R_bins, profile, err = radial_binning(stack, R_profiles, patch_size = np.rad2deg(width), weighted = weighted, errors = self.stacking_errors)
        pixel_width = np.rad2deg(width) * 60 / np.shape(stack)[0]
        x,y = np.indices(np.shape(stack))
        center = np.shape(stack)[0]//2,np.shape(stack)[1]//2
        x,y = (x - center[0])*pixel_width,(y - center[1])*pixel_width
        if background_err == True:
            err = np.sqrt(err**2 + background_err**2)
        if replace_mean:
            self.mean_profile = np.array(profile)
            self.error_in_mean = np.array(err)
            self.R = np.array(R_bins) * R_profiles.unit
            self.SNr = np.sqrt(np.sum(np.array(profile)**2/np.array(err)**2))
        if plot == True:
            fig, ax = plt.subplots(1,2, figsize = (12,6))
            ax[0].errorbar(R_bins.value, profile, yerr = err, label = "stacked profile", color = 'blue', fmt = "o", capsize = 3, alpha = 0.75)
            ax[0].set(xlabel = f"R {R_profiles.unit}", ylabel = r"$\langle y\rangle$", yscale = "log", title = r"stacked profile with $\lambda \in [%.1f,%.1f]$" % (np.min(self.richness), np.max(self.richness)))
            ax[0].grid(True)
            ax[0].legend()
            im = ax[1].imshow(stack, cmap = 'viridis', origin = 'lower', interpolation = 'nearest', extent = (x[0][0],x[-1][-1],y[0][0],y[-1][-1]))
            ax[1].set_aspect("auto")
            plt.colorbar(im, ax = ax[1], label = "$y$")
            fig.tight_layout()
            fig.savefig(f"{self.output_path}/stacking.png")
    def save_and_plot(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        label = f"RICHNESS={np.round(np.min(self.richness))}-{np.round(np.max(self.richness))}" + f"REDSHIFT={np.round(np.min(self.z),2)}-{np.round(np.max(self.z),2)}"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            self.richness, self.y_c, label="$y_c$", color="black", s=1, alpha=0.2
        )
        size = 30
        sort_richness = np.sort(self.richness)
        sort_y_c = np.array(self.y_c)[np.argsort(self.richness)]
        padded_y_c = np.pad(sort_y_c, (size // 2, size // 2), mode="edge")
        moving_avg_y = np.zeros_like(sort_y_c, dtype=np.float64)
        moving_err_y = np.zeros_like(sort_y_c, dtype=np.float64)
        for i in range(len(sort_y_c)):
            moving_avg_y[i] = np.mean(padded_y_c[i : i + size])
            moving_err_y[i] = np.std(padded_y_c[i : i + size], ddof=1) / np.size(
                padded_y_c[i : i + size]
            )
        ax.plot(sort_richness, moving_avg_y, color="red", label="moving average $y_c$")
        ax.fill_between(
            sort_richness,
            moving_avg_y - moving_err_y,
            moving_avg_y + moving_err_y,
            alpha=0.5,
            color="red",
        )

        self.avg_yc = moving_avg_y
        self.err_yc = moving_err_y
        yc = np.zeros((4, len(moving_avg_y)))
        yc[0] = sort_richness
        yc[1] = sort_y_c
        yc[2] = moving_avg_y
        yc[3] = moving_err_y
        np.save(f"{self.output_path}/yc.npy", yc)
        ax.legend()
        ax.set(
            title="$y_c$ in function of $\\lambda$",
            ylabel="$y_c$",
            xlabel="$\\lambda$",
            yscale="log",
        )
        ax.grid(True)
        fig.savefig(f"{self.output_path}/richness_y_c.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(self.richness,histtype='barstacked', edgecolor='black', alpha=0.7,log=True,color='green', label = 'DES-Y3')
        ax.set(
            title="distribution of richness $\\lambda$",
            xlabel="richness $\\lambda$",
            ylabel="N of clusters",
            yscale='log'
        )
        ax.grid(True)
        fig.savefig(f"{self.output_path}/richness_distribution.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.cm.seismic
        norm = plt.Normalize(np.min(self.richness), np.max(self.richness))
        SNr = []
        for i in range(len(self.profiles)):
            current_SNr = np.sqrt(
                np.sum(
                    [
                        self.profiles[i][j] ** 2 / self.errors[i][j] ** 2
                        for j in range(len(self.errors[i]))
                    ]
                )
            )
            SNr.append(current_SNr)
            ax.plot(
                self.R,
                self.profiles[i] / cosmo.efunc(self.z[i]),
                color=cmap(norm(self.richness[i])),
                alpha = 0.1,
                lw = 0.5,

            )
            ax.plot(self.R, self.profiles[i], color=cmap(norm(self.richness[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, label="richness $\\lambda$",cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]))
        ax.grid(True)
        ax.set(xlabel=f"$R$ ({self.R.unit})", ylabel=r"$\langle y \rangle$", title=f"Individual Profiles")
        ax.plot([], [], " ", label=rf"$N^o$ clusters = {len(self.profiles)}")
        ax.plot([], [], " ", label=rf'$z = [{np.round(np.min(self.z),2)} , {np.round(np.max(self.z),2)}]$')
        fig.savefig(f"{self.output_path}/profiles.png")
        fig.savefig(f"{data_path}{config['FILES']['GROUPED_CLUSTERS_PATH']}/profiles/profiles.png")
        if hasattr(self, "mean"):
            output_data = np.zeros((3, len(self.mean_profile)))
            output_data[0] = self.R
            output_data[1] = self.mean_profile
            output_data[2] = self.error_in_mean
            np.save(f"{self.output_path}/mean_profile.npy", output_data)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(
                self.R,
                self.mean_profile,
                yerr=self.error_in_mean,
                capsize=4,
            )
            if hasattr(self.R, "value"):
                ax.fill_between(
                    self.R.value,
                    self.mean_profile - self.error_in_mean,
                    self.mean_profile + self.error_in_mean,
                    alpha=0.2,
                    color="grey",
                )
            else:
                ax.fill_between(
                    self.R,
                    self.mean_profile - self.error_in_mean,
                    self.mean_profile + self.error_in_mean,
                    alpha=0.2,
                    color="grey",
                )
            ax.set(
                yscale="log",
                xlabel=f"$R$ ({self.R.unit})",
                ylabel="$\\langle y \\rangle$",
                title=f"Mean Profile of $y$ in $\\lambda \\in $ [{np.round(np.min(self.richness))},{np.round(np.max(self.richness))}]",
            )
            ax.plot([], [], " ", label=f"$S/N ratio = ${np.round(np.median(SNr),2)}")
            ax.plot([], [], " ", label=rf"$N^o$ clusters = {len(self.profiles)}")
            ax.plot([], [], " ", label=rf'$z = [{np.round(np.min(self.z),2)} , {np.round(np.max(self.z),2)}]$')
            ax.grid(True)
            ax.legend()
        fig.savefig(f"{self.output_path}/mean_profile.png")
        fig.savefig(f"{data_path}{config['FILES']['GROUPED_CLUSTERS_PATH']}/profiles/mean_profile{label}.png")
        about_group = {
            "mean redshift": np.mean(self.z),
            "min richness": np.min(self.richness),
            "max richness": np.max(self.richness),
            "mean richness": np.mean(self.richness),
            "N clusters": len(self.richness),
            "SNr": self.SNr,
        }
        pd.DataFrame(about_group, index=[0]).to_csv(
            f"{self.output_path}/about_group.csv"
        )
        plt.close("all")
        with h5py.File(f"{self.output_path}/data.h5", "w") as f:
            f.create_dataset("richness", data=self.richness)
            f.create_dataset("redshift", data=self.z)
            f.create_dataset("profiles", data=self.profiles)
            f.create_dataset("errors", data=self.errors)
            f.create_dataset("MASK_FLAG", data=self.MASK_FLAG)
            f.create_dataset("N", data=self.N)
            f.create_dataset("y_c", data=self.y_c)
            f.create_dataset("y_c_err", data=self.y_c_err)
            f.create_dataset("ra", data = self.ra)
            f.create_dataset("dec", data = self.dec)
        try:
            fig = plt.figure(figsize=(5,5))
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1,0])
            ax_histx = fig.add_subplot(gs[0,0],sharex=ax)
            ax_histy = fig.add_subplot(gs[1,1],sharey=ax)
            k = gaussian_kde(np.vstack([self.z,self.richness]))
            x = np.array(self.z)
            y = np.array(self.richness)
            xi,yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            ax.scatter(self.z,self.richness,color='red',alpha=0.8,s=5)
            ax_histx.hist(self.z,color='red',histtype='step', log = True)
            ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5,cmap='Reds',vmin=np.min(zi)*5,levels=25)
            ax_histx.tick_params(axis='x',labelbottom=False)
            ax_histy.hist(self.richness,color='red',histtype='step',orientation='horizontal', log = True)
            ax_histy.tick_params(axis='y',labelleft=False)
            ax.set(ylabel='Richness $\\lambda$',xlabel='Redshift $z$', title = "Richness-Redshift distribution")
            ax.grid(True)
            ax_histy.grid(True)
            ax_histx.grid(True)
            fig.tight_layout()
            fig.savefig(f'{self.output_path}/richness_redshift_scatter{label}.png')
        except:
            return
    def mass_richness_func(self, pivot=40, slope=1.29, normalization=10**14.45):
        return lambda l: (normalization * (l / pivot) ** slope)

    def completeness_and_halo_func(self, plot = False):
        zbins = int(config["STACKED_HALO_MODEL"]["zbins"])
        Mbins = int(config["STACKED_HALO_MODEL"]["Mbins"])
        df = pd.read_csv(
            completeness_file, delimiter="|", usecols=[1, 2, 3, 4]
        ).to_numpy()  # completeness file from DES
        richness_obs = df[:, 2]  # observed richness
        richness_cut = np.where(
            (richness_obs >= np.min(np.round(self.richness)))
            & (richness_obs < np.max(np.round(self.richness)))
        )
        completeness = df[richness_cut[0]]
        closest_redshift = np.unique(completeness[:, 0])[
            np.argmin(np.abs(np.unique(completeness[:, 0]) - np.mean(self.z)))
        ]
        completeness = completeness[completeness[:, 0] == closest_redshift]
        richness_true = completeness[:, 1]
        prob_distribution = []
        for i in range(len(richness_true)):
            sr = completeness[completeness[:, 1] == richness_true[i]]
            prob_distribution.append(sr[:, -1])
        P_true = np.array(prob_distribution).transpose()
        richness_obs = np.linspace(
            np.min(richness_obs), np.max(richness_obs), np.shape(P_true)[0]
        )
        sigmaML = 0.25
        richness_true = np.linspace(
            np.min(richness_true), np.max(richness_true), np.shape(prob_distribution)[0]
        )
        mass_range = np.log10(
            np.array(
                [
                    10 ** (14.45) * (np.min(self.richness) / 40) ** 1.29,
                    10 ** (14.45) * (np.max(self.richness) / 40) ** 1.29,
                ]
            )
        )
        mass_arr = np.logspace(mass_range[0], mass_range[1], Mbins)
        self.mass_range = mass_range
        richnes2mass = self.mass_richness_func()
        mass2richness = self.mass_richness_func(3 * 10**14, 0.75, 1.29)
        richess_true, mass = np.meshgrid(richness_true, mass_arr)
        richness_model = mass2richness(mass)
        P_mass = (
            1
            / (np.sqrt(2 * np.pi * sigmaML))
            * np.exp(
                -((np.log(richness_true) - np.log(richness_model)) ** 2)
                / (2 * sigmaML**2)
            )
        )
        product = P_true[:, :, None] * P_mass.transpose()
        richness_true = np.linspace(
            np.min(richness_true), np.max(richness_true), np.shape(prob_distribution)[0]
        )
        P_obs = trapz(product, x=richness_true, axis=1)
        P_MR = trapz(P_obs, x=richness_obs, axis=0)
        self.P_MR = P_MR

        # halo func
        mass_arr = np.logspace(12, 16, Mbins)
        z_arr = np.linspace(np.min(self.z),np.max(self.z),zbins)
        cosmo = ccl.Cosmology(**cosmological_model)
        mdef = ccl.halos.massdef.MassDef(500, "critical")
        a = 1 / (1 + z_arr)
        mfunc = ccl.halos.mass_function_from_name("Tinker10")
        mfunc = mfunc(cosmo, mdef)
        dndM = np.array([[mfunc(cosmo, M, ai) for M in mass_arr] for ai in a])
        dndM = dndM * 1/(mass_arr * np.log(10))
        self.dndM = dndM
        if plot == True:
            fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (8,8))
            im = ax1.imshow(np.log(np.abs(np.array(P_true))),interpolation='nearest',origin='lower',extent=(richess_true.min(),richess_true.max(),richness_obs.min(),richness_obs.max()))
            plt.colorbar(im,label='prob.',ax=ax1)
            ax1.set(title=f'Probability function $P(\lambda_o|\lambda_t)$',
                    ylabel='$\\lambda_{obs}$', xlabel='$\\lambda_{true}$')

            yticks = np.round(np.linspace(richness_obs.min(),richness_obs.max(),8))
            xticks = np.round(np.linspace(richness_true.min(),richness_true.max(),8))
            ax1.xaxis.set_ticks(xticks)
            ax1.yaxis.set_ticks(yticks)
            im = ax2.imshow(np.log(np.abs(np.array(P_obs))),interpolation='nearest',origin='lower',extent=(mass_arr.min(),mass_arr.max(),richness_obs.min(),richness_obs.max()))
            plt.colorbar(im,label='prob.',ax=ax2)
            ax2.set(title=f'Probability function $P(\lambda_o|M)$',
                    ylabel='$\\lambda_{obs}$', xlabel='$M_{\\odot}$')
            yticks = np.round(np.linspace(richness_obs.min(),richness_obs.max(),8))
            xticks = np.round(np.linspace(mass_arr.min(),mass_arr.max(),8)) 
            ax2.xaxis.set_ticks(xticks)
            ax2.yaxis.set_ticks(yticks)       
            ax3.plot(mass_arr,P_MR,color = 'black', label = '$P(M)$')
            ax3.plot(mass_arr,trapz(dndM,x = z_arr, axis = 0),color = 'red', label = 'halo func.')
            ax3.legend()
            ax3.grid(True)
            ax3.set(xlabel = "Mass ($M_{\\odot}$)", xscale = 'log', ylabel = "$P(M)$", yscale = 'log', title = 'Probability function $P(M)$ and $dn/dM$')   
            ax1.set_aspect('auto')
            ax2.set_aspect('auto')
            ax3.set_aspect('auto')
            fig.tight_layout()
            fig.savefig(f"{self.output_path}/Prob.png")
    def stacked_halo_model_func(self, profile_model,units = "arcmin"):
        zbins = int(config["STACKED_HALO_MODEL"]["zbins"])
        Mbins = int(config["STACKED_HALO_MODEL"]["Mbins"])
        P_MR = self.P_MR
        dndM = self.dndM
        M_range = self.mass_range
        mass_arr = np.logspace(M_range[0], M_range[1], Mbins).astype(np.float64)
        z_arr = np.linspace(np.min(self.z),np.max(self.z),zbins)
        dV = cosmo.differential_comoving_volume(z_arr).to(u.kpc**3 / u.sr)
        norm = trapz(dV * trapz(dndM * P_MR, x=mass_arr, axis=1), x=z_arr,axis = 0)
        #FWHM = 1.4
        #sigma = FWHM/(2*np.sqrt(2*np.log(2)))
        #gaussian = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-R**2 / (2*sigma**2))
        global func
        def func(R, params):        
            y_model = np.array([[profile_model(R, M, z, params) for M in mass_arr] for z in z_arr]).astype(np.float64)
            I = trapz(dV[:,None] * trapz(dndM[:, :, None] * (P_MR[None,:, None] * y_model), x=mass_arr, axis=1), x=z_arr, axis=0)
            mean_profile = np.array(I / norm).astype(np.float64)
            #if units == "arcmin": 
            #    mean_profile = np.convolve(mean_profile,gaussian,mode="same")
            return mean_profile
        return func

    def stacked_halo_model_func_by_bins(self, profile_model, units = 'arcmin' , zb = None):
        if zb is None:
            return
        grouped_by_richness = self.split_optimal_richness()
        grouped = []
        data = []
        err = []
        for c in grouped_by_richness:
            sc = c.split_by_redshift(zb)
            grouped.append([])
            data.append([])
            err.append([])
            for s in sc:
                if len(s.richness) > 1:
                    s.completeness_and_halo_func()
                    grouped[-1].append(s)
                    s.mean(from_path = True, search_closest = True)
                    data[-1].append(np.array(s.mean_profile))
                    err[-1].append(np.array(s.error_in_mean))
            data[-1] = np.array(data[-1], dtype = 'object')
            err[-1] = np.array(err[-1], dtype = 'object')
        func = []
        for i in range(len(grouped)):
            func.append([])
            for j in range(len(grouped[i])):
                func[-1].append(grouped[i][j].stacked_halo_model_func(profile_model, units))
        global general_func
        def general_func(R,params):
                res = []
                for i in range(len(func)):
                    res.append([])
                    for j in range(len(func[i])):
                        res[-1].append(np.array(func[i][j](R,params)))
                    res[-1] = np.array(res[-1],dtype = 'object')
                return np.array(res, dtype = 'object')
        return general_func, np.array(data, dtype = 'object'), np.array(err, dtype = 'object')
    def load_from_h5(self):
        if hasattr(self, "output_path"):
            with h5py.File(f"{self.output_path}/data.h5", "r") as f:
                self.z = f["redshift"][:]
                self.richness = f["richness"][:]
                self.profiles = f["profiles"][:]
                self.errors = f["errors"][:]
                self.MASK_FLAG = f["MASK_FLAG"][:]
                self.y_c = f["y_c"][:]
                self.ra = f["ra"][:]
                self.dec = f["dec"][:]
                self.N = len(self.richness)

    def yc_richness_relationship(self, model,random_tests = 30):
        model_name = model.__name__.replace('_',' ')
        if os.path.exists(self.output_path + "/yc relationship/") == False:
            os.mkdir(self.output_path + "/yc relationship/")
        richness = np.array(self.richness)
        if not hasattr(self,"avg_yc") and os.path.exists(self.output_path + "/yc.npy"):
            file = np.load(self.output_path + "/yc.npy")
            self.avg_yc = file[2]
            self.err_yc = file[3]
        redshift = np.array(self.z)
        Ez = cosmo.efunc(self.z) # in Mpc
        yc = np.array(self.y_c)
        yc = yc * Ez**(-1)
        moving_average = np.array(self.avg_yc)*Ez**(-1)
        moving_error = np.array(self.err_yc)*Ez**(-1)
        cut = np.mean(richness) + np.std(richness) #mean + 1std
        small_clusters,small_richness,z1 = yc[richness < cut],richness[richness < cut],redshift[richness < cut]
        big_clusters,big_richness,z2 = yc[richness >= cut],richness[richness >= cut],redshift[richness >= cut]
        fig = plt.figure(figsize = (5,4))
        n, bins, _ = plt.hist([small_richness, big_richness], histtype='barstacked', edgecolor='black', alpha=0.7,log=True,color=['green','red'], bins = 30)
        plt.axvline(x = np.mean(richness),color='blue',ls='--',label='mean richness')
        plt.axvline(x = np.mean(richness) + np.std(richness),color='black',ls='--',label='upper limit')
        plt.legend()
        plt.xlabel("richness $\\lambda$")
        plt.ylabel("N clusters")
        plt.title("Distribution of richness after categorization")
        fig.savefig(self.output_path + "/yc relationship/histrogram.png")
        params,cov = curve_fit(model,richness, yc, maxfev = 50000, p0 = (1e-4, -0.3)) #, -0.2, 50, 1e-3))
        err = np.sqrt(np.diag(cov))
        labels = [r'$\log_{10}(A)$',r'$\alpha$'] #,r'$\alpha_2$',r'$\lambda_{\mathrm{break}}$', r'$\omega$']
        fig = plt.figure(figsize = (8,6))
        ax = plt.axes()
        ax.scatter(small_richness,small_clusters,marker="^",c = z1, cmap = 'magma',label=f'small clusters N = {len(small_clusters)}',s=2,alpha=0.1)
        ax.scatter(big_richness,big_clusters,marker="s",c = z2, cmap = 'magma', label=f'big clusters N = {len(big_clusters)}',s=2,alpha=0.1)
        ax.errorbar(np.sort(richness), moving_average, yerr = moving_error, capsize = 0.2, color = 'red', alpha = 0.4)
        norm = Normalize(vmin=redshift.min(), vmax=redshift.max())
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='magma'),cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]))
        cbar.set_label('Redshift')
        plt.plot(np.sort(richness),model(np.sort(richness),*params),label = model_name, color = 'blue')
        text = []
        for i in range(len(labels)):
            if labels[i].split('_')[0] == r'$\log':
                text.append(f'{labels[i]} : {np.round(np.log10(params[i]),5)} $\pm$ {np.round(err[i]/(np.log(10) * params[i]),5)}')
            else:
                text.append('%s' % labels[i] + ': $%.5f$ ' % params[i] + '$\pm %.5f$' % err[i] )
#        chi2_r = np.sum(( (yc - model(richness, *params))**2) / moving_error ** 2) / len(moving_average)
#        text.append(r'$\chi^{2}_r = %.3f$' % chi2_r)
        text = '\n'.join(text)
        props = dict(boxstyle='round', facecolor='lightgreen', edgecolor = 'black', alpha=0.5)
        ax.text(0.05, (0.001 + len(params) * 0.05), text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
        lower_bound = model(np.sort(richness), *(params - err))
        upper_bound = model(np.sort(richness), *(params + err))
        ax.fill_between(np.sort(richness), lower_bound, upper_bound, color='blue', alpha=0.3)
        ax.grid(True)
        if 'broken' in model_name.split(' '):
            if model_name == 'smoothly broken power law':
                xbreak = params[-2]
            elif model_name == 'broken power law':
                xbreak = params[-1]
            ax.axvline(x = xbreak, ls = '--', color = 'red', label = 'break')

        ax.legend(fontsize = 8, loc = 'lower right')
        ax.set(xscale = 'log', xlabel = r'richness $\lambda$', ylabel = r'$y_c E(z)^{-1}$', yscale = 'log', title = 'distribution of central compton parameter $y_c$ in DES-Y3 + ACT-DR6')
        fig.savefig(self.output_path + f"/yc relationship/yc_{model_name}.png", dpi = 800)
        real_params = params
        real_error = err
        test_parameters, test_errors = [], []
        n = 0
        # while n < random_tests:
        #     if os.path.exists(self.output_path + "/yc relationship/tests") == False:
        #         os.mkdir(self.output_path + "/yc relationship/tests/")
        #     random_indx = np.random.randint(0,len(small_clusters),2500)
        #     small_clusters_random_sample = small_clusters[random_indx]
        #     small_richness_random_sample = small_richness[random_indx]
        #     redshift_random_sample = z1[random_indx]
        #     yc = np.concatenate((small_clusters_random_sample,big_clusters))
        #     richness = np.concatenate((small_richness_random_sample,big_richness))
        #     redshift = np.concatenate((redshift_random_sample,z2))
        #     try:
        #         params,cov = curve_fit(model,richness,yc, p0 = (real_params))
        #     except:
        #         continue
        #     test_parameters.append(params)
        #     errors = np.sqrt(np.diag(cov))
        #     test_errors.append(errors)
        #     fig = plt.figure(figsize = (6,5))
        #     plt.scatter(small_richness,small_clusters,marker="^",c = z1, cmap = 'magma',label=f'small clusters N = {len(small_clusters_random_sample)}',s=4,alpha=0.3)
        #     plt.scatter(big_richness,big_clusters,marker="s",c = z2, cmap = 'magma', label=f'big clusters N = {len(big_clusters)}',s=4,alpha=0.3)
        #     norm = Normalize(vmin=redshift.min(), vmax=redshift.max())
        #     cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='magma'))
        #     cbar.set_label('Redshift')
        #     plt.plot(richness,model(richness,*params),label='best fitting',color='blue')
        #     plt.plot([],[]," ",label=r"$\log_{10}(A)$ = " + str(np.round(np.log10(params[0]),2))+"±"+str(np.round(np.log10(errors[0]),2)))
        #     plt.plot([],[]," ",label=r"slope = "+str(np.round(params[1],2))+"±"+ str(np.round(errors[1],2)))
        #     plt.plot([],[]," ",label=r"$\log_{10}(pivot)$ = "+str(np.round(np.log10(params[2]),2)) +"±"+str(np.round(np.log10(errors[2]),2)))
        #     plt.grid(True)
        #     lower_bound = model(richness, *(params - errors))
        #     upper_bound = model(richness, *(params + errors))
        #     plt.fill_between(richness, lower_bound, upper_bound, color='blue', alpha=0.3)
        #     plt.legend()
        #     plt.ylabel(r"$\langle y_c \rangle$")
        #     plt.xlabel(r"richness $\lambda$")
        #     plt.title("distribution of mean central compton parameter \n$y_c$ in DES-Y3 + ACT-DR6")
        #     plt.yscale('log')           
        #     fig.savefig(self.output_path + f"/yc relationship/tests/test_n={n}.png")
        #     plt.close()
        #     n += 1
        # if len(test_errors) > 1:
        #     weights = 1 / np.array(test_errors)**2
        #     mean_parameters = np.mean(test_parameters, axis = 0)
        #     mean_errors = 1 / np.sqrt(np.sum(weights, axis = 0))
        #     y = np.arange(0,len(mean_parameters),1)
        #     labels = ["$A$","slope","pivot"]
        #     colors = ["purple","pink","green"]
        #     fig = plt.figure(figsize = (4,8))
        #     ax = plt.axes()
        #     for i in range(len(mean_parameters)):
        #         ax.errorbar(mean_parameters[i]/real_params[i],y[i],xerr = np.abs(mean_errors[i]/real_error[i]),color=colors[i],fmt= "o",alpha=0.5, capsize = 4)
        #         ax.errorbar(1,y[i],xerr = 1,color="black",fmt = "o",alpha=0.3, capsize = 4)
        #     ax.set_xlabel("parameter normalized by real parameter")
        #     ax.set_yticks(y , labels)
        #     ax.set_ylabel("Parameter")
        #     ax.grid(True)
        #     ax.set_xlim((1 - 3, 1 + 3))
        #     ax.set_title("Parameter distribution with random tests")
        #     fig.savefig(self.output_path + "/yc relationship/parameters_distribution.png")
        # #yc redshift vs richness
        # Ez = cosmo.efunc(self.z) # in Mpc
        # yc = np.array(self.y_c)*Ez**(-2/3)
        # richness = np.array(self.richness)
        # redshift = np.array(self.z)
        # moving_average = np.array(self.avg_yc)
        # moving_error = np.array(self.err_yc)
        # cut = np.mean(richness) + np.std(richness) #mean + 1std
        # small_clusters,small_richness,z1 = yc[richness < cut],richness[richness < cut],redshift[richness < cut]
        # big_clusters,big_richness,z2 = yc[richness >= cut],richness[richness >= cut],redshift[richness >= cut]
        # params,cov = curve_fit(model,np.sort(richness),moving_average)
        # err = np.sqrt(np.diag(cov))
        # fig = plt.figure(figsize = (6,5))
        # plt.scatter(small_richness,small_clusters,marker="^",c = z1, cmap = 'magma',label=f'small clusters N = {len(small_clusters)}',s=4,alpha=0.3)
        # plt.scatter(big_richness,big_clusters,marker="s",c = z2, cmap = 'magma', label=f'big clusters N = {len(big_clusters)}',s=4,alpha=0.3)
        # norm = Normalize(vmin=redshift.min(), vmax=redshift.max())
        # cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='magma'))
        # cbar.set_label('Redshift')

        # plt.errorbar(np.sort(richness),moving_average,yerr=moving_error,color='grey',alpha = 0.5, label = 'moving average')
        # plt.plot(richness,model(richness,*params),label = 'best fitting', color = 'blue')
        # plt.plot([],[]," ",label=r"$\log_{10}(A)$ = " + str(np.round(np.log10(params[0]),2))+"±"+str(np.round(np.log10(err[0]),2)))
        # plt.plot([],[]," ",label=r"slope = "+str(np.round(params[1],2))+"±"+ str(np.round(err[1],2)))
        # plt.plot([],[]," ",label=r"$\log_{10}(pivot)$ = "+str(np.round(np.log10(params[2]),2)) +"±"+str(np.round(np.log10(err[2]),2)))
        # lower_bound = model(richness, *(params - err))
        # upper_bound = model(richness, *(params + err))
        # plt.fill_between(richness, lower_bound, upper_bound, color='blue', alpha=0.3)
        # plt.grid(True)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # order = [0,1,2,6,3,4,5]  # Change the order of labels as desired
        # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        # plt.ylabel(r"$\langle y_c \rangle E(z)^{-2/3}$")
        # plt.xlabel(r"richness $\lambda$")
        # plt.title("distribution of mean central compton parameter \n$y_cE(z)^{-2/3}$ in DES-Y3 + ACT-DR6")
        # plt.yscale('log')
        # fig.savefig(self.output_path + "/yc relationship/ycEz_vs_richness.png")
        # real_params = params
        # real_error = err
        # test_parameters, test_errors = [], []
        # n = 0
        #redshift-yc relationship
        redshift = np.array(self.z)
        yc = np.array(self.y_c) * Ez**(-1)
        richness = np.array(self.richness)
        labels = [r'$\log_{10}(A)$',r'$\alpha$']
        fig = plt.figure(figsize = (8,6))
        ax = plt.axes()
        ax.scatter(redshift,yc,c = np.log(richness), cmap = 'Reds',label=f'N = {len(small_clusters)}',s=4,alpha=0.6,edgecolors='black')
        norm = Normalize(vmin=np.log(richness).min(), vmax=np.log(richness).max())
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'),cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]))
        cbar.set_label('$\\ln \\lambda$')
        ax.set(yscale = 'log', xscale = 'log', xlabel = 'redshift $z$', ylabel = '$y_c$', title = "redshift $z$ vs $y_c$ distribution in ACT-DR6 + DES-Y3")
        size = 30
        sort_redshift = np.sort(redshift)
        sort_y_c = np.array(self.y_c)[np.argsort(redshift)]
        padded_y_c = np.pad(sort_y_c, (size // 2, size // 2), mode="edge")
        moving_avg_y = np.zeros_like(sort_y_c, dtype=np.float64)
        moving_err_y = np.zeros_like(sort_y_c, dtype=np.float64)
        for i in range(len(sort_y_c)):
            moving_avg_y[i] = np.mean(padded_y_c[i : i + size])
            moving_err_y[i] = np.std(padded_y_c[i : i + size], ddof=1) / np.size(
                padded_y_c[i : i + size]
            )
        ax.plot(sort_redshift, moving_avg_y, color="purple", alpha = 0.8, label="moving average $y_c$")
        ax.fill_between(
            sort_redshift,
            moving_avg_y - moving_err_y,
            moving_avg_y + moving_err_y,
            alpha=0.5,
            color="purple",
            )
        params,cov = curve_fit(model,redshift,yc, maxfev = 50000, p0 = (1e-4, 0))
        err = np.sqrt(np.diag(cov))
        text = []
        for i in range(len(labels)):
            if labels[i].split('_')[0] == r'$\log':
                text.append(f'{labels[i]} : {np.round(np.log10(params[i]),5)} $\pm$ {np.round(err[i]/(np.log(10) * params[i]),5)}')
            else:
                text.append('%s' % labels[i] + ': $%.5f$ ' % params[i] + '$\pm %.5f$' % err[i] )
        #chi2_r = np.sum(( (moving_average - model(np.sort(redshift), *params))**2) / moving_error ** 2) / len(moving_average)
        #text.append(r'$\chi^{2}_r = %.3f$' % chi2_r)
        text = '\n'.join(text)
        props = dict(boxstyle='round', facecolor='lightgreen', edgecolor = 'black', alpha=0.5)
        ax.text(0.05, (0.001 + len(params)*0.05), text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
        if 'broken' in model_name.split(' '):
            if model_name == 'smoothly broken power law':
                xbreak = params[-2]
            elif model_name == 'broken power law':
                xbreak = params[-1]
            ax.axvline(x = xbreak, ls = '--', color = 'red', label = 'break')

        ax.plot(sort_redshift,model(sort_redshift,*params),label = 'best fitting', color = 'blue')
        lower_bound = model(sort_redshift, *(params - err))
        upper_bound = model(sort_redshift, *(params + err))
        plt.fill_between(sort_redshift, lower_bound, upper_bound, color='blue', alpha=0.3)
        plt.grid(True)
        plt.legend(loc = "lower right", fontsize = 8)
        fig.savefig(self.output_path + f"/yc relationship/redshift_{model_name}.png")

        #using act-dr5 to find Y500 propto richness
        fit = fits.open("/data2/javierurrutia/szeffect/data/ilc_SZ_yy_noKspaceCor.fits")[0]
        wcs = WCS(fit.header)
        arcmin_per_pixel_x = (wcs.pixel_scale_matrix[0, 0] * u.degree).to(u.arcmin)
        arcmin_per_pixel_y = (wcs.pixel_scale_matrix[1, 1] * u.degree).to(u.arcmin)
        dr5_clusters = []
        saved_names = []
        redmapper_path = data_path + config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]
        for n,redmapper_cluster in enumerate(os.listdir(redmapper_path)):
            if "match.csv" in os.listdir(redmapper_path + '/' + redmapper_cluster):
                ID = int(redmapper_cluster.split('=')[-1])
                cluster = sz_cluster(None)
                cluster.ID = ID
                cluster.output_path = redmapper_path + "/" + redmapper_cluster
                cluster.generate_profile(from_path = True,full_data = True)
                shape = np.shape(cluster.szmap)
                box = ((shape[0]//2 - 30, shape[1]//2 - 30),
                        (shape[0]//2 + 30, shape[1]//2 + 30))
                cut = np.array(cluster.szmap)[box[0][0]:box[1][0],box[0][1]:box[1][1]]
                center = np.where(cut == np.max(cut))
                center = (box[1][0] - center[0], box[1][1] - center[1]) #given by brighter pixell
                x,y = np.indices(np.shape(cluster.szmap))
                r = np.sqrt(((x - center[0])*arcmin_per_pixel_x)**2 + (((y - center[1]))*arcmin_per_pixel_y)**2)
                cluster.theta = r
                cluster.cluster_radius = r* cosmo.kpc_proper_per_arcmin(cluster.z)
                dr5_clusters.append(cluster)
                saved_names.append(cluster.name_match.to_numpy()[0])
        dr5_richness = np.array([dr5_clusters[i].richness for i in range(len(dr5_clusters))])
        M500_Msun = np.array([dr5_clusters[i].M500c_match[0] * 1e14 * const.M_sun.value for i in range(len(dr5_clusters))]) * u.kg
        M500 = M500_Msun.to(u.Msun)
        dr5_redshift = np.array([dr5_clusters[i].z for i in range(len(dr5_clusters))])
        #richness vs m500

        fig,ax = plt.subplots(figsize = (8,6))
        pivot = np.round(np.mean(M500),1)
        def power_law(mass, norm, slope):
            return norm*(mass/1e13)**slope
        param, cov = curve_fit(power_law, M500.value, dr5_richness, maxfev = 50000, p0 = (10e-2,2/3), method = 'dogbox')
        err = np.diag(np.sqrt(cov))
        self.RM_params = [param,err]
        ax.scatter(M500, dr5_richness, color = 'blue', s = 2, alpha = 0.8, label = "DR5")
        ax.plot(np.sort(M500),power_law(np.sort(M500),*param), color = 'purple', alpha = 0.8, label = 'best fitting')
        ax.set(xlabel = "$M_{500}$", ylabel = r'richness $\lambda$', yscale = 'log', xscale = 'log', title = "richness - $M_{500}$ distribution")
        lower_bound = power_law(np.sort(M500), *(param - err)).value
        upper_bound = power_law(np.sort(M500), *(param + err)).value
        ax.fill_between(np.sort(M500).value, lower_bound, upper_bound, color = 'purple', alpha = 0.3)
        labels = [r'$\log_{10}(A)$',r'$\alpha$']
        text = []
        for i in range(len(labels)):
            if labels[i].split('_')[0] == r'$\log':
                text.append(f'{labels[i]} : {np.round(np.log10(param[i]),5)} $\pm$ {np.round(err[i]/(np.log(10) * param[i]),5)}')
            else:
                text.append('%s' % labels[i] + ': $%.5f$ ' % param[i] + '$\pm %.5f$' % err[i])
        txt = '\n'.join(text)
        ax.text(0.60, 0.001 + len(text)*0.05, txt, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
        ax.legend()
        ax.grid(True)
        fig.savefig(f"{self.output_path}/yc relationship/RM.png")
        #Y vs M500
        rho_c = cosmo.critical_density(dr5_redshift).to(u.kg / u.m**3)
        R500_m = np.array([( (( 3 * M500_Msun[i] ) / (4 * np.pi * 500 * rho_c[i]) )**(1/3) ).value for i in range(len(M500_Msun))]) * u.m
        R500_kpc = R500_m.to(u.kpc)
        kpc2arcmin = cosmo.arcsec_per_kpc_proper(dr5_redshift).to(u.arcmin/u.kpc)
        R500_in_arcmin = R500_kpc * kpc2arcmin
        Y = np.array([dr5_clusters[i].calculate_Y(R500_in_arcmin[i]) for i in range(len(dr5_clusters))])
        Y_Mpc = (Y*cosmo.efunc(dr5_redshift)**(-2/3)*cosmo.angular_diameter_distance(dr5_redshift)**2)
        def virial_pred(x,a):
            return a*x**(5/3)
        param,cov = curve_fit(virial_pred,M500, Y_Mpc)
        err = np.sqrt(np.diag(cov))
        pred = virial_pred(np.sort(M500), *param)
        lower_bound = virial_pred(np.sort(M500), *(param - err)).value
        upper_bound = virial_pred(np.sort(M500), *(param + err)).value
        fig, ax = plt.subplots(figsize = (8,6))
        fig.subplots_adjust(bottom = 0.25)
        ax.scatter(M500, Y_Mpc, s = 4, color = 'black', alpha = 0.3, label = 'data')
        ax.plot(np.sort(M500), pred, color = 'blue', alpha = 0.8, label = '$\propto M^{5/3}$')
        ax.fill_between(np.sort(M500).value, lower_bound, upper_bound, color = 'blue', alpha = 0.6)
        ax.grid(True)
        text = [r'$\log_{10}(M_0)$ =' + f"${np.round(np.log10(param[0]),5)} \pm {np.round( 1/(np.log(10) * param[0]) * err[0],5)}$"]
        props = dict(boxstyle='round', facecolor='lightgreen', edgecolor = 'black', alpha=0.5)
        ax.set(yscale = 'log', xscale = 'log', xlabel = r'$M_{500}$ $M_{\odot}$', ylabel = '$D_{a}(z)^{2} Y_{500}E(z)^{-2/3}$ $Mpc^{2}$')
        ax2 = fig.add_axes([ax.get_position().x0, 0.05, ax.get_position().width, 0.2])
        ax2.scatter(np.sort(M500), Y_Mpc[np.argsort(M500)]/pred, s = 2, color = 'darkblue', alpha = 0.3)
        ax2.plot(np.sort(M500), np.ones(np.shape(M500)), ls = '--', color = 'black')
        ax2.set(yscale = 'log', xscale = 'log', xlabel = r'$M_{500}$ $M_{\odot}$', ylabel = 'ratio')
        ax2.grid(True)
        fig.suptitle("$Y_{500} - M_{500}$ relationship from ACT-DR6 + DES-Y3")

        params,cov = curve_fit(model, M500.value, Y_Mpc, maxfev = 50000, p0 = (2.5e-3, -4, -1.6667, 3e14,0.1), method = 'dogbox')
        err = np.sqrt(np.diag(cov))
        pred = model(np.sort(M500).value, *params)
        lower_bound = model(np.sort(M500).value, *(params - err))
        upper_bound = model(np.sort(M500).value, *(params + err))
        if 'broken' in model_name.split(' '):
            if model_name == 'smoothly broken power law':
                xbreak = params[-2]
            elif model_name == 'broken power law':
                xbreak = params[-1]
            ax.axvline(x = xbreak, ls = '--', color = 'red', label = 'break')
            ax2.axvline(x = xbreak, ls = '--', color = 'red')
        labels = [r'$\log_{10}(A_1)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10}(M_{\mathrm{break}})$',r'$\omega$']

        for i in range(len(labels)):
            if labels[i].split('_')[0] == r'$\log':
                text.append(f'{labels[i]} : {np.round(np.log10(params[i]),5)} $\pm$ {np.round(err[i]/(np.log(10) * params[i]),5)}')
            else:
                text.append('%s' % labels[i] + ': $%.5f$ ' % params[i] + '$\pm %.5f$' % err[i])

        txt = '\n'.join(text)
        ax.text(0.05, 0.001 + len(text)*0.05, txt, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
        ax.plot(np.sort(M500), pred, color = 'purple', alpha = 0.8, label = model_name)
        ax.fill_between(np.sort(M500).value, lower_bound, upper_bound, color = 'purple', alpha = 0.6)
        ax.legend(loc = 'lower right')
        fig.savefig(self.output_path + f"/yc relationship/YM_{model_name}.png")

        #richness
        labels = [r'$\log_{10}(A_1)$',r'$\alpha_1$',r'$\alpha_2$',r'$\lambda_{\mathrm{break}}$',r'$\omega$']
        params,cov = curve_fit(model, dr5_richness, Y_Mpc, maxfev = 50000, p0 = (1e-2, -2.1, -1.3, 103, 1e-3))
        richness_break = params[-2]
        err = np.sqrt(np.diag(cov))
        pred = model(np.sort(dr5_richness), *params)
        lower_bound = model(np.sort(dr5_richness), *(params - err))
        upper_bound = model(np.sort(dr5_richness), *(params + err))
        fig, ax = plt.subplots(figsize = (8,6))
        fig.subplots_adjust(bottom = 0.25)
        ax.scatter(dr5_richness, Y_Mpc, s = 4, color = 'black', alpha = 0.3, label = 'data')
        ax.plot(np.sort(dr5_richness), pred, color = 'blue', alpha = 0.8, label = model_name)
        ax.fill_between(np.sort(dr5_richness), lower_bound, upper_bound, color = 'blue', alpha = 0.6)
        ax.grid(True)
        text = []
        for i in range(len(labels)):
            if labels[i].split('_')[0] == r'$\log':
                text.append(f'{labels[i]} : {np.round(np.log10(params[i]),5)} $\pm$ {np.round(err[i]/(np.log(10) * params[i]),5)}')
            else:
                text.append('%s' % labels[i] + ': $%.5f$ ' % params[i] + '$\pm %.5f$' % err[i])
        text = '\n'.join(text)
        props = dict(boxstyle='round', facecolor='lightgreen', edgecolor = 'black', alpha=0.5)
        ax.text(0.05, (0.001 + len(params)*0.05), text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
        ax.set(yscale = 'log', xscale = 'log', xlabel = r'richness $\lambda$', ylabel = '$D_{a}(z)^{2}Y_{500} E(z)^{-2/3}$ $Mpc^{2}$')
        ax2 = fig.add_axes([ax.get_position().x0, 0.05, ax.get_position().width, 0.2])
        ax2.scatter(np.sort(dr5_richness), Y_Mpc[np.argsort(dr5_richness)]/pred, s = 2, color = 'darkblue', alpha = 0.3)
        ax2.plot(np.sort(dr5_richness), np.ones(np.shape(dr5_richness)), ls = '--', color = 'black')
        ax2.set(yscale = 'log', xscale = 'log', xlabel = r'richness $\lambda$', ylabel = 'ratio')
        ax2.axvline(x = richness_break, ls = '--', color = 'red')
        if 'broken' in model_name.split(' '):
            if model_name == 'smoothly broken power law':
                xbreak = params[-2]
            elif model_name == 'broken power law':
                xbreak = params[-1]
            ax.axvline(x = xbreak, ls = '--', color = 'red', label = 'break')
            ax2.axvline(x = xbreak, ls = '--', color = 'red')

        ax.legend(loc = 'lower right')
        ax2.grid(True)
        fig.suptitle(r"$Y_{500} - \lambda$ relationship from ACT-DR6 + DES-Y3")
        fig.savefig(self.output_path + f"/yc relationship/YR_{model_name}.png")

def prueba(n_samples):
#    path = "GROUPED_CLUSTER_RICHNESS=20.0-208.0REDSHIFT=0.1-0.92"
#    g = grouped_clusters(None)
#    g.output_path = data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + "GROUPED_CLUSTER_RICHNESS=20.0-208.0REDSHIFT=0.1-0.92"
#    g.load_from_h5()
#    g.yc_richness_relationship(getattr(helpers, "power_law"))
    from matplotlib.colors import ListedColormap
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from astropy.table import Table
    DR6 = config["FILES"]["DR6-ACT-map"]
    DR5 = config["FILES"]["DR5-ACT-map"]
    DES_Y3 = config["FILES"]["Y3-REDMAPPER"]
    MASK_DR6 = config["FILES"]["MASK_DR6-ACT-map"]
    szmap = fits.open(data_path + DR6)[0].data
    redmapper = fits.open(data_path + DES_Y3)[1].data
    mask = fits.open(data_path + MASK_DR6)[0].data
    mask[np.where((mask > 0) & (mask < 1/2))] = 1/2
    mask[mask > 1/2] = 1
    mask_enmap = enmap.read_map(data_path + MASK_DR6)
    RA,DEC = np.array(redmapper['RA']),np.array(redmapper['DEC'])
    RA[RA > 180] = RA[RA > 180] - 360
    sz_enmap = enmap.read_map(data_path + DR6)
    milliquas = fits.open(data_path + agn_catalog)
    milliquas_catalog = Table(milliquas[1].data)
    types = ["Q","A","B","K","N"]
    names = ["QSO type I broad-line core-dominated","AGN type I Seyferts/host-dominated",
            "BL Lac", "Narrow-Line Type II", "Seyferts/host-dominated Type II"]
    colors = ["purple", "green", "slategrey", "blue", "cyan"]
    milliquas_catalog["grouped_type"] = group_types(np.array(milliquas_catalog["TYPE"]), types, names)
    grouped_type = np.unique(milliquas_catalog["grouped_type"])
    #milliquas_catalog = milliquas_catalog.group_by("grouped_type")
    bool_with2lobes = ["2" in t.split() for t in milliquas_catalog["TYPE"]]
    with2lobes = milliquas_catalog[bool_with2lobes]
    Z_2lobes = with2lobes["Z"]
    Z = milliquas_catalog["Z"]
    QSO = milliquas_catalog[milliquas_catalog["grouped_type"] == grouped_type[0]]
    AGN = milliquas_catalog[milliquas_catalog["grouped_type"] == grouped_type[1]]
    BL = milliquas_catalog[milliquas_catalog["grouped_type"] == grouped_type[2]]
    K = milliquas_catalog[milliquas_catalog["grouped_type"] == grouped_type[3]]
    N = milliquas_catalog[milliquas_catalog["grouped_type"] == grouped_type[4]]
    splitted = [QSO, AGN, BL, K, N]
    fig,ax = plt.subplots(figsize = (8,5))
    ax.hist(Z, label = "entire catalog", color = 'red', linestyle = '--', lw = 2, histtype = "step")
    [ax.hist(splitted[i]["Z"], label = names[i], color = colors[i], linestyle = '--', lw = 2, histtype = "step") for i in range(len(names))]
    ax.hist(Z_2lobes, label = "sources with 2 radio lobes", color = 'gold', linestyle = '--', lw = 2, histtype = "step")
    ax.grid(True)
    ax.set(xlabel = 'redshift $z$', ylabel = "N of sources", yscale = 'log', title = 'redshift distribution on milliquas')
    ax.legend()
    fig.savefig("milliquas_z_distribution.png")
    return with2lobes
    """
    fig,ax = plt.subplots(figsize = (10,5))
    cmap = ListedColormap(['white','black' ,'crimson'])
    ax.imshow(mask, extent = [-180, 180, -90, 20], interpolation = 'nearest', origin = 'lower', cmap = cmap, label = 'ACT-DR6')
    ax.set(xlabel = 'RA (degrees)', ylabel = 'DEC (degrees)', title = 'ACT-DR6 + DES-Y3 Gold',
           yticks = np.arange(-90,90,15), xticks = np.arange(-180, 180, 30))
    RA,DEC = np.array(redmapper['RA']),np.array(redmapper['DEC'])
    RA[RA > 180] = RA[RA > 180] - 360
    kde = gaussian_kde(np.vstack([RA,DEC]))
    density = kde(np.vstack([RA,DEC]))
    ax.scatter(RA, DEC, c = density, cmap = 'autumn', alpha = 0.4, s = 3, label = 'DES-Y3 RM clusters')
    ax.grid(True, color = 'black')
    ax.legend()
    fig.savefig('map.png')
    """
    ra,dec = (RA[25],DEC[25])
    ypix, xpix = enmap.sky2pix(sz_enmap.shape, sz_enmap.wcs, np.deg2rad((dec,ra)))
    widht = 2 #degrees
    box = np.deg2rad([[dec - widht/2, ra - widht/2], [dec + widht/2, ra + widht/2]])
    submap = sz_enmap.submap(box)
    submask = mask_enmap.submap(box)
    print(np.shape(submap))
    box_px = [xpix - np.shape(submap)[0]//2, ypix - np.shape(submap)[1]//2,
              xpix + np.shape(submap)[0]//2, ypix + np.shape(submap)[1]//2]
    mask[mask != 1] = 0
    fig, ax = plt.subplots(figsize = (10,5))
    im = ax.imshow((szmap*mask), extent = [-180, 180, -60, 20], interpolation = 'nearest', cmap = 'bwr',
                    vmax = 5e-6, vmin = -5e-6, origin = 'lower')
    ax.scatter(RA, DEC, c = 'purple', s = 3, alpha = 0.2, label = 'DES-Y3 RM clusters')
    ax.set(xlabel = 'RA (degrees)', ylabel = 'DEC (degrees)', title = 'ACT-DR6 y-compton map',
           yticks = np.arange(-90,90,15), xticks = np.arange(-180, 180, 30))
    ax.grid(True, color = 'black')
    box = np.rad2deg(box)
    axins = ax.inset_axes([0.85, 0.6, 0.5,0.5], xlim = (box[0][1],box[1][1]), ylim = (box[0][0],box[1][0]),
                          xticklabels=[], yticklabels=[])
    axins.imshow((submap*submask), cmap = 'bwr', extent = (box[0][1],box[1][1],box[0][0],box[1][0]))
    ax.indicate_inset_zoom(axins, edgecolor = 'black', lw = 4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    plt.colorbar(im ,label = '$y$-compton parameter', orientation = 'horizontal', cax = cax)
    ax.set_aspect('auto')
    fig.tight_layout()
    fig.savefig('ymap.png')
    fig, ax = plt.subplots()
    ax.imshow(submap*submask, cmap = 'bwr')
    fig.savefig("submap.png")
if __name__ == "__main__":
    if with_pool == False:
        print("Running \033[91mwithout multiprocessing...\033[0m")
        if only_stacking == True:
            R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
            R_units = config["CLUSTER PROPERTIES"]["r_units"]
            R_profiles = R_profiles * getattr(u, R_units)
            path = os.listdir(data_path + "GROUPED_CLUSTERS/")
            output_paths = [data_path + "GROUPED_CLUSTERS/" + p for p in path if (p.split("_")[0] == "GROUPED" and p != "GROUPED_CLUSTER_RICHNESS=20.0-224.0REDSHIFT=0.1-0.92")]
            print("The script was inicialized only \033[91mstacking profiles\033[0m")
            t1 = time()
            for i in range(len(output_paths)):
                c = grouped_clusters(None)
                c.output_path = output_paths[i]
                c.load_from_h5()
                c.stacking(R_profiles, plot = True, weighted = True, background_err = True, szmap = szmap)
            t2 = time() - t1
            print("Stacking profiles was ended and took %.5f seconds." % t2)
        else:
            extract_cluster_data(None)
    elif with_pool == True:
        print("Running \033[91mwith multiprocessing...\033[0m")
        richness_bins = prop2arr(config["EXTRACT"]["RICHNESS BINS"], dtype=np.float64)
        intervals = []
        use_starmap = False
        for i in range(len(richness_bins) - 1):
            intervals.append([richness_bins[i], richness_bins[i + 1]])
        with Pool(processes=len(intervals)) as pool:
            R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
            R_units = config["CLUSTER PROPERTIES"]["r_units"]
            R_profiles = R_profiles * getattr(u, R_units)
            t1 = time()
            print(
                f"extract_cluster_data was inicialized with \033[92mn_nodes = {len(intervals)}\033[0m"
            )

            szmap = enmap.read_map(data_path + DR6)
            sz_clusters = fits.open(data_path + DR5)[1].data
            cluster_catalog = fits.open(data_path + DES_Y3)[1].data
            agn_catalog = Table(fits.open(data_path + MILLIQUAS)[1].data)

            if use_starmap == True:
                args = [(intervals[i], szmap, sz_clusters, cluster_catalog, agn_catalog) for i in range(len(intervals))]
                groups = pool.starmap(extract_cluster_data, args)
            else:
                args = intervals
                groups = pool.map(extract_cluster_data, args)

            pool.close()
            pool.join()
            grouped = np.concatenate(groups)
            grouped_clusters = np.sum(grouped)
            grouped_clusters.mean("weighted")
            grouped_clusters.stacking(R_profiles, plot = True, weighted = True, background_err = True, szmap = szmap)
            grouped_clusters.save_and_plot()
            split_by_richness = grouped_clusters.split_optimal_richness()
            data = []
            for c in split_by_richness:
                sc = c.split_by_redshift(redshift_bins)
                for s in sc:
                    if len(s)>1:
                        data.append(s)
            [d.stacking(R_profiles, plot = True, weighted = True, background_err = True, szmap = szmap) for d in data]
            [d.save_and_plot() for d in data]
            t2 = time()
            print(f"multiprocessing took {t2 - t1} seconds.")


