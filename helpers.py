import numpy as np
import re
import argparse
import astropy.units as u
import numpy as np
import astropy
import configparser
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from matplotlib import colors as mcolors
from scipy import stats
from matplotlib.ticker import MaxNLocator
import requests
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, KDTree
from bs4 import BeautifulSoup as bs
from difflib import SequenceMatcher
import os
import healpy as hp
from astropy.io import fits
from scipy.stats.kde import gaussian_kde
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import norm as norm_scipy
from tqdm import tqdm
able_colors = list(mcolors.CSS4_COLORS.keys())
import importlib
from pixell import enmap
from scipy.integrate import simpson as simp
from scipy.integrate import trapezoid as trapz
from astropy import constants as const
from astropy import units as u
import pyccl as ccl
import matplotlib.ticker as ticker
import h5py
import shutil 
import sys
from multiprocessing import Manager, pool
from astropy.cosmology import Planck18 as planck18

global check_none
def check_none(cls):
	class Wrapper(cls):
		def __init__(self,*args,**kwargs):
			if any(arg is None for arg in args) or any(value is None for value in kwargs.values()):
				args = (None,) * (getattr(cls, '__init__').__code__.co_argcount - 1)
				kwargs = {key: None for key in kwargs}
			super(Wrapper, self).__init__(*args, **kwargs)
	return Wrapper

class Found_Error_Config(Exception):
	pass

class Empty_Data(Exception):
	pass

class Attr_error(Exception):
	pass

def prop2arr(prop,delimiter=',',dtype=np.float64, remove_white_spaces = True):
	"""
	convert a property from a configuration file to a numpy array
	"""
	arr = prop.replace(' ','').split(delimiter) if remove_white_spaces else prop.split(delimiter)
	return np.array(arr,dtype=dtype)


def power_law(x,a,b):
    return a*(x/45)**b

def double_power_law(x, A1,alpha1, A2, alpha2, x0):
    y = np.zeros_like(x)
    y[x < x0] = A1 * x[x < x0]**alpha1
    y[x >= x0] = A2 * x[x >= x0]**alpha2 - A2*x0**alpha2 + A1 * x0 ** alpha1
    return y

def smooth_power_law(x, A1, alpha1, A2, alpha2, x0, width):
    t = np.tanh((x - x0) / width)
    y = A1 * (x ** alpha1) * (1 - t) + A2 * (x ** alpha2) * t
    return y

def broken_power_law(x, A, alpha1, alpha2, x0):
    y = np.zeros_like(x)
    mask = x <= x0
    y[mask] = A*(x[mask]/x0)**(alpha1)
    y[np.logical_not(mask)] = A*(x[np.logical_not(mask)]/x0)**(alpha2)
    return y

def smoothly_broken_power_law(x, A, alpha1, alpha2, x0, delta):
    return A*(x/x0)**(-alpha1)*(1/2 * (1 + (x/x0)**(1/delta)))**((alpha1 - alpha2)*delta)

def moving_average_func(x,y,size, median = False):
    sort_x = np.sort(x)
    sort_y = np.array(y)[np.argsort(sort_x)]
    padded_y = np.pad(sort_y, (size // 2, size // 2), mode="edge")
    moving_avg_y = np.zeros_like(sort_y, dtype=np.float64)
    moving_err_y = np.zeros_like(sort_y, dtype=np.float64)
    for i in range(len(sort_y)):
        m = np.mean(padded_y[i : i + size]) if median == False else np.median(padded_y[i : i + size])
        if m <= 0 or m < np.min(y):
            m = moving_avg_y[-1] 
        moving_avg_y[i] = m
        moving_err_y[i] = np.std(padded_y[i : i + size], ddof=1) / np.size(padded_y[i : i + size])
    return sort_x,moving_avg_y, moving_err_y

def str2bool(string):
	return True if string.lower() == "true" else False

def extract_values(s):
    match = re.match(r'GROUPED_CLUSTER_RICHNESS=(\d+\.\d+)-(\d+\.\d+)REDSHIFT=(\d+\.\d+)-(\d+\.\d+)', s)
    if match:
        richness_range = (float(match.group(1)), float(match.group(2)))
        redshift_range = (float(match.group(3)), float(match.group(4)))
        return richness_range, redshift_range
    else:
        return None, None

def closest_path(target, paths):
    target_richness, target_redshift = extract_values(target)
    closest_dist = float('inf')
    closest_str = None
    for s in paths:
        richness_range, redshift_range = extract_values(s)
        if richness_range is not None and redshift_range is not None:
            richness_dist = abs(target_richness[0] - richness_range[0]) + abs(target_richness[1] - richness_range[1])
            redshift_dist = abs(target_redshift[0] - redshift_range[0]) + abs(target_redshift[1] - redshift_range[1])
            total_dist = richness_dist + redshift_dist
            if total_dist < closest_dist:
                closest_dist = total_dist
                closest_str = s

    return closest_str


def log_inhomogeneous(arr):
    log = np.array([np.array([np.log(list(arr[i][j])) for j in range(len(arr[i]))]) for i in range(len(arr))], dtype = 'object')
    return log

def flatten(array):
    flattened = []
    for item in array:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def extract_parameters(result):
    parameters,errors = [], []
    best = list(result.params.values())
    for i in range(len(best)):
        parameters.append(best[i].value)
        errors.append(best[i].stderr)
    return parameters, errors

def group_types(table, names, types):
    grouped_types = np.full_like(table, 'Other', dtype='U10')
    for i in range(len(names)):
        grouped_types[np.char.startswith(table, names[i])] = types[i]
    return grouped_types

def list_array_or_tuple_type(arg):
    try:
        value = eval(arg)
        if isinstance(value, (list, np.ndarray, tuple)):
            return value
        else:
            raise argparse.ArgumentTypeError("Argument must be a list, tuple or numpy array.")
    except:
        raise argparse.ArgumentTypeError("Invalid argument format.")

def random_initial_steps(limits, n):
    params = np.zeros(n)
    for i in range(len(params)):
            if np.all(limits < 0):
                lower = limits[0]*0.9
                upper = limits[1]*1.1
            elif limits[0] < 0 and limits[1] > 0:
                lower = limits[0]*0.9
                upper = limits[1]*0.9
            else:
                lower = limits[0]*1.1
                upper = limits[1]*0.9
            params[i] = np.random.uniform(lower, upper)
    return params

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    if norm:
        acf /= acf[0]

    return acf

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def Hartlap_unbiased_estimator(C, Nb, Nran):
    return (Nran - Nb - 2)/ (Nran - 1) * np.linalg.inv(C)

def min_separation(ra, dec, deg = True):
    """ Compute the minimum angular separation using KDTree """
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    xyz = np.vstack(coords.cartesian.xyz).T 
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=2) 
    min_sep_rad = np.min(distances[:, 1])
    min_sep = (min_sep_rad * u.rad).to(u.arcmin).value if deg == False else (min_sep_rad * u.rad).to(u.deg).value
    return min_sep

def compute_separation(ra0, dec0, ra_list, dec_list):

    coords = SkyCoord(ra=ra_list, dec=dec_list, unit="deg")
    ref_coord = SkyCoord(ra=ra0, dec=dec0, unit="deg")

    xyz_list = np.array([coords.cartesian.x, coords.cartesian.y, coords.cartesian.z]).T
    xyz_ref = np.array([ref_coord.cartesian.x, ref_coord.cartesian.y, ref_coord.cartesian.z])
    tree = KDTree(xyz_list)
    distances, indices = tree.query(xyz_ref, k=len(ra_list))

    separations = 2 * np.arcsin(distances / 2) * (180 / np.pi)

    return separations


def extract_patch(ra0, dec0, hdu = None, data = None, wcs = None, dtheta = 5, dtype = "fits"):

    if dtype == "fits":
        data = hdu.data if (hdu is not None and data is None) else data
        wcs = WCS(hdu.header) if (hdu is not None and wcs is None) else wcs

        center = SkyCoord(ra0*u.deg, dec0*u.deg, frame='icrs')
        px, py = wcs.world_to_pixel(center)

        dtheta_deg = dtheta / 60 

        pix_scale = np.abs(hdu.header["CDELT1"]) 
        dtheta_pix = int(dtheta_deg / pix_scale)

        x_min, x_max = int(px - dtheta_pix//2), int(px + dtheta_pix//2)
        y_min, y_max = int(py - dtheta_pix//2), int(py + dtheta_pix//2)

        return data[y_min:y_max, x_min:x_max]
    elif dtype == "healpy":
        nside = hp.get_nside(data)
        center = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
        theta = np.radians(90 - center.dec.deg)
        phi = np.radians(center.ra.deg)

        dtheta_rad = np.deg2rad(dtheta / 60)

        pix_indices = hp.query_disc(nside, hp.ang2vec(theta, phi), dtheta_rad)
        return data[pix_indices]


def generate_sequence(N0, N, N_max):
    first_multiple = ((N0 + N - 1) // N) * N  # Closest multiple of N ≥ N0
    if first_multiple == N0:
        res = np.arange(N0, N_max + 1, N)
    else:
        res =  np.arange(first_multiple, N_max + 1, N)
    return np.array(list([N0]) + list(res))

def autocorr_time_from_chain(chain, Nbins = 20):
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), Nbins)).astype(int)
    tau = np.empty(len(N))
    for i, n in enumerate(N):
        tau[i] = autocorr_new(chain[:, :n])
    return N,tau

import numpy as np

def cov_matrices_from_maps(maps, R_profiles, width=None, dtype=np.float32):
    """
    Compute covariance matrices for a batch of maps.

    Parameters
    ----------
    maps : array_like
        Array of shape (N, H, W) containing N map realizations.
    R_profiles : array_like
        Radius bin edges in arcmin, shape (Nr+1,).
    width : float
        Full width of each map in degrees.
    dtype : data-type
        Desired dtype for intermediate arrays.

    Returns
    -------
    covs : ndarray
        Array of shape (N, Nr, Nr) containing covariance matrices.
    """
    maps = np.asarray(maps, dtype=dtype)
    N, H, W = maps.shape
    Nr = len(R_profiles) - 1

    y, x = np.indices((H, W))
    cy, cx = H // 2, W // 2
    pix_size = (width / H) * 60.0 
    r_grid = np.sqrt(((y - cy) * pix_size) ** 2 + ((x - cx) * pix_size) ** 2)

    masks = [(r_grid > R_profiles[i]) & (r_grid <= R_profiles[i+1]) for i in range(Nr)]

    covs = np.empty((N, Nr, Nr), dtype=dtype)

    for idx in range(N):
        data = maps[idx]
        binned = [data[mask] for mask in masks]
        maxlen = max(arr.size for arr in binned)
        mat = np.full((Nr, maxlen), np.nan, dtype=dtype)
        for i, arr in enumerate(binned):
            mat[i, :arr.size] = arr
        mmat = np.ma.masked_invalid(mat)
        cov = np.empty((Nr, Nr), dtype=dtype)
        for i in range(Nr):
            for j in range(i, Nr):
                cov[i, j] = np.ma.cov(mmat[[i, j], :])[0, 1] if i != j else np.ma.var(mmat[i, :])
                cov[j, i] = cov[i, j]
        covs[idx] = cov

    return covs


import numpy as np

def compute_covariance_per_map(maps, R_profiles, width, full = False, dtype = np.float32):
    """
    For each map in `maps`, bin its pixels radially and compute
    its covariance matrix of radial-bin values via numpy.ma.cov.

    Parameters
    ----------
    maps : list of 2D numpy arrays or MaskedArrays
        Square maps (all same shape N×N).
    R_profiles : array-like
        Radial bin edges, in arcminutes (length = nbins+1).
    width_deg : float
        Angular width of each map side, in degrees.

    Returns
    -------
    cov_list : list of 2D numpy arrays
        Each entry is the (nbins×nbins) covariance matrix for that map.
    mean_profiles : list of 1D numpy arrays
        Each entry is the length-nbins mean radial profile for that map.
    counts : 1D numpy array
        Number of valid pixels in each bin (same for all maps).
    radial_bin_indices : 2D integer array
        Bin index (0…nbins−1) for each pixel of shape (N,N).
    """
    n_maps = len(maps)
    N = maps[0].shape[0]
    arcmin_per_deg = 60.0
    pix_size = (width * arcmin_per_deg) / N
    Rpix = np.array(R_profiles) / pix_size
    nbins = len(Rpix) - 1

    # build one radial-index map
    y, x = np.indices((N, N))
    center = (N - 1) / 2
    r = np.hypot(y - center, x - center)
    radial_bin_indices = np.digitize(r, bins=Rpix) - 1

    # precompute counts per bin (using first map’s mask for valid pix)
    first = maps[0]
    if not isinstance(first, np.ma.MaskedArray):
        first = np.ma.masked_invalid(first)
    counts = np.array([
        np.sum((radial_bin_indices == b) & ~first.mask)
        for b in range(nbins)
    ])

    cov_list = np.zeros((len(maps), len(R_profiles) - 1, len(R_profiles) - 1), dtype = np.float32)
    if full == True:
        mean_profiles = []
        data_list = []
    # loop over maps
    for i,M in enumerate(maps):
        # ensure masked array
        if not isinstance(M, np.ma.MaskedArray):
            Mma = np.ma.masked_invalid(M)
        else:
            Mma = M

        # build data array: rows=bins, cols up to max count
        Nmax = counts.max()
        data = np.ma.masked_all((nbins, Nmax))

        # fill each row
        for b in range(nbins):
            vals = Mma[radial_bin_indices == b]
            data[b, : vals.size] = vals

        # compute this map’s mean profile and covariance
        mean_prof = np.array([row.mean() for row in data])
        cov = np.ma.cov(data, rowvar=True, bias=False, allow_masked=True)
        cov_list[i] = cov.astype(np.float32)
        if full == True:
            data_list.append(data)
            mean_profiles.append(mean_prof)
    if full == True:
        return cov_list, mean_profiles, data_list, radial_bin_indices, counts
    else:
        return cov_list


def compute_radial_covariance(maps, R_profiles_arcmin, width_deg):
    """
    Compute the radial profile covariance matrix from multiple maps.

    Parameters
    ----------
    maps : list of 2D numpy arrays or MaskedArrays
        Input maps, all assumed square and same size (N x N).
    R_profiles : array_like
        Radial bin edges in arcminutes.
    width_deg : float
        Width of the full map (in degrees).

    Returns
    -------
    cov : 2D numpy array
        Covariance matrix between radial bins (shape: nbins x nbins)
    mean_profile : 1D array
        Average radial profile over maps (length: nbins)
    profiles : 2D array
        Array of shape (n_maps, nbins), the radial profiles per map
    counts : 1D array
        Number of valid pixels per bin
    """

    # --- Setup and constants
    n_maps = len(maps)
    N = maps[0].shape[0]  # assuming square maps
    arcmin_per_deg = 60.0
    pixel_size_arcmin = (width_deg * arcmin_per_deg) / N
    R_profiles_pix = np.array(R_profiles) / pixel_size_arcmin
    nbins = len(R_profiles_pix) - 1

    # --- Compute radial index map (same for all maps)
    y, x = np.indices((N, N))
    center = (N - 1) / 2
    r_pix = np.hypot(y - center, x - center)
    radial_bin_indices = np.digitize(r_pix, bins=R_profiles_pix) - 1  # 0-based bin indices

    # --- Prepare profiles
    profiles = np.empty((n_maps, nbins))
    profiles[:] = np.nan

    # Count valid pixels per bin (use first map for this)
    counts = np.array([
        np.sum((radial_bin_indices == i) & ~np.ma.getmaskarray(maps[0]))
        for i in range(nbins)
    ])

    # --- Extract profile from each map
    for i, M in enumerate(maps):
        if not isinstance(M, np.ma.MaskedArray):
            M = np.ma.masked_invalid(M)  # treat NaNs as masked
        for b in range(nbins):
            mask = (radial_bin_indices == b)
            values = M[mask]
            if values.count() > 0:
                profiles[i, b] = values.mean()
    
    # --- Compute mean profile and covariance matrix
    mean_profile = np.nanmean(profiles, axis=0)
    cov = np.ma.cov(profiles, rowvar=False, ddof=1, )  # shape: (nbins, nbins)

    return cov, mean_profile, profiles, counts


def radial_binning2(x, R, width=None, r=None, full=False):
    """
    Compute the mean (and optionally standard deviation) radial profile 
    for one or multiple 2D arrays (patches).

    Parameters:
        x (2D array or list/array of 2D arrays): Input data patch(es).
        R (array): Radial bin edges (in arcmin).
        width (float, optional): Total width of the image in degrees.
        r (2D array, optional): Precomputed radial distance array (in arcmin).
        full (bool, optional): If True, also return the standard deviation 
            per bin and the number of pixels in each bin.

    Returns:
        If full is False:
            averages: 1D array (or 2D array with shape (N, n_bins) for multiple patches)
                      with the mean values in each radial bin.
        If full is True:
            R_bins: Bin centers.
            averages: as above.
            stds: Standard deviation per bin.
            counts: Number of pixels in each radial bin (common to all patches).
    """
    # If x is a list, or a 2D array, ensure we work with a 3D array.
    if isinstance(x, list):
        # Convert list of 2D arrays to a 3D array: shape (N, ny, nx)
        X = np.stack(x)
    elif x.ndim == 2:
        # Single patch: add a leading dimension
        X = x[None, ...]
    else:
        # Assume already a 3D array (multiple patches)
        X = x

    N, ny, nx = X.shape

    # Compute the radial coordinate array r (in arcmin) if not provided.
    if r is None:
        if width is None:
            raise ValueError("Either r or width must be provided.")
        # Build pixel grid (assuming center is the patch center)
        py, px = np.indices((ny, nx))
        cy, cx = ny // 2, nx // 2
        py, px = py - cy, px - cx
        # Convert pixel size: (width in degrees / ny) * 60 = arcmin per pixel
        pix_size = (width / ny) * 60.0  
        r = np.sqrt((px * pix_size) ** 2 + (py * pix_size) ** 2)
    # r is common to all patches; flatten it.
    r_flat = r.ravel()
    
    # Get bin indices for each pixel.
    # np.digitize returns values in 1...len(R), so subtract 1 to get 0-indexed bin numbers.
    bin_idx = np.digitize(r_flat, bins=R) - 1  
    n_bins = len(R) - 1

    # Create a boolean matrix of shape (n_pix, n_bins) where each column corresponds to a bin.
    # This vectorizes the operation of assigning each pixel to a bin.
    bin_mask = (np.arange(n_bins)[None, :] == bin_idx[:, None]).astype(np.float64)
    
    # Flatten each patch into a row vector: shape (N, n_pix)
    X_flat = X.reshape(N, -1)
    
    # Compute the sum in each bin for each patch via matrix multiplication.
    # sums: shape (N, n_bins)
    sums = X_flat @ bin_mask  
    # Compute number of pixels in each bin (same for all patches)
    counts = np.sum(bin_mask, axis=0)
    
    # Avoid division by zero.
    counts_safe = np.where(counts == 0, 1, counts)
    
    # Compute the average per bin for each patch.
    averages = sums / counts_safe  # shape (N, n_bins)
    
    if full:
        # For standard deviation, compute the sum of squares per bin.
        sq_sums = (X_flat**2) @ bin_mask
        # variance = (mean of squares) - (square of mean)
        variance = sq_sums / counts_safe - averages**2
        # Clip negative variance due to numerical errors, then take sqrt.
        stds = np.sqrt(np.clip(variance, a_min=0, a_max=None))
    
    # Compute bin centers.
    R_bins = (np.array(R[:-1]) + np.array(R[1:])) / 2.0

    # If only one patch was provided, squeeze the arrays to 1D.
    if N == 1:
        averages = averages[0]
        if full:
            stds = stds[0]

    if full:
        # Return counts as integers (number of pixels in each bin)
        return R_bins.astype(float), averages.astype(float), stds.astype(float), counts.astype(int)
    else:
        return averages.astype(float)



def radial_binning(data, R, weighted = False, errors = None, wcs = None):
    #data is assumed as a enmap
    if np.shape(data)[0] == np.shape(data)[1]:
        area = enmap.area(data.shape, wcs)
        width = np.rad2deg(np.sqrt(area))
        pixwidth = (width / np.shape(data)[0]) * 60
        center = np.shape(data)[0]//2 , np.shape(data)[1]//2
        R_pixel = np.array(R / pixwidth).astype(int)
        profile, err, R_cent = [], [], []
        arrs = []
        for i in range(0, len(R_pixel) - 1):
            r_in = R_pixel[i]
            r_out = R_pixel[i + 1]
            bin_cent = (r_in + r_out)/2.
            jmin = center[0] - r_out
            jmax = center[0] + r_out + 1
            kmin = center[1] - r_out
            kmax = center[1] + r_out + 1
            target = []
            target_err = []
            for j in np.arange(jmin, jmax):
                for k in np.arange(kmin, kmax):
                    jk = np.array([j,k])
                    dist = np.linalg.norm(jk - np.array(center))
                    if dist > r_in and dist <= r_out:
                        target.append(data[j][k])
                        if weighted == True and errors is not None:
                            target_err.append(errors[j][k])
            if weighted == True and errors is not None:
                w = 1/np.array(target_err)**2
                mean = np.average(target,weights = w)
                error = np.sqrt(1/np.sum(w))
            else:
                mean = np.mean(target)
                error = np.std(target)
            arrs.append(target)
            profile.append(mean)
            err.append(error)
            R_cent.append(bin_cent)
        R_cent = [R[i] + R[i + 1] for i in range(len(R) - 1)]
        return np.array(R_cent)/2, np.array(profile), np.array(err), arrs

def parse_value(value):
    """Convert config values to appropriate types."""
    if value.lower() == "none":
        return None
    elif value.lower() in ["true", "false"]:
        return value.lower() == "true"
    elif "," in value:  # Handle comma-separated values (e.g., lists)
        return [v.strip() for v in value.split(",")]
    try:
        return float(value) if "." in value else int(value)  # Convert numbers
    except ValueError:
        return value  # Return as string if conversion fails


def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    config_dict = {}
    
    for section in config.sections():
        for key, value in config[section].items():
            config_dict[key] = parse_value(value)

    return config_dict

def power_law_model(r,z,r0,z0,A,B,C):
    return A * (r / r0) ** B * ((1 + z)/(1 + z0))**C

def bootstrap_p_value(x, y, num_resamples=10000):
    # Compute the observed correlation coefficient
    observed_corr = np.corrcoef(x, y)[0, 1]
    
    # Initialize an array to store bootstrap correlation coefficients
    bootstrap_corrs = np.zeros(num_resamples)
    
    # Generate bootstrap samples and compute correlation for each
    for i in range(num_resamples):
        # Resample with replacement
        indices = np.random.choice(len(x), size=len(x), replace=True)
        x_resampled = x[indices]
        y_resampled = y[indices]
        
        # Compute the correlation for the resampled data
        bootstrap_corrs[i] = np.corrcoef(x_resampled, y_resampled)[0, 1]
    
    # Compute the p-value as the fraction of bootstrap correlations greater than or equal to the observed correlation
    p_value = np.mean(np.abs(bootstrap_corrs) >= np.abs(observed_corr))
    
    return p_value

def split_multimodal(data, min_chi2 = None, nbins = 200, threshold = 0.5, output_path = "param",
                     bins_ratio = 0.25, aim = None, only_aim = False, distance = None, height = None, **kwargs):
    fig_kwargs = kwargs.pop('fig_kwargs', {})
    hist_kwargs = kwargs.pop('hist_kwargs', {})
    scatter_kwargs = kwargs.pop('scatter_kwargs', {})
    axes_kwargs = kwargs.pop('axes_kwargs', {})
    q = np.percentile(data,[16, 50, 84])
    best = q[1]
    lower,upper = np.diff(q)
    fig,ax = plt.subplots(1,**fig_kwargs)
    hist = plt.hist(data, bins = nbins, density = True, **hist_kwargs);
    counts, bins = hist[0],hist[1]
    cut = 1.35 * np.mean(counts) if height is None else height
    distance = len(bins)//20 if distance is None else distance
    aim = min_chi2 if aim is None else aim
    peaks,_ = find_peaks(counts, height = cut, distance = distance)
    ax.axhline(cut, color = 'black')
    ax.axhline(cut/2, color = 'black', ls = '--')
    ax.axvline(best, color = 'green', lw = 1)
    if min_chi2 is not None:
        ax.axvline(min_chi2, color = 'red', lw = 1)
    ax.axvline(best - lower, color = 'green', lw = 1, ls = '--')
    ax.axvline(best + upper, color = 'green', lw = 1, ls = '--')
    ax.scatter(bins[peaks],counts[peaks], **scatter_kwargs)
    ax.grid(True)
    n_components = len(peaks)
    gmm = GaussianMixture(n_components=n_components, means_init=bins[peaks][:, np.newaxis])
    gmm.fit(data[:, np.newaxis])
    probabilities = gmm.predict_proba(data[:,None])
    predicted_labels = np.argmax(probabilities, axis=1)
    selected_data = []
    means,lowers,uppers = [],[],[]
    for i in range(len(peaks)):
        component_index = i
        selected_indices = np.where((predicted_labels == component_index) & (probabilities[:, component_index] > threshold))[0]
        selected_data.append(data[selected_indices])
        s = selected_data[-1]
        if len(s) <= 2:
            continue
        color = np.random.choice(available_colors)
        ax.fill_between((np.min(s),np.max(s)), 0, np.max(counts), 
                        color = color, alpha = 0.3, edgecolors = 'black')
        q = np.percentile(selected_data[i],[16,50,84])
        df = np.diff(q)
        means.append(q[1])
        lowers.append(df[0])
        uppers.append(df[1])
        text = r"$%.2f^{+%.2f}_{-%.2f}$" % (q[1],df[1],df[0])
        ylim = ax.get_ylim()
        y_position = ylim[0] - (0.15 + i%2 * 0.05) * (ylim[1] - ylim[0])
        plt.text(q[1],y_position, text, horizontalalignment='center', verticalalignment='bottom', 
                 fontsize=10, bbox=dict(facecolor=color, alpha=0.5, boxstyle = "round"))
    ax.set(**axes_kwargs)
    fig.tight_layout()
    fig.savefig(f"{output_path}")
    indx = np.argmin(np.abs(np.array(means) - aim))
    if only_aim == True:
        return [means[indx]], [lowers[indx]], [uppers[indx]]
    return means, lowers, uppers

def pte(chi2, cov, cinv=None, n_samples=10000, return_samples=False, return_realizations = False):
    """Probability to exceed chi2 by Monte Carlo sampling the
    covariance matrix

    Parameters
    ----------
    chi2 : float
        measured chi2
    cov : ndarray, shape (N[,N])
        if 2d, then this is the covariance matrix. If 1d, then it
        represents errorbars, i.e., the sqrt of the diagonals of the
        covariance matrix. A diagonal covariance matrix will be
        constructed in this case.
    cinv : ndarray, optional
        inverse of covariance matrix, used to calculate chi2 of MC
        samples. If not provided will be calculated with
        ``np.linalg.pinv``
    n_samples : int, optional
        Number of Monte Carlo samples to draw.
    return_samples : bool, optional
        Whether to return the full Monte Carlo chi2 vector

    Returns
    -------
    pte : float
        probability to exceed measured chi2
    chi2_mc : np.ndarray, optional
        array of sampled chi2 values. Only returned if
        ``return_samples==True``
    """
    if len(cov.shape) == 1:
        cov = np.eye(cov.size) * cov
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    if cinv is None:
        cinv = np.linalg.pinv(cov)
    mc = stats.multivariate_normal.rvs(cov=cov, size=n_samples)
    chi2_mc = np.array([np.dot(i, np.dot(cinv, i)) for i in mc])
    pte = (chi2_mc > chi2).sum() / n_samples
    if return_samples == False and return_realizations == False:
        return pte
    else:
        output = [pte]
        if return_samples == True:
            output.append(chi2_mc)
        if return_realizations == True:
            output.append(mc)
        return output



def weighted_median(values, weights):

    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)

    half_total_weight = 0.5 * np.sum(weights)
    median_index = np.where(cumulative_weights >= half_total_weight)[0][0]
    
    return sorted_values[median_index]

def set_default(dic, default_values):
    for key, default in default_values:
        dic.setdefault(key, default)
    return dic

def text2latex(param):
    greek_letters = {"alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
                     "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
                     "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
                     "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
                     "nu": r"\nu", "xi": r"\xi", "omicron": r"o", "pi": r"\pi",
                     "rho": r"\rho", "sigma": r"\sigma", "tau": r"\tau",
                     "upsilon": r"\upsilon", "phi": r"\phi", "chi": r"\chi",
                     "psi": r"\psi", "omega": r"\omega"}
    if len(param.split("log")) > 1:
        return "$"+param+"$"
    if param in greek_letters and param[-1].isdigit() == False:
        return f"${greek_letters[param]}$"
    if param[:-1] in greek_letters and param[-1].isdigit():
        return f"${greek_letters[param[:-1]]}_{param[-1]}$"
    elif len(param) > 1 and param[-1].isdigit():
        return f"${param[:-1]}_{param[-1]}$"
    elif len(param) > 1:
        return "$" +param[0] + "_{"+param[1::]+ "}$"
    else:
        return param

def inverse_abell_integral(r, R, projected_profile):
    dSigma_dR = np.gradient(projected_profile, R)
    res = []
    for ri in r:
        valid_indices = R > ri
        R_valid = R[valid_indices]
        dSigma_dR_valid = dSigma_dR[valid_indices]
        integrand = dSigma_dR_valid / np.sqrt(R_valid**2 - ri**2)
        return -1 / np.pi * simps(integrand, R_valid)
    return res

def change_ticks(axes, max_label_size, min_label_size, max_bins):
    if np.iterable(axes) == True:
        for ax in axes.flatten():
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_bins))
            ax.tick_params(axis='both', which='major', labelsize=max_label_size)
            ax.tick_params(axis='both', which='minor', labelsize=min_label_size)
    else:
            ax = axes
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_bins))
            ax.tick_params(axis='both', which='major', labelsize=max_label_size)
            ax.tick_params(axis='both', which='minor', labelsize=min_label_size)   
               
def extract_params(chain, labels = None, best = None, method = "median", percentiles = [16,84], show_results = True, latex_format = False):
    print(f"Computing params using \033[91m{method}\033[0m with \033[35m{percentiles} % uncertainties \033[0m\n")
    if labels is None:
        show_results = False
    if method == "median":
        me = np.median(chain, axis = 0)
        pl, ph = np.abs(np.percentile(chain, percentiles, axis = 0) - me)
    elif method == "mean":
        me = np.mean(chain, axis = 0)
        pl, ph = np.abs(np.percentile(chain, percentiles, axis = 0) - me)

    for i in range(len(me)):
        if show_results:
            pname = labels[i].replace("$","").replace("\\","")
            if latex_format == False:    
                print(pname,f": \033[92m{me[i]}\033[0m - \033[92m{pl[i]}\033[0m + \033[92m{ph[i]}\033[0m")
            else:
                print(pname,":\033[92m%.3f\033[0m_{-\033[92m%.3f\033[0m}^{+ \033[92m%.3f\033[0m}" % (me[i], pl[i], ph[i]))
    return me,pl,ph

def scatter_hist(x,y, fig = None, bins = 25, add_contours = True, **kwargs):
    default_fig_kwargs = (
        ("figsize",(8,8)),
    )
    default_ax_kwargs = (
        ("xlabel", "x"),
        ("ylabel","y"),
        ("xscale", "linear"),
        ("yscale", "linear"),
        ("title", None)
    )
    default_gs_kwargs = (
        ("width_ratios",(4,1)),
        ("height_ratios", (1,4)),
        ("left", 0.1),
        ("right", 0.9),
        ("bottom", 0.1),
        ("top", 0.9),
        ("wspace", 0.05),
        ("hspace", 0.05)
    )
    default_scatter_kwargs = (
        ("color","red"),
        ("label","data"),
        ("s", 2),
        ("alpha", 0.5),
    )
    default_histx_kwargs = (
        ("histtype","step"),
        ("color","red"),
        ("log", True)
    )
    default_histy_kwargs = (
        ("histtype","step"),
        ("color","red"),
        ("log", True)
    )
    default_ax_histx_kwargs = (
        ("ylabel", "N"),
    )
    default_ax_histy_kwargs = (
        ("xlabel", "N"),
    )
    default_contour_kwargs = (
        ("alpha", 0.85),
        ("colors", 'white'),
        ("levels", 25),
        ("min_ratio", 2),
        ("linestyles", "solid"),
        ("linewidths", 2.5)
    )
    default_contourf_kwargs = (
        ("alpha", 0.55),
        ("cmap", 'Reds'),
        ("levels", 25),
        ("min_ratio", 2)
    )
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs) 
    contour_kwargs = set_default(kwargs.pop("contour_kwargs",{}), default_contour_kwargs) 
    contourf_kwargs = set_default(kwargs.pop("contourf_kwargs",{}), default_contourf_kwargs) 
    ax_kwargs = set_default(kwargs.pop("ax_kwargs",{}), default_ax_kwargs) 
    scatter_kwargs = set_default(kwargs.pop("scatter_kwargs",{}), default_scatter_kwargs)
    histx_kwargs = set_default(kwargs.pop("histx_kwargs",{}), default_histx_kwargs)
    histy_kwargs = set_default(kwargs.pop("histy_kwargs",{}), default_histy_kwargs)
    ax_histx_kwargs = set_default(kwargs.pop("ax_histx_kwargs",{}), default_ax_histx_kwargs)
    ax_histy_kwargs = set_default(kwargs.pop("ax_histy_kwargs",{}), default_ax_histy_kwargs)
    gs_kwargs = set_default(kwargs.pop("gs_kwargs",{}), default_gs_kwargs)


    fig = fig if fig is not None else plt.figure(**fig_kwargs)
    gs = fig.add_gridspec(2,2, **gs_kwargs)
    ax = fig.add_subplot(gs[1,0])

    ax_histx = fig.add_subplot(gs[0,0], sharex = ax)
    ax_histy = fig.add_subplot(gs[1,1], sharey = ax)
    
    ax.scatter(x,y, **scatter_kwargs)
    ax.set(**ax_kwargs)
    ax_histx.tick_params(axis = "x", labelbottom = False, labeltop = True)
    ax_histy.tick_params(axis = "y", labelleft = False, labelright = True)
    ax_histx.hist(x , bins = bins, **histx_kwargs)
    ax_histx.set(**ax_histx_kwargs)
    ax_histy.hist(y, bins = bins, orientation = 'horizontal', **histy_kwargs)
    datay = ax_histy.get_yticklabels()
    ax_histy.set_yticks(ax.get_yticks())
    ax_histy.set_yticklabels(datay, rotation = 90)
    ax_histy.set(**ax_histy_kwargs)

    if ax_kwargs["xscale"] == "log":
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, numticks=6)) 
        ax_histx.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax_histx.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, numticks=6))

    if ax_kwargs["yscale"] == "log":
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, numticks=6)) 
        ax_histy.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax_histy.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, numticks=6))
    if add_contours == True:
        k = gaussian_kde(np.vstack([x,y]))
        xi,yi = np.mgrid[min(x):max(x):np.size(x)**0.5*1j,min(y):max(y):np.size(y)**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        min_ratio1 = contour_kwargs.pop("min_ratio", 2)
        min_ratio2 = contourf_kwargs.pop("min_ratio", 2)
        extent = [np.min(x), np.max(x), np.min(y),  np.max(y)]
        ax.contour(xi, yi, zi.reshape(xi.shape), vmin = min_ratio1*np.min(zi), extent = extent, **contour_kwargs)
        ax.contourf(xi, yi, zi.reshape(xi.shape), vmin = min_ratio2*np.min(zi), **contourf_kwargs)
    return fig

def scatter2imshow(x,y,z, fig = None, num_bins = 50, interpolate = False, smooth = None, add_scatter = False, 
                    moving_average = None, median = False, nx = 500, ny = 500, add_contours = False,
                    fit = (False, False), add_hist = False, **kwargs):
    from scipy.stats import binned_statistic_2d
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import matplotlib.ticker as ticker

    default_fig_kwargs = (
        ("figsize", (12,12)),
    )
    default_pcolormesh_kwargs = (
        ("shading", "auto"),
        ("cmap", "viridis"),
    )

    default_imshow_kwargs = (
        ("cmap", "viridis"),
        ("interpolation", "gaussian"),
        ("origin", "lower"),
        )

    default_hist_kwargs = (
        ("bins", 20),
        ("lw", 2),
        ("histtype", "step"),
        ("color", "darkred"),
        ("ls", "--"),
    )
    
    default_ax_hist_kwargs = (
        ("ylabel", "y"),
    )
    default_ax_kwargs = (
        ("xlabel", "x"),
        ("ylabel", "y"),
        ("yscale", "linear"),
        ("xscale", "linear")
    )

    default_scatter_x_kwargs = (
        ("s", 1),
        ("color", "black"),
        ("alpha", 0.5)
    )

    default_scatter_y_kwargs = (
        ("s", 1),
        ("color", "black"),
        ("alpha", 0.5)
    )

    default_ax_scatter_x_kwargs = (
        ("ylabel", "z"),
    )
    default_ax_scatter_y_kwargs = (
        ("xlabel", "z"),
    )

    default_x_fit_kwargs = (
        ("func", lambda x,a,b: a*x**b),
        ("plot", True),
        ("plot_kwargs", dict(color = "darkblue", lw = 3, alpha = 0.9)),
        ("show_params", False)
    )

    default_y_fit_kwargs = (
        ("func", lambda x,a,b: a*x**b),
        ("plot", True),
        ("plot_kwargs", dict(color = "darkblue", lw = 3, alpha = 0.9)),
        ("show_params", False)
    )

    default_gs_kwargs = (
        ("width_ratios",(0.1, 4, 1)), 
        ("height_ratios",(1,1, 4)),
        ("left",0.1), 
        ("right",0.9), 
        ("bottom",0.1), 
        ("top",0.9), 
        ("wspace",0.05), 
        ("hspace",0.05)
    )
    default_contour_kwargs = (
        ("alpha", 0.85),
        ("colors", 'black'),
        ("levels", 10),
        ("min_ratio", 2),
        ("linestyles", "solid"),
        ("linewidths", 2.5)
    )

    default_cbar_kwargs = (
        ("label", "intensity"),
        ("fontweight", "bold")
    )

    default_cbar_ticks_kwargs = (
        ("N", 4),

    )
    
    pcolormesh_kwargs = set_default(kwargs.pop("pcolormesh_kwargs",{}), default_pcolormesh_kwargs)
    fig_kwargs = set_default(kwargs.pop("fig_kwargs", {}), default_fig_kwargs)
    imshow_kwargs = set_default(kwargs.pop("imshow_kwargs", {}), default_imshow_kwargs)
    ax_scatter_x_kwargs = set_default(kwargs.pop("ax_scatter_x_kwargs",{}), default_ax_scatter_x_kwargs)
    ax_scatter_y_kwargs = set_default(kwargs.pop("ax_scatter_y_kwargs",{}), default_ax_scatter_y_kwargs)
    scatter_x_kwargs = set_default(kwargs.pop("scatter_x_kwargs",{}), default_scatter_x_kwargs)
    scatter_y_kwargs = set_default(kwargs.pop("scatter_y_kwargs",{}), default_scatter_y_kwargs)
    gs_kwargs = set_default(kwargs.pop("gs_kwargs",{}), default_gs_kwargs)
    ax_kwargs = set_default(kwargs.pop("ax_kwargs",{}), default_ax_kwargs)
    contour_kwargs = set_default(kwargs.pop("contour_kwargs",{}), default_contour_kwargs) 
    cbar_kwargs = set_default(kwargs.pop("cbar_kwargs", {}), default_cbar_kwargs)
    cbar_ticks_kwargs = set_default(kwargs.pop("cbar_ticks_kwargs", {}), default_cbar_ticks_kwargs)

    hist_kwargs = set_default(kwargs.pop("hist_kwargs", {}), default_hist_kwargs)
    ax_hist_kwargs = set_default(kwargs.pop("ax_hist_kwargs", {}), default_ax_hist_kwargs)

    x_fit_kwargs = set_default(kwargs.pop("x_fit_kwargs",{}), default_x_fit_kwargs)
    y_fit_kwargs = set_default(kwargs.pop("y_fit_kwargs",{}), default_y_fit_kwargs)

    x_bins = np.linspace(x.min(), x.max(), num_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), num_bins + 1)

    stat, x_edges, y_edges, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_bins, y_bins])
    stat = np.nan_to_num(stat, nan=np.nanmean(stat))

    Z = stat.flatten()

    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

    ax = None
    ax_colorbar = None
    ax_hist = None
    if interpolate == True and nx is not None and ny is not None:
        x_fine = np.linspace(x.min(), x.max(), nx)
        y_fine = np.linspace(y.min(), y.max(), ny)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        points = np.array([(xi, yi) for xi in (x_edges[:-1]) for yi in (y_edges[:-1])])
        Z = griddata(points, Z, (X_fine, Y_fine), method='cubic')


    fig = fig if fig is not None else plt.figure(**fig_kwargs)
    if add_scatter == True:


        gs = fig.add_gridspec(3,3, **gs_kwargs)

        ax_colorbar = fig.add_subplot(gs[2:,0])
        ax = fig.add_subplot(gs[2,1])
        ax_scatter_x = fig.add_subplot(gs[1,1], sharex = ax)
        ax_scatter_y = fig.add_subplot(gs[2,2], sharey = ax)
        pos = ax_colorbar.get_position()  # Get current position
        ax_colorbar.set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])

        ax_hist = fig.add_subplot(gs[1,2]) if add_hist == True else None

        ax_scatter_x.scatter(x, z, **scatter_x_kwargs)
        ax_scatter_y.scatter(z, y, **scatter_y_kwargs)

        ax_scatter_x.tick_params(axis = "x", labelbottom = False, labeltop = True)
        ax_scatter_y.tick_params(axis = "y", labelleft = False, labelright = True)

        ax_scatter_x.set(**ax_scatter_x_kwargs)
        ax_scatter_y.set(**ax_scatter_y_kwargs)

        if ax_hist is not None:
            ax_hist.hist(z, orientation = "horizontal", **hist_kwargs)
            ax_hist.yaxis.set_label_position("right")
            ax_hist.xaxis.set_label_position("top")
            ax_hist.yaxis.tick_right()
            ax_hist.xaxis.tick_top()
            ax_hist.set(**ax_hist_kwargs)

        if np.any(fit) == True:
            if fit[0] == True:
                func = x_fit_kwargs["func"]
                plot_kwargs = x_fit_kwargs["plot_kwargs"]
                plot = x_fit_kwargs["plot"]
                pars,cov = curve_fit(func, x, z)
                if plot == True:
                    xnew = np.linspace(x.min(), x.max(), 100)
                    ax_scatter_x.plot(xnew, func(xnew, *pars), **plot_kwargs)
            if fit[1] == True:
                func = y_fit_kwargs["func"]
                plot_kwargs = y_fit_kwargs["plot_kwargs"]
                plot = y_fit_kwargs["plot"]
                pars,cov = curve_fit(func, y, z)
                if plot == True:
                    ynew = np.linspace(y.min(), y.max(), 100)
                    ax_scatter_y.plot(func(ynew, *pars), ynew, **plot_kwargs)
        
        if moving_average is not None and type(moving_average) == int:
            x_sort, xz_moving_avg, xz_moving_err = moving_average_func(x, z, moving_average, median = median)
            y_sort, yz_moving_avg, yz_moving_err = moving_average_func(y, z, moving_average, median = median)

            ax_scatter_x.plot(x_sort, xz_moving_avg, ls = "--", label = "moving average", color = scatter_x_kwargs.pop("color", "black"))
            ax_scatter_y.plot(yz_moving_avg, y_sort, ls = "--", label = "moving average", color = scatter_y_kwargs.pop("color", "black"))

    Z = np.nan_to_num(Z, nan=np.nanmean(Z))
    Z = gaussian_filter(Z, smooth) if smooth is not None else Z

    ax = plt.axes() if ax is None else ax

    im = ax.imshow(Z, extent = extent, **imshow_kwargs) if interpolate == True else ax.pcolormesh(x_bins, y_bins, stat.T, **pcolormesh_kwargs)
    min_ratio = contour_kwargs.pop("min_ratio", 2)

    ax.contour(x_fine, y_fine, Z, extent = extent, vmin = min_ratio*np.min(Z), **contour_kwargs) if (interpolate == True and add_contours == True) else None
    ax.set(**ax_kwargs)
    ax.set_aspect("auto")

    if ax_colorbar is None:
        divider = make_axes_locatable(ax)
        ax_colorbar = divider.append_axes('left', size='5%', pad=0.05)
        
    cbar = plt.colorbar(im, cax = ax_colorbar) 

    nticks = cbar_ticks_kwargs.pop("N", 4)

    #cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nticks)) 
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15, subs = "all"))
    cbar.ax.yaxis.set_ticks_position("left")  
    cbar.ax.tick_params(rotation=90)

    cbar_label = cbar_kwargs.pop("label", "Intensity")
    cbar.ax.set_title(cbar_label, **cbar_kwargs)

    return fig
def search_fit_in_repo(repo_url, fits_name, search_in_paths = True, search_similar = False, output_path = None, filters = None):
    response = requests.get(repo_url) #repo contect
    if response.status_code == 200:
        soup = bs(response.content, 'html.parser')
        links = soup.find_all('a')
        files = [link.get('href').strip() for link in links if link.get('href') and not link.get('href').startswith('/')]
        if filters is not None:
            if np.iterable(filters) == False:
                files = filters(files)
            else:
                files = filters[0](files)
        if search_in_paths == False:
            if search_similar == False:
                if fits_name in files or fits_name + ".fits" in files:
                    fit = fits.open(repo_url + fits_name)
                else:
                    fit = None
            elif search_similar == True:
                similarities = [(f, SequenceMatcher(None, fits_name, f).ratio()) for f in files]
                most_similar = max(similarities, key=lambda x: x[1])[0]
                try:
                    fit = fits.open(repo_url + most_similar)
                except:
                    fit = None
        elif search_in_paths == True:
            fit = None
            for n,f in enumerate(files):
                current_link = repo_url + f if repo_url[-1] == "/" else repo_url + "/" + f
                response = requests.get(current_link)
                print(f"searching on: [\033[34m\033[4m{current_link}\033[0m\033[0m] ({n + 1} / {len(files)})")
                if response.status_code == 200:
                        soup = bs(response.content, 'html.parser')
                        slinks = soup.find_all('a')
                        sfiles = [link.get('href') for link in slinks if link.get('href') and not link.get('href').startswith('/')]
                        if np.iterable(filters) == True and len(filters) > 1:
                            sfiles = filters[1](sfiles)
                        if search_similar == False:
                            if fits_name in sfiles or fits_name + ".fits" in sfiles:
                                fit_link = repo_url + f + fits_name if repo_url[-1] == "/" else repo_url + "/" + f + fits_name
                                fit_link = fit_link + ".fits" if fit_link[-3::] != "fits" else fit_link
                                print(10*"=",f"\ndownloading fit from [\033[34m\033[4m{fit_link}\033[0m\033[0m]",10*"=")
                                fit = fits.open(fit_link)
                                break
                            else:
                                continue
                        elif search_similar == True:
                            similarities = [(f, SequenceMatcher(None, fits_name, f).ratio()) for f in sfiles]
                            most_similar = max(similarities, key=lambda x: x[1])[0]
                            try:
                                fit_link = repo_url + f + "/" + most_simila if repo_url[-1] == "/" else repo_url + "/" + f + "/" + most_simila
                                fit = fits.open(fit_link)
                            except:
                                continue

    if fit is not None:
        if output_path is not None:
            if os.path.exists(output_path) == False:
                fit.writeto(output_path)
        return fit
    else:
        return None

def norm(x,u,sigma):
    return 1/np.sqrt(2*np.pi * sigma**2) * np.exp(- (x - u)**2 / (2*sigma**2))
def log_norm(x,u,sigma):
    return 1/(np.sqrt(2*np.pi * sigma**2) * x) * np.exp(- (np.log(x) - np.log(u))**2 / (2*sigma**2))



def load_function_from_file(file_path, function_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, function_name):
        func = getattr(module, function_name)
        return func
    else:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'")

def cumulative_mass(r,kappa, zs, zd, cosmo = None):
    if cosmo is None:
        cosmo = astropy.cosmology.Planck18
    D_d = cosmo.angular_diameter_distance(zd).to("m")
    D_s = cosmo.angular_diameter_distance(zs).to("m")
    D_ds = cosmo.angular_diameter_distance_z1z2(zd, zs).to("m")
    c = const.c
    G = const.G
    Msun = const.M_sun
    sigma_crit =  c**2 / (4*np.pi * G) * (D_s / (D_d * D_ds))
    sigma = kappa * sigma_crit
    r = r*cosmo.kpc_proper_per_arcmin(zd).to(u.m/u.arcmin)
    r = r.to("m")
    M = 2 * np.pi * cumtrapz(sigma * r, r, initial = 0) * u.kg
    return M/Msun

def Mpc2arcmin(r, z, cosmo = "planck18"):
    if cosmo == "planck18":
        from astropy.cosmology import Planck18 as planck18
        arcsec2kpc = planck18.arcsec_per_kpc_proper(z)
        arcmin2Mpc = arcsec2kpc.to(u.arcmin / u.Mpc)
        return (r*arcmin2Mpc).value
def arcmin2Mpc(r, z, cosmo = "planck18"):
    if cosmo == "planck18":
        from astropy.cosmology import Planck18 as planck18
        kpc2arcmin = planck18.kpc_proper_per_arcmin(z)
        Mpc2arcmin = kpc2arcmin.to(u.Mpc / u.arcmin)
        return (r * Mpc2arcmin).value
def scientific2float(num):
    if type(num) == str:
        if "e" in list(num):
            base, exp = num.split("e")
            as_float = float(base) * 10**float(exp)
        else:
            as_float = float(num)
        return as_float
    else:
        return num

def compute_projected_offset(ra1, dec1, z1, ra2, dec2, z2, cosmo, sep = 1 * u.arcmin, 
                sep_min = None, max_sep_kpc = None, use_redshift = False, zerr1 = None, zerr2 = None):
    from astropy.coordinates import SkyCoord

    d1 = cosmo.comoving_distance(z1)
    d2 = cosmo.comoving_distance(z2)

    c1 = SkyCoord(ra1, dec1, unit = "deg") if use_redshift == False else SkyCoord(ra1, dec1, distance = d1, unit = [u.deg, u.deg, u.Mpc])
    c2 = SkyCoord(ra2, dec2, unit = "deg") if use_redshift == False else SkyCoord(ra2, dec2, distance = d2, unit = [u.deg, u.deg, u.Mpc])

    idx1, idx2, d2d, d3d = c2.search_around_sky(c1, sep)

    if sep_min is not None:
        idx = np.where(d2d.to(sep_min.unit) >= sep_min)
        idx1, idx2, d2d = idx1[idx], idx2[idx], d2d[idx]

    zbar = []
    for i in range(len(idx1)):
        z1_i = z1[idx1[i]]
        z2_i = z2[idx2[i]]
        if (zerr1 is not None) and (zerr2 is not None):
            w1, w2 = 1/(zerr1[idx1[i]]**2), (1/zerr**2)
            zbar.append(np.average((z1_i, z2_i)), weights = (w1,w2))
        else:
            zbar.append(np.mean((z1_i, z2_i)))
    
    d_proj = (d2d.to(u.arcmin) * cosmo.kpc_proper_per_arcmin(zbar).to(u.Mpc / u.arcmin))
    return d_proj


def plot_grouped_clusters(grouped_clusters, labels = None, fig = None, suptitle = "", split_by_richness = True,
         split_by_redshift = False, contours = False, colors = None, sort_colors = False, add_snr = False, 
         labels_coords = (0.5,0.75),**kwargs):
    default_fig_kwargs = (
        ("figsize", (10, 6*len(grouped_clusters))),
    )

    default_suptitle_kwargs = (
        ("fontsize", 14),
        ("fontweight","bold"),
        ("color","black")
    )

    default_ax_kwargs = (
        ("xlabel", "R (arcmin)"),
        ("ylabel", "y"),
    )

    default_labels_kwargs = (
        ("fontsize", 16),
        ("ha", "center"),
        ("va", "center"),
        ("fontweight", "bold")
    )
    colors = ["darkred", "darkgreen", "darkblue", "grey", "orange"] if colors is None else colors
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)
    suptitle_kwargs = set_default(kwargs.pop("suptitle_kwargs",{}), default_suptitle_kwargs)
    labels_kwargs = set_default(kwargs.pop("labels_kwargs",{}), default_labels_kwargs)
    ax_kwargs = set_default(kwargs.pop("ax_kwargs",{}), default_ax_kwargs)
    fig,ax = plt.subplots(len(grouped_clusters), 1, sharex = True, sharey = True, **fig_kwargs) if fig is None else fig
    for i,g in enumerate(grouped_clusters):
        if len(g) > 1:
            for j,sg in enumerate(g):
                prof = sg.mean_profile
                errs = sg.error_in_mean
                cov = sg.cov
                snr = np.sqrt(np.dot(prof, np.dot(np.linalg.inv(cov), prof.T)))
                richness = [np.min(sg.richness), np.max(sg.richness)]
                redshift = [np.min(sg.z), np.max(sg.z)]
                label = r"$\lambda \in [%.i, %.i]$" % tuple(richness) if split_by_redshift else r"$z \in [%.2f, %.2f]$" % tuple(redshift)
                color = colors[i]
                if sort_colors == True:
                    k = 0
                    if split_by_redshift == True and split_by_richness == False:
                        while k < len(colors):
                            rmin, rmax = colors[k][0]
                            if richness[0] >= rmin and richness[1] <= rmax:
                                color = colors[k][1]
                                break
                            k+=1
                    elif split_by_richness == True and split_by_redshift == False:
                        while k < len(colors):
                            zmin, zmax = colors[k][0]
                            if redshift[0] >= zmin and redshift[1] <= zmax:
                                color = colors[k][1]
                            k+=1
                if add_snr == True:
                    label = label + " , $\mathrm{SNr} = %.2f$" % snr
                if contours == False:
                    ax[i].errorbar(sg.R, prof, yerr = errs, fmt = "o", color = color, label = label)
                else:
                    ax[i].plot(sg.R, prof, color = color, lw = 3, alpha = 0.7, label = label)
                    ax[i].fill_between(sg.R, prof - errs, prof + errs, color = color, alpha = 0.2, edgecolor = "black")
        ax[i].text(labels_coords[0], labels_coords[1], labels[i], transform = ax[i].transAxes, **labels_kwargs)
        ax[i].legend(fontsize = 12)
        ax[i].set(**ax_kwargs)
    fig.suptitle(suptitle, **suptitle_kwargs)
    return fig
def load_kappa_data(source_path, folder, output_path, load_filter_only = True):
    source_path = source_path[:-1] if source_path[-1] == "/" else source_path
    folder = folder[:-1] if folder[-1] == "/" else folder
    output_path = output_path[:-1] if output_path[-1] == "/" else output_path
    from cluster_data import grouped_clusters
    c = grouped_clusters.empty()
    c.output_path = output_path + "/" + folder 
    if os.path.exists(c.output_path) == False:
        os.mkdir(c.output_path)

    kfilter = source_path +"/"+ folder + f"/{folder}_kmask.fits"
    shutil.copy(kfilter, c.output_path)
    code_content = f"""
from profiley.filtering import Filter
import numpy as np
filt = Filter('{output_path}{folder}/{folder}_kmask.fits')

def apply_k_space_filter(R, data, units = "arcmin"):
    if hasattr(R,"value"):
        R = R.value
    delta_R = (R[1] - R[0])
    R_edges = np.arange(R[0] - delta_R/2, R[-1] + delta_R, delta_R)
    theta_filtered, kappa_filtered = filt.filter(R, data, R_edges, units = units)
    return kappa_filtered
    """
    with open(f"{output_path}/{folder}/filters.py", "w") as f:
        f.write(code_content)
    
    if load_filter_only == True:
        return
    
    catalog_data = np.loadtxt(source_path + "/" + folder + f'/{folder}_catalog_data.txt')
    cov_matrix = np.loadtxt(source_path + "/" + folder + f'/{folder}_opt_covm.txt')
    mean_prof = np.loadtxt(source_path + "/" + folder + f'/{folder}_opt_profile.txt')
    errors = np.loadtxt(source_path + "/" + folder + f'/{folder}_opt_profile_errs.txt')


    richness = catalog_data[:,0]
    redshift = catalog_data[:,2]
    c.richness = richness
    c.z = redshift
    c.cov = cov_matrix
    c.R = mean_prof[:,0]
    c.mean_profile = mean_prof[:,1]
    c.error_in_mean = errors[:,1]
    c.profiles = np.tile(c.mean_profile, len(richness)).reshape((len(richness),-1))
    c.errors = np.tile(c.error_in_mean, len(richness)).reshape((len(richness),-1))
    c.plot()
    c.save()

def offset(R, Roff, P, params):
    theta = np.linspace(0, 2*np.pi, 20)
    R = np.sqrt(R**2 + Roff**2 + 2*R*Roff*np.cos(theta))
    I = P(R, *params) 
    return trapz(I, x = theta) / (2*np.pi)

def two_halo_term(P, M, z, R, cosmo, params, delta = 500, background = "matter", 
        r_units = "Mpc", **kwargs):
    default_mass_kwargs = (
        ("Mmin", 12),
        ("Mmax", 15.5),
        ("Nbins", 40)
        )
    default_k_kwargs = (
        ("kmin", -15),
        ("kmax", 15),
        ("Nbins", 500)
    )
    default_r_kwargs = (
        ("rmin", -3),
        ("rmax", 3),
        ("Nbins", 40)
    )
    
    extra_kwargs = kwargs.pop("extra_kwargs", {})

    M_arr = extra_kwargs["M_arr"] if "M_arr" in list(extra_kwargs.keys()) else None
    r = extra_kwargs["r_arr"] if "r_arr" in list(extra_kwargs.keys()) else None
    k = extra_kwargs["k_arr"] if "k_arr" in list(extra_kwargs.keys()) else None
    dndM = extra_kwargs["dndM"] if "dndM" in list(extra_kwargs.keys()) else None
    Pk = extra_kwargs["Pk"] if "Pk" in list(extra_kwargs.keys()) else None
    ki_r = extra_kwargs["ki_r"] if "ki_r" in list(extra_kwargs.keys()) else None
    bh = extra_kwargs["bh"] if "bh" in list(extra_kwargs.keys()) else None
    bM = np.asarray(extra_kwargs["bM"]) if "bM" in list(extra_kwargs.keys()) else None 
    mass_kwargs = set_default(kwargs.pop("mass_kwargs",{}), default_mass_kwargs)
    k_kwargs = set_default(kwargs.pop("k_kwargs", {}), default_k_kwargs)
    r_kwargs = set_default(kwargs.pop("r_kwargs", {}), default_r_kwargs)
    r = np.logspace(r_kwargs["rmin"], r_kwargs["rmax"], r_kwargs["Nbins"]) if r is None else r
    M_arr = np.logspace(mass_kwargs["Mmin"], mass_kwargs["Mmax"], mass_kwargs["Nbins"]) if M_arr is None else M_arr
    k = np.logspace(k_kwargs["kmin"], k_kwargs["kmax"], k_kwargs["Nbins"]) if k is None else k

    mdef = ccl.halos.MassDef(delta, background)
    if dndM is None:
        mfunc = ccl.halos.mass_function_from_name("Tinker10")
        mfunc = mfunc(cosmo, mdef)
        dndM = np.array([mfunc(cosmo, Mi, 1/(z + 1)) for Mi in M_arr])
        dndM = dndM * 1/(M_arr * np.log(10))
    if bh is None or bM is None:
        bias = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef) 
        bh = bias.get_halo_bias(cosmo, M, 1/(1 + z)) if bh is None else bh
        bM = np.array([bias.get_halo_bias(cosmo, Mi, 1/(1 + z)) for Mi in M_arr]) if bM is None else bM
    Pk = ccl.linear_matter_power(cosmo, k, 1/(1+z)) if Pk is None else Pk

    factor = 4 * np.pi * r**2
    ki_r = np.outer(k, r) if ki_r is None else ki_r
    sin_term = np.sin(ki_r) / np.where(ki_r != 0, ki_r, 1)
    P_values = np.array([P(r, Mi, z, params) for Mi in M_arr])
    uP = trapz(factor * sin_term[None, :, :] * P_values[:, None, :], x = r, axis = 2)
    PhP = bh*Pk*trapz((bM * dndM)[:,None] * uP , x = M_arr, axis = 0)

    if r_units == "Mpc":
        ki_R = np.outer(R, k)
        sin_term2 = np.sin(ki_R) / np.where(ki_R != 0, ki_R, 1)
        P2h = k**2/(2*np.pi**2) * sin_term2 * PhP
    elif r_units == "arcmin":
        R = R if hasattr(R, "unit") else R*u.arcmin
        Da_co = planck18.angular_diameter_distance(z) * (1 + z)
        R_Mpc = ((R.to(u.rad)) * Da_co).value
        ki_R = np.outer(R_Mpc, k)
        sin_term2 = np.sin(ki_R) / np.where(ki_R != 0, ki_R, 1)
        P2h = k**2/(2*np.pi**2) * sin_term2 * PhP
    return trapz(P2h, x = k, axis = 1)


global worker
def worker(P, R, r_units, params, M_arr, z_arr, counter, N_total, nworker = 0, two_halo_term_kwargs = None):
    cosmo = ccl.CosmologyVanillaLCDM()
    two_halo_term_kwargs = dict() if two_halo_term_kwargs is None else two_halo_term_kwargs
    M2_arr = two_halo_term_kwargs["M_arr"] if "M_arr" in list(two_halo_term_kwargs.keys()) else None
    r_arr = two_halo_term_kwargs["r_arr"] if "r_arr" in list(two_halo_term_kwargs.keys()) else None
    k_arr = two_halo_term_kwargs["k_arr"] if "k_arr" in list(two_halo_term_kwargs.keys()) else None
    dndM = two_halo_term_kwargs["dndM"] if "dndM" in list(two_halo_term_kwargs.keys()) else None
    Pk = two_halo_term_kwargs["Pk"] if "Pk" in list(two_halo_term_kwargs.keys()) else None
    ki_r = two_halo_term_kwargs["ki_r"] if "ki_r" in list(two_halo_term_kwargs.keys()) else None
    bh = two_halo_term_kwargs["bh"] if "bh" in list(two_halo_term_kwargs.keys()) else None
    bM = two_halo_term_kwargs["bM"] if "bM" in list(two_halo_term_kwargs.keys()) else None 

    extra_kwargs = dict(M_arr = M2_arr, r_arr = r_arr, k_arr = k_arr, ki_r = ki_r)

    evals = np.zeros((len(params), len(z_arr), len(M_arr), len(R)))
    for i,p in enumerate(params):
        for j,zj in enumerate(z_arr):
            extra_kwargs["dndM"] = np.array(dndM[j,:])
            extra_kwargs["Pk"] = np.array(Pk[j,:])
            extra_kwargs["bM"] = np.array(bM[j,:])
            for k, Mk in enumerate(M_arr):
                extra_kwargs["bh"] = np.array(bh[j,k])
                evals[i,j,k,:] = two_halo_term(P, Mk, zj, R, cosmo, p, r_units = r_units, extra_kwargs = extra_kwargs)
        counter.value+=1
        sys.stdout.write(f"\rEvaluating 2h model: ({counter.value} / {N_total})")
        sys.stdout.flush()
    print(f"\nWorker {nworker} is already finished!\n")
    return evals


def make_2halo_term_interpolator(P = None, M_arr = None, z_arr = None, R = None, cosmo = None, params = None, pool = None, delta = 500,
                n_cores = None, overwrite = True, background = "matter", output_file = None, interpolation = "linear", r_units = "Mpc", return_samples = False,
                **kwargs):

    default_two_halo_term_kwargs = (
    ("mass_kwargs", {
        "Mmin" : 12,
        "Mmax" :15.5,
        "Nbins": 70
    }
    ),
    ("k_kwargs", {
        "kmin" :-15,
        "kmax" : 15,
        "Nbins" : 500
        }
    ),
    ("r_kwargs", {
        "rmin" : -3,
        "rmax" : 3,
        "Nbins" : 70
    }
    )
    )

    from scipy.interpolate import RegularGridInterpolator
    if (interpolation in ("linear", "cubic", "nearest")) == False:
        raise ValueError(f"{interpolation} isn't a valid method. Interpolation mush be 'linear', 'cubic' or 'nearest'.")
    if overwrite == False:
        print(f"Loading interpolator from {output_file}.\n")
        with h5py.File(output_file, "r") as f:
            M_arr = f["Mass"][:]
            z_arr = f["z"][:]
            evals = f["evals"][:]
            R = f["R"][:]
            N = len(f.keys()) - 4
            params = []
            for i in range(N):
                params.append(f["param " + str(i)][:])
        meshgrid = np.meshgrid(*params, indexing = "ij")
        params_arr = np.stack([grid.flatten() for grid in meshgrid], axis = -1)
        shape = [len(grid) for grid in params] + [len(z_arr), len(M_arr), len(R)]
        evals = np.reshape(evals, shape)
        full_grid = (*params, z_arr, np.log10(M_arr), R)
        interpolator = RegularGridInterpolator(
            full_grid,
            np.log10(evals),
            method=interpolation,
            bounds_error=False,
            fill_value=None
        )
        if return_samples == False:
            return interpolator
        else:
            return (M_arr, z_arr, params, R, evals), interpolator
    meshgrid = np.meshgrid(*params, indexing = "ij")
    params_arr = np.stack([grid.flatten() for grid in meshgrid], axis = -1)
    if pool is None:
        evals = []
        for n in tqdm(range(len(params_arr)), desc = "Evaluating 2h model..."):
            evals.append([[two_halo_term(P, Mi, zj, R, cosmo , params_arr[n]) for Mi in M_arr] for zj in z_arr])
    else:
        two_halo_term_kwargs = set_default(kwargs.pop("two_halo_term_kwargs", {}), default_two_halo_term_kwargs)

        mass_kwargs = two_halo_term_kwargs["mass_kwargs"]
        k_kwargs = two_halo_term_kwargs["k_kwargs"]
        r_kwargs = two_halo_term_kwargs["r_kwargs"]

        r2 = np.logspace(r_kwargs["rmin"], r_kwargs["rmax"], r_kwargs["Nbins"])
        M2_arr = np.logspace(mass_kwargs["Mmin"], mass_kwargs["Mmax"], mass_kwargs["Nbins"])
        k2 = np.logspace(k_kwargs["kmin"], k_kwargs["kmax"], k_kwargs["Nbins"])

        mdef = ccl.halos.MassDef(delta, background)
        mfunc = ccl.halos.mass_function_from_name("Tinker10")
        mfunc = mfunc(cosmo, mdef)
        dndM = np.array([[ 1/(Mi * np.log(10)) * mfunc(cosmo, Mi, 1/(zi + 1)) for Mi in M2_arr] for zi in z_arr])
        bias = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef) 
        bh = np.array([[bias.get_halo_bias(cosmo, Mi, 1/(1 + zi)) for zi in z_arr] for Mi in M_arr])
        bM = np.array([[bias.get_halo_bias(cosmo, Mi, 1/(1 + zi)) for Mi in M2_arr] for zi in z_arr])
        Pk = np.array([ccl.linear_matter_power(cosmo, k2, 1/(1+zi)) for zi in z_arr])
        ki_r = np.outer(k2, r2)


        extra_kwargs = dict(
            M_arr = M2_arr,
            r_arr = r2,
            k_arr = k2,
            dndM = dndM,
            Pk = Pk,
            ki_r = ki_r, 
            bh = bh, 
            bM = bM
        )
        n_cores = pool._processes if n_cores is None else n_cores
        new_params = np.array_split(np.array(params_arr), n_cores)
        manager = Manager()
        counter = manager.Value("i", 0 )
        N_total = len(params_arr)
        args = [(P,R, r_units, p, M_arr, z_arr, counter, N_total, i, extra_kwargs) for i,p in enumerate(new_params)]
        print(f"Making new 2h term interpolator with {len(params_arr)} parameters.\n")
        res = pool.starmap(worker, args)
        pool.close()
        pool.join()
        manager.shutdown() 
        evals = np.concatenate(res, axis = 0)
    if overwrite == True:
        if output_file is not None:
            print("Saving interpolator to file " + output_file)
            output_file = "interpolator.h5" if output_file == "" else output_file
            with h5py.File(output_file, "w") as f:
                f.create_dataset("Mass", data = M_arr)
                f.create_dataset("z", data = z_arr)
                f.create_dataset("evals", data = evals)
                f.create_dataset("R", data = R)
                for i in range(np.shape(params)[0]):
                    p = np.array(params)[i,:]
                    f.create_dataset(f"param {i}", data = p)
    shape = [len(grid) for grid in params] + [len(z_arr), len(M_arr), len(R)]
    evals = np.reshape(evals, shape)
    full_grid = (*params, z_arr, np.log10(M_arr), R)
    interpolator = RegularGridInterpolator(
        full_grid,
        np.log10(evals),
        method=interpolation,
        bounds_error=False,
        fill_value=None
    )
    if return_samples == False:
        return interpolator
    else:
        return M_arr, z_arr, params, R, evals
def is_running_via_nohup():
    import sys
    if not sys.stdout.isatty():
        return True
    if not sys.stdin.isatty():
        return True
    if not sys.stderr.isatty():
        return True
    
    return False

def calculate_sigma_intervals(array, sigma):
    """
    Calculate the median and +/- sigma intervals of an array.

    Parameters:
        array (array-like): The input data array.
        sigma (float): The sigma level (e.g., 1 for 1σ, 2 for 2σ, etc.).

    Returns:
        tuple: Median, sigma_plus, sigma_minus values.
    """
    lower_percentile = 100 * (0.5 - norm_scipy.cdf(-sigma))
    upper_percentile = 100 * (0.5 + norm_scipy.cdf(-sigma))
    lower_bound, median, upper_bound = np.percentile(array, [50 - lower_percentile, 50, 100 - (upper_percentile - 50)])
    sigma_minus = median - lower_bound
    sigma_plus = upper_bound - median

    return sigma_minus, median, sigma_plus
def null_test_map(x, chain, null_test_idx, func, N_realizations = 10, bins = 30, ncores = 30,
                  cut = False, sigma = 1):
    from multiprocessing import Pool
    pool = Pool(ncores)
    params = dict(x = x, chain = chain, null_test_idx = null_test_idx, func = func, 
                N_realizations = N_realizations, bins = bins, sigma = sigma, cut = cut)
    params = list(params.values())
    args = []
    for i in range(ncores):
        args.append(params)
    print(f"running null test for \033[35m param index = {null_test_idx}\033[0m and \033[35m{N_realizations} realizations.\033[0m")
    results = pool.starmap(null_test, args)
    pool.close()
    pool.join()
    return results
def null_test(x, chain, null_test_idx, func, N_realizations = 100, bins = 30, sigma = 1, cut = False):
    null_test_results = []
    for i in range(N_realizations):
        new_params = np.zeros(chain.shape[1])
        for j in range(len(new_params)):
            if j!=null_test_idx:
                new_params[j] = random_from_histogram(chain[:,j], bins = bins, sigma = sigma, cut = cut)
            profs = func(x, new_params)
            null_test_results.append(profs)
    return np.array(null_test_results)
def random_from_histogram(data = None, prob = None, bin_edges = None, counts = None, bins=100, 
                          cut = False, sigma = 1):
    """
    Returns a random number sampled from the histogram distribution of the input data.

    Parameters:
    - data (list or np.array): The list of data points to generate the histogram.
    - bins (int): The number of bins for the histogram. Default is 30.

    Returns:
    - float: A random number sampled from the distribution of the data.
    """
    if data is not None and prob is None:
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        le, _, ue = np.percentile(data, [16, 50, 84])
        idx_le, idx_ue = np.argmin(np.abs(bin_edges - le)), np.argmin(np.abs(bin_edges - ue))
        probabilities = counts / counts.sum() if (sigma is None or cut is False) else counts[idx_le:idx_ue]/np.sum(counts[idx_le:idx_ue])
        bins_range = np.arange(0,len(counts),1) if (sigma is None or cut is False) else np.arange(idx_le, idx_ue, 1)
        random_bin = np.random.choice(bins_range, p=probabilities)
        random_value = np.random.uniform(bin_edges[random_bin], bin_edges[random_bin + 1])
        return random_value
    elif data is None and prob is not None and bin_edges is not None and counts is not None:
        probabilities = prob
        random_bin = np.random.choice(len(counts), p=probabilities)
        random_value = np.random.uniform(bin_edges[random_bin], bin_edges[random_bin + 1])
        return random_value
        # #check convergence of numerical  integrals
        # check_convergence = False
        # plot_convergence = False
        # if check_convergence == True:
        #     convergence_ratio = 0.03
        #     rbins = 10
        #     zbins = 6
        #     Mbins = 9
        #     results = []
        #     diffs = []
        #     means = []
        #     parameters = [-8, 0.15, 5, 1, 2.6]
        #     N = 0
        #     times = []
        #     window = 5
        #     indx = 0
        #     while True:
        #         t1 = time_ns()
        #         func = empty_group.stacked_halo_model_func(getattr(profiles_module, profile_stacked_model), rbins, zbins, Mbins)
        #         res = func(empty_group.R.value, parameters)
        #         results.append(res)
        #         times.append(time_ns() - t1)
        #         diff = np.sum( np.abs( (results - res ) / results) ) if len(results) < 2 else np.sum( np.abs( (results[-2::] - res ) / results[-2::]) )
        #         if diff == 0 or diff == np.nan:
        #             diff = diffs[-1] if len(diffs) > 1 else np.nan
        #         diffs.append(diff)
        #         if len(diffs) < window:
        #             means.append(diff)
        #         else:
        #             current_mean = np.nanmean(diffs[-window::])
        #             means.append(current_mean)
        #             if current_mean <= convergence_ratio*0.95 and indx == 0:
        #                 indx = len(means)
        #             if current_mean <= convergence_ratio * 0.75:
        #                 break
        #         rbins+=1
        #         if np.random.randint(0,4) == 2 and indx == 0:
        #             Mbins+=1
        #         if np.random.randint(0,6) == 4 and indx == 0:
        #             zbins+=1
        #     if plot_convergence == True:
        #         ratios = np.array(means)
        #         text = f"R bins = {rbins}\nz bins = {zbins}\nM bins = {Mbins}"
        #         bbox = dict(color = "lightgreen", boxstyle = "round", edgecolor = "black")
        #         ratios = np.array(ratios)
        #         fig,(ax,ax2) = plt.subplots(2, figsize = (5,12), sharex = True)
        #         ax.axhline(convergence_ratio, color = 'black', ls = '--', label = "target ratio")
        #         ax.plot(ratios, label = "ratio for each bins size", lw = 1, alpha = 0.8, color = 'black')
        #         convergence_threshold = indx
        #         ax.text(convergence_threshold*0.7, np.nanmax(np.log10(ratios))*0.45, text, bbox = bbox, fontsize = 8)
        #         ax.fill_between(np.arange(convergence_threshold,len(ratios)),0,np.nanmax(ratios), color = 'grey',alpha = 0.8, label = "fully converged")
        #         ax.legend()
        #         ax.grid(True)
        #         ax.set(xlabel = "step", ylabel = f"mean diff. between last {window} step", title = "Convergence of integral", yscale = "log")
        #         ax2.plot(times, label = "integration time", lw = 1, alpha = 0.8, color = 'black')
        #         ax2.fill_between(np.arange(convergence_threshold,len(times)),np.min(times),np.max(times), color = 'grey',alpha = 0.8, label = "fully converged")
        #         ax2.legend()
        #         ax2.set(xlabel = "step", ylabel = "time (in nanoseconds)", title = "Time for each integral")
        #         ax2.text(len(times)*0.10, np.max(times)*0.6, f"time (seconds) = {times[convergence_threshold]/1e9}", fontsize = 8, bbox = bbox)
        #         ax2.grid(True)
        #         fig.tight_layout()
        #         fig.savefig("convergence.png")
        # else:

    # R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
    # R_units = config["CLUSTER PROPERTIES"]["r_units"]
    # grouped_clusters_list = [
    #     path
    #     for path in os.listdir(data_path + grouped_clusters_path)
    #     if os.path.isdir(data_path + grouped_clusters_path + path) 
    #     and
    #     path.split('_')[0] == 'GROUPED'
    # ]
    # intervals = []
    # for g in grouped_clusters_list:
    #     pattern_richness = r'GROUPED_CLUSTER_RICHNESS=(\d+\.\d+)-(\d+\.\d+)'
    #     pattern_redshift = r'REDSHIFT=(\d+\.\d+)-(\d+\.\d+)'
    #     match_richness = re.search(pattern_richness, g)
    #     match_redshift = re.search(pattern_redshift, g)
    #     if match_richness:
    #         intervals.append([[float(match_richness.group(1)),float(match_richness.group(2))],[float(match_redshift.group(1)),float(match_redshift.group(2))]])
    # clusters = []
    # try:
    #     R_profiles = R_profiles * getattr(u, R_units)
    # except:
    #     R_profiles = R_profiles
    # for n in range(len(grouped_clusters_list)):
    #     current_interval = intervals[n][0]
    #     current_redshift_interval = intervals[n][1]
    #     if (current_interval[0] == 20 and current_interval[1] == 224) or current_interval[0] < r_interval[0] or current_interval[1] > r_interval[1] or current_redshift_interval[0] < z_interval[0] or current_redshift_interval[1] > z_interval[1]:
    #         continue

    #     if individual_bool == True and (float(current_interval[0]) != float(r_interval[0]) 
    #                                     or float(current_interval[1]) != float(r_interval[1]) 
    #                                     or float(current_redshift_interval[0]) != float(z_interval[0]) 
    #                                     or float(current_redshift_interval[1]) != float(z_interval[1])):
    #         continue

    #     empty_group = grouped_clusters.empty()
    #     empty_group.output_path = (
    #         data_path + grouped_clusters_path + grouped_clusters_list[n]
    #     )
    #     output_path = empty_group.output_path
    #     try:
    #         empty_group.load_from_h5(search_closest = True)
    #     except FileNotFoundError:
    #         continue
    #     empty_group.mean(from_path = True)
    #     empty_group.R = np.array([(R_profiles[i + 1] + R_profiles[i]).value/2 for i in range(len(R_profiles) - 1)]) * R_profiles.unit
    #     profile = np.array(empty_group.mean_profile)
    #     errors = np.array(empty_group.error_in_mean)
    #     positive_values = profile[profile > 0]
    #     SNr_total = np.sqrt(np.sum(profile**2 / errors**2))
    #     output_paths = []

    #     if hasattr(empty_group,"cov_matrix"):
    #         cov = empty_group.cov_matrix
    #         cov_inv = np.linalg.inv(cov)
    #         sigma = cov_inv
    #     else:
    #         sigma = empty_group.error_in_mean
    #         cov = None
    #         cov_inv = None
    #     clusters.append(empty_group)
    #     output_paths.append(empty_group.output_path)

    #     if whole_data == True:
    #         continue

    #     rbins = 25
    #     Mbins = 10
    #     zbins = 11

    #     np.save(output_path + "/convergence.npy",np.array([rbins, zbins, Mbins]))
    #     func = empty_group.stacked_halo_model_func(getattr(profiles_module, profile_stacked_model), rbins = rbins, zbins = zbins, Mbins = Mbins)
    #     global ln_prior
    #     def ln_prior(theta):
    #         prior = 0.0
    #         i_theta = 0
    #         for i in range(len(prior_parameters)):
    #             if "free" in prior_parameters[i]:
    #                 args = np.array(prior_parameters[i][-1].split("|")).astype(
    #                     np.float64
    #                 )
    #                 prior += getattr(MCMC_func, prior_parameters[i][1])(
    #                     theta[i_theta], *args
    #                 )
    #                 i_theta += 1
    #         return prior

    #     global ln_likelihood
    #     if likelihood_func == 'gaussian':
    #         if np.array(sigma).ndim == 1:
    #             def ln_likelihood(theta, x, y, sigma, **kwargs):
    #                 model = kwargs["model"]
    #                 mu = model(x, theta)
    #                 log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
    #                     (y - mu) ** 2
    #                 ) / (sigma**2)
    #                 return np.sum(log_likelihood),mu
    #         else:
    #             log_det_C = np.linalg.slogdet(cov)[1]
    #             def ln_likelihood(theta, x, y, sigma, **kwargs):
    #                 model = kwargs["model"]
    #                 y1 =  model(x, theta)
    #                 residual = y - y1
    #                 current_chi2 = np.dot(residual.T, np.dot(sigma, residual))
    #                 X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
    #                 return X, y1
    #     elif likelihood_func == "chi2":
    #         if np.array(sigma).ndim == 1:
    #             def ln_likelihood(theta, x, y, sigma, **kwargs):
    #                 model = kwargs["model"]
    #                 y1 =  model(x, theta)
    #                 res = np.sum(((y - y1) / sigma)**2)
    #                 return -0.5 * res, y1
    #         else:
    #             log_det_C = np.linalg.slogdet(cov)[1]
    #             inv_cov_matrix = np.linalg.inv(cov)
    #             def ln_likelihood(theta, x, y, sigma, **kwargs):
    #                 model = kwargs["model"]
    #                 y1 = model(x, theta)
    #                 residual = y - y1
    #                 chi2 = np.dot(residual, np.dot(inv_cov_matrix, residual))
    #                 log_likelihood_value = -0.5 * chi2
    #                 return log_likelihood_value, y1
    #     param_limits = [
    #         np.array(prior_parameters[i][-1].split("|")).astype(float)[-2::]
    #         for i in range(len(prior_parameters))
    #         if "free" in prior_parameters[i]
    #     ]
    #     ndims = len(param_limits)
    #     initial_guess = np.array([random_initial_steps(param_limits[i], nwalkers) for i in range(len(param_limits))]).T
    #     """
    #     initial_guess = np.array(
    #         [
    #            np.array(
    #                [
    #                     np.random.uniform(*param_limits[j] * 0.90)
    #                     for j in range(len(param_limits))
    #                 ]
    #             )
    #             for i in range(nwalkers)
    #         ]
    #     )
    #     """
    #     if demo:
    #         print(20*"="+"\nRunning demo...\n"+20*"=")
    #         demo_path = data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + "demo"
    #         if os.path.exists(demo_path) == False:
    #             os.mkdir(demo_path)
    #         #initial guess
    #         fig,ax = plt.subplots()
    #         good_parameters = []
    #         ax.errorbar(
    #             empty_group.R.value,
    #             empty_group.mean_profile,
    #             yerr=empty_group.error_in_mean,
    #             capsize=4,
    #             fmt="o",
    #             label="data",
    #             color = 'red',
    #             lw = 1.5
    #         )
    #         for sample in initial_guess:
    #             p = func(empty_group.R,sample) 
    #             if np.any(p > np.max(empty_group.mean_profile)) or np.any(p <= 1e-10) or p[0] <= p[-1]:
    #                 continue
    #             else:
    #                 good_parameters.append(sample)
    #                 ax.plot(empty_group.R.value,p,alpha = 0.3, color='black')
    #         ax.set(yscale = 'log', ylabel = '$\\langle y \\rangle$', xlabel = f'R {empty_group.R.unit}', title = 'DEMO')
    #         ax.plot([],[]," ",label = f'N of good parameters = {len(good_parameters)}')
    #         ax.grid(True)
    #         ax.legend()
    #         fig.savefig(demo_path + "/initial_sample.png")
    #         #demo of ln_posterior
    #         with cProfile.Profile() as pr:
    #             theta_demo = initial_guess[np.random.randint(0,len(initial_guess))]
    #             x_demo, y_demo, sigma_demo = empty_group.R.value, empty_group.mean_profile, empty_group.error_in_mean
    #             N = config["STACKED_HALO_MODEL"]["N demos"]
    #             for demos in range(len(N)):
    #                 ln_posterior(theta_demo, x_demo, y_demo, sigma_demo, model = func, ln_prior = ln_prior, ln_likelihood = ln_likelihood, chi2 = chi2)
    #                 #pr.print_stats()
    #         continue
    #     pool = Pool(38)
    #     filename = output_path + "/" + fil_name + "." + ext
    #     if skip == True and os.path.exists(filename) == True:
    #         if os.path.getsize(filename)/1e9 >= 1:
    #             continue
    #     print(f"running MCMC in interval richness: {intervals[n][0]} , redshift {intervals[n][1]}")
    #     print("saving sample in "+filename)
    #     backend = emcee.backends.HDFBackend(filename)
    #     #blobs
    #     blobs_config = config["BLOBS"]
    #     dtype = []
    #     for i,key in enumerate(list(blobs_config.keys())):
    #         b = blobs_config[key]
    #         dt, shape = prop2arr(b, dtype = str)
    #         if dt == "np.float64":
    #             dt = np.float64
    #         if len(shape.split('|'))==1:
    #             shape = int(shape.replace('(','').replace(')',''))
    #             if shape == 1:
    #                 dtype.append((key,dt))
    #             elif shape > 1:
    #                 dtype.append((key,np.dtype((dt, shape))))
    #     sampler = emcee.EnsembleSampler(
    #         nwalkers,
    #         ndims,
    #         ln_posterior,
    #         args=(
    #             empty_group.R.value,
    #             empty_group.mean_profile,
    #             sigma,
    #         ),
    #         kwargs={
    #             "model": func,
    #             "ln_prior": ln_prior,
    #             "ln_likelihood": ln_likelihood,
    #             "chi2": chi2
    #         },
    #         pool = pool,
    #         backend=backend,
    #         blobs_dtype = dtype
    #     )
    #     if rewrite == True:
    #         print("rewriting backend")
    #         backend.reset(nwalkers, ndims)
    #     elif rewrite == False and os.path.exists(filename):
    #        try:
    #            initial_guess = None
    #            last_chain = backend.get_last_sample()
    #            sampler._previous_state = last_chain.coords
    #        except Exception as e:
    #            initial_guess =  np.array(
    #             [
    #             np.array(
    #                 [
    #                     np.random.uniform(*param_limits[j] * 0.90)
    #                     for j in range(len(param_limits))
    #                     ]
    #                 )
    #                 for i in range(nwalkers)
    #                 ]
    #             )
    #            print("Chain can't open the last sample. return the exception: \n",e)
        
    #     t1 = time()
    #     sampler.run_mcmc(initial_guess, nsteps, progress=True, store = True)
    #     t2 = time()
    #     print(f"The sampler of richess {intervals[i][0]} and redshift {intervals[i][1]} was finished in {t2 - t1} seconds.")