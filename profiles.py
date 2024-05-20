import numpy as np
from scipy.integrate import simps, trapz
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from helpers import *
from configparser import ConfigParser
import os

current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path + "/config.ini"
config = ConfigParser()
config.optionxform = str

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

rbins = int(config["STACKED_HALO_MODEL"]["rbins"])

def projected_GNFW(R, M, z, params):
    p0,gamma,beta,alpha,rs = params
    R_los = np.logspace(-7,9,rbins)
    R = (np.expand_dims(R_los, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    #R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
    profile = 2 * trapz(GNFW(R,M,z,p0,gamma,beta,alpha,rs),R_los,axis=0)
    return profile

def GNFW(r, M, z, p0, gamma, beta, alpha,  rs):
    RS = 10**rs
    P0 = 10**p0
    x = r / RS
    profile = P0 / ( ((x) ** gamma) * (1 + (x) ** alpha)**((beta - gamma)/alpha) )
    return profile

def projected_GNFW_fixed_c(R,M,z,params):
    p0,gamma,beta,alpha,rs = params
    R_los = np.logspace(-7,9,rbins)
    R = (np.expand_dims(R_los, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    #R = np.transpose([np.hypot(*np.meshgrid(R_los[:,i],R[:])) for i in range(R_los.shape[1])],axes=(1,2,0))
    profile = 2 * trapz(GNFW(R,M,z,p0,gamma,beta,alpha,rs),R_los,axis=0)
    return profile

def projected_GNFW_arcmin(theta,M,z,params):
    p0, gamma, beta, alpha, rs = params
    R_kpc = theta * cosmo.kpc_proper_per_arcmin(z)
    R_kpc = R_kpc.value
    R_los_kpc = np.logspace(-7, 9, rbins)
    R_proj_kpc = (np.expand_dims(R_los_kpc, tuple(range(1, R_kpc.ndim + 1))) ** 2 + R_kpc**2) ** 0.5
    #R_proj_kpc = np.transpose([np.hypot(*np.meshgrid(R_los_kpc[:, i], R_kpc[:])) for i in range(R_los_kpc.shape[1])], axes=(1, 2, 0))
    profile = 2 * trapz(GNFW(R_proj_kpc, M, z, p0, gamma, beta, alpha, rs), R_los_kpc, axis=0)
    return profile

def projected_GNFW_arcmin_fixed_c(theta,M,z,params):
    p0,gamma,beta,alpha,rs = params
    R_kpc = theta * cosmo.kpc_proper_per_arcmin(z)
    R_kpc = R_kpc.value
    R_los_kpc = np.logspace(-7, 9, rbins)
    R_proj_kpc = (np.expand_dims(R_los_kpc, tuple(range(1, R_kpc.ndim + 1))) ** 2 + R_kpc**2) ** 0.5
    #R_proj_kpc = np.transpose([np.hypot(*np.meshgrid(R_los_kpc[:, i], R_kpc[:])) for i in range(R_los_kpc.shape[1])], axes=(1, 2, 0))
    profile = 2 * trapz(GNFW(R_proj_kpc, M, z, p0, gamma, beta, alpha, rs), R_los_kpc, axis=0)
    return profile

def projected_GNFW_arcmin_fixed_c_whole_data(theta, M, z, params):
    pm_M,p0_M, pm_Z, p0_Z, gm1_M, g01_M, gm1_Z, g01_Z, bm_M, b0_M, bm_Z, b0_Z, alpha, rs = params
    richness = 30*(M / (3**14 / 0.7)) ** 0.75
    p0 = pm_M * richness + p0_M + pm_Z*z + p0_Z
    gamma = gm1_M*richness + g01_M + gm1_Z * z + g01_Z
    beta = bm_M * richness + b0_M + bm_Z * z + b0_Z
    R_kpc = theta * cosmo.kpc_proper_per_arcmin(z)
    R_kpc = R_kpc.value
    R_los_kpc = np.logspace(-7, 9, rbins)
    R_proj_kpc = (np.expand_dims(R_los_kpc, tuple(range(1, R_kpc.ndim + 1))) ** 2 + R_kpc**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_kpc, M, z, p0, gamma, beta, alpha, rs), R_los_kpc, axis=0)
    return profile
