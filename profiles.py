import numpy as np
from scipy.integrate import simpson as simp
from scipy.integrate import trapezoid as trapz
from astropy import units as u
from helpers import *
from configparser import ConfigParser
from config import *
import os
import pyccl as ccl
from scipy.integrate import cumulative_trapezoid as cum_trapz
from icecream import ic
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.interpolate import RegularGridInterpolator
from numba import njit
from numba import prange

from numba_func import * 

cosmology.setCosmology('planck18') 

from astropy.cosmology import Planck18 as planck18
import astropy.constants as const

current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path + "/config.ini"
config = ConfigParser()
config.optionxform = str


ccl_cosmo = ccl.CosmologyVanillaLCDM()

global ycompton_factor
ycompton_factor = ((const.sigma_T / (const.m_e * const.c**2))).value
global kappa_factor
kappa_factor = (const.c**2 / (4 * np.pi * const.G)).to(u.Msun / u.Mpc)
import pyccl as ccl


mdef_500c = ccl.halos.massdef.MassDef(500, 'critical')
mdef_200c = ccl.halos.massdef.MassDef(200, 'critical')

c500c = ccl.halos.concentration.ishiyama21.ConcentrationIshiyama21(mass_def=mdef_500c)  
c200c = ccl.halos.concentration.ishiyama21.ConcentrationIshiyama21(mass_def=mdef_200c) 

m500tom200 = ccl.halos.massdef.mass_translator(
    mass_in=mdef_500c,
    mass_out=mdef_200c,
    concentration=c500c
) 
m200tom500 = ccl.halos.massdef.mass_translator(
    mass_in=mdef_200c,
    mass_out=mdef_500c,
    concentration=c200c
) 



def create_mass_interpolator(plot = False, background_500 = "critical", background_200 = "critical"):

    mdef_500c = ccl.halos.massdef.MassDef(500, background_500)
    mdef_200c = ccl.halos.massdef.MassDef(200, background_200)

    if background_200 == "matter":
        c200c = ccl.halos.concentration.bhattacharya13.ConcentrationBhattacharya13(mass_def = mdef_200c)
    else:
        c200c = ccl.halos.concentration.ishiyama21.ConcentrationIshiyama21(mass_def=mdef_200c) 
    if background_500 == "matter":
        c500c = ccl.halos.concentration.bhattacharya13.ConcentrationBhattacharya13(mass_def = mdef_500c)
    else:
        c500c = ccl.halos.concentration.ishiyama21.ConcentrationIshiyama21(mass_def=mdef_500c)  
    
    m500tom200 = ccl.halos.massdef.mass_translator(
        mass_in=mdef_500c,
        mass_out=mdef_200c,
        concentration=c500c
    ) 
    m200tom500 = ccl.halos.massdef.mass_translator(
        mass_in=mdef_200c,
        mass_out=mdef_500c,
        concentration=c200c
    ) 

    M200 = np.logspace(12, 16.1, 100)
    M500 = np.logspace(12, 16.1, 100)
    z = np.linspace(1e-5, 3, 100)

    m200tom500_values = np.log10([m200tom500(ccl_cosmo, M200, 1/(zi+1)) for zi in z])
    m500tom200_values = np.log10([m500tom200(ccl_cosmo, M500, 1/(zi+1)) for zi in z])
    
    fM200toM500 = RegularGridInterpolator((np.log10(M200), z), m200tom500_values.T, bounds_error = False)
    fM500toM200 = RegularGridInterpolator((np.log10(M500), z), m500tom200_values.T, bounds_error = False)

    if plot == True:
        m500 = 10**fM200toM500((np.log10(M200), 0.3))
        m500_exact = m200tom500(ccl_cosmo, M200, 0.3)
        fig, ax = plt.subplots(figsize = (8, 4))
        ax.plot(M200, m500, label = "Interpolated")
        ax.plot(M200, m500_exact, label = "Exact")
        ax.loglog()
        ax.legend(fontsize = 20)
        fig.savefig("m200tom500.png")
    return fM200toM500, fM500toM200

fM200toM500, fM500toM200 = create_mass_interpolator()
fM200mtoM500c, fM500ctoM200m = create_mass_interpolator(background_500 = "critical", background_200 = "matter")

def create_concentration_interpolator(plot = False):
    M200 = np.logspace(12, 16.1, 500)
    z = np.linspace(0.01, 2, 500)
    M500c = fM200toM500((np.log10(M200), z))
    c500c_ishi = np.array([concentration.concentration(M500c ,"500c", zi, model = "ishiyama21") for zi in z])
    c200m_duff_z = [concentration.concentration(M200, "200m", zi, model = "duffy08") for zi in z]

    fc500c2c200m = RegularGridInterpolator((np.log10(M200), z), np.log10(c500c_ishi / c200m_duff_z).T, bounds_error = False)
    return fc500c2c200m

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

def NFW(R, M, z, rh0, rs):
    x = R / rs
    return rh0 / (x * (1 + x)**2)

LOG10 = np.log(10)

@njit(fastmath=True, cache = True)
def GNFW(r, p0, gamma, beta, alpha, rs):
    x = r / rs
    xa = x ** alpha
    xg = x ** gamma
    return p0 / (xg * (1.0 + xa) ** ((beta - gamma) / alpha))

def GNFW2(r, M, z, p0, gamma, beta, rs):
    x = r / rs
    profile = p0 / ( (x**gamma) * (1 + x)**(beta - gamma))
    return profile

def sigma_crit_cmb(z):
    D_d  = planck18.angular_diameter_distance(z)
    D_s  = planck18.angular_diameter_distance(1100)
    D_ds = planck18.angular_diameter_distance_z1z2(z, 1100)
    return kappa_factor * D_s / (D_d * D_ds)

def analytic_nfw(R, M, z, rho, rs, dtype = np.float32):

    rho_s = 10**rho
    r_s = 10**rs
    
    R = np.asarray(R, dtype)
    r_s = np.asarray(r_s, dtype)
    rho_s = np.asarray(rho_s, dtype)
    x = R / r_s

    r_s_b = np.broadcast_to(r_s, x.shape).astype(dtype)
    rho_s_b = np.broadcast_to(rho_s, x.shape).astype(dtype)

    Sigma = np.empty_like(x, dtype = dtype)
    prefac = 2 * rho_s_b * r_s_b
    mask_lt = x < 1
    if np.any(mask_lt):
        xm = x[mask_lt]
        A = np.arctan(np.sqrt((1 - xm) / (1 + xm)))
        Sigma[mask_lt] = prefac[mask_lt] / (xm**2 - 1) * (
            1 - 2 / np.sqrt(1 - xm**2) * A
        )
    mask_gt = x > 1
    if np.any(mask_gt):
        xm = x[mask_gt]
        B = np.arctanh(np.sqrt((xm - 1) / (xm + 1)))
        Sigma[mask_gt] = prefac[mask_gt] / (xm**2 - 1) * (
            1 - 2 / np.sqrt(xm**2 - 1) * B
        )
    mask_eq = np.isclose(x, 1.0)
    if np.any(mask_eq):
        Sigma[mask_eq] = (2.0 / 3.0) * rho_s_b[mask_eq] * r_s_b[mask_eq]

    return Sigma


def M_enclosed(r_m, M200, R200, c):
    r_s = R200 / c
    x = r_m / r_s
    norm = M200 / (np.log(1+c) - c/(1+c))
    return norm * (np.log(1+x) - x/(1+x))


def Pandey25(R, richness, M200, z, params, rbins = 35, dtype = np.float32, redshift_pivot = 0.465, richness_pivot = 32.68,
    ):
    
    R200 = (M200 / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    unique_M, inv_idxM = np.unique(M200, return_inverse=True)
    
    c200 = np.asarray([concentration.concentration(unique_M, '200c', zi, model = 'ishiyama21') for zi in unique_z]).astype(dtype).T

    c200 = auto_broadcast_axes(c200, M200)

    rs = R200 / c200 #scale radius in Mpc
    rho0 = M200 / (4 * np.pi * rs**3) * (np.log((rs + R200)/rs) + rs/(rs+R200) - 1)**(-1)
    #Truncated NFW
    Rlos_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj = np.sqrt(np.expand_dims(Rlos_Mpc, tuple(range(1, R.ndim + 1)))**2 + R[None, ...]**2)

    x = R_proj / rs
    y = R_proj / (4*R200)
    
    rho_nfw = trapz(rho0 / (x*(1 + x)**2)*(1 / (1 + y**2)**2), axis = 0, x = R_proj)
    Mtot = trapz(2*np.pi*R*rho_nfw, axis = 0, x = R)

    #central stellar profile from Pandey et al 2024
    Rh = 0.015*R200 #stellar half-light radius
    f_cga = 0.09*(2.5*1e11 / 0.7 / unique_M)**0.6 #stellar abundance in central galaxy from Moster et al 2013
    f_star = 0.09*(2.5e11 / 0.7 / unique_M)**0.19 #stellar abundance in satellites + central galaxy from Shivam et al 2024
    f_sga = 1 - f_cga - f_star #stellar abundance in satellites


def Battaglia16(R, richness, M200, z, params, rbins = 35, dtype = np.float64, redshift_pivot = 0.465, richness_pivot = 32.68):
    AP0, alpha_mP0, alpha_zP0, gamma, alpha, Abeta, alpha_mbeta, alpha_zbeta, Axc, alpha_mxc, alpha_zxc = params
    fb = ccl_cosmo["Omega_b"]/ccl_cosmo["Omega_m"]
    M200 = np.float64(M200)
    rho_c = planck18.critical_density(z).to(u.Msun / u.Mpc**3) 
    Om = planck18.Om(z)
    R200 = (M200 / (4*np.pi/3 * 200 * rho_c * Om))**(1/3)
    G = const.G.to(u.Mpc**3/(u.Msun * u.s**2))
    P200_val = P200(M200, z, planck18)
    P0 = AP0 * (M200/(10**(14.35)))**alpha_mP0 * ((1+z))**alpha_zP0
    beta = Abeta * (M200/(10**(14.35)))**alpha_mbeta * ((1+z))**alpha_zbeta
    xc = Axc * (M200/(10**(14.35)))**alpha_mxc * (1+z)**alpha_zxc
    y = (const.sigma_T / (const.m_e * const.c**2)).to(u.s**2/u.Msun)
    
    R_los = np.logspace(-9, 6, rbins)
    R_proj = np.sqrt(np.expand_dims(R_los, tuple(range(1, R.ndim + 1)))**2 + R[None, ...]**2)

    x = R_proj / R200.value
    Pth_3d = P200_val * P0 * (x/xc)**gamma * ((1 + (x/xc)**alpha)**(-beta))
    Pth = 2*np.trapz(Pth_3d, axis = 0, x = R_los)
    Pe = Pth/1.932
    Y = (Pe*y).value
    return Y


def P200(M200, z, cosmo):
    
    rho_c = cosmo.critical_density(z).to(u.Msun / u.Mpc**3)
    fb = cosmo.Ob0 / cosmo.Om0

    M200 = np.float64(M200) * u.Msun

    R200 = ((M200 / (4*np.pi/3 * 200 * rho_c)).to(u.Mpc**3))**(1/3)

    G = const.G.to(u.Mpc**3/(u.Msun * u.s**2))

    P200 = (G * M200 * fb * 200 * rho_c) / (2 * R200)
    return P200.to(u.Msun/(u.Mpc * u.s**2))


def kappa_general(R, richness, M, z, params, rbins = 35, dtype = np.float32, redshift_pivot = 0.465, richness_pivot = 32.68):
    Arho, Brho, Crho, Ac, Bc, Cc = params

    rho = np.log10( 10**Arho*(richness/richness_pivot)**Brho*((1+z)/(1+redshift_pivot))**Crho )
    c = Ac*(1.37 / (1+ z))**Cc * (M / (8e14/0.7))**Bc

    R200 = (M / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    rs = R200 / c #scale radius in Mpc

    R_los = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj = np.sqrt(R_los[:, None, None, None, None]**2 + R[None, ...]**2)
    sigma_vals = 2*np.trapz(NFW(R_proj, M, z, rho, np.log10(rs)), R_los, axis=0)

    z = np.asarray(z, dtype = dtype)

    unique_z, inv_idx = np.unique(z, return_inverse=True)
    sigma_crit_unique = sigma_crit_cmb(unique_z)
    sigma_crit = sigma_crit_unique[inv_idx].reshape(z.shape)

    return (sigma_vals * u.Msun / u.Mpc**2 / sigma_crit).value



def kappa_general_richness(R, richness, z, params, rbins = 35):

    Arho, Brho, Crho, Ars, Brs, Crs = params
    rho = np.log10(power_law_model(richness, z, 32.68, 0.4737, 10**Arho, Brho, Crho))
    rs = np.log10(power_law_model(richness, z, 32.68, 0.4737, 10**Ars, Brs, Crs))
    
    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = np.float32)

    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    sigma = 2 * trapz(NFW(R_proj_Mpc, M, z, rho, rs), R_los_Mpc, axis=0)
    sigma_crit = sigma_crit_cmb(z)

    return sigma / sigma_crit

def kappa(R, richness, M, z, params, rbins = 45, dtype = np.float32, richness_pivot = 32.68, redshift_pivot = 0.465):
    Ac,Bc,Cc = 3.66, -0.14, -0.32
    rho, c200 = params
    R200 = (M / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc

    c = c200*(Ac*((1.37 / (1+ z))**Cc) * (M / (8e14/0.7))**Bc)
    
    rs = R200 / c #scale radius in Mpc
    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    sigma = 2 * trapz(NFW(R_proj_Mpc, M, z, rho, np.log10(rs)), R_los_Mpc, axis=0)
    z = np.asarray(z, dtype = dtype)
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    sigma_crit_unique = sigma_crit_cmb(unique_z)
    sigma_crit = sigma_crit_unique[inv_idx].reshape(z.shape)
    return (sigma * u.Msun / u.Mpc**2 / sigma_crit).value

def ycompton_general(R, richness, z, params, rbins = 35, redshift_pivot = 0.465, richness_pivot = 32.68):
    Arho, Brho, Crho, gamma, beta, alpha, Ars, Brs, Crs = params
    P0 = np.log10(power_law_model(richness, z, richness_pivot, redshift_pivot, 10**Arho, Brho, Crho))
    rs = np.log10(power_law_model(richness, z, richness_pivot, redshift_pivot, 10**Ars, Brs, Crs)) 

    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_Mpc, richness, z, P0, gamma, beta, alpha, rs), R_los_Mpc, axis=0)
    
    #pressure = P500(M, z, planck18) #add P500 assuming arnaud et al 2010 
    return np.asarray(profile*ycompton_factor, dtype = np.float32)

@njit(fastmath = True, cache = True)
def mis_centering_model(M, z, params, redshift_pivot = 0.465):
    fmis, sigma0 = params
    mass_pivot = 10**(14.35)
    sigma = sigma0 * (M/mass_pivot)**(-0.63)
    return fmis, sigma

def ycompton_individualM200(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.465, richness_pivot = 32.68):
    
    P0, gamma, beta, alpha, rs = params
    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_Mpc, richness, z, P0, gamma, beta, alpha, rs), R_los_Mpc, axis=0)
    return profile*ycompton_factor

def ycompton_generalM200(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.465, richness_pivot = 32.68):
    Arho, Brho, Crho, gamma, beta, alpha, Ars, Brs, Crs = params

    mass_pivot = 10**(14.31)
    P0 = np.log10(10**Arho*((M200/mass_pivot)**Brho*((1 + z)/(1 + redshift_pivot))**Crho))
    rs = np.log10(10**Ars*(M200/mass_pivot)**Brs*((1 + z)/(1 + redshift_pivot))**Crs)
    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_Mpc, richness, z, P0, gamma, beta, alpha, rs), R_los_Mpc, axis=0)
    return np.asarray(profile*ycompton_factor, dtype = np.float32)



def ycompton_general2M200(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.465, richness_pivot = 32.68):
    Arho, Brho, Crho, gamma0, gammaM, gammaZ, beta, alpha, rs = params
    mass_pivot = 10**(14.31)
    gamma = gamma0*(M200/mass_pivot)**(gammaM)*((1 + z)/(1 + redshift_pivot))**gammaZ
    P0 = np.log10(10**Arho*((M200/mass_pivot)**Brho*(planck18.efunc(z))**Crho))
    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_Mpc, richness, z, P0, gamma, beta, alpha, np.log10(rs)), R_los_Mpc, axis=0)
    return np.asarray(profile*ycompton_factor, dtype = np.float32)


@njit(fastmath=True, cache = True)
def ycompton_general3M200_4d(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.4737, richness_pivot = 32.68,
                            projected = True):
    
    mass_pivot = 10**(14.35)

    Arho, Brho, Crho, gamma0, gammaM, gammaZ, beta, betaM, betaZ, alpha, alphaM, alphaZ, logc200, c200M, c200z = params

    NR = R.shape[0]
    Nl = R.shape[1]
    NM = R.shape[2]
    Nz = R.shape[3]

    R3 = R[:, 0, :, :]    
    M2 = M200[0, :, :] 
    z2 = z[0, :, :]      

    om    = z2Om(z2)
    rho   = z2rho(z2)
    E     = z2E(z2)

    R200   = (M2 / (4.0 * np.pi / 3.0 * 200.0 * om * rho)) ** (1.0 / 3.0)

    c200   = 10.0**logc200 * (M2/ mass_pivot)**c200M* ((1.0 + z2) / (1.0 + redshift_pivot))**c200z
    rs     = R200 / c200

    gamma  = (10.0**gamma0
            * (M2 / mass_pivot)**gammaM
            * ((1.0 + z2) / (1.0 + redshift_pivot))**gammaZ)
    
    P0 = 10**Arho * (M2/mass_pivot)**Brho * (E)**Crho * ycompton_factor
    
    #P200_val = P200_self_similar(M2, z2, Brho, Crho)*2.4863e-18 * 1.615e+15

    #P0    =  10.0**Arho * P200_val
    

    R_los_Mpc = np.logspace(-2, 2, rbins)

    NR1 = R.shape[0]
    NR2 = R.shape[1]
    NM  = R.shape[2]
    Nz  = R.shape[3]

    R3 = R[:, 0, :, :] 

    if projected == True:
        R_proj_Mpc = np.zeros((rbins, NR1, NM, Nz))
        P0_4d      = np.zeros((rbins, NR1, NM, Nz))
        gamma_4d   = np.zeros((rbins, NR1, NM, Nz))
        rs_4d      = np.zeros((rbins, NR1, NM, Nz))

        for i in prange(rbins):
            for j in range(NR1):
                for k in range(NM):
                    for l in range(Nz):
                        R_proj_Mpc[i, j, k, l] = (R_los_Mpc[i]**2 + R3[j, k, l]**2)**0.5
                        P0_4d     [i, j, k, l] = P0   [k, l]
                        gamma_4d  [i, j, k, l] = gamma[k, l]
                        rs_4d     [i, j, k, l] = rs[k, l]
        
        profile_3d = 2.0 * trapz_axis0(
            GNFW(R_proj_Mpc, P0_4d, gamma_4d, beta, alpha, rs_4d),
            R_los_Mpc
        ) 

        out = np.empty((NR, Nl, NM, Nz), dtype=np.float32)
        for li in range(Nl):
            for ri in range(NR):
                for mi in range(NM):
                    for zi in range(Nz):
                        out[ri, li, mi, zi] = profile_3d[ri, mi, zi]
        return out
    else:
        P0_3d = np.zeros((NR1, NM, Nz))
        gamma_3d = np.zeros((NR1, NM, Nz))
        rs_3d = np.zeros((NR1, NM, Nz))

        for i in prange(NR1):
            for j in range(NM):
                for k in range(Nz):
                    P0_3d[i, j, k] = P0[j, k]
                    gamma_3d[i, j, k] = gamma[j, k]
                    rs_3d[i, j, k] = rs[j, k]

        profile_3d = GNFW(R3, P0_3d, gamma_3d, beta, alpha, rs_3d)

        out = np.empty((NR, Nl, NM, Nz), dtype=np.float32)
        for li in range(Nl):
            for ri in range(NR):
                for mi in range(NM):
                    for zi in range(Nz):
                        out[ri, li, mi, zi] = profile_3d[ri, mi, zi]
        return out

@njit(fastmath = True, cache = True)
def ycompton_general3M200_3d(R, richness, M200, z, params, rbins=35, redshift_pivot = 0.4737, richness_pivot=32.68,
                             projected = True):

    mass_pivot = 10.0**(14.35)

    Arho, Brho, Crho, gamma0, gammaM, gammaZ, beta, betaM, betaZ, alpha, alphaM, alphaZ, logc200, c200M, c200z = params

    NR = R.shape[0]
    NM = R.shape[1]
    Nz = R.shape[2]

    M2 = M200[0, :, :]
    z2 = z[0, :, :]

    om  = z2Om(z2)
    rho = z2rho(z2)
    E   = z2E(z2)

    R200  = (M2 / (4.0 * np.pi / 3.0 * 200.0 * om * rho)) ** (1.0 / 3.0)
    c200  = 10.0**logc200 * (M2 / mass_pivot)**c200M * ((1.0 + z2) / (1.0 + redshift_pivot))**c200z
    rs    = R200 / c200
    gamma = (10.0**gamma0
            * (M2 / mass_pivot)**gammaM
            * ((1.0 + z2) / (1.0 + redshift_pivot))**gammaZ)

    #P200_val = P200_self_similar(M2, z2, Brho, Crho)*2.4863e-18 * 1.615e+15

    #P0    = 10.0**Arho * P200_val

    P0 = 10**Arho * (M2/mass_pivot)**Brho * (E)**Crho * ycompton_factor

    if projected == True:
        R_los_Mpc = np.logspace(-2, 2, rbins)

        R_proj_Mpc = np.zeros((rbins, NR, NM, Nz))
        P0_4d      = np.zeros((rbins, NR, NM, Nz))
        gamma_4d   = np.zeros((rbins, NR, NM, Nz))
        rs_4d      = np.zeros((rbins, NR, NM, Nz))

        for i in prange(rbins):
            for j in range(NR):
                for k in range(NM):
                    for l in range(Nz):
                        R_proj_Mpc[i, j, k, l] = (R_los_Mpc[i]**2 + R[j, k, l]**2)**0.5
                        P0_4d     [i, j, k, l] = P0   [k, l]
                        gamma_4d  [i, j, k, l] = gamma[k, l]
                        rs_4d     [i, j, k, l] = rs[k, l]
        out =  2.0 * trapz_axis0(
            GNFW(R_proj_Mpc, P0_4d, gamma_4d, beta, alpha, rs_4d),
            R_los_Mpc
        )
        return out
    else:
        P0_3d = np.zeros((NR, NM, Nz))
        gamma_3d = np.zeros((NR, NM, Nz))
        rs_3d = np.zeros((NR, NM, Nz))
        for i in prange(NR):
            for j in range(NM):
                for k in range(Nz):
                    P0_3d [i, j, k] = P0[j, k]
                    gamma_3d [i, j, k] = gamma[j, k]
                    rs_3d [i, j, k] = rs[j, k]
        return GNFW(R, P0_3d, gamma_3d, beta, alpha, rs_3d)



def ycompton_general3M200_1d(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.4737, richness_pivot = 32.68,
                        projected = True):
    P0, Pm, Pz, gamma0, gammaM, gammaz, beta0, betaM, betaz, alpha0, alpha_M, alpha_z, logc200, c200M, c200z = params
    
    om  = z2Om(z)
    rho = z2rho(z)
    E   = z2E(z)
    
    mass_pivot = 10**(14.35)
    R200 = (M200/(4*np.pi / 3 * 200 * planck18.Om(z) * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    
    c200 = np.log10(10**logc200 * (M200/mass_pivot)**c200M * ((1 + z)/(1 + redshift_pivot))**c200z)
    alpha = alpha0 * (M200/mass_pivot)**alpha_M * ((1 + z)/(1 + redshift_pivot))**alpha_z
    beta = beta0 * (M200/mass_pivot)**betaM * ((1 + z)/(1 + redshift_pivot))**betaz
    gamma = 10**gamma0 * (M200/mass_pivot)**gammaM * ((1 + z)/(1 + redshift_pivot))**gammaz

    rs = R200/(10**c200)

    #P200_val = P200_self_similar(M200, z, Pm, Pz)*2.4863e-18 * 1.615e+15

    #P    = 10.0**P0 * P200_val
    
    P = 10**P0 * (M200/mass_pivot)**Pm * (E)**Pz * ycompton_factor

    if projected == True:
        R_los_Mpc = np.logspace(-9, 6, rbins)
        R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5

        profile_3d = P / ((R_proj_Mpc / rs) ** gamma * (1 + (R_proj_Mpc / rs) ** alpha) ** ((beta - gamma) / alpha))
        profile = 2 * np.trapz(profile_3d, R_los_Mpc, axis=0)
        return profile
    else:
        profile_3d = P / (R / rs) ** gamma * (1 + (R / rs) ** alpha) ** ((beta - gamma) / alpha)
        return profile_3d


def ycompton_general3M200(R, richness, M200, z, params,
                          rbins=35, redshift_pivot=0.4737, richness_pivot=32.68,
                          projected = True):
    if R.ndim == 4:
        return ycompton_general3M200_4d(R, richness, M200, z, params, rbins, redshift_pivot, richness_pivot, projected)
    elif R.ndim == 1:
        return ycompton_general3M200_1d(R, richness, M200, z, params, rbins, redshift_pivot, richness_pivot, projected)
    else:
        return ycompton_general3M200_3d(R, richness, M200, z, params, rbins, redshift_pivot, richness_pivot, projected)




def ycompton_general4M200(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.4737, richness_pivot = 32.68):
    Arho, Brho, Crho, gamma, beta, alpha, logc200, c200M, c200z = params
    mass_pivot = 10**(14.31)
    R200 = (M200/(4*np.pi / 3 * 200 * planck18.Om(z) * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    c200 = np.log10(10**logc200 * (M200/mass_pivot)**c200M * ((1 + z)/(1 + redshift_pivot))**c200z)
    rs = R200/(10**c200)
    P0 = np.log10(10**Arho*((M200/mass_pivot)**Brho*(planck18.efunc(z))**Crho))
    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW2(R_proj_Mpc, richness, z, P0, gamma, beta, alpha, np.log10(rs)), R_los_Mpc, axis=0)
    return np.asarray(profile*ycompton_factor, dtype = np.float32)

def ycompton_general5M200(R, richness, M200, z, params, rbins = 35, redshift_pivot = 0.47, richness_pivot = 32.68, 
                        projected = True):
    Arho, Brho, Crho, gamma0, gammaM, gammaz, beta0, betaM, betaz, alpha0, alpha_M, alpha_z, logc200, c200M, c200z = params
    mass_pivot = 10**(14.31)
    R200 = (M200/(4*np.pi / 3 * 200 * planck18.Om(z) * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    c200 = 10**logc200 * (M200/mass_pivot)**c200M * ((1 + z)/(1 + redshift_pivot))**c200z
    alpha = alpha0 * (M200/mass_pivot)**alpha_M * ((1 + z)/(1 + redshift_pivot))**alpha_z
    beta = beta0 * (M200/mass_pivot)**betaM * ((1 + z)/(1 + redshift_pivot))**betaz
    gamma = 10**gamma0 * (M200/mass_pivot)**gammaM * ((1 + z)/(1 + redshift_pivot))**gammaz
    rs = R200/c200
    P0 = np.log10(10**Arho*((M200/mass_pivot)**Brho*(planck18.efunc(z))**Crho))
    if projected == True:
        R_los_Mpc = np.logspace(-9, 6, rbins)
        R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
        profile_3d = 10**P0 / ((R_proj_Mpc / rs) ** gamma * (1 + (R_proj_Mpc / rs) ** alpha) ** ((beta - gamma) / alpha))
        return 2 * np.trapz(profile_3d, R_los_Mpc, axis=0) * ycompton_factor
    else:
        profile_3d = 10**P0 / ((R/rs) ** gamma * (1 + (R/rs) ** alpha) ** ((beta - gamma) / alpha))
        return profile_3d * ycompton_factor
 
def Arnaud10(R, richness, M200, z, params, rbins=200, redshift_pivot=0.4737, richness_pivot=32.68):
    rho_c = planck18.critical_density(z).to(u.Msun / u.Mpc**3).value
    P0, c500, gamma, beta, alpha = params
    M500 = 10**fM200toM500((np.log10(M200), z))
    M500 = np.float64(M500)*0.67
    R500 = (M500 / (4 * np.pi / 3 * 500 * rho_c))**(1/3)
    P500_val = P500(M500, z, planck18)*(u.keV / u.cm**3).to(u.Msun/(u.Mpc * u.s**2))
    conversion_factor = (const.sigma_T / (const.m_e * const.c**2)).to(u.s**2/u.Msun)
    R_los_Mpc = np.logspace(-4, 4, rbins)
    r_3d = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    x = c500 * r_3d / R500
    P_3d = P0 / (x**gamma * (1 + x**alpha)**((beta - gamma) / alpha))

    P_integrated = 2 * np.trapz(P_3d, R_los_Mpc, axis=0)
    Y = P_integrated * P500_val * conversion_factor
    Y = Y.value
    return Y 
def P500(M, z, cosmo, Pm = 2/3, Pz = 8/3, alpha = 0.12):
    return 1.65*10**(-3)*(M / (3e14 / 0.67 / (0.67/0.70)))**(Pm)*z2E(z)**(Pz)*(M/(3e14 / 0.67 /  (0.67/0.70)))**alpha * (0.67/0.70)**2

@njit()
def P200_self_similar(M200m, z, Pm = 2/3, Pz = 8/3, alpha = 0):
    return 1.65*10**(-3)*(M200m / (10**(14.35)))**(Pm)*z2E(z)**(Pz)*(M200m/(10**(14.35)))**alpha


def NFW_cored(R, M, z, rho, rs, rc):
    Rho = 10**rho
    Rs = 10**rs
    Rc = 10**rc
    x = R / Rs
    xc = Rc / Rs
    return Rho/((x + xc) * (1 + x)**2)

def Ishiyama21M200Burkert(R, richness, M200, z, params, rbins = 35, dtype = np.float32, redshift_pivot = 0.465, 
    richness_pivot = 32.68):
    A_ratio, c_ratio = params
    z = np.asarray(z).astype(dtype)
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    unique_M, inv_idxM = np.unique(M200, return_inverse=True)
    sigma_crit = sigma_crit_cmb(unique_z)[inv_idx].reshape(z.shape)
    c200 = np.asarray([concentration.concentration(
        unique_M, '200c', zi, model = 'ishiyama21') for zi in unique_z]).astype(dtype).T
    if M200.ndim == 3:
        c200 = np.repeat(c200[None, ...],np.shape(M200)[0], axis = 0).reshape(M200.shape)
    else:
        c200 = np.repeat(c200[None, ...],np.shape(M200)[0], axis = 0)
        c200 = np.repeat(c200[...,None],np.shape(M200)[-1], axis = -1).reshape(M200.shape)

    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)
    c200 = c200 * c_ratio
    R200 = (M200 / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    rs = R200 / c200 #scale radius in Mpc
    X = R200 / rs
    x = R_proj_Mpc / rs
    I =  0.5 * np.log1p(X) + 0.25 * np.log1p(X**2) - 0.5 * np.arctan(X)
    rho_s = M200 / (4.0 * np.pi * rs**3 * I)

    profile = 2 * np.trapz(rho_s/((1 + x)*(1+x**2)), x = R_los_Mpc, axis=0)
    
    convergence = (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value
    res = A_ratio * convergence
    return res
    
def Ishiyama21M200cored(R, richness, M200, z, params, rbins = 35, dtype = np.float32, redshift_pivot = 0.465, 
    richness_pivot = 32.68):

    A_ratio, c_ratio, gamma = params
    c_ratio = 10**c_ratio
    z = np.asarray(z).astype(dtype)
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    unique_M, inv_idxM = np.unique(M200, return_inverse=True)
    sigma_crit = sigma_crit_cmb(unique_z)[inv_idx].reshape(z.shape)

    c200 = np.asarray([concentration.concentration(
        unique_M, '200c', zi, model = 'ishiyama21') for zi in unique_z]).astype(dtype).T

    c200 = auto_broadcast_axes(c200, M200)

    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1)))**2 + R[None, ...]**2)

    c200 = c200 * c_ratio
    R200 = (M200 / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    rs = R200 / c200 #scale radius in Mpc
    x = R_proj_Mpc / rs

    r200_grid = np.linspace(0.1, R200, 10)
    r_proj_mpc = np.sqrt(np.expand_dims(R_los_Mpc, tuple(range(1, r200_grid.ndim + 1)))**2 + r200_grid[None, ...]**2)
    rho = M200 / np.trapz(2*np.pi*r200_grid *np.trapz(A_ratio/((r_proj_mpc/rs)**gamma * (1 + r_proj_mpc/rs)**(3 - gamma)), x = R_los_Mpc, axis = 0), x = r200_grid, axis = 0)
    
    profile = 2 * np.trapz(rho /((x)*gamma * (1 + (x))**(3 - gamma)), x = R_los_Mpc, axis=0)

    convergence = (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value
    res = convergence
    return res

def Ishiyama21M200(R, richness, M200, z, params, rbins = 35, dtype = np.float32, redshift_pivot = 0.465, 
    richness_pivot = 32.68):

    A_ratio, B_ratio, C_ratio, A_c, B_c, C_c = params
    ratio = 10**A_ratio * (richness/richness_pivot)**B_ratio * (z/redshift_pivot)**C_ratio
    c_ratio = 10**A_c * (richness/richness_pivot)**B_c * (z/redshift_pivot)**C_c
    z = np.asarray(z).astype(dtype)
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    unique_M, inv_idxM = np.unique(M200, return_inverse=True)
    sigma_crit = sigma_crit_cmb(unique_z)[inv_idx].reshape(z.shape)

    c200 = np.asarray([concentration.concentration(
        unique_M, '200c', zi, model = 'ishiyama21') for zi in unique_z]).astype(dtype).T

    c200 = auto_broadcast_axes(c200, M200)

    c200 = c200 * c_ratio
    R200 = (M200 / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc
    rs = R200 / c200 #scale radius in Mpc
    
    rho0 =  ratio * M200 / (4 * np.pi * rs**3) * (np.log((rs + R200)/rs) + rs/(rs+R200) - 1)**(-1)
    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)

    x = R_proj_Mpc / rs
    
    profile = 2 * np.trapz(rho0 /(x * (1 + x**2)), x = R_los_Mpc, axis=0)
    convergence = (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value

    res = convergence

    return res


def linear_redshift(z, params):
    A1, A2 = params
    return A1*(1 + (1+z)**A2)

def linear(params):
    A1 = params
    return A1

def Ishiyama21(R, richness, M, z, params, rbins = 35, dtype = np.float32, isM500c = True):

    c_ratio = params
    z = np.asarray(z).astype(dtype)
    unique_z, inv_idx = np.unique(z, return_inverse=True)
    unique_M, inv_idxM = np.unique(M, return_inverse=True)
    sigma_crit = sigma_crit_cmb(unique_z)[inv_idx].reshape(z.shape)

    c500 = np.asarray([concentration.concentration(
        unique_M, '500c', zi, model = 'ishiyama21') for zi in unique_z]).astype(dtype).T

    if M.ndim == 3:
        c500 = np.repeat(c500[None, ...],np.shape(M)[0], axis = 0).reshape(M.shape)
    else:
        c500 = np.repeat(c500[None, ...],np.shape(M)[0], axis = 0)
        c500 = np.repeat(c500[...,None],np.shape(M)[-1], axis = -1).reshape(M.shape)

    c500 = c500 * c_ratio
    R500 = (M / (4 * np.pi / 3 * 500 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R500 in Mpc
    rs = R500 / c500 #scale radius in Mpc
    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)
    
    f_c = np.log(1 + c500) - c500 / (1 + c500)
    rho = M / (4 * np.pi * rs**3 * f_c)
    x = R_proj_Mpc / rs
    profile = 2 * trapz( rho[None, ...]/ (x*(1 + x**2)), x = R_los_Mpc, axis=0)
    return (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value

def merten_et_al_2015_cored(R, richness, M, z, params, rbins = 35, dtype = np.float32, isM200c = False):
    
    A,B,C = 3.66, -0.14, -0.32 #concentration-mass relationship parameters from Merten et al. 2015
    Arho0, Brho0, Crho0, rc = params #normalization parameters for the NFW profile

    rho = np.log10(power_law_model(richness, z, 32.68, 0.4737, 10**Arho0, Brho0, Crho0))

    c = A*(1.37 / (1+ z))**B * (M / (8e14/0.7))**C

    R200 = (M / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc

    rs = np.log10(R200 / c) #scale radius in Mpc   

    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)

    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)

    profile = 2 * trapz(NFW_cored(R_proj_Mpc, M, z, rho, rs, rc), R_los_Mpc, axis=0)

    z = np.asarray(z).astype(dtype)

    unique_z, inv_idx = np.unique(z, return_inverse=True)
    sigma_crit_unique = sigma_crit_cmb(unique_z)

    sigma_crit = (sigma_crit_unique[inv_idx].reshape(z.shape)).astype(dtype)
    return (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value


def merten_et_al_2015(R, richness, M, z, params, rbins = 30, dtype = np.float32, isM200c = True):
    A,B,C = 3.66, -0.14, -0.32 #concentration-mass relationship parameters from Merten et al. 2015
    Arho0, Brho0, Crho0 = params #normalization parameters for the NFW profile

    if isM200c == False:
        unique_M, inv_idxM = np.unique(M, return_inverse=True)
        unique_z, inv_idx = np.unique(z, return_inverse=True)
        M200c = np.asarray([m500tom200(ccl_cosmo, unique_M, 1/(zi + 1)) for zi in unique_z])
        if M.ndim == 3:
            M200c = np.repeat(M500[None, ...],np.shape(M)[0], axis = 0).reshape(M.shape)
        else:
            M200c = np.repeat(c500[None, ...],np.shape(M)[0], axis = 0)
            M200c = np.repeat(c500[...,None],np.shape(M)[-1], axis = -1).reshape(M.shape)
        M = M200c

    rho = np.log10(power_law_model(richness, z, 30.68, 0.4737, 10**Arho0, Brho0, Crho0))

    c = A*(1.37 / (1+ z))**B * (M / (8e14/0.7))**C

    R200 = (M / (4 * np.pi / 3 * 200 * planck18.critical_density(z).to(u.Msun / u.Mpc**3).value))**(1/3) #R200 in Mpc

    rs = np.log10(R200 / c) #scale radius in Mpc

    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)
    profile = 2 * trapz(NFW(R_proj_Mpc, M, z, rho, rs), R_los_Mpc, axis=0)

    z = np.asarray(z).astype(dtype)

    unique_z, inv_idx = np.unique(z, return_inverse=True)
    sigma_crit_unique = sigma_crit_cmb(unique_z)

    sigma_crit = (sigma_crit_unique[inv_idx].reshape(z.shape)).astype(dtype)
    return (profile * u.Msun / u.Mpc**2 / sigma_crit).decompose().value

def ycompton(R, M, z, params, rbins = 35, dtype = np.float32):
    P0, gamma, beta, alpha, rs = params
    richness = 30*(M / (3e14 / 0.7)) ** 0.75
    R_los_Mpc = np.logspace(-10, 6, rbins, dtype = dtype)
    R_proj_Mpc = np.sqrt(R_los_Mpc[:, None, None, None, None]**2 + R[None, ...]**2)
    profile = 2 * trapz(GNFW(R_proj_Mpc, M, z, P0, gamma, beta, alpha, rs), R_los_Mpc, axis=0).astype(dtype)
    return profile*ycompton_factor

def ycompton_fixed_alp_bet(R, M, z, params, rbins = 35):
    P0, gamma, rs = params
    beta,alpha = 6.3, 2.54
    factor = (sigma_T / (m_e * c**2)).value
    richness = 30*(M / (3e14 / 0.7)) ** 0.75
    R_los_Mpc = np.logspace(-10, 6, rbins)
    R_proj_Mpc = (np.expand_dims(R_los_Mpc, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    profile = 2 * trapz(GNFW(R_proj_Mpc, M, z, P0, gamma, beta, alpha, rs), R_los_Mpc, axis=0)
    return profile*ycompton_factor
    
