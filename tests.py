#this script search which is the best match ratio between DES-Y3 and milliquas catalog

from cluster_data import *
from config import *
from astropy.table import Table
from helpers import *



"""
milliquas = Table(fits.open(data_path + 'milliquas.fits')[1].data)
redmapper  = Table(fits.open(data_path + DES_Y3)[1].data)
types = ["Q","A","B","K","N"]
names = ["QSO type I broad-line core-dominated","AGN type I Seyferts/host-dominated",
            "BL Lac", "Narrow-Line Type II", "Seyferts/host-dominated Type II"]
milliquas["grouped_type"] = group_types(np.array(milliquas["TYPE"]), types, names)
grouped_types = np.unique(milliquas["grouped_type"])
#only agn type

milliquas = milliquas[milliquas["grouped_type"] == grouped_types[1]]
print(milliquas)
rm_RA,rm_DEC = redmapper["RA"],redmapper["DEC"]
mq_RA,mq_DEC = milliquas["RA"],milliquas["DEC"]

rm_RA[rm_RA > 180] = rm_RA[rm_RA > 180] - 360
mq_RA[mq_RA > 180] = mq_RA[mq_RA > 180] - 360

mq_coords = SkyCoord(mq_RA,mq_DEC, unit = (u.degree, u.degree))
rm_coords = SkyCoord(rm_RA,rm_DEC, unit = (u.degree, u.degree))

R = np.arange(0.5, 5 , 0.0001) * u.arcmin
matched_clusters = []
for i,r in enumerate(R):
    r = r.to(u.degree)
    mq_indx, rm_indx, d2d, d3d = rm_coords.search_around_sky(mq_coords, r)
    rm_matched_clusters = redmapper[rm_indx]
    matched_clusters.append(rm_matched_clusters)

R_stairs = np.array([0,*R.value])
print(R_stairs)
lenghts = np.array([len(m) for m in matched_clusters])
fig,ax = plt.subplots()
ax.stairs(lenghts, R_stairs)
ax.set(xlabel = "R match (arcmin)", ylabel = "N of matched AGNs")
ax.grid(True)
fig.savefig("hist_match_test2.png")
"""

#test of the error and S/N ratio to differents size of radial binnings
import os
import matplotlib.pyplot as plt
import numpy as np
"""
path = os.listdir(data_path + "GROUPED_CLUSTERS")
cluster_path = [p for p in path if p.split("_")[0] == "GROUPED"]
clusters = []
stacks = []
errors = []
background_errors = []
width = 0.8
for i in range(1):
    c = grouped_clusters(None)
    c.output_path = data_path + f"GROUPED_CLUSTERS/{cluster_path[i]}"
    c.load_from_h5()
    clusters.append(c)
    s,e,b = c.stacking(return_stack = True, szmap = szmap)
    stacks.append(s)
    errors.append(e)
    background_errors.append(b)
    clusters.append(c)
"""

path = data_path + "GROUPED_CLUSTERS/GROUPED_CLUSTER_RICHNESS=56.0-224.0REDSHIFT=0.1-0.5"
c = grouped_clusters(None)
c.output_path = path
c.load_from_h5()
s,e,b = c.stacking(return_stack = True, szmap = szmap)

Rmax = 0.8 * 60 / 2
dr = np.linspace(0.8,6,10)
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,4))
SNr = []
for i in range(len(dr)):
    R_profiles = np.arange(0,Rmax, dr[i]) * u.arcmin
    Rbins, profile, err = radial_binning(s,R_profiles, weighted = False)
    SNr.append(  np.sqrt(np.sum( np.array(profile)**2/np.array(err)**2 )) / np.sqrt(len(profile)))
    ax1.errorbar(Rbins, profile, yerr = err, fmt = "o", label = dr[i], alpha = 0.4)
ax2.plot(dr, SNr)
ax2.grid(True)
ax2.set(ylabel = "$S/N$ ratio", xlabel = "$dR$")
ax1.set(ylabel = "R (arcmin)", xlabel = "$y$")
ax1.set(yscale = "log")
ax1.grid(True)
ax1.legend()
fig.savefig("stacking_example.png")

aa
#this script test the effect of the parameters in the profiles, this with the aim to constrain correlation between parameters and r_s
from profiles import *
from plot_profiles import *
from astropy.cosmology import Planck18 as cosmo
path = "/data2/javierurrutia/szeffect/data/GROUPED_CLUSTERS/GROUPED_CLUSTER_RICHNESS=56.0-208.0REDSHIFT=0.1-0.5"
results = plot_mcmc(path, return_cluster = True, plot = True)
cluster = results[-1]
R_profiles = cluster.R
func = cluster.stacked_halo_model_func(projected_GNFW_arcmin)
params = np.copy(results[0])
rs_arr = [params[-1] , 3, 3.3, 3.6]

for rs in rs_arr:
    indx = 3 #alpha
    dp = 0.25
    best_param = params[indx]
    param_name = labels[indx]
    n_evals = 50
    params_arr = np.linspace(best_param - dp, best_param + dp, n_evals)
    params_evaluated = np.tile(np.copy(params),n_evals).reshape((n_evals, -1))
    params_evaluated[:,indx] = params_arr
    cmap = plt.cm.viridis
    norm = plt.Normalize(np.min(params_arr),np.max(params_arr))
    best_fit = func(R_profiles, params)
    fig, ax = plt.subplots(figsize = (7,4))
    ax.errorbar(R_profiles.value, cluster.mean_profile, yerr = cluster.error_in_mean, label = "data", color = 'black', fmt = "o")
    z = np.mean(cluster.z)
    rs_arcmin = (10**rs) * u.kpc * (cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.kpc))

    r_s5 = np.linspace(0, 5 * rs_arcmin.value, 50) * u.arcmin

    ax.plot(r_s5, func(r_s5, params), label = "fit", color = cmap(norm(best_param)), lw = 2)
    [ax.plot(r_s5, func(r_s5, params_evaluated[i]), lw = 1, alpha = 0.6, color = cmap(norm(params_arr[i]))) for i in range(len(params_evaluated))]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label=r"$\%.s$" % (param_name) ,cax=fig.add_axes([0.92, 0.1, 0.02, 0.8]))
    text = r"$\%.s \in [%.2f, %.2f]$" % (param_name,np.min(params_arr), np.max(params_arr))
    ax.axvline(rs_arcmin.value, color = 'darkred', ls = '--',label = r"$r_s \text{arcmin} = %.4f$" % rs_arcmin.value)
    ax.legend()
    ax.grid(True)
    ax.set(yscale = 'log', xlabel = "R (arcmin)", ylabel = "$y$", title = r"test with $\log_{10}(r_s) = %.4f $"  % np.round(rs,4), xlim = (0, 5 * rs_arcmin.value))
    fig.savefig(f"test_{param_name}_rs={np.round(rs,2)}.png")
