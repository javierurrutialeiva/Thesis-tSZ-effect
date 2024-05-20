from multiprocessing import Pool, cpu_count
import emcee
import astropy.units as u
import os
import corner
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from configparser import ConfigParser
from helpers import *
import profiles
import importlib
import warnings
import emcee
import shutil
import time
from cluster_data import *
import sys
from plottery.plotutils import colorscale
import argparse
from scipy.signal import find_peaks
from config import *
from sklearn.mixture import GaussianMixture
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-f', type = str, help = "Name of path where is the samples file.")
parser.add_argument('--plot', '-p', action = 'store_true', help = "Plot best fitting and show parameters.")
parser.add_argument('--discard', '-d', type = int, default = 0, help = "Discarted steps.")
parser.add_argument('--steps', '-e', action = 'store_true', help = "Plot paramreters steps.")
parser.add_argument('--corner', '-c', action = 'store_true', help = "If true plot the corners.")
parser.add_argument('--save', '-s', action = 'store_true', help = "If true save the parameters and his priors.")
parser.add_argument('--make_copy', '-m', action = 'store_false', help = 'If passed is not created a copy of the sample file.')
parser.add_argument('--range_sigma_ratio', '-r', type = float, default = None, help = "If passed with a number the range of corner plot is changed to the quantiles of sigma r times.")
args = parser.parse_args()

labels_text = ['$P_0$',r"$\gamma$",r"$\beta$",r"$\alpha$",r"$r_s$"]

def calculate_chi2(y,y1,sigma):
    y,y1,sigma = np.array(y),np.array(y1),np.array(sigma)
    return np.abs(np.sum((y - y1)**2 / sigma **2))


def main():
    source_path = args.path
    plot = args.plot

    if type(source_path) is str:
        if source_path != "all":
            plot_mcmc(data_path + grouped_clusters_path + source_path, plot = plot)
        if source_path == "all":
            richness_pattern = r"RICHNESS=(\d+\.\d+)-(\d+\.\d+)"
            redshift_pattern = r"REDSHIFT=(\d+\.\d+)-(\d+\.\d+)"
            redshift, richness = [],[]
            redhisft_err, richness_err = [], []
            parameters, lower_errors, upper_errors = [], [], []
            all_grouped_cluster = [f"{data_path}{grouped_clusters_path}{path}" for
                                   path in os.listdir(f"{data_path}{grouped_clusters_path}")
                                   if os.path.isdir(f"{data_path}{grouped_clusters_path}{path}") == True]
            with_samples = [path for path in all_grouped_cluster if f"{fil_name}.{ext}" in os.listdir(path)]
            for i,path in enumerate(with_samples):
                print(30*"=","\n")
                print(f"loading parameters from  \033[92m{path.split('/')[-1]}\033[0m.\n")
                try:
                    p, le, ue, r, re, z, ze = plot_mcmc(with_samples[i], plot = plot)
                    #richness_match = re.search(richness_pattern, path.split('/')[-1])
                    #redshift_match = re.search(redshift_pattern, path.split('/')[-1])
                    richness.append(r)
                    richness_err.append(re)
                    redshift.append(z)
                    parameters.append(p), lower_errors.append(le), upper_errors.append(ue)
                except Exception as e:
                    print(f"{path} return exception {e}")
                print("")
            redshift = np.reshape(redshift,(len(redshift),1))
            richness = np.array(richness)
            parameters = np.array(parameters)
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin = np.min(redshift), vmax = 1)
            lower_errors = np.array(lower_errors)
            upper_errors = np.array(upper_errors)
            fig,axes = plt.subplots(len(labels),1,figsize = (12,8),sharex = True)
            for i in range(len(labels)):
                ax = axes[i]
                lower_bound = parameters[:,i] - lower_errors[:,i]
                upper_bound = parameters[:,i] + upper_errors[:,i]
                ax.scatter(richness, parameters[:,i], color = cmap(norm(redshift[:,0])), alpha = 0.5)
                ax.errorbar(richness, parameters[:, i], xerr = richness_err, yerr = (lower_errors[:,i],upper_errors[:,i]), fmt = 'None',
                            capsize = 2, color = 'black', alpha = 0.3)
                #ax.fill_between(richness,lower_bound, upper_bound, color = 'grey', alpha = 0.5)
                ax.grid(True)
                ax.set(ylabel = labels[i])
            axes[-1].set(xlabel = "richness $\\lambda$", xscale = 'log')
            fig.suptitle("Parameter distribution in function of richness $\\lambda$", fontsize = 24)
            cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', label = "redshift $z$")
            fig.savefig(f"{data_path}/param_distribution.png")
            file = []
            file.append(parameters)
            file.append(lower_errors)
            file.append(upper_errors)
            np.save("parameeters.npy", file)
    else:
        print("xd")
    # all_parameters = np.array(all_parameters)
    # lower_limits = np.array(lower_limits)
    # upper_limits = np.array(upper_limits)
    # fig, axes = plt.subplots(len(all_parameters),sharex = True)
    # for i in range(len(all_parameters[0])):
    #     ax = axes[i]
    #     ax.errorbar(list(range(len(all_parameters[:,i]))),all_parameters[:,i],yerr=[lower_limits[:,i],upper_limits[:,i]])
    #     ax.set(ylabel=labels[i],xlabel='richness')
    # fig.savefig("parameters_distribution.png")

def plot_mcmc(source_path, return_cluster = False, plot = False):
    samples_file = source_path +'/'+ fil_name + '.' + ext
    empty_group = grouped_clusters(None)
    empty_group.output_path = source_path
    empty_group.load_from_h5()
    empty_group.mean(from_path = True)
    empty_group.R = np.array([(R_profiles[i + 1] + R_profiles[i]).value/2 for i in range(len(R_profiles) - 1)]) * R_profiles.unit
    empty_group.completeness_and_halo_func()
    func = empty_group.stacked_halo_model_func(getattr(profiles_module, profile_stacked_model))
    output_file = f"_RICHNESS={np.round(np.min(empty_group.richness))}-{np.round(np.max(empty_group.richness))}" + f"REDSHIFT={np.round(np.min(empty_group.z),2)}-{np.round(np.max(empty_group.z),2)}"
    richness, richness_err = np.mean(empty_group.richness),np.std(empty_group.richness)/np.sqrt(len(empty_group))
    redshift, redshift_err = np.mean(empty_group.z),np.std(empty_group.z)/np.sqrt(len(empty_group))
    if os.path.exists(samples_file):
        make_copy = args.make_copy
        if make_copy:
            copy = f"{source_path}/{fil_name}_copy.{ext}"
            shutil.copy(samples_file, copy)
            samples_file = copy
        discard = args.discard
        ndims = len(labels)
        nwalkers = int(config["STACKED_HALO_MODEL"]["nwalkers"])
        backend = emcee.backends.HDFBackend(samples_file, read_only = True)

        #walkers plot

        blobs = backend.get_blobs(discard = discard)
        chi2_values = blobs['CHI2'].flatten()
        signal = blobs['SIGNAL'].flatten()
        ln_prior = blobs['LN_PRIOR'].flatten()
        ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
        chain = backend.get_chain(discard = discard, flat = True)
        tsteps, npar = chain.shape
        stepbin = nwalkers
        stepbins = np.arange(0, tsteps, stepbin) + stepbin / 2
        runmeb = np.transpose(
        [
            np.median(chain[k : k + stepbin], axis=0)
            for k in range(0, tsteps, stepbin)
        ]
        )
        runlo, runhi = np.transpose(
        [
            np.percentile(chain[k : k + stepbin], [16, 84], axis=0)
            for k in range(0, tsteps, stepbin)
        ],
        axes=(1, 2, 0),
        )
        log_prob = backend.get_log_prob(discard = discard, flat = True)
        plot_steps = args.steps
        if plot_steps == True:
            cmap_samples = 20000
            chi2_values = np.log(chi2_values)
            j = np.arange(tsteps)
            colors, cmap = colorscale(
            chi2_values,
            vmin=chi2_values[-cmap_samples:].min(),
            vmax=np.percentile(chi2_values[-cmap_samples:], 99),
            cmap="viridis",
            )
            fig, axes = plt.subplots(
                ndims, 1, sharex=True, constrained_layout=True, figsize=(12, 15)
            )
            for ax, p, m, lo, hi, label in zip(axes, chain.T, runmeb, runlo, runhi, labels):
                ax.scatter(j[::1], p[::1], c=colors, marker=".", s=0.2)
                #ax.plot(stepbins, m, "k-", lw=1.5)
                #ax.plot(stepbins, lo, "k--", lw=1.2)
                #ax.plot(stepbins, hi, "k--", lw=1.2)
                #ax.set(ylabel=label)
            plt.colorbar(cmap, ax=axes, label="Log Likelihood")
            axes[-1].set(xlabel="MCMC step")
            #fig.savefig(f"{data_path}parameters_steps.png")
            fig.savefig(f"{source_path}/parameters_steps{output_file}.png")

        #profiles
        params_min_chi2 = chain[chi2_values >= 1][np.where(chi2_values[chi2_values >= 1] == np.min(chi2_values[chi2_values >= 1]))][0]
        param = []
        lower = []
        upper = []
        range_sigma_ratio = args.range_sigma_ratio
        sigma_ranges = [] #99.73
        for i in range(ndims):
            mcmc = np.percentile(chain[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            param.append(mcmc[1])
            lower.append(q[0])
            upper.append(q[1])
            print(labels[i],f": \033[92m{mcmc[1]}\033[0m - \033[92m{q[1]}\033[0m + \033[92m{q[0]}\033[0m")
            if range_sigma_ratio is not None:
                sigma_ranges.append(np.percentile(chain[:, i], [0.005,99.990]))
            else:
                sigma_ranges = None
        if plot == True:
            R = empty_group.R
            fit = func(R.value,param)
            data = np.array(empty_group.mean_profile)
            err = np.array(empty_group.error_in_mean)
            """
            mean_z = np.mean(empty_group.z)
            P0,gamma,alpha,beta,rs,c = param
            P0, rs = 10**P0, 10**rs * u.kpc
            rs = (rs * cosmo.arcsec_per_kpc_proper(mean_z)).to(u.arcmin)
            rs_err = (rs * (10 ** upper[-1] * u.kpc * cosmo.arcsec_per_kpc_proper(mean_z)).to(u.arcmin)).value,  (rs * (-10 ** lower[-1] * u.kpc * cosmo.arcsec_per_kpc_proper(mean_z)).to(u.arcmin)).value
            P0_fit = 2 ** ((beta - gamma)/ alpha)* func([rs.value * 1.177], param)[0] #expected P0
            P0_fit_err = [2 ** ((beta - gamma)/ alpha)* func([rs.value * 1.177], param + np.array(upper))[0],2 ** ((beta - gamma)/ alpha)* func([rs.value * 1.177], param + np.array(lower))[0]]
            indx = np.argmin(np.abs(rs.value - R.value))
            R_P0 = (R.value[indx + 1] + R.value[indx]) / 2
            R_P0_err = np.std([R.value[indx + 1],R.value[indx]])/ np.sqrt(2)
            P0_data = (data[indx + 1] + data[indx]) / 2
            P0_data_err = np.sqrt(err[indx + 1]**2 + err[indx]**2)

            """
            fig,ax = plt.subplots(figsize=(8,6))
            #ax.scatter(rs.value * 1.177, P0_fit, color = 'red', edgecolor = 'black', s = 40)
            #ax.scatter(R_P0, P0_data, color = 'blue', edgecolor = 'black', s = 40)
            #ax.errorbar(R_P0, P0_data, fmt = " ", xerr = R_P0_err, yerr = P0_data_err, capsize = 3)
            #ax.plot([rs.value * 1.177,rs.value * 1.177], [0,P0_fit], ls = '--', color = 'red', alpha = 0.3)
            #ax.plot([0,rs.value * 1.177], [P0_fit, P0_fit], ls = '--', color = 'red', alpha = 0.3)
            #ax.plot([R_P0,R_P0], [0,P0_data], ls = '--', color = 'blue', alpha = 0.3)
            #ax.plot([0,R_P0], [P0_data, P0_data], ls = '--', color = 'blue', alpha = 0.3)
            ax.errorbar(R,data,yerr=err,capsize=3,color='black',fmt='o',label='data')
            ax.plot(R,fit,label='fit',color='green')
            ax.plot(R,func(R.value, params_min_chi2), label = r'$\min{\chi^2}$', ls = '--', color = 'darkgreen', alpha = 0.6, lw = 2)
            lower_bound = func(R.value,np.array(param) - np.array(lower))
            upper_bound = func(R.value,np.array(param) + np.array(upper))
            ax.fill_between(R.value, lower_bound, upper_bound, color='green', alpha=0.3,label = '$1 \\sigma$')
            chi2 = np.sum((fit - data)**2/err**2) / (len(data) - len(param))
            text  = [
                        r'$\chi_r^{2} = %.5f$ $(%.5f)$' % (chi2,np.min(chi2_values[chi2_values >= 1])/(len(data) - len(param)) ),
                        r'N clusters = $%.i$' % (len(empty_group.richness)),
                        r'$\mathrm{richness} = [%.i,%.i]$' %   (np.min(empty_group.richness), np.max(empty_group.richness)),
                        r'$\mathrm{redshift} = [%.2f, %.2f]$' % (np.min(empty_group.z),np.max(empty_group.z))]
                        #r'N samples = $%.i$' % (len(blobs),),
            for i in range(len(labels_text)):
                if labels[i].split('_')[0] == r'$\log':
                    text.append(f'{labels_text[i]} : {np.round(np.log10(params[i]),2)} $\pm$ {np.round(err[i]/(np.log(10) * params[i]),2)}')
                else:
                    text.append('%s' % labels_text[i] + ': $%.2f' % param[i] + '^{+%.2f}_{-%.2f}$' % (upper[i],lower[i]) + "$({%.2f})$" % params_min_chi2[i])
            text = '\n'.join(text)
            props = dict(boxstyle = 'round', facecolor = 'darkgreen', edgecolor = 'black', alpha = 0.65)
            ax.text(0.03, 0.07 + len(labels_text)*0.07, text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props, color = 'blue')
            ax.legend(loc = 'upper right')
            ax.grid(True)
            ax.set(xlabel=f"$R $({R_profiles.unit})",ylabel="$\\langle y \\rangle$",yscale='log', title = 'best fitting')
            plt.savefig(f"{data_path}best_fitting_profile.png")
            plt.savefig(f"{source_path}/best_fitting_profile{output_file}.png")

            #autocorr_time
            fig, axes = plt.subplots(len(labels),figsize = (5,10), sharex = True)
            for i in range(len(axes)):
                chain_unflatted = backend.get_chain()[:,:, i].T
                ax = axes[i]
                N,tau = autocorr_time_from_chain(chain_unflatted)
                ax.loglog(N, tau, label = r"$\tau $ estimation", color = 'red')
                ax.loglog(N, N/50, label = r"$\tau = N/50$", ls = "--", color = 'black')
                ax.grid(True)
                ax.set_ylabel(labels_text[i], fontsize = 8)
            axes[-1].set(xlabel = "number of samples N")
            axes[-1].legend()
            fig.suptitle(r"$\tau$ estimator for each parameter")
            fig.tight_layout()
            fig.savefig(f"{source_path}/tau.png")

            #ln_prior and ln_likelihood
            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (12,6))
            hist_prior = ln_prior[ln_prior > -np.inf]
            hist_likelihood = ln_likelihood[ln_likelihood > -np.inf]
            hist_likelihood = hist_likelihood[np.where((hist_prior < np.max(hist_prior)) & (hist_prior > np.min(hist_prior)))]
            hist_posterior = log_prob[log_prob> -np.inf]
            hist_posterior = hist_posterior[np.where((hist_prior < np.max(hist_prior)) & (hist_posterior > np.min(hist_prior)))]
            ax1.hist(hist_prior,histtype='barstacked',label = 'ln_prior', edgecolor='black', alpha=0.5,color='green',bins = 50)
            ax2.hist(hist_likelihood,histtype='barstacked',label = 'ln_likelihood', edgecolor='black', alpha=0.5,color='red',bins = 50)
            ax3.hist(hist_posterior,histtype='barstacked',label = 'ln_posterior', edgecolor='black', alpha=0.5,color='blue',bins = 50)
            fig.suptitle("prior, likelihood and posterior distribution",fontsize = 24)
            [ax.grid(True) for ax in [ax1,ax2,ax3]]
            [ax.set(xlabel = "log value", ylabel = "N",yscale = 'log', title = ["prior","likelihood","ln_posterior"][n]) for n,ax in enumerate([ax1,ax2,ax3])]
            fig.savefig(f"{source_path}/hist{output_file}.png")
            fig.savefig(f"{data_path}hist.png")

            #corner plot

            plot_corner = args.corner
            if plot_corner:
                figure = corner.corner(
                    chain,
                    labels = labels_text,
                    quantiles = [0.16, 0.5, 0.84],
                    show_titles = True,
                    title_kwargs = {"fontsize": 12},
                    levels = (1-np.exp(-0.5), 1-np.exp(-2) ),
                    truths = param,
                    plot_density = True,
                    plot_datapoints = False,
                    color = 'blue',
                    truth_color='green',
                    #range = sigma3,
                    bins = 40,
                    hist_bin_factor = 5,
                    smooth = 1.0,
                    smooth1d = 3.0,
                    fill_contours = True,
                    n_max_ticks = 8,
                    range = sigma_ranges,
#                    hist_kwargs = {'log' : True},
                    fill_contour_kwargs={'colors': ['darkblue', 'lightblue']}
                )
                axes = np.array(figure.axes).reshape((len(labels),len(labels)))

                for yi in range(len(labels)):
                    for xi in range(yi + 1):
                        ax = axes[yi,xi]
                        if xi == yi:
                            ax.axvline(params_min_chi2[yi], color = 'red')
                        else:
                            ax = axes[yi,xi]
                            ax.axvline(params_min_chi2[xi], color = 'red')
                            ax.axhline(params_min_chi2[yi], color = 'red')
                            ax.plot(params_min_chi2[xi], params_min_chi2[yi], "sr")
                figtxt = []
                xt,yt = 0.7, 0.88
                figtxt.append(f"richness = $[{np.round(np.min(empty_group.richness))}-{np.round(np.max(empty_group.richness))}]$")
                figtxt.append(f"redshift = $[{np.round(np.min(empty_group.z),2)}-{np.round(np.max(empty_group.z),2)}]$")
                figtxt.append(r"   $\min{\chi^2}$")
                figtxt.append(r"   mean")
                figtxt = '\n'.join(figtxt)
                figure.text(xt,yt, figtxt, fontsize = 16, color = 'black', bbox = dict(facecolor='white', alpha=0.5, boxstyle = 'round'))
                figure.text(xt,yt+0.03, r"$\blacksquare$", color = 'red')
                figure.text(xt,yt+0.001, r"$\blacksquare$", color = 'green')
                figure.savefig(f"{source_path}/corner{output_file}.png")
                figure.savefig(f"{data_path}corner.png")


        fig,axes = plt.subplots(1,len(labels),figsize = (12,4), sharey = True)
        for i,ax in enumerate(axes):
            ax.hist(chain[:,i], bins = 100, log = True, histtype = 'step', lw = 2, edgecolor = 'blue')
            ax.grid(True)
            ax.set(xlabel = labels_text[i], ylabel = "counts N")
            ax.axvline(param[i], color = 'green')
            ax.axvline(param[i] - lower[i], color = 'green', ls = '--')
            ax.axvline(param[i] + upper[i], color = 'green', ls = '--')
            ax.axvline(params_min_chi2[i], color = 'red', ls = '--')
        fig.savefig(f"{source_path}/histograms.png")
        if make_copy:
            os.remove(samples_file)
        plt.close('all')

        save_params = args.save
        aims = [None, None, None, None, 2.5]
        aim_bools = [False, False, False, False, True]
        if save_params:
            import pandas as pd
            used_priors = [p[1] for p in prior_parameters]
            prior_limits = [np.array(p[-1].split('|')[-2:]).astype(float) for p in prior_parameters]
            errors = [(lower[i], upper[i]) for i in range(len(upper))]
            param_dict = {"used priors": used_priors, "prior limits": prior_limits, "mean params": param , "errors": errors, "min chi2 params": params_min_chi2}
            df = pd.DataFrame(param_dict)
            df.to_csv(f"{source_path}/parameters{output_file}.csv", index=False)
            np.save("chain.npy",chain)
            fig_properties = {'figsize': (16, 6)}
            hist_properties = {'color': 'blue', 'histtype': "step"}
            scatter_kwargs = {'color':'yellow', 'edgecolors': 'black', 's' : 100, "marker" : "*"}
            hist_colors = ["red","blue","green","orange","yellow","cyan"]
            multimodal = []
            for i in range(len(labels)):
                name = labels[i]
                filename = f"{source_path}/{name}.png"
                means, lowers, uppers = split_multimodal(chain[:,i], params_min_chi2[i], aim = aims[i], only_aim = aim_bools[i], output_path = filename , fig_kwargs=fig_properties, hist_kwargs=hist_properties, scatter_kwargs = scatter_kwargs)
                multimodal.append((means, lowers, uppers))
            R = empty_group.R
            data = np.array(empty_group.mean_profile)
            err = np.array(empty_group.error_in_mean)
            all_parameters = np.array(multimodal, dtype = 'object')[:,0]
            print(all_parameters)
            permutations = np.array(list(product(*all_parameters)))
            print(permutations)
            chi2_array = [calculate_chi2(data, func(R.value,p), err) for p in permutations]
            print(chi2_array)
            best_permutation_indx = np.argmin(chi2_array)
            best_permutation = permutations[best_permutation_indx]
            best_params_indx = [np.argmin(np.abs(all_parameters[i] - best_permutation[i])) for i in range(len(best_permutation))]
            print(best_permutation)
            best_params = np.array([multimodal[i][0][best_params_indx[i]] for i in range(len(multimodal))])
            lower_errors  = np.array([multimodal[i][1][best_params_indx[i]] for i in range(len(multimodal))])
            upper_errors = np.array([multimodal[i][2][best_params_indx[i]] for i in range(len(multimodal))])
            print(best_params)
            fig,ax = plt.subplots(figsize=(8,6))
            ax.errorbar(R,data,yerr=err,capsize=3,color='black',fmt='o',label='data')
            fit = func(R.value,best_params)
            lower_bound = func(R.value,np.array(best_params) - np.array(lower_errors))
            upper_bound = func(R.value,np.array(best_params) + np.array(upper_errors))
            ax.plot(R.value, fit, color = 'red', lw = 1, label = 'estimate via multimodal algorithm')
            ax.fill_between(R.value, lower_bound, upper_bound, color = 'red', label = "$1\\sigma$", alpha = 0.3, edgecolor = "darkred")
            ax.set(yscale = 'log', xlabel = "R (arcmin)", ylabel = "$y$", title = "best fitting with multimodal posteriors", ylim = (1e-11, 1e-4))
            ax.grid(True)
            fig.savefig(f"{source_path}/multimodal.png")
        output =  [param, lower, upper, richness, richness_err, redshift, redshift_err]
        if return_cluster:
            output.append(empty_group)
        return output
if __name__ == "__main__":
    main()



