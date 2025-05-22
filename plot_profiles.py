from multiprocessing import Pool, cpu_count
import emcee
import astropy.units as u
import os
import corner
import re as rec
import matplotlib.pyplot as plt
from tqdm import tqdm
from configparser import ConfigParser
from helpers import *
import profiles
import importlib
import warnings
import matplotlib.colors as mcolors
import emcee
import shutil
import time
from cluster_data import *
from matplotlib.lines import Line2D
import sys
from scipy.stats import pearsonr
from plottery.plotutils import colorscale
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse
from config import *
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib import cm


def sigma_to_percentiles(sigma):
    lower_percentile = stats.norm.cdf(-sigma)*100
    upper_percentile = stats.norm.cdf(sigma)*100
    return [lower_percentile, upper_percentile]

from plottery.plotutils import update_rcParams
update_rcParams()

def calculate_chi2(y,y1,sigma):
    if len(np.shape(sigma)) == 1:
        y,y1,sigma = np.array(y),np.array(y1),np.array(sigma)
        return np.abs(np.sum((y - y1)**2 / sigma **2))
    else:
        cov_inv = sigma
        residual = np.array(y - y1)
        chi2 = np.dot(residual, np.dot(cov_inv, residual))
        return chi2

def BIC(n,k,lnL):
    return k*np.log(n) - 2*lnL

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-f', type = str, help = "Name of path where is the samples file.")
parser.add_argument('--plot', '-p', action = 'store_true', help = "Plot best fitting and show parameters.")
parser.add_argument('--all_data', '-a', action = "store_true", help = "Fit all the available cluster data in the path.")
parser.add_argument('--general', '-g', action = "store_true", help = "Fit all the available clusrer data but given a general model.")
parser.add_argument('--cov_matrix', '-CM', action = "store_true", help = "Compute covariance matrix of parameters.")
parser.add_argument('--corr_matrix', '-CRM', action = "store_true", help = "Compute correlation matrix of parameters.")
parser.add_argument('--discard', '-d', default = 0, help = """
                    Discarted steps. If is an iterable with 2 elements will take the first element as lower floor and the second one as an upper floor. 
                    If is an iterable with 3 elements the argument will be interprete as lower, middle and upper floot, respectivally.""")
parser.add_argument('--thin', '-t', type = int, default = 1, help = "Only select the n-th  steps.")
parser.add_argument('--tau', '-T', action = 'store_true', help = "Plot autocorrelation time.")
parser.add_argument('--steps', '-e', action = 'store_true', help = "Plot paramreters steps.")
parser.add_argument('--corner', '-c', action = 'store_true', help = "If true plot the corners.")
parser.add_argument('--save', '-s', action = 'store_true', help = "If true save the parameters and his priors.")
parser.add_argument('--make_copy', '-m', action = 'store_false', help = 'If passed is not created a copy of the sample file.')
parser.add_argument('--range_sigma_ratio', '-r', type = float, default = None, help = "If passed with a number the range of corner plot is changed to the quantiles of sigma r times.")
parser.add_argument('--fit_parameters', '-F', action='store_false', help = 'If passed the parameters distributin are not fitted')
parser.add_argument('--extract_method','-M', default = 'mean', 
                    help = """
Compute the way to compute the paramaeters.
    \n* mean: the parameters and their uncertainties are compute using 16th, 50th and 84th percentiles.
    \n* median: is used the np.median and np.std function to compute parameters and uncertainties.
    \n* min_chi2: the best combination of parameters (i.e maximum likelihood or min chi2) is used.
    \n--note: in other cases is used the mean method--
                    """
                    )
parser.add_argument("--use_obs_chi2", "-U", action = "store_true", help = "compute the PTE value using chi_obs instead of chi_min.")
parser.add_argument("--verbose","-v", action = "store_true", help = "verbose results of fit as latex table format.")
parser.add_argument("--dpi","-D",type = int, default = 600, help = "dpi (quality) of matplotlib figure.")
parser.add_argument("--redshift_bins", "-R", default = None, help = "Redshift bin to general fit.")
parser.add_argument("--dont-show_results", "-W", action = "store_true", help = "If pass don't show the results of fit in the plot.")
parser.add_argument("--share_plot", "-P", action = "store_true", help = "if pass plot all the profile in the same axe.")
parser.add_argument('--signal', '-S', action = "store_true", help = "if pass compute lower and upper bound using signal obtained from MCMC chain.")
parser.add_argument("--PRIOR_CONFIG","-C", type = str, default = "PRIORS", help = "Key in config.ini file that define the priors.")
parser.add_argument("--EMCEE_CONFIG", "-E", type = str, default = "EMCEE CONFIG", help = "Key in config.ini file that determinate parameters about emcee sampler.")
parser.add_argument("--MODEL_CONFIG", "-MC", type = str, default = "MODEL CONFIG", help = "Which key in the config.ini file have information about the physical profile model.")
parser.add_argument("--ask_to_add", "-Y", action = "store_false", help = "If passed wont be asked to add new clusters to the group and will be assumed the existence of a ignore.txt file in the path")
#parser.add_argument("--BLOBS_CONFIG", "-B", type = str, default = "BLOBS", help = "This argument specifies which key in config.ini file correspond to emcee blobs")
parser.add_argument("--CONFIG_FILE", "-CF", default = None, help = "Configuration file to extract the PRIORS, EMCEE, MODEL and BLOBS config.")
args = parser.parse_args()

verbose = args.verbose
ask_to_add = args.ask_to_add
discard = args.discard
source_path = args.path
plot = args.plot
all_data = args.all_data
general = args.general

if args.CONFIG_FILE is None:
    thin = args.thin
    priors_config = args.PRIOR_CONFIG
    model_config = args.MODEL_CONFIG
    emcee_config = args.EMCEE_CONFIG

    prior_parameters = dict(config[priors_config])
    prior_parameters_dict = {
        key: list(prop2arr(prior_parameters[key], dtype=str))
        for key in list(prior_parameters.keys())
    }
    prior_parameters = list(prior_parameters_dict.values())
    labels = np.array(
        [
            list(prior_parameters_dict.keys())[i]
            for i in range(len(prior_parameters))
            if "free" in prior_parameters[i]
        ]
    ).astype(str)
    n_parameters = len(labels)
    profile_stacked_model = config[model_config]["profile"]
    filters = config[model_config]["filters"].split("|")
    use_filters = str2bool(config[model_config]["use_filters"])
    fil_name, ext = list(prop2arr(config[emcee_config]["output_file"], dtype=str))
    # try:
    ylabel, ylabel_unit = prop2arr(config[model_config]["y_physical_propertie"], dtype = str, remove_white_spaces = False)
    xlabel, xlabel_unit = prop2arr(config[model_config]["x_physical_propertie"], dtype = str, remove_white_spaces = False)
    xlabel_unit, ylabel_unit = xlabel_unit.replace(" ", ""), ylabel_unit.replace(" ", "")
    ylabel = ylabel + f" ({ylabel_unit})" if (ylabel_unit is not None and ylabel_unit != "None") else ylabel
    xlabel = xlabel + f" ({xlabel_unit})" if (xlabel_unit is not None and xlabel_unit != "None") else xlabel
    title = config[model_config]["model_name"]
    yscale = config[model_config["yscale"]] if "yscale" in list(config[model_config].keys()) else "log"
    xscale = config[model_config["xscale"]] if "xscale" in list(config[model_config].keys()) else "linear"

else:
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_filepath = current_path +"/"+ str(args.CONFIG_FILE)
    config = ConfigParser()
    config.optionxform = str
    if os.path.exists(config_filepath):
        config.read(config_filepath)
    else:
        raise Found_Error_Config(f"The config file {str(args.CONFIG_FILE)} doesn't exist")
    priors_config = config["PRIORS"]
    model_config = config["MODEL"]
    output_config = config["EMCEE"]
    profile_stacked_model = model_config["profile"]
    filters = model_config["filters"].split("|")
    use_filters = str2bool(model_config["use_filters"])
    fil_name, ext = list(prop2arr(config["EMCEE"]["output_file"], dtype = str))

    ylabel, ylabel_unit = prop2arr(config["MODEL"]["y"], dtype = str, remove_white_spaces = False)
    xlabel, xlabel_unit = prop2arr(config["MODEL"]["x"], dtype = str, remove_white_spaces = False)
    xlabel_unit, ylabel_unit = xlabel_unit.replace(" ", ""), ylabel_unit.replace(" ", "")
    ylabel = ylabel + f" ({ylabel_unit})" if (ylabel_unit is not None and ylabel_unit != "None") else ylabel
    xlabel = xlabel + f" ({xlabel_unit})" if (xlabel_unit is not None and xlabel_unit != "None") else xlabel

    yscale = model_config["yscale"] if "yscale" in list(model_config.keys()) else  "log"
    xscale = model_config["xscale"] if "xscale" in list(model_config.keys()) else "linear"

    title = config["MODEL"]["title"]

    #PRIORS
    prior_parameters = dict(priors_config)
    prior_parameters_dict = {
        key: list(prop2arr(prior_parameters[key], dtype=str))
        for key in list(prior_parameters.keys())
    }
    prior_parameters = list(prior_parameters_dict.values())
    labels = np.array(
        [
            list(prior_parameters_dict.keys())[i]
            for i in range(len(prior_parameters))
            if "free" in prior_parameters[i]
        ]
    ).astype(str)
    n_parameters = len(labels)


def main():

    if all_data == False and general == False:
        model = getattr(profiles_module, profile_stacked_model)
        plot_mcmc(source_path, model, labels, plot = plot, steps = args.steps, 
                    corner_ = args.corner, make_copy = args.make_copy, discard = args.discard, thin = args.thin,
                    tau = args.tau, method = args.extract_method, use_signal = args.signal, filters = filters,
                    fil_name = fil_name, ext = ext, use_filters = use_filters, model_name = title
                    )    
    elif all_data == True and general == False:
        main_path = args.path
        main_path = main_path + "/" if main_path [-1] != "/" else main_path 
        ignore = np.loadtxt(main_path + "ignore.txt", dtype = str).T if os.path.exists(main_path + "ignore.txt") else []
        available_paths = [path for path in os.listdir(main_path) if (os.path.isdir(main_path + path) and (path in ignore) == False)]
        redshift, richness = [],[]
        redshift_err, richness_err = [], []
        parameters, lower_errors, upper_errors = [], [], []
        min_chi2 = []
        redshift_bin = []
        richness_bin = []
        chi2 = []
        BIC = []
        signals = []
        labels_latex = [text2latex(l) for l in labels]
        model = getattr(profiles_module, profile_stacked_model)
        clusters = []
        for i,path in enumerate(available_paths):
            print(30*"=","\n")
            print(f"loading parameters from  \033[92m{path.split('/')[-1]}\033[0m.\n")
            #try:
            cluster = grouped_clusters.load_from_path(main_path + available_paths[i])
            clusters.append(cluster)
            p, le, ue, r, re, z, ze, mc2, bic, s = plot_mcmc(main_path + available_paths[i], model, labels, plot = plot, steps = args.steps, 
                    corner_ = args.corner, make_copy = args.make_copy, discard = args.discard, thin = args.thin, use_signal = True,
                    tau = args.tau, method = args.extract_method, fil_name = fil_name, ext = ext,
                    xlabel = xlabel, ylabel = ylabel, model_name = model, plot_cov = args.cov_matrix, plot_corr = args.corr_matrix
                    )
            signals.append(s)
            redshift_bin.append([np.min(cluster.z), np.max(cluster.z)])
            richness_bin.append([int(np.min(cluster.richness)), int(np.max(cluster.richness))])
            current_richness = cluster.richness
            current_richness_err = cluster.richness_err
            current_redshift = cluster.z
            current_redshift_err = cluster.z
            mean_richness = np.average(current_richness, weights = 1/current_richness_err**2)
            mean_redshift = np.average(current_redshift, weights = 1/current_redshift_err**2)
            mean_richness_err = np.sqrt(np.sum(1/current_richness_err**2))
            mean_redshift_err = np.sqrt(np.sum(1/current_redshift_err**2))
            richness.append(mean_richness)
            richness_err.append(mean_richness_err)
            redshift.append(mean_redshift)
            redshift_err.append(mean_redshift_err)
            chi2.append(mc2)
            BIC.append(bic)
            parameters.append(p), lower_errors.append(le), upper_errors.append(ue)
            min_chi2.append(mc2)
            #except Exception as e:
            #    print(f"{path} return exception {e}")
            print("")
        
        redshift = np.array(redshift).reshape(len(richness))
        parameters = np.array(parameters)
        lower_errors = np.array(lower_errors)
        upper_errors = np.array(upper_errors)
        fig, axes = plt.subplots(np.shape(parameters)[1],1, figsize = (20,3*len(parameters)), sharex = True)
        N_params = np.shape(parameters)[1]
        norm = Normalize(vmin=min(redshift), vmax = 0.8)
        colormap = cm.autumn
        colors = colormap(norm(redshift))
        colors = [tuple(c) for c in colors]
        if verbose == True:
            head = r"$\lambda$ | $z$ |"
            for i in range(len(parameters[1])):
                head+=labels_latex[i]
                head+="|"
            head+=r"$\chi^2$"
            print(head + "\\")
            for i in range(len(richness)):
                line = r" $[%.i, %.i]$ & $[%.2f,%.2f]$ & " % (richness_bin[i][0],richness_bin[i][1], 
                                                          redshift_bin[i][0],redshift_bin[i][1])
                for j in range(len(parameters[i])):
                    line+=r" $%.2f^{%.2f}_{%.2f}$ &" % (parameters[i][j], upper_errors[i][j], lower_errors[i][j])
                line+= r" $%.2f$" % chi2[i]
                print(line + "\\")
        for i in range(N_params):
            ax = axes[i]
            sp = parameters[:,i]
            le,ue = lower_errors[:,i],upper_errors[:,i]
            for j in range(len(richness)):
                ax.errorbar(richness[j], sp[j], yerr =[[le[j]],[ue[j]]], fmt = "None", c = colors[j])
                ax.scatter(richness[j], sp[j], marker = "o", s = 100, color = colors[j])
            ax.plot(np.sort(richness), np.array(sp[np.argsort(richness)]), ls = '--', color = 'black', alpha = 0.3, lw = 0.5)
            ax.set_ylabel(labels_latex[i], fontsize  = 16)
        ax.set(xscale = "log")
        ax.set_xlabel(xlabel = "richness $\\lambda$", fontsize = 20)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = plt.colorbar(sm, label="redshift $z$",cax=fig.add_axes([0.92, 0.1, 0.01, 0.8]))
        cbar.ax.tick_params(labelsize=12)
        fig.savefig(main_path + "parameters.png", transparent = False)
        if args.share_plot == False:
            [c.mean(from_path = True) for c in clusters]
            profiles = np.array([c.mean_profile for c in clusters])
            pmin = np.min(profiles[profiles > 0])
            pmax = np.max(profiles)
            errors = [c.error_in_mean for c in clusters]
            num_profiles = len(clusters)
            profiles_per_row = 2
            num_rows = (num_profiles + profiles_per_row - 1) // profiles_per_row
            fig, axes = plt.subplots(num_rows, profiles_per_row, figsize=(18, 4 * num_rows), sharey = False, sharex = True)
            axes = axes.flatten()
            for i,ax in zip(range(num_profiles), axes):
                signal = signals[i]
                c = clusters[i]
                zmin, zmax = np.min(c.z), np.max(c.z)
                rmin, rmax = int(np.min(c.richness)), int(np.max(c.richness))
                default_profiles_kwargs = (
                        ("output_file", None),
                        ("ax_kwargs", dict(xlabel = xlabel, ylabel = xlabel, 
                                            title = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))),
                        ("show_legend", False),
                        ("show_results", False)
                        )  
                R = c.R
                lower_bound, upper_bound = np.percentile(signal, [16,84], axis = 0)
                fit = np.median(signal, axis = 0)
                for j in range(len(profiles[i])):
                    p = profiles[i][j]
                    if errors[i][j] > 0  and p > 0:
                        ax.errorbar(R[j], p, yerr = errors[i][j], capsize = 3, fmt = "o", color = "black")
                    else:
                        ax.errorbar(R[j], pmin*1e-1, yerr = np.abs(errors[i][j]), capsize = 3,  fmt = "x", color = "black")
                ax.plot(R, fit, color = "black", lw = 2)
                ax.fill_between(R, lower_bound, upper_bound, color = "grey", alpha = 0.4)
                if i == 0  or ((i)%profiles_per_row) == 0:
                    ax.set(xlabel = xlabel, ylabel = "y-compton", yscale = yscale, xscale = xscale,
                        title = r"$\lambda \in [%.i, %.i]\;z\in[%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))
                else:
                    ax.set(xlabel = xlabel, yscale = yscale, xscale = xscale,
                        title = r"$\lambda \in [%.i, %.i]\;z\in[%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))                  
                ax.grid(True)
            fig.tight_layout()
            for i in range(num_profiles, len(axes)):
                axes[i].axis('off')
            if num_profiles % profiles_per_row != 0:
                last_row_start = num_profiles - (num_profiles % profiles_per_row)
                last_row_profiles = num_profiles % profiles_per_row
                if (profiles_per_row % 2 != 0 and last_row_profiles % 2 != 0) or (profiles_per_row % 2 == 0 and last_row_profiles % 2 == 0):
                    for j in range(last_row_profiles):
                        idx = last_row_start + j
                        ax = axes[idx]
                        pos = ax.get_position()
                        new_x0 = pos.x0 + 1.2*pos.width
                        new_pos = [new_x0, pos.y0, pos.width, pos.height]
                        axes[idx].set_position(new_pos)
                elif (profiles_per_row % 2 != 0 and last_row_profiles % 2 == 0) or (profiles_per_row % 2 == 0 and last_row_profiles % 2 != 0):
                    for j in range(last_row_profiles):
                        idx = last_row_start + j
                        ax = axes[idx]
                        pos = ax.get_position()
                        new_x0 = pos.x0 + 0.6*pos.width
                        new_pos = [new_x0, pos.y0, pos.width, pos.height]
                        axes[idx].set_position(new_pos)
                fig.savefig(main_path + "shared_plot_profiles.png", transparent = False)
    elif general == True and all_data == False:
        main_path = args.path
        source_file = main_path + "general_fit_" + fil_name + "." + ext
        rbins, zbins, Mbins = model_config["rbins"], model_config["zbins"], model_config["Mbins"]
        rbins = int(rbins)
        zbins = int(zbins)
        Mbins = int(Mbins)
        plot_general_mcmc(main_path, source_file, profile_stacked_model, labels, plot = args.plot, steps = args.steps, 
                    corner_ = args.corner, make_copy = args.make_copy, discard = args.discard, thin = args.thin,
                    tau = args.tau, use_signal = args.signal, method = args.extract_method, share_plot = args.share_plot,
                    output_path = main_path, rbins = rbins, zbins = zbins, Mbins = Mbins, model_name = title,
                    xlabel = xlabel, ylabel = ylabel, plot_cov = args.cov_matrix, plot_corr = args.corr_matrix
                    )
    else:
        print("xd")

def plot_general_mcmc(main_path, source_file, model, labels, ndims = None, nwalkers = None,
                       plot = False, discard = 0, steps = False, corner_ = False, make_copy = True, 
                       method = "median", tau = False, thin = 1, output_path = './', use_signal = False, 
                       rbins = None, Mbins = None, zbins = None, share_plot = False, xlabel = None,
                       ylabel = None, model_name = "", plot_cov = False, plot_corr = False, **kwargs):
    ndims = len(labels) if ndims is None else ndims
    labels_latex = [text2latex(l) for l in labels]
    list_of_clusters = []
    main_path = main_path + "/" if main_path [-1] != "/" else main_path 
    ignore = np.loadtxt(main_path + "ignore.txt", dtype = str).T if os.path.exists(main_path + "ignore.txt") else []
    available_paths = [path for path in os.listdir(main_path) if os.path.isdir(main_path + path)]
    richness_bins = []
    paths = []
    for path in available_paths:
        if path in ignore:
            continue
        current_path = main_path + path
        try:
            clusters = grouped_clusters.load_from_path(current_path)
            clusters.mean(from_path = True)
            if is_running_via_nohup() == False and ask_to_add == True:
                add_cluster = input(f"Do you want to add the next cluster? (Y, yes or enter to add it):\n {clusters}from: \033[35m{path}\033[0m\n").strip().lower()
                if add_cluster in ["y","yes",""]:
                    list_of_clusters.append(clusters)
                    paths.append(clusters.output_path)
                    richness_bins.append(np.round(np.min(clusters.richness)))
                else:
                    continue
            else:
                list_of_clusters.append(clusters)
                paths.append(clusters.output_path)
                richness_bins.append(np.round(np.min(clusters.richness)))
        except Exception as e:
            print(f"The next exception occurred trying to load {current_path}: \033[31m{e}\033[0m")
            continue
    all_clusters = list_of_clusters[0]
    for i in range(1,len(list_of_clusters)):
        all_clusters+=list_of_clusters[i]
    clusters = all_clusters
    richness_bins.append(int(np.max(clusters.richness)))
    richness_bins = np.unique(richness_bins)
    richness_bins = np.load(main_path + "intervals.npy") if os.path.exists(main_path + "intervals.npy") else richness_bins
    if args.redshift_bins is not None:
        try:
            redshift_bins = np.tile(prop2arr(args.redshift_bins, dtype = float),len(richness_bins)).reshape((len(richness_bins),2))
        except:
            if type(args.redshift_bins) == str:
                redshift_bins = []
                with open(args.redshift_bins) as f:
                    for l in f.readlines():
                        ll = l.strip().replace('[','').replace(']','').split(',')
                        if len(ll) == 2:
                            rl,rb = float(ll[0]),float(ll[1])
                            redshift_bins.append([rl,rb])
                        else:
                            rl, rb = [float(ll[0]),float(ll[1])],[float(ll[2]),float(ll[3])]
                            redshift_bins.append([rl[0], rl[1], rb[1]])
    else:
        redshift_bins = prop2arr(args.redshift_bins, dtype = np.float64) if args.redshift_bins is not None else np.tile([np.min(clusters.z),np.max(clusters.z)],len(richness_bins)).reshape((len(richness_bins),2))

    func,cov_matrix, about_clusters, clusters, raw_profiles, funcs = clusters.stacked_halo_model_func_by_bins(getattr(profiles_module, model),
                                                zb = redshift_bins, rb = richness_bins, full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths)
    N_clusters = len(clusters)
    R = list_of_clusters[0].R
    profiles = np.array(raw_profiles).reshape((N_clusters, len(R)))
    errors = np.sqrt(np.diag(cov_matrix)).reshape((N_clusters, len(R)))
    if make_copy:
        copy = f"{source_file.split('.')[0]}_copy.h5"
        shutil.copy(source_file, copy)
        source_file = copy
    backend = emcee.backends.HDFBackend(source_file, read_only = True)
    try:
        autocorr_time = backend.get_autocorr_time()
        max_tau = int(2*np.max(autocorr_time))
        print(f"The chain converged at {max_tau} steps using get_autorr_time function.")
    except:
        print("\033[91mThe chain is not fully converged yet!\033[0m\n")
    discard = int(discard) if len(str(discard).split(',')) == 1 else np.array(discard.split(','), dtype = int)
    if np.iterable(discard) == False:
        blobs = backend.get_blobs()
        chi2_values = blobs['CHI2'].flatten()
        signal = blobs['SIGNAL'].reshape((len(chi2_values),-1))
        chi2_values = chi2_values[np.where((chi2_values < np.inf) & (chi2_values > -np.inf))]
        signal2bound = np.reshape(signal,(-1,N_clusters, len(R))) if use_signal else None
        ln_prior = blobs['LN_PRIOR'].flatten()
        ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
        max_ln_likelihood = np.max(ln_likelihood)
        chain = backend.get_chain(discard = discard, thin = thin, flat = True)
        params, lower, upper = extract_params(chain, labels, method = method)
        if verbose == True:
            print("Parameter & value")
            for j in range(len(params)):
                print(labels_latex[j] + r"& $%.2f_{%.2f}^{%.2f}$ \\" % (params[j],lower[j],upper[j]))

        if steps:     
            default_steps_kwargs = (
                ("labels", labels_latex),
                ("plot_tracers", True),
                ("output_file", f"{output_path}/parameters_steps.png"),
                ("tracers_aspect", ('k-','k--','k--'))
            )
            steps_kwargs = set_default(kwargs.pop("steps_kwargs",{}), default_steps_kwargs)
            plot_steps(chain, backend, nwalkers, chi2_values,**steps_kwargs)
        if corner_:
            default_corner_kwargs = (
                ("truths", params),
                ("truths_color","black"),
                ("corner_color", "grey"),
                ("output_file", f"{output_path}/corner.png"),
                ("labels",labels_latex),
                ("fontsize", 16),
                ("range_sigma_ratio", float(args.range_sigma_ratio)),
                ("title_kwargs", {"fontsize":12})
            )
            corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
            plot_corner(chain, **corner_kwargs)
        if tau:
            default_tau_kwargs = (
                ("show_convergence", True),
                ("output_file", f"{output_path}/tau.png")
            )
            tau_kwargs = set_default(kwargs.pop("tau_kwargs", {}), default_tau_kwargs)
            plot_tau(backend, labels_latex, **tau_kwargs)
        if plot_cov == True or plot_corr == True:
            default_cov_kwargs = (
                ("output_file", f"{output_path}/cov_params.png"),
            )
            cov_kwargs = set_default(kwargs.pop("cov_kwargs", {}), default_cov_kwargs)   
            plot_cov_matrix(chain, labels_latex, corr = plot_corr, **cov_kwargs)       
        if plot == True:
            
            fit = func(R,params).reshape((N_clusters, len(R)))

            upper_params = np.array(params) + np.array(upper)
            lower_params = np.array(params) - np.array(lower)
            lower_bound = func(R,lower_params).reshape((N_clusters, len(R)))
            upper_bound = func(R,upper_params).reshape((N_clusters, len(R)))

            num_profiles = len(fit)
            if share_plot == False:
                profiles_per_row = 3
                num_rows = (num_profiles + profiles_per_row - 1) // profiles_per_row
                fig, axes = plt.subplots(num_rows, profiles_per_row, figsize=(20, 8 * num_rows), sharey = True, sharex = True)
                axes = axes.flatten()
                for i,ax in zip(range(num_profiles), axes):
                    c = clusters[i]
                    zmin, zmax = np.min(c.z), np.max(c.z)
                    rmin, rmax = int(np.min(c.richness)), int(np.max(c.richness))

                    default_profiles_kwargs = (
                            ("output_file", None),
                            ("ax_kwargs", dict(xlabel = xlabel, ylabel = xlabel, 
                                                title = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))),
                            ("show_legend", False),
                            ("show_results", False)
                            )  
                    profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)                
                    plot_profiles(R, profiles[i], func, params, np.array(errors[i])**2, labels_latex, lower, upper,
                                np.max(ln_likelihood), np.mean(all_clusters.z), ax = ax, fit = fit[i], lower_bound = lower_bound[i],
                                upper_bound = upper_bound[i], signal = signal2bound[:,i,:], **profiles_kwargs)
                    ax.grid(True)
                for i in range(num_profiles, len(axes)):
                    axes[i].axis('off')
                if num_profiles % profiles_per_row != 0:
                    last_row_start = num_profiles - (num_profiles % profiles_per_row)
                    last_row_profiles = num_profiles % profiles_per_row
                    if (profiles_per_row % 2 != 0 and last_row_profiles % 2 != 0) or (profiles_per_row % 2 == 0 and last_row_profiles % 2 == 0):
                        for j in range(last_row_profiles):
                            idx = last_row_start + j
                            ax = axes[idx]
                            pos = ax.get_position()
                            new_x0 = pos.x0 + 1.2*pos.width
                            new_pos = [new_x0, pos.y0, pos.width, pos.height]
                            axes[idx].set_position(new_pos)
                    elif (profiles_per_row % 2 != 0 and last_row_profiles % 2 == 0) or (profiles_per_row % 2 == 0 and last_row_profiles % 2 != 0):
                        for j in range(last_row_profiles):
                            idx = last_row_start + j
                            ax = axes[idx]
                            pos = ax.get_position()
                            new_x0 = pos.x0 + 0.6*pos.width
                            new_pos = [new_x0, pos.y0, pos.width, pos.height]
                            axes[idx].set_position(new_pos)
            elif share_plot == True:
                fig, ax = plt.subplots(figsize = (14,8))
                colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), size  = num_profiles)
                colors = ["darkgreen", "purple","darkblue","darkseagreen","darkred","coral","brown","orange","cyan"]
                profile_labels = []
                show_redshift = True
                for i in range(num_profiles):
                    color = colors[i]
                    c = clusters[i]
                    zmin, zmax = np.min(c.z), np.max(c.z)
                    rmin, rmax = int(np.min(c.richness)), int(np.max(c.richness))
                    label = r"$\lambda = [%.i , %.i] \;,\; z = [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax) if show_redshift==True \
                            else r"$\lambda = [%.i$ , %.i]$" % (rmin, rmax) 

                    default_profiles_kwargs = (
                            ("output_file", None),
                            ("show_legend", False),
                            ("fit_plot_kwargs", {"color": color, "label" : None}),
                            ("bounds_plot_kwargs", {"color": color, "alpha": 0.2, "label" : None}),
                            ("data_plot_kwargs", {"color": color,"label": label}),
                            ("show_results", False),
                            ("ax_kwargs", dict(xlabel = xlabel, ylabel = ylabel, yscale = yscale, xscale = xscale))
                        ) 
                    profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)    
                    signal2bound_i = signal2bound[:,i,:] if signal2bound is not None else None
                    plot_profiles(R + i*0.1, profiles[i], func, params, c.cov, labels_latex, lower, upper,
                                np.max(ln_likelihood), np.mean(c.z), ax = ax, fit = fit[i], lower_bound = lower_bound[i],
                                upper_bound = upper_bound[i], signal = signal2bound_i, show_labels = False, show_error_bars = True, **profiles_kwargs)
                ax.legend(loc = "upper right", fontsize = 10)
                ax.set_title("Best Fitting " + model_name)
                #ax.set_ylim((np.min(np.array(profiles)[np.array(profiles) > 0])*0.1, np.max(profiles)*1.5))
                ax.grid(True)
            if args.dont_show_results == False:
                chi2 = np.min(chi2_values)#calculate_chi2(raw_profiles, func(R.value,params), np.linalg.inv(cov_matrix))
                if args.use_obs_chi2 == True:
                    observed_profiles = np.median(signal, axis = 0)
                    res = raw_profiles - observed_profiles
                    chi2_obs = np.dot(np.dot(res, np.linalg.inv(cov_matrix)), res.T)
                    p_value = pte(chi2_obs, cov_matrix)
                else:
                    p_value = pte(chi2, cov_matrix)
                bic = BIC(np.size(raw_profiles), len(params), max_ln_likelihood)
                text  = [
                        r'$\chi^{2} = %.2f$' % chi2,
                        r'$PTE = %.2f$' % p_value,
                ]
                for i in range(len(labels)):
                    if labels[i].split('_')[0] == r'$\log':
                        text.append(f'{labels_latex[i]} : {np.round(np.log10(params[i]),2)} $\pm$ {np.round(err[i]/(np.log(10) * paramss[i]),2)}')
                    else:
                        text.append('%s' % labels_latex[i] + ': $%.2f' % params[i] + '^{+%.2f}_{-%.2f}$' % (np.abs(params[i] - upper_params[i]),np.abs(params[i] - lower_params[i])))   
                s = '\n'.join(text)
                if share_plot:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    #0.15, 0.6
                    ax.text(0.17, 0.1, s, fontsize=11, verticalalignment='bottom', ha = "right", transform=ax.transAxes, bbox=props, color = 'black')
                else:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    fig.text(0.7, 0.95, s, fontsize=11, verticalalignment='top', ha = "right", bbox=props, color = 'black')       
            fig.tight_layout()           
            fig.savefig(f"{output_path}/best_fitting.png", dpi = args.dpi, transparent = False)   
    elif np.iterable(discard) == True:
        blobs = backend.get_blobs()
        chain = backend.get_chain()
        corner_fig = plt.figure(figsize = (12,12))
        for i in range(len(discard) - 1):
            d1,d2 = discard[i],discard[i + 1]
            schain  = chain[d1:d2,:,:].reshape((-1, ndims))
            chi2_values = blobs['CHI2'][int(d1):int(d2),:].flatten()
            signal = blobs['SIGNAL'][int(d1):int(d2),:].reshape((len(chi2_values),-1))
            ln_prior = blobs['LN_PRIOR'][int(d1):int(d2),:].flatten()
            ln_likelihood = blobs['LN_LIKELIHOOD'][int(d1):int(d2),:].flatten()  
            params, lower, upper = extract_params(schain, labels, method = method)              
            if corner_:
                corner_color = ["darkgreen", "purple","darkblue","darkseagreen","darkred"][i]
                default_corner_kwargs = (
                    ("corner_color", corner_color),
                    ("labels",labels_latex),
                    ("fontsize", 6),
                    ("fig", corner_fig),
                    ("range_sigma_ratio", float(args.range_sigma_ratio)),
                    ("title_kwargs", {"fontsize":7})
                )
                corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
                axes, corner_fig = plot_corner(schain, **corner_kwargs)
                axes[-1][-1].scatter([],[], color = corner_color, marker = "s", label = f"N steps $=[{d1},{d2}]$")
        if corner_:
            corner_fig.legend(fontsize = 16)
            corner_fig.savefig(f"{output_path}/corner.png", dpi = args.dpi)   
    if make_copy == True:
        os.remove(copy)
def plot_mcmc(source_path, model, labels, ndims = None, nwalkers = None, fil_name = 'mcmc_samples', ext = 'h5', fig_corner = None, fig_profile = None,
              return_cluster = False, plot = False, use_signal = False, discard = 0, steps = False, corner_ = False, make_copy = True,
              method = "median", tau = False, thin = 1, filters = None, use_filters = False, rbins = None, zbins = None, Mbins = None,
              xlabel = None, ylabel = None, model_name = "", return_signal = True, plot_cov = False, plot_corr = False, **kwargs):
    if use_filters == True and filters is not None:
        filters_dict = {}
        for f in filters:
            f = np.array(f.replace('(','').replace(')','').split(','))
            f = [k for k in f if len(k)>1]
            func_name = f[0]
            if len(f) > 1:
                func_params = {k.split(':')[0]:k.split(':')[1] for k in f[1::]}
            else:
                func_params = {}
        filters_dict[func_name] = func_params
    else:
        filters_dict = None
    labels_latex = [text2latex(l) for l in labels]
    samples_file = source_path + '/' + fil_name + '.' + ext
    group = grouped_clusters.empty()
    group.output_path = source_path
    group.load_from_h5()
    group.mean(from_path = True)
    R = group.R * u.arcmin 
    output_file = f"_RICHNESS={np.round(np.min(group.richness))}-{np.round(np.max(group.richness))}" + f"REDSHIFT={np.round(np.min(group.z),2)}-{np.round(np.max(group.z),2)}"
    if rbins is None or zbins is None or Mbins is None:
        rbins, zbins, Mbins = 40, 20, 20
    rbins = int(rbins)
    zbins = int(zbins)
    Mbins = int(Mbins)
    func = group.stacked_halo_model_func(model, rbins = rbins, zbins = zbins, Mbins = Mbins,
                                        use_filters = use_filters, filters = filters_dict  
    )
    richness, richness_err = np.mean(group.richness),np.std(group.richness)/np.sqrt(len(group))
    redshift, redshift_err = np.mean(group.z),np.std(group.z)/np.sqrt(len(group))
    if os.path.exists(samples_file):
        if make_copy:
            copy = f"{source_path}/{fil_name}_copy.{ext}"
            shutil.copy(samples_file, copy)
            samples_file = copy           
    if ndims is None:
        ndims = len(labels)
    if nwalkers is None:
        nwalkers = int(config["EMCEE"]["nwalkers"])
    backend = emcee.backends.HDFBackend(samples_file, read_only = True)
    try:
        autocorr_time = backend.get_autocorr_time()
        max_tau = int(2*np.max(autocorr_time))
        print(f"The chain converged at {max_tau} steps using get_autorr_time function.")
    except:
        print("\033[91mThe chain is not fully converged yet!\033[0m\n")
    if np.iterable(discard) == False:
        blobs = backend.get_blobs(discard = discard, thin = thin)
        chi2_values = blobs['CHI2'].flatten()
        chi2 = np.min(chi2_values)
        signal = blobs['SIGNAL'].reshape((len(chi2_values),-1))
        signal2bound = np.reshape(signal,(-1,len(R))) if use_signal else None
        ln_prior = blobs['LN_PRIOR'].flatten()
        ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
        chain = backend.get_chain(discard = discard, thin = thin, flat = True)
        params, lower, upper = extract_params(chain, labels, method = method)
        
        upper_params = np.array(params) + np.array(upper)
        lower_params = np.array(params) - np.array(lower)
        
        upper_bound = func(R,upper_params)
        lower_bound = func(R,lower_params)
        
        bic = BIC(len(group.R), len(params), np.max(ln_likelihood))

        if steps:     
            default_steps_kwargs = (
                ("labels", labels_latex),
                ("plot_tracers", True),
                ("output_file", f"{source_path}/parameters_steps.png"),
                ("tracers_aspect", ('k-','k--','k--'))
            )
            steps_kwargs = set_default(kwargs.pop("steps_kwargs",{}), default_steps_kwargs)
            plot_steps(chain, backend, nwalkers, chi2_values,**steps_kwargs)
        
        if plot:
            extra_text = [
                r'N clusters = $%.i$' % (len(group.richness)),
                r'$\mathrm{richness} = [%.i,%.i]$' %   (np.min(group.richness), np.max(group.richness)),
                r'$\mathrm{redshift} = [%.2f, %.2f]$' % (np.min(group.z),np.max(group.z))
            ]
            color = "black"
            default_profiles_kwargs= (("output_file", None),
                ("show_legend", False),
                ("fit_plot_kwargs", {"color": color, "label" : None}),
                ("bounds_plot_kwargs", {"color": color, "alpha": 0.2, "label" : None}),
                ("data_plot_kwargs", {"color": color}),
                ("show_results", True),
                ("ax_kwargs", dict(xlabel = xlabel, ylabel = ylabel, yscale = yscale, xscale = xscale, 
                    title = r"$\lambda \in [%.i, %.i]\;,\; z\in[%.2f, %.2f]$" % (np.min(group.richness), np.max(group.richness),
                    np.min(group.z), np.max(group.z))))
            )
            profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)
            ax,fig, chi2, bic = plot_profiles(group.R, group.mean_profile, func, params, group.cov, labels_latex, lower, upper, 
                          np.max(ln_likelihood), np.mean(group.z), min_chi2 = chi2, lower_bounrd = lower_bound, upper_bound = upper_bound, signal = signal2bound, **profiles_kwargs)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(f"{source_path}/best_fitting.png", dpi = args.dpi)   
        if corner_:
            default_corner_kwargs = (
                ("truths", params),
                ("truths_color","black"),
                ("corner_color", "grey"),
                ("output_file", f"{source_path}/corner.png"),
                ("labels",labels_latex),
                ("fontsize", 16),
                ("range_sigma_ratio", float(args.range_sigma_ratio)),
                ("title_kwargs", {"fontsize":12})
            )
            corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
            plot_corner(chain, **corner_kwargs)
        if tau:
            default_tau_kwargs = (
                ("show_convergence", True),
                ("output_file", f"{source_path}/tau{output_file}.png")
            )
            tau_kwargs = set_default(kwargs.pop("tau_kwargs", {}), default_tau_kwargs)
            plot_tau(backend, labels_latex, **tau_kwargs)
        output =  [params, lower, upper, richness, richness_err, redshift, redshift_err, chi2, bic]
        if make_copy == True:
            os.remove(copy)
        if return_cluster:
            output.append(group)
        if return_signal:
            output.append(signal2bound)
        return output

def plot_corner(chain, fig = None, truths = None, truths_color = "black", truths_label = None, output_file = None, 
                corner_label = None, corner_color = "grey", other = None, other_label = None, 
                other_color = None, levels = (1-np.exp(-0.5), 1-np.exp(-2) ), bins = 40,  fontsize = 14,
                labels = None, show_labels = False, quantiles = [0.16, 0.5, 0.84] , alpha = 0.5,
                range_sigma_ratio = 4, **kwargs):
    default_title_kwargs = (
        ("fontsize", 20),
    )
    default_fig_kwargs = (
        ("figsize", (2*len(truths),2*len(truths))),
    )
    default_bins_kwargs = (
        ('bins',40),
    )
    default_ticks_kwargs = (
        ('max_label_size', 14),
        ('min_label_size',10),
        ('max_bins', 4)
    )
    default_legend_kwargs = (
        ('fontsize', 20),
        ('loc', 'upper right'),
    )
    ticks_kwargs = set_default(kwargs.pop("default_ticks_kwargs",{}), default_ticks_kwargs)
    title_kwargs = set_default(kwargs.pop("title_kwargs",{}), default_title_kwargs)
    fig_kwargs = set_default(kwargs.pop("fig_kwargs", {}), default_fig_kwargs)
    bins_kwargs = set_default(kwargs.pop("bins_kwargs",{}), default_bins_kwargs)
    legend_kwargs = set_default(kwargs.pop("legend_kwargs",{}), default_legend_kwargs)
    ndims = np.shape(chain)[1]
    plt.rcParams['axes.labelpad'] = 20
    if fig is None:
        fig = plt.figure(**fig_kwargs)
    fill_contour_kwargs = {'colors': [f'dark{corner_color}', f'light{corner_color}'], 'alpha': 0.1}
    corner_plot = corner.corner(
        chain,
        fig = fig,
        labels = labels,
        quantiles = quantiles,
        show_titles = True,
        levels = levels,
        truths = truths,
        truth_color = truths_color,
        plot_density = True,
        no_fill_contours = True,
        plot_datapoints = False,
        bins = bins,
        smooth = 3,
        smooth1d = 5.0,
        fill_contours = True,
        #hist_bin_factor= 3,
        #n_max_ticks = 6,
        color = corner_color,
        title_kwargs = title_kwargs,
        fill_contour_kwargs = fill_contour_kwargs,
        bins_kwargs = bins_kwargs
    )
    axes = fig.get_axes()
    axes = np.array(axes).reshape((ndims, ndims))
    if other is not None:
        for i in range(ndims):
            for j in range(i + 1):
                ax = axes[i][j]
                if i == j:
                    ax.axvline(other[i], color = other_color)
                else:
                    ax.axvline(other[j], color = other_color)
                    ax.axhline(other[i], color = other_color)
                    ax.scatter(other[j],other[i], color = other_color, marker = 's')
    if range_sigma_ratio is not None:
        for i in range(ndims):
            for j in range(i + 1):
                pminx, pmedianx, pmaxx = calculate_sigma_intervals(chain[:,j], sigma = range_sigma_ratio)
                pminy, pmediany, pmaxy = calculate_sigma_intervals(chain[:,i], sigma = range_sigma_ratio)
                if i != j:
                    axes[i][j].set_ylim((pmediany - pminy, pmediany + pmaxy))
                    axes[i][j].set_xlim((pmedianx - pminx, pmedianx + pmaxx))
                elif i == j:
                    axes[i][j].set_xlim((pmedianx - pminx, pmedianx + pmaxx))                    
    ax = fig.get_axes()[0]
    ax.scatter([],[], color = corner_color, label = corner_label, marker = 's')
    ax.scatter([],[], color = truths_color, label = truths_label, marker = 's')
    if other is not None and other_label is not None:
        ax.scatter([],[], color = other_color, label =  other_label, marker = 's')
    if show_labels == True:
        fig.legend(**legend_kwargs)
    for i in range(ndims):
        for j in range(i + 1):
            change_ticks(axes[i][j],**ticks_kwargs)
    for ax in axes.flatten():
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize) 
    if output_file is not None:
        fig.savefig(output_file , dpi = args.dpi, transparent = False)   
    return axes, fig


def plot_steps(chain, backend, nwalkers, chi2_values, labels, plot_tracers = False, tracers_aspect = 'k-', 
               output_file = None, cmap = 'viridis', **kwargs):
    default_fig_kwargs = (
        ("figsize",(5,15)),
        ("sharex", True),
        ("constrained_layout", True),
    )
    default_colorbar_kwargs = (
        ("label",r"Log $\chi^2$"),
        )
    default_title_kwargs = (
        ("t", "MCMC steps plot"),
        ("fontsize", 18),
    )
    print(nwalkers)
    nwalkers = 100
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)
    colorbar_kwargs = set_default(kwargs.pop("colorbar_kwargs", {}),default_colorbar_kwargs)
    title_kwargs = set_default(kwargs.pop("title_kwargs", {}),default_title_kwargs)

    tracers_aspect = np.tile(tracers_aspect, 3) if np.iterable(tracers_aspect) == False else tracers_aspect
    
    tsteps, npar = chain.shape
    stepbin = nwalkers
    print(nwalkers)
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
    ndims = len(labels)
    plot_steps = args.steps
    cmap_samples = 20000
    chi2_values = np.log(chi2_values)
    j = np.arange(tsteps)
    colors, cmap = colorscale(
    chi2_values,
    vmin = chi2_values[-cmap_samples:].min(),
    vmax = np.percentile(chi2_values[-cmap_samples:], 99),
    cmap = cmap,
    )
    fig, axes = plt.subplots(
        ndims, 1,**fig_kwargs
    )
    for ax, p, m, lo, hi, label in zip(axes, chain.T, runmeb, runlo, runhi, labels):
        ax.scatter(j[::1], p[::1], c=colors, marker=".", s=0.2)
        if plot_tracers == True:
            ax.plot(stepbins, m, tracers_aspect[0], lw=1.5)
            ax.plot(stepbins, lo, tracers_aspect[1], lw=1.2)
            ax.plot(stepbins, hi, tracers_aspect[2], lw=1.2)
        ax.set(ylabel=label)
    plt.colorbar(cmap, ax = axes, **colorbar_kwargs)
    axes[-1].set(xlabel="MCMC step")
    fig.suptitle(**title_kwargs)
    if plot_tracers == True:
        axes[-1].plot([],[], tracers_aspect[0], label = "median")
        axes[-1].plot([],[], tracers_aspect[1], label = "lower bound")
        axes[-1].plot([],[], tracers_aspect[2], label = "upper bound")
        fig.legend(fontsize = 20)
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi)   
    return axes, fig

def plot_profiles(R, data, model, params, cov, labels, lower, upper, max_ln_likelihood = 0 , z = 0, fig = None, ax = None,
                  output_file = None, fit = None, lower_bound = None, upper_bound = None, show_legend = False, 
                  show_results = False, signal = None, plot_bounds = True, min_chi2 = None, show_error_bars = True, **kwargs):
    default_fig_kwargs = (
        ("figsize",(8,6)),    
    )
    default_data_plot_kwargs = (
        ("capsize", 3),
        ("color", "black"),
        ("fmt", 'o'),
        ("label", "data")
    )
    default_fit_plot_kwargs = (
        ("label", "fit"),
        ("color", "black"),
        ("ls", "solid")
    )
    default_bounds_plot_kwargs = (
        ("color","grey"),
        ("alpha",0.3),
        ("label",r"$1 \sigma$")
    )
    default_text_kwargs = (
        ("fontsize", 11),
        ("verticalalignment", 'bottom'),
        ("color","black"),
        ("extra_text", None),
        ("x0", 0.025),
        ("y0", 0.05),
        ("dy", 0.065),
        ("bbox", dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8))
    )
    default_ax_kwargs = (
        ("xlabel", "R (arcmin)"),
        ("ylabel", "profile (unknown)"),
        ("yscale", "log"),
        ("title", "Best Fitting"),
    )
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)    
    data_plot_kwargs = set_default(kwargs.pop("data_plot_kwargs",{}), default_data_plot_kwargs)
    fit_plot_kwargs = set_default(kwargs.pop("fit_plot_kwargs",{}), default_fit_plot_kwargs)
    text_kwargs = set_default(kwargs.pop("text_kwargs",{}), default_text_kwargs)
    ax_kwargs = set_default(kwargs.pop("ax_kwargs",{}), default_ax_kwargs)

    R = R.value if hasattr(R,"value") == True else R
    fit = np.array(model(R, params)) if fit is None else np.array(fit)
    if len(np.shape(cov)) == 1:
        new_cov = np.zeros((len(cov),len(cov)))
        np.fill_diagonal(new_cov, cov)
        cov = new_cov
    err = np.sqrt(np.diag(cov))
    dof = len(data) - len(params)
    chi2 = calculate_chi2(data,fit,np.linalg.inv(cov)) if min_chi2 is None else min_chi2
    p_value = pte(chi2, cov)
    bic = BIC(len(data), len(params), max_ln_likelihood)
    if fig is None and ax is None:
        fig = plt.figure(**fig_kwargs)
        ax = plt.axes()
    elif fig is None and ax is not None:
        fig = ax.get_figure()
    elif fig is not None and ax is None:
        ax = plt.axes()
    if plot_bounds == True:
        if signal is None:
            lower_bound = model(R,np.array(params) - np.array(lower)) if lower_bound is None else np.array(lower_bound)
            upper_bound = model(R,np.array(params) + np.array(upper)) if upper_bound is None else np.array(upper_bound)
        elif signal is not None:
            lower_bound, upper_bound = np.percentile(signal, [16,84], axis = 0)
            fit = np.median(signal, axis = 0)
        if show_error_bars == True:
            ax.errorbar(R, data, yerr = err, **data_plot_kwargs)
        else:
            data_plot_kwargs.pop("capsize", None)
            data_plot_kwargs["marker"] = data_plot_kwargs["fmt"]
            data_plot_kwargs.pop("fmt", None)
            ax.scatter(R, data, **data_plot_kwargs)
        ax.plot(R, fit, **fit_plot_kwargs)
        bounds_plot_kwargs = set_default(kwargs.pop("bounds_plot_kwargs",{}), default_bounds_plot_kwargs)
        ax.fill_between(R, lower_bound, upper_bound, **bounds_plot_kwargs)
    if show_results == True:
        text  = [
                r'$\chi^{2} = %.4f$' % round(chi2,2),
                r'$PTE = %.4f$' % round(p_value,2)
        ]
        text = text + list(text_kwargs["extra_text"]) if text_kwargs["extra_text"] is not None else text
        for i in range(len(labels)):
            text.append('%s' % labels[i] + ': $%.2f' % params[i] + '^{+%.2f}_{-%.2f}$' % (np.abs(upper[i] - params[i]),np.abs(params[i] - lower[i])))   
        text = '\n'.join(text)
        text_kwargs.pop("extra_text",None)
        x0,y0,dy = text_kwargs["x0"],text_kwargs["y0"],text_kwargs["dy"]
        text_kwargs.pop("x0",None),text_kwargs.pop("y0",None),text_kwargs.pop("dy",None)
        if args.dont_show_results == False:
            ax.text(0.3, 0.1, text, ha = "right", transform=ax.transAxes, **text_kwargs)
    ax.set(**ax_kwargs)
    if show_legend == True:
        ax.legend(loc = "lower left")
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi, transparent = False)   
    return ax, fig, chi2, bic

def plot_cov_matrix(chain, labels, output_file = None, corr = False, norm = "linear", fig = None, 
                    compute_p_values = True, ax = None, **kwargs):
    default_fig_kwargs = (
        ("figsize",(12,12)),    
    )
    default_ax_kwargs = (
        ("xlabel", "Parameter"),
        ("ylabel", "Parameter")
    )
    default_imshow_kwargs = (
        ("cmap", "seismic"),
    )
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs) 
    ax_kwargs = set_default(kwargs.pop("ax_kwargs",{}), default_ax_kwargs)
    imshow_kwargs = set_default(kwargs.pop("imshow_kwargs",{}), default_imshow_kwargs)
    if fig is None:
        fig = plt.figure(**fig_kwargs)
    if ax is None:
        ax = plt.axes()
    cov = np.cov(chain, rowvar = False) if corr == False else np.corrcoef(chain, rowvar = False)
    if norm == "linear":
        im = ax.imshow(cov, **imshow_kwargs)
    elif norm == "log":
        im = ax.imshow(cov, norm = LogNorm, **imshow_kwargs)
    ax.set_xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    ax.set_yticks(ticks=np.arange(len(labels)), labels=labels)
    ax.tick_params(axis='both', which='both', pad=25)
    lines = np.arange(-0.5,len(labels) + 0.5,1)
    [ax.axhline(l, color = "black", lw = 3) for l in lines]
    [ax.axvline(l, color = "black", lw = 3) for l in lines]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax = cax)
    if corr == False:
        ax.set_title("Covariance matrix of parameters")
        cbar.set_label(r"Cov")
    else:
        ax.set_title("Correlation matrix of parameters")
        cbar.set_label(r"Corr")
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi)
    if compute_p_values == True:
        N = np.shape(cov)[0]
        p_value_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                p_value = bootstrap_p_value(chain[:, i], chain[:, j])
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value
        fig = plt.figure(**fig_kwargs)
        ax = plt.axes()
        im = ax.imshow(p_value_matrix, norm = LogNorm(), **imshow_kwargs)
        ax.set_xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
        ax.set_yticks(ticks=np.arange(len(labels)), labels=labels)
        ax.tick_params(axis='both', which='both', pad=25)
        lines = np.arange(-0.5,len(labels) + 0.5,1)
        [ax.axhline(l, color = "black", lw = 3) for l in lines]
        [ax.axvline(l, color = "black", lw = 3) for l in lines]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax = cax)
        ax.set_title("p-value for each parameter")
        output_file_pvalue = output_file.split(".")[0] + "_pvalues." + output_file.split(".")[1]
        fig.tight_layout()
        fig.savefig(output_file_pvalue)
def plot_tau(backend, labels, output_file = None, fig = None, ax = None, show_convergence = False, Nbins = 20, **kwargs):
    default_fig_kwargs = (
        ("figsize", (24, 2*len(labels))),
        ("sharex", True),
    )
    default_ylabel_kwargs = (
        ("fontsize", 24),
    )
    default_legend_kwargs = (
        ("fontsize", 18),
    )
    default_tau_plot_kwargs = (
        ("color", "red"),
    )
    default_N50_plot_kwargs = (
        ("color", "black"),
        ("ls", '--')
    )   
    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs) 
    ylabel_kwargs = set_default(kwargs.pop("ylabel_kwargs",{}), default_ylabel_kwargs) 
    legend_kwargs = set_default(kwargs.pop("legend_kwargs",{}), default_legend_kwargs) 
    tau_plot_kwargs = set_default(kwargs.pop("tau_plot_kwargs",{}), default_tau_plot_kwargs) 
    N50_plot_kwargs = set_default(kwargs.pop("N50_plot_kwargs",{}), default_N50_plot_kwargs) 
    suptitle_kwargs = kwargs.pop("suptitle_kwargs",{})
    xlabel_kwargs = kwargs.pop("xlabel_kwargs",{})
    
    if fig is None:
        fig,axes = plt.subplots(len(labels), **fig_kwargs)
    elif fig is None and ax is not None:
        fig = ax.get_figure()
    N_arr, tau_arr = [],[]
    for i in range(len(axes)):
        chain_unflatted = backend.get_chain()[:,:, i].T
        ax = axes[i]
        N,tau = autocorr_time_from_chain(chain_unflatted, Nbins)
        N_arr.append(N)
        tau_arr.append(tau)
        ax.loglog(N, tau, label = r"$\tau $ estimation", **tau_plot_kwargs)
        ax.loglog(N, N/50, label = r"$\tau = N/50$", **N50_plot_kwargs)
        ax.grid(True)
        ax.set_ylabel(labels[i], **ylabel_kwargs)
    if show_convergence == True:
        convergence_indx = []
        for ax,N,tau, label in zip(axes,N_arr,tau_arr,labels):
            N,tau = np.array(N),np.array(tau)
            indx = np.nan
            for i in range(len(N)):
                ratio = N[i]/50
                if tau[i] <= ratio:
                    indx = i
                    break
            if str(indx) != "nan":
                print(label, "converged at", N[indx],"steps")
                convergence_indx.append(indx)
                ax.axvline(N[indx], ls = "--", label = "parameter convergence")
            else:
                print(label, "is not converged yet.")
    axes[-1].set(xlabel = "number of samples N", **xlabel_kwargs)
    axes[-1].legend(**legend_kwargs)
    fig.suptitle(r"$\tau$ estimator for each parameter", **suptitle_kwargs)
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi, transparent = False)   
    return axes, fig


if __name__ == "__main__":
    main()



