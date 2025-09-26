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
from scipy.stats import invgauss
from matplotlib import cm
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA

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
parser.add_argument('--range_sigma_ratio', '-r', type = float, default = 2, help = "If passed with a number the range of corner plot is changed to the quantiles of sigma r times.")
parser.add_argument('--fit_parameters', '-F', action='store_false', help = 'If passed the parameters distributin are not fitted')
parser.add_argument('--extract_method','-M', default = 'median', 
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
parser.add_argument("--dpi","-D",type = int, default = 100, help = "dpi (quality) of matplotlib figure.")
parser.add_argument("--redshift_bins", "-R", default = None, help = "Redshift bin to general fit.")
parser.add_argument("--dont-show_results", "-W", action = "store_true", help = "If pass don't show the results of fit in the plot.")
parser.add_argument("--share_plot", "-P", action = "store_true", help = "if pass plot all the profile in the same axe.")
parser.add_argument('--signal', '-S', action = "store_true", help = "if pass compute lower and upper bound using signal obtained from MCMC chain.")
parser.add_argument("--PRIOR_CONFIG","-C", type = str, default = "PRIORS", help = "Key in config.ini file that define the priors.")
parser.add_argument("--EMCEE_CONFIG", "-E", type = str, default = "EMCEE CONFIG", help = "Key in config.ini file that determinate parameters about emcee sampler.")
parser.add_argument("--MODEL_CONFIG", "-MC", type = str, default = "MODEL CONFIG", help = "Which key in the config.ini file have information about the physical profile model.")
parser.add_argument("--ask_to_add", "-Y", action = "store_false", help = "If passed wont be asked to add new clusters to the group and will be assumed the existence of a ignore.txt file in the path")
#parser.add_argument("--BLOBS_CONFIG", "-B", type = str, default = "BLOBS", help = "This argument specifies which key in config.ini file correspond to emcee blobs")
parser.add_argument("--smooth_corner","-O", action = "store_true", help = "If passed the corner plot will be smoothened.")
parser.add_argument("--CONFIG_FILE", "-CF", default = None, help = "Configuration file to extract the PRIORS, EMCEE, MODEL and BLOBS config.")
parser.add_argument("--joint", "-j", action = "store_true")
parser.add_argument("--plot-chi2", "-PC", action = "store_true", help = "Plot min-chi2 vs step")
parser.add_argument("--calculate-masses", "-CMS", action = "store_true", help = "Interpretate the profiles as density mass and compute the respective masses.")
parser.add_argument("--plot_mis_centering", "-pmc", action = "store_true", help = "Plot miscentering")
parser.add_argument("--plot_hm_relationship", "-phmr", action = "store_true", help = "Plot hm relationship")
parser.add_argument("--infere-mass","-I", action = "store_true")
parser.add_argument("--show-individuals-chi2", "-SIC", action = "store_false", help = "If passed plot the chi2 of each cluster.")
args = parser.parse_args()

verbose = args.verbose
ask_to_add = args.ask_to_add
discard = args.discard
source_path = args.path
plot = args.plot
all_data = args.all_data
general = args.general
joint = args.joint

calculate_masses = args.calculate_masses

plot_mis_centering = args.plot_mis_centering
plot_hm_relationship = args.plot_hm_relationship

if args.CONFIG_FILE is None and joint == False:
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



elif args.CONFIG_FILE is not None and joint == False:
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

    params_indxs = [(0, len(prior_parameters))]


    use_mis_centering = str2bool(model_config["mis_centering"])
    mis_centering_kwargs = dict(config["MIS_CENTERING"]) if "MIS_CENTERING" in list(dict(config).keys()) else {}
    for k in list(mis_centering_kwargs.keys()):
        if k == "params":
            p = mis_centering_kwargs["params"]
            if "," not in p:
                mis_centering_kwargs[k] = [float(p)]
            else:
                mis_centering_kwargs[k] = str2bool(p, dtype = float)
        if mis_centering_kwargs[k] in ("True", "False"):
            mis_centering_kwargs[k] = str2bool(mis_centering_kwargs[k])
        elif "," in mis_centering_kwargs[k]:
            arr = prop2arr(mis_centering_kwargs[k], dtype = str)
            if arr[-1] in ["log", "linear"]:
                arg = list(np.array(arr[:-1], dtype = float))
                arg[-1] = int(arg[-1])
                mis_centering_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
            else:
                mis_centering_kwargs[k] = np.array(arr, dtype = float)
        elif "dict" in mis_centering_kwargs[k]:
            mis_centering_kwargs[k] = eval(mis_centering_kwargs[k])        
    fixed_mis_centering = mis_centering_kwargs["fixed"]
    if fixed_mis_centering == True and use_mis_centering == True:
        mis_centering_params = [float(mis_centering_kwargs["fmis"]), *mis_centering_kwargs["params"]]
    elif fixed_mis_centering == False and use_mis_centering == True:
        prior_config_mc = config["PRIORS_MIS_CENTERING"]
        prior_parameters_mc = dict(prior_config_mc)
        prior_parameters_dict_mc = {
            key: list(prop2arr(prior_parameters_mc[key], dtype=str))
            for key in list(prior_parameters_mc.keys())
        }
        prior_parameters_mc = list(prior_parameters_dict_mc.values())
        mis_centering_params = [0,0]
        labels_mc = np.array(
        [
            list(prior_parameters_dict_mc.keys())[i]
            for i in range(len(prior_parameters_mc))
            if "free" in prior_parameters_mc[i]
        ]
        ).astype(str)
        n_parameters_mc = len(prior_parameters_mc)
        params_indxs.append([params_indxs[-1][1] + 1, params_indxs[-1][1] + n_parameters_mc])

    completeness_config = dict(config["COMPLETENESS"])
    for k in list(completeness_config.keys()):
        if completeness_config[k] in ("True", "False"):
            completeness_config[k] = str2bool(completeness_config[k])
        elif "," in completeness_config[k]:
            completeness_config[k] = prop2arr(completeness_config[k], dtype = float)
        elif "dict" in completeness_config[k]:
            completeness_config[k] = eval(completeness_config[k])
    fixed_halo_model = completeness_config["fixed_RM_relationship"]
    if fixed_halo_model == False:
        prior_config_hm = config["PRIORS_HALO_MODEL"]
        prior_parameters_hm = dict(prior_config_hm)
        prior_parameters_dict_hm = {
            key: list(prop2arr(prior_parameters_hm[key], dtype=str))
            for key in list(prior_parameters_hm.keys())
        }
        prior_parameters_hm = list(prior_parameters_dict_hm.values())
        labels_hm = np.array(
        [
            list(prior_parameters_dict_hm.keys())[i]
            for i in range(len(prior_parameters_hm))
            if "free" in prior_parameters_hm[i]
        ]
        ).astype(str)
        n_parameters_hm = len(prior_parameters_hm)
        params_indxs.append([params_indxs[-1][1] + 1, params_indxs[-1][1] + n_parameters_hm])

    use_two_halo_term = str2bool(model_config["two_halo_term"])
    two_halo_kwargs = dict(config["TWO_HALO_TERM"]) if "TWO_HALO_TERM" in list(dict(config).keys()) else {}
    for k in list(two_halo_kwargs.keys()):
        if two_halo_kwargs[k] in ("True", "False"):
            two_halo_kwargs[k] = str2bool(two_halo_kwargs[k])
        elif "," in two_halo_kwargs[k]:
            arr = prop2arr(two_halo_kwargs[k], dtype = str)
            if arr[-1] in ["log", "linear"]:
                arg = list(np.array(arr[:-1], dtype = float))
                arg[-1] = int(arg[-1])
                two_halo_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
            else:
                two_halo_kwargs[k] = np.array(arr, dtype = float)
        elif "dict" in two_halo_kwargs[k]:
            two_halo_kwargs[k] = eval(two_halo_kwargs[k])

    two_halo_power = two_halo_kwargs["two_halo_power"]
    if two_halo_power == True:
        prior_config_2h = config["PRIORS_TWO_HALO_POWER"]
        prior_parameters_2h = dict(prior_config_2h)
        prior_parameters_dict_2h = {
            key: list(prop2arr(prior_parameters_2h[key], dtype=str))
            for key in list(prior_parameters_2h.keys())
        }
        two_halo_prior = {
            key: list(prior_parameters_dict_2h[key])
            for key in list(prior_parameters_dict_2h.keys())
        }
        prior_parameters_2h = list(prior_parameters_dict_2h.values())
        n_parameters_2h = len(prior_parameters_2h)
        labels_2h = np.array(
        [
            list(prior_parameters_dict_2h.keys())[i]
            for i in range(len(prior_parameters_2h))
            if "free" in prior_parameters_2h[i]
        ]
        ).astype(str)
        params_indxs.append(
            [params_indxs[-1][1] + 1, params_indxs[-1][1] + n_parameters_2h]
        )
    two_halo_kwargs["cosmo"] = ccl.CosmologyVanillaLCDM()
    profile_stacked_model = model_config["profile"]
    filters = model_config["filters"].split("|")
    use_filters = str2bool(model_config["use_filters"])
    fil_name, ext = list(prop2arr(config["EMCEE"]["output_file"], dtype = str))

    xlabel_config, ylabel_config = config["MODEL"]["x"], config["MODEL"]["y"]

    ylabel_config = ylabel_config.split(",") if len(ylabel_config.split(",")) == 2 else (ylabel_config,"")
    xlabel_config = xlabel_config.split(",") if len(xlabel_config.split(",")) == 2 else (xlabel_config,"")

    xlabel, xlabel_unit = xlabel_config[0], xlabel_config[1]
    ylabel, ylabel_unit = ylabel_config[0], ylabel_config[1]

    xlabel_unit = xlabel_unit.replace(" ", "") if xlabel_unit is not None else ""
    ylabel_unit = ylabel_unit.replace(" ", "") if ylabel_unit is not None else ""

    ylabel = f"{ylabel} ({ylabel_unit})" if ylabel_unit != "" else ylabel
    xlabel = f"{xlabel} ({xlabel_unit})" if xlabel_unit != "" else xlabel

    yscale = model_config["yscale"] if "yscale" in list(model_config.keys()) else "log"
    xscale = model_config["xscale"] if "xscale" in list(model_config.keys()) else "linear"

    title = config["MODEL"]["title"]

    ymin = eval(config["MODEL"]["ymin"]) if "ymin" in list(config["MODEL"].keys()) else None
    ymax = eval(config["MODEL"]["ymax"]) if "ymax" in list(config["MODEL"].keys()) else None

    off_diag = str2bool(model_config["off_diag"]) if "off_diag" in list(model_config.keys()) else False

    if use_mis_centering == True and fixed_mis_centering == False:
        labels = np.array(list(labels) + list(labels_mc))
        prior_parameters = list(prior_parameters) +  list(prior_parameters_mc)

    if fixed_halo_model == False:
        labels = np.array(list(labels) + list(labels_hm))
        prior_parameters = list(prior_parameters) +  list(prior_parameters_hm)
    
    if use_two_halo_term == True and two_halo_power == True:
        labels = np.array(list(labels) + list(labels_2h))
        prior_parameters = list(prior_parameters) +  list(prior_parameters_2h)

    n_parameters = len(labels)

    priors_funcs = []
    priors_args = []
    
    for i in range(len(prior_parameters)):
        if 'free' in prior_parameters[i]:
            args_ = np.array(prior_parameters[i][-1].split("|")).astype(
                np.float64
            ) if "|" in prior_parameters[i][-1] else np.array(prior_parameters[i][-2].split("|")).astype(np.float64)

            priors_funcs.append(getattr(MCMC_func, prior_parameters[i][1]))
            priors_args.append(args_)
        else:
            pass
    rebinning_config = dict(config["REBINNING"])

    use_rebinning = str2bool(model_config["rebinning"])
    nbins_rebinning = int(rebinning_config["Nbins"])
    pixel_size_rebinning = float(rebinning_config["pixel_size"])
    rebinning_method = rebinning_config["method"]
    interpolation_kwargs = eval(rebinning_config["interpolation_kwargs"])

    rebinning_kwargs = dict(
        method = rebinning_method,
        interpolation_kwargs = interpolation_kwargs,
        rmin = 0,
        nbins = nbins_rebinning,
        pixel_size = pixel_size_rebinning,
    )

elif args.CONFIG_FILE is not None and joint == True:
    paths = args.path.split(",")
    config_files = args.CONFIG_FILE.split(",")
    jprior_parameters = []
    jfuncs = []
    jcovs = []
    jprofiles = []
    jclusters_list = []
    X = []
    params_indx = []        
    models = []
    configs = []
    profile_models = []
    hm_model_added = False
    hm_params_indx = None
    jrbins = []
    jzbins = []
    jMbins = []
    labels = []
    xlabels, ylabels = [], []
    yscales, xscales = [], [] 
    titles = []
    jpriors_funcs = []
    jpriors_args = []
    for (j,c),path in zip(enumerate(config_files), paths):
        current_path = os.path.dirname(os.path.realpath(__file__))
        config_filepath = current_path +"/"+ str(c)
        print("Loading config from :",config_filepath)
        config = ConfigParser()
        config.optionxform = str
        if os.path.exists(config_filepath):
            config.read(config_filepath)
        else:
            raise Found_Error_Config(f"The config file {str(c)} doesn't exist")
        configs.append(config)

        prior_config = config["PRIORS"]
        emcee_config = config["EMCEE"]
        model_config = config["MODEL"]
        blobs_config = config["BLOBS"]

        try: 
            model_id = model_config["name"]
        except:
            model_id = "model " + j

        xlabel_config, ylabel_config = config["MODEL"]["x"], config["MODEL"]["y"]

        ylabel_config = ylabel_config.split(",") if len(ylabel_config.split(",")) == 2 else (ylabel_config,"")
        xlabel_config = xlabel_config.split(",") if len(xlabel_config.split(",")) == 2 else (xlabel_config,"")

        xlabel, xlabel_unit = xlabel_config[0], xlabel_config[1]
        ylabel, ylabel_unit = ylabel_config[0], ylabel_config[1]

        xlabel_unit = xlabel_unit.replace(" ", "") if xlabel_unit is not None else ""
        ylabel_unit = ylabel_unit.replace(" ", "") if ylabel_unit is not None else ""

        ylabel = f"{ylabel} ({ylabel_unit})" if ylabel_unit != "" else ylabel
        xlabel = f"{xlabel} ({xlabel_unit})" if xlabel_unit != "" else xlabel

        yscale = model_config["yscale"] if "yscale" in list(model_config.keys()) else "log"
        xscale = model_config["xscale"] if "xscale" in list(model_config.keys()) else "linear"

        title = config["MODEL"]["title"]

        xlabels.append(xlabel)
        ylabels.append(ylabel)

        xscales.append(xscale)
        yscales.append(yscale)

        titles.append(title)
        if j == 0:
            rewrite = str2bool(emcee_config["rewrite"])
            likelihood_func = emcee_config["likelihood"]
            nwalkers = int(emcee_config["nwalkers"])
            nsteps = int(emcee_config["nsteps"])
            del_backend = str2bool(emcee_config["delete"])

        completeness_config = dict(config["COMPLETENESS"])
        for k in list(completeness_config.keys()):
            if completeness_config[k] in ("True", "False"):
                completeness_config[k] = str2bool(completeness_config[k])
            elif "," in completeness_config[k]:
                completeness_config[k] = prop2arr(completeness_config[k], dtype = float)
            elif "dict" in completeness_config[k]:
                completeness_config[k] = eval(completeness_config[k])
        fixed_halo_model = completeness_config["fixed_RM_relationship"]
        if fixed_halo_model == False:
            prior_config_hm = config["PRIORS_HALO_MODEL"]
            prior_parameters_hm = dict(prior_config_hm)
            prior_parameters_dict_hm = {
                key: list(prop2arr(prior_parameters_hm[key], dtype=str))
                for key in list(prior_parameters_hm.keys())
            }
            prior_parameters_hm = list(prior_parameters_dict_hm.values())

        ncores = int(emcee_config["ncores"]) if "ncores" in list(emcee_config.keys()) else None
        use_two_halo_term = str2bool(model_config["two_halo_term"])
        two_halo_kwargs = dict(config["TWO_HALO_TERM"]) if "TWO_HALO_TERM" in list(dict(config).keys()) else {}

        for k in list(two_halo_kwargs.keys()):
            if two_halo_kwargs[k] in ("True", "False"):
                two_halo_kwargs[k] = str2bool(two_halo_kwargs[k])
            elif "," in two_halo_kwargs[k]:
                arr = prop2arr(two_halo_kwargs[k], dtype = str)
                if arr[-1] in ["log", "linear"]:
                    arg = list(np.array(arr[:-1], dtype = float))
                    arg[-1] = int(arg[-1])
                    two_halo_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                else:
                    two_halo_kwargs[k] = np.array(arr, dtype = float)
            elif "dict" in two_halo_kwargs[k]:
                two_halo_kwargs[k] = eval(two_halo_kwargs[k])
        use_mis_centering = str2bool(model_config["mis_centering"])
        mis_centering_kwargs = dict(config["MIS_CENTERING"]) if "MIS_CENTERING" in list(dict(config).keys()) else {}
        for k in list(mis_centering_kwargs.keys()):
            if k == "params":
                p = mis_centering_kwargs["params"]
                if "," not in p:
                    mis_centering_kwargs[k] = [float(p)]
                else:
                    mis_centering_kwargs[k] = str2bool(p, dtype = float)
            if mis_centering_kwargs[k] in ("True", "False"):
                mis_centering_kwargs[k] = str2bool(mis_centering_kwargs[k])
            elif "," in mis_centering_kwargs[k]:
                arr = prop2arr(mis_centering_kwargs[k], dtype = str)
                if arr[-1] in ["log", "linear"]:
                    arg = list(np.array(arr[:-1], dtype = float))
                    arg[-1] = int(arg[-1])
                    mis_centering_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                else:
                    mis_centering_kwargs[k] = np.array(arr, dtype = float)
            elif "dict" in mis_centering_kwargs[k]:
                mis_centering_kwargs[k] = eval(mis_centering_kwargs[k])
        fixed_mis_centering = mis_centering_kwargs["fixed"]
        if fixed_mis_centering == True and use_mis_centering == True:
            mis_centering_params = [float(mis_centering_kwargs["fmis"]), *mis_centering_kwargs["params"]]
        else:
            prior_config_mc = config["PRIORS_MIS_CENTERING"]
            prior_parameters_mc = dict(prior_config_mc)
            prior_parameters_dict_mc = {
                key: list(prop2arr(prior_parameters_mc[key], dtype=str))
                for key in list(prior_parameters_mc.keys())
            }
            prior_parameters_mc = list(prior_parameters_dict_mc.values())
            mis_centering_params = [0,0]
        two_halo_kwargs["cosmo"] = ccl.CosmologyVanillaLCDM()

        prior_parameters = dict(prior_config)
        prior_parameters_dict = {
            key: list(prop2arr(prior_parameters[key], dtype=str))
            for key in list(prior_parameters.keys())
        }
        prior_parameters = list(prior_parameters_dict.values())

        if fixed_mis_centering == False and use_mis_centering == True:
            prior_parameters += prior_parameters_mc
            prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_mc}
        if fixed_halo_model == False and hm_model_added == False:
            hm_model_added = True
            prior_parameters += prior_parameters_hm
            prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_hm}

        if len(params_indx) == 0:
            params_indx.append((0,len(prior_parameters)))
        else:
            params_indx.append((params_indx[-1][1], len(prior_parameters) + params_indx[-1][1]))

        if fixed_halo_model == False and hm_params_indx is None:
            hm_params_indx = [params_indx[-1][1] - len(prior_parameters_hm), params_indx[-1][1]]
        
        priors_funcs = []
        priors_args = []
        
        for i in range(len(prior_parameters)):
            if 'free' in prior_parameters[i]:
                args_ = np.array(prior_parameters[i][-1].split("|")).astype(
                    np.float64
                )
                priors_funcs.append(getattr(MCMC_func, prior_parameters[i][1]))
                priors_args.append(args_)
            else:
                priors_funcs.append(None)
                priors_args.append(None)
            
        jpriors_funcs.append(priors_funcs)
        jpriors_args.append(priors_args)
        jprior_parameters = jprior_parameters + prior_parameters

        models.append(model_id)
        profile_models.append(model_config["profile"])
        rbins, zbins, Mbins = model_config["rbins"], model_config["zbins"], model_config["Mbins"]
        jrbins.append(int(rbins))
        jzbins.append(int(zbins))
        jMbins.append(int(Mbins))

        clabels = np.array(
        [
            list(prior_parameters_dict.keys())[i]
            for i in range(len(prior_parameters))
         if "free" in prior_parameters[i]]
        ).astype(str)

        labels = np.concatenate((labels, clabels))

    fixed_params = [(i,p[1]) for i,p in enumerate(jprior_parameters) if "fixed" in p]
    free_params = [(i,p) for i,p in enumerate(jprior_parameters) if "free" in p]

    fil_name = "x".join(models)
    output_path = "/".join(paths[-1].split("/")[:-2])

    samples_file = output_path + "/" + fil_name + "_joint_fit.h5" 
def main():
    if all_data == False and general == False and joint == False:
        model = getattr(profiles_module, profile_stacked_model)
        plot_mcmc(source_path, model, labels, plot = plot, steps = args.steps, 
                    corner_ = args.corner, make_copy = args.make_copy, discard = args.discard, thin = args.thin,
                    tau = args.tau, method = args.extract_method, use_signal = args.signal, filters = filters,
                    fil_name = fil_name, ext = ext, use_filters = use_filters, model_name = title, 
                    priors = priors_funcs, priors_args = priors_args)
    elif all_data == True and general == False and joint == False:
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
            redshift_bin.append([np.nanmin(cluster.z), np.max(cluster.z)])
            richness_bin.append([int(np.nanmin(cluster.richness)), int(np.max(cluster.richness))])
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
            pmin = np.nanmin(profiles[profiles > 0])
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
                zmin, zmax = np.nanmin(c.z), np.max(c.z)
                rmin, rmax = int(np.nanmin(c.richness)), int(np.max(c.richness))
                default_profiles_kwargs = (
                        ("output_file", None),
                        ("ax_kwargs", dict(xlabel = xlabel, ylabel = xlabel, 
                                            title = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))),
                        ("show_legend", False),
                        ("show_results", False)
                        )  
                R = c.R
                lower_bound, upper_bound = np.nanpercentile(signal, [16,84], axis = 0)
                fit = np.nanmedian(signal, axis = 0)
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
    elif general == True and all_data == False and joint == False:
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
                    xlabel = xlabel, ylabel = ylabel, plot_cov = args.cov_matrix, plot_corr = args.corr_matrix,
                    priors = priors_funcs, priors_args = priors_args, chi2_ = args.plot_chi2, params_indxs = params_indxs,
                    plot_mis_centering = args.plot_mis_centering, plot_hm_relationship = args.plot_hm_relationship,
                    off_diag = off_diag, ymin = ymin, ymax = ymax
                    )
    elif joint == True:
        plot_joint_mcmc(paths, samples_file, labels, params_indx, profile_models, 
                Mbins = jMbins, rbins = jrbins, zbins = jzbins, plot = args.plot, steps = args.steps, 
                corner_ = args.corner, make_copy = args.make_copy, discard = args.discard, thin = args.thin,
                tau = args.tau, use_signal = True, method = args.extract_method, share_plot = args.share_plot,
                plot_cov = args.cov_matrix, plot_corr = args.corr_matrix, xlabels = xlabels, ylabels = ylabels,
                titles = titles, _chi2 = args.plot_chi2, fixed_params = fixed_params, free_params = free_params,
                jpriors = jpriors_funcs, jpriors_args = jpriors_args, xscales = xscales, yscales = yscales)

def plot_joint_mcmc(data_paths, source_file, labels, params_indx, profile_models = None, ndims = None, nwalkers = None, plot = False, discard = 0,
                steps = False, corner_ = False, make_copy = True, method = "median", tau = False, thin = 1,
                output_paths = "./", use_signal = True, rbins = None, Mbins = None, zbins = None, share_plot = False,
                xlabels = None, ylabels = None, xscales = None, yscales = None, models_name = None, titles = None, 
                plot_cov = False, plot_corr = False,  plot_shared_versions = True, _chi2 = False, 
                fixed_params = None, free_params = None, jpriors = None, jpriors_args = None, **kwargs):

    free_indx = [p[0] for p in free_params]
    fixed_indx = [p[0] for p in fixed_params]
    fixed_values = [p[1] for p in fixed_params]

    xscales = ["log" for i in range(len(data_paths))] if xscales is None else xscales
    yscales = ["log" for i in range(len(data_paths))] if yscales is None else yscales

    discard = int(discard) if len(str(discard).split(',')) == 1 else np.array(discard.split(','), dtype = int)
    discard = int(discard)

    jclusters = []
    funcs = []
    jprofiles = []
    X = []
    jsignal = []
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
    blobs = backend.get_blobs(discard = discard)
    nwalkers = backend.get_chain().shape[1] if nwalkers is None else nwalkers
    ndims = len(labels)
    chi2_values = blobs['CHI2'].flatten()
    signal = blobs['SIGNAL'].reshape((len(chi2_values),-1))
    ln_prior = blobs['LN_PRIOR'].flatten()
    ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
    max_ln_likelihood = np.max(ln_likelihood)

    chain = backend.get_chain(discard = discard, thin = thin, flat = True)  
    unflatten_chain = backend.get_chain(discard = discard, thin = thin)

    nsteps, nwalkers, n_free = unflatten_chain.shape
    n_total = n_free + len(fixed_params)

    chain_full = np.empty((nsteps, nwalkers, n_total), dtype=unflatten_chain.dtype)
    chain_full[:, :, fixed_indx] = fixed_values
    chain_full[:, :, free_indx] = unflatten_chain
    chain_full = np.reshape(chain_full, (nwalkers * nsteps, n_total))

    for i,p in enumerate(data_paths):
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        xscale = xscales[i]
        yscale = yscales[i]
        title = titles[i]
        print("Loading data from ", p)
        indx = params_indx[i]
        indx_i, indx_f = indx
        sub_chain = chain[:, int(indx_i):int(indx_f)]
        slabels = labels[int(indx_i):int(indx_f)]
        list_of_clusters = []
        main_path = p
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
                        richness_bins.append(np.round(np.nanmin(clusters.richness)))
                    else:
                        continue
                else:
                    list_of_clusters.append(clusters)
                    paths.append(clusters.output_path)
                    richness_bins.append(np.round(np.nanmin(clusters.richness)))
            except Exception as e:
                print(f"The next exception occurred trying to load {current_path}: \033[31m{e}\033[0m")
                continue
        all_clusters = list_of_clusters[0]
        for j in range(1,len(list_of_clusters)):
            all_clusters+=list_of_clusters[j]

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
            redshift_bins = prop2arr(args.redshift_bins, dtype = np.float64) if args.redshift_bins is not None else np.tile([np.nanmin(clusters.z),np.max(clusters.z)],len(richness_bins)).reshape((len(richness_bins),2))

        func,cov_matrix, about_clusters, clusters, raw_profiles, funcs = grouped_clusters.stacked_halo_model_func_by_paths(
                                getattr(profiles_module, profile_models[i]), fixed_RM_relationship = fixed_halo_model, use_mis_centering = use_mis_centering,
                                zb = redshift_bins, rb = richness_bins, full = True, Mbins = Mbins[i], Rbins = rbins[i], Zbins = zbins[i], paths = paths) 
        N_clusters = len(list_of_clusters)
        funcs.append(func)
        jclusters.append(clusters)
        jprofiles.append(raw_profiles)
        X.append(clusters[0].R)
        R = clusters[0].R
        profiles = np.reshape(raw_profiles, (-1, len(R)))
        chi2_values2 = blobs['CHI2']
        if len(jsignal) == 0:
            signal_i = signal[:, 0:len(raw_profiles)]
            jsignal.append(signal_i)
        else:
            signal_i = signal[:,len(jsignal[-1][0]): len(jsignal[-1][0]) + len(raw_profiles)]
            jsignal.append(signal_i)
        signal_i2 = np.reshape(signal_i, (np.shape(chi2_values2)[0], np.shape(chi2_values2)[1], -1))
        sub_chain2 = sub_chain.reshape(len(chi2_values2), -1, len(slabels))
        min_indices = np.argmin(chi2_values2, axis = 1)

        signal_at_min_chi2 = signal_i2[np.arange(signal_i2.shape[0]), min_indices, :]
        signal_at_abs_min_chi2 = signal_at_min_chi2[np.argmin(np.min(chi2_values2, axis = 1))]
        signal_at_abs_min_chi2 = np.reshape(signal_at_abs_min_chi2, (-1, len(R)))
        
        residuals = (profiles - signal_at_abs_min_chi2).flatten()
        specific_chi2 = np.dot(residuals, np.dot(np.linalg.inv(cov_matrix), residuals.T))
        specific_pte = pte(specific_chi2, cov_matrix)

        signal2bound = np.reshape(signal_i,(-1, N_clusters, len(R))) if use_signal else None

        params, lower, upper = extract_params(sub_chain, slabels, method = method)
        labels_latex = [text2latex(l) for l in slabels]
        output_path = p
        if _chi2 == True:
            default_chi2_kwargs = (
                ("zoom_in", False),
                ("path", f"{output_path}/chi2"),
                ("zoom", 0.15),
                ("output_file", "chi2_joint.png"),
                ("labels", slabels)
            )
            plot_chi2(chi2_values2, chain = sub_chain2, **set_default(kwargs.pop("chi2_kwargs",{}), default_chi2_kwargs)
            )
        if verbose == True:
            print("Parameter & value")
            for j in range(len(params)):
                print(labels_latex[j] + r"& $%.2f_{%.2f}^{%.2f}$ \\" % (params[j],lower[j],upper[j]))

        if steps:     
            default_steps_kwargs = (
                ("labels", labels_latex),
                ("plot_tracers", True),
                ("nwalkers", nwalkers),
                ("discard", discard),
                ("output_file", f"{output_path}/parameters_steps_joint.png"),
                ("tracers_aspect", ('k-','k--','k--'))
            )
            steps_kwargs = set_default(kwargs.pop("steps_kwargs",{}), default_steps_kwargs)
            plot_steps(sub_chain, backend,chi2_values,**steps_kwargs)

        if corner_:
            default_corner_kwargs = (
                ("truths", params),
                ("truths_color","black"),
                ("corner_color", "blue"),
                ("output_file", f"{output_path}/corner_joint.png"),
                ("labels",labels_latex),
                ("fontsize", 16),
                ("range_sigma_ratio", float(args.range_sigma_ratio)),
                ("title_kwargs", {"fontsize":12})
            )
            corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
            plot_corner(sub_chain, priors = jpriors[i], priors_args = jpriors_args[i], **corner_kwargs)
        if tau:
            default_tau_kwargs = (
                ("show_convergence", True),
                ("output_file", f"{output_path}/tau_joint.png")
            )
            tau_kwargs = set_default(kwargs.pop("tau_kwargs", {}), default_tau_kwargs)
            plot_tau(backend, labels_latex, **tau_kwargs)

        if plot_cov == True or plot_corr == True:
            default_cov_kwargs = (
                ("output_file", f"{output_path}/cov_params_joint.png"),
            )
            cov_kwargs = set_default(kwargs.pop("cov_kwargs", {}), default_cov_kwargs)   
            plot_cov_matrix(sub_chain, labels_latex, corr = plot_corr, **cov_kwargs)    

        if plot == True:
            num_profiles = N_clusters
            if share_plot == False:
                profiles_per_row = 3
                num_rows = (num_profiles + profiles_per_row - 1) // profiles_per_row
                fig, axes = plt.subplots(num_rows, profiles_per_row, figsize=(20, 8 * num_rows), sharey = True, sharex = True)
                axes = axes.flatten()
                val = [np.nanmedian(c.richness) + np.nanmedian(c.z) for c in clusters]
                sorted_idx = np.argsort(val)
                for i, idx in enumerate(sorted_idx):
                    ax = axes[i]
                    c = clusters[i]
                    zmin, zmax = np.nanmin(c.z), np.max(c.z)
                    rmin, rmax = int(np.nanmin(c.richness)), int(np.max(c.richness))
                    rmin = rmin + 1 if abs(int(rmin)) % 10 == 9 else rmin
                    rmax = rmax + 1 if abs(int(rmax)) % 10 == 9 else rmax
                    default_profiles_kwargs = (
                            ("output_file", None),
                            ("ax_kwargs", dict(xlabel = xlabel, ylabel = xlabel, 
                                                title = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))),
                            ("show_legend", False),
                            ("show_results", False)
                            )  
                    profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)  
                    fit = np.median(signal2bound[:,i,:], axis = 0)
                    lower_bound, upper_bound = np.percentile(signal2bound[:,i,:], [16, 84], axis = 0)
                    lower_bound = fit - lower_bound
                    upper_bound = upper_bound - fit              
                    plot_profiles(R, profiles[i], func, params, c.cov, labels_latex, lower, upper,
                                np.max(ln_likelihood), np.mean(all_clusters.z), ax = ax, fit = fit, lower_bound = lower_bound,
                                upper_bound = upper_bound, signal = signal2bound[:,i,:], 
                                specific_pte = specific_pte, specific_chi2 = specific_chi2,
                                **profiles_kwargs)
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
                val = [np.nanmedian(c.richness) + np.nanmedian(c.z) for c in clusters]
                sorted_idx = np.argsort(val)
                for i, idx in enumerate(sorted_idx):
                    color = colors[i]
                    c = clusters[idx]
                    zmin, zmax = np.nanmin(c.z), np.max(c.z)
                    rmin, rmax = int(np.nanmin(c.richness)), int(np.max(c.richness))
                    rmin = rmin + 1 if abs(int(rmin)) % 10 == 9 else rmin
                    rmax = rmax + 1 if abs(int(rmax)) % 10 == 9 else rmax
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

                    fit = np.median(signal2bound[:,idx,:], axis = 0)
                    lower_bound, upper_bound = np.percentile(signal2bound[:,idx,:], [16, 84], axis = 0)
                    lower_bound = fit - lower_bound
                    upper_bound = upper_bound - fit 
                    profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)    
                    signal2bound_i = signal2bound[:,i,:] if signal2bound is not None else None
                    plot_profiles(R + i*0.1, profiles[idx], func, params, c.cov, labels_latex, lower, upper,
                                np.max(ln_likelihood), np.mean(c.z), ax = ax, fit = fit, lower_bound = lower_bound,
                                upper_bound = upper_bound, signal = signal2bound_i, show_labels = False, show_error_bars = True,
                                specific_pte = specific_pte, specific_chi2 = specific_chi2, **profiles_kwargs)
                ax.legend(loc = "upper right", fontsize = 10)
                ax.set_title("Best Fitting " + title)
                #ax.set_ylim((np.nanmin(np.array(profiles)[np.array(profiles) > 0])*0.1, np.max(profiles)*1.5))
                ax.grid(True)
            if args.dont_show_results == False:
                chi2 = np.nanmin(chi2_values) #calculate_chi2(raw_profiles, func(R.value,params), np.linalg.inv(cov_matrix))
                if args.use_obs_chi2 == True:
                    observed_profiles = np.nanmedian(signal, axis = 0)
                    res = raw_profiles - observed_profiles
                    chi2_obs = np.dot(np.dot(res, np.linalg.inv(cov_matrix)), res.T)
                    p_value = pte(chi2_obs, cov_matrix)
                else:
                    p_value = pte(chi2, cov_matrix)
                bic = BIC(np.size(raw_profiles), len(params), max_ln_likelihood)
                text  = [
                        r'$\chi^{2} = %.2f\; (%.2f)$' % (chi2, specific_chi2),
                        r'$PTE = %.4f\; (%.4f)$' % (p_value, specific_pte),
                ]
                for i in range(len(labels_latex)):
                    if labels[i].split('_')[0] == r'$\log':
                        text.append(f'{labels_latex[i]} : {np.round(np.log10(params[i]),2)} $\pm$ {np.round(err[i]/(np.log(10) * paramss[i]),2)}')
                    else:
                        text.append('%s' % labels_latex[i] + ': $%.2f' % params[i] + '^{+%.2f}_{-%.2f}$' % (np.abs(upper[i]),np.abs(lower[i])))   
                s = '\n'.join(text)
                if share_plot:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    #0.15, 0.6
                    ax.text(0.17, 0.025, s, fontsize=13, verticalalignment='bottom', ha = "right", transform=ax.transAxes, bbox=props, color = 'black')
                else:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    fig.text(0.7, 0.95, s, fontsize=11, verticalalignment='top', ha = "right", bbox=props, color = 'black')       
            fig.tight_layout()           
            fig.savefig(f"{output_path}/best_fitting_joint.png", dpi = args.dpi, transparent = False) 

def plot_general_mcmc(main_path, source_file, model, labels, ndims = None, nwalkers = None,
                       plot = False, discard = 0, steps = False, corner_ = False, make_copy = True, 
                       method = "median", tau = False, thin = 1, output_path = './', use_signal = False, 
                       rbins = None, Mbins = None, zbins = None, share_plot = False, xlabel = None,
                       ylabel = None, model_name = "", plot_cov = False, plot_corr = False, chi2_ = True,
                       priors = None, priors_args = None, params_indxs = None, 
                       plot_mis_centering = False, plot_hm_relationship = False, 
                       compute_mass = False, off_diag = False, ymin = None, ymax = None,
                       **kwargs):


    ndims = len(labels) if ndims is None else ndims
    labels_latex = [text2latex(l) for l in labels]
    list_of_clusters = []
    main_path = main_path + "/" if main_path [-1] != "/" else main_path 
    ignore = np.loadtxt(main_path + "ignore.txt", dtype = str).T if os.path.exists(main_path + "ignore.txt") else []
    available_paths = [path for path in os.listdir(main_path) if os.path.isdir(main_path + path) and path not in ignore]
    print("Available paths:")
    [print(f"\t* \033[35m{p}\033[0m") for p in available_paths]
    richness_bins = []
    paths = []
    
    apply_filter_per_profile = str2bool(model_config["apply_filter_per_profile"])

    for path in available_paths:
        if path in ignore:
            continue
        current_path = main_path + path
        try:
            if is_running_via_nohup() == False and args.ask_to_add == True:
                add_cluster = input(f"Do you want to add the next cluster? (Y, yes or enter to add it):\n {clusters}from: \033[35m{path}\033[0m\n").strip().lower()
                if add_cluster in ["y","yes",""]:
                    paths.append(current_path)
                else:
                    continue
            else:
                paths.append(current_path)
        except Exception as e:
            print(f"The next exception occurred trying to load {current_path}: \033[31m{e}\033[0m")
            continue

    rbins, zbins, Mbins = completeness_config.pop("rbins", 25), completeness_config.pop("zbins", 25), completeness_config.pop("Mbins", 25)
    rbins = int(rbins)
    zbins = int(zbins)
    Mbins = int(Mbins)
    sort_by_redshift = False
    func,cov, about_clusters, clusters, _, funcs = grouped_clusters.stacked_halo_model_func_by_paths(getattr(profiles_module, profile_stacked_model),
                                        full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths, verbose = True)
                                        # #use_filters = use_filters, filters = filters_dict,
                                        # completeness_kwargs = dict(completeness_config), use_two_halo_term = use_two_halo_term, off_diag = off_diag,
                                        # two_halo_kwargs = two_halo_kwargs, use_mis_centering = use_mis_centering, fixed_RM_relationship = fixed_halo_model
                                        # , background = background, delta = delta, eval_mass = eval_mass, apply_filter_per_profile = apply_filter_per_profile
                                        # ,rebinning = use_rebinning, rebinning_kwargs = rebinning_kwargs)
    bins = np.array([[*c.richness_bin, *c.redshift_bin] for c in clusters])
    sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
    bins = bins[sorted_idx]
    clusters = [clusters[i] for i in sorted_idx]

    cluster, cov = grouped_clusters.compute_joint_cov(off_diag = off_diag, groups = clusters, corr = False)

    profiles = np.array([c.mean_profile for c in clusters])
    R = np.loadtxt(f"{main_path}/xobs.txt")
    sigma = np.loadtxt(f"{main_path}/sigma.txt")
    cbins = np.loadtxt(f"{main_path}/bins.txt", skiprows = 1)
    cov = np.loadtxt(f"{main_path}/cov.txt") if os.path.exists(f"{main_path}/cov.txt") else cov
    redshift_bins = cbins[:, 2:4]
    richness_bins = cbins[:, 0:2]

    discard = int(discard) if len(str(discard).split(',')) == 1 else np.array(discard.split(','), dtype = int)

    N_clusters = len(clusters)
    errors = np.sqrt(np.diag(cov)).reshape((N_clusters, len(R)))
    if make_copy:
        copy = f"/{source_file.split('.')[0]}_copy.h5"
        shutil.copy(source_file, copy)
        source_file = copy
    backend = emcee.backends.HDFBackend(source_file, read_only = True)
    unflatten_chain = backend.get_chain(discard = discard, thin = thin)

    nsteps, nwalkers, ndims = unflatten_chain.shape

    try:
        autocorr_time = backend.get_autocorr_time()
        max_tau = int(2*np.max(autocorr_time))
        print(f"The chain converged at {max_tau} steps using get_autorr_time function.")
    except:
        print("\033[91mThe chain is not fully converged yet!\033[0m\n")
    if np.iterable(discard) == False:
        blobs = backend.get_blobs(discard = discard, thin = thin)
        blobs_keys = blobs.dtype.base.names
        chi2_values = blobs['CHI2'].flatten()
        signal = blobs['SIGNAL'].reshape((len(chi2_values),-1))
        P1halo = blobs['ONE_HALO'].reshape((len(chi2_values),-1)).reshape((-1,N_clusters, len(R))) if "ONE_HALO" in blobs_keys else np.full(signal.shape, np.nan)
        P2halo = blobs['TWO_HALO'].reshape((len(chi2_values),-1)).reshape((-1,N_clusters, len(R))) if "TWO_HALO" in blobs_keys else np.full(signal.shape, np.nan)
        Masses = blobs['MASS'].reshape((len(chi2_values),-1)).reshape((-1,N_clusters)) if "MASS" in blobs_keys else np.full(signal.shape, np.nan)
        P1halo = np.reshape(P1halo, (-1, N_clusters, len(R)))
        P2halo = np.reshape(P2halo, (-1, N_clusters, len(R)))
        signal2bound = np.reshape(signal,(-1,N_clusters, len(R))) if use_signal else None
        signal2bound = P1halo + P2halo if "ONE_HALO" in blobs_keys and "TWO_HALO" in blobs_keys else signal2bound
        ln_prior = blobs['LN_PRIOR'].flatten()
        ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
        max_ln_likelihood = np.max(ln_likelihood)
        chain = backend.get_chain(discard = discard, thin = thin, flat = True)
        params, lower, upper = extract_params(chain, labels, method = method)
        for i in range(len(chain.T)):
            c = chain[:,i]
            _, nsigma = nsigma_from_posterior(c, 0)
            print(f"{labels[i]} is {nsigma} aways from 0.")
        if np.all(np.isnan(Masses)) == False and args.infere_mass == True:
            profiles_per_row = 4
            num_profiles = len(clusters)
            num_rows = (num_profiles + profiles_per_row - 1) // profiles_per_row
            fig = plt.figure(figsize=(18 + num_rows, 6 * num_rows))
            gs = fig.add_gridspec(num_rows, profiles_per_row, wspace=0, hspace=0)
            axs = []
            nrows = num_rows
            ncols = profiles_per_row
            sharex_row = [None] * nrows
            sharey_col = [None] * ncols
            for i in range(nrows):
                for j in range(ncols):
                    ax = fig.add_subplot(
                        gs[i, j], 
                        sharex=sharex_row[i],
                        sharey=sharey_col[j]
                    )
                    if sharex_row[i] is None:
                        sharex_row[i] = ax
                    if sharey_col[j] is None:
                        sharey_col[j] = ax
                    axs.append(ax)

            fig.add_subplot(gs[:, 0]).axis('off')
            axs = np.reshape(axs, (nrows, ncols))     
            axes = axs.flatten()
            counter = 0
            richness = []
            richness_errs = []
            infered_masses = []
            redshift = []
            infered_masses = np.zeros((len(clusters), 3, 3), dtype = object)
            for i in range(len(clusters)):
                if counter <= num_profiles:
                    ax = axes[i]
                    c = clusters[i]
                    zmin, zmax = redshift_bins[i]
                    rmin, rmax = richness_bins[i]

                    weights = c.weights[0:len(c.richness)]
                    richness.append(np.average(c.richness, weights = weights))
                    richness_errs.append(np.sqrt(np.sum(1/weights)))

                    redshift.append(np.average(c.z, weights = weights))

                    M = Masses[:, i]
                    M = M[np.where((M > 1e14) & (M < 1e16))]
                    Mbins = np.logspace(14, 16, 50)
                    hist, bins_edges = np.histogram(M, bins = Mbins, density = True)
                    hist = np.nan_to_num(hist)
                    hist = gaussian_filter1d(hist, sigma = 3, mode = "constant", cval = 0.0)
                    hist = hist/np.max(hist)
                    ax.stairs(hist, Mbins, color = "black", lw = 3, alpha = 0.7)
                    median_M = np.nanmedian(M)
                    ax.axvline(median_M, ls = "--", color = "C3", lw = 3, alpha = 0.7)
                    #gaussian mixture to check if mass histogram is multivariable
                    bin_centers = 0.5*(Mbins[:-1] + Mbins[1:])
                    g = np.log10(bin_centers).reshape(-1,1)
                    N_synth     = 20000
                    props       = np.clip(hist, 0, None)
                    props      /= props.sum()
                    n_repeat    = np.round(props * N_synth).astype(int)

                    g_synth = np.repeat(g, n_repeat, axis=0)
                    g_synth = np.reshape(g_synth, (-1, 1))

                    bics = []
                    min_bic = 0
                    for i in range(2):
                        gmm = GMM(i+1, max_iter = 1000, covariance_type = "full")
                        gmm.fit(g_synth)
                        labels_ = gmm.predict(g_synth)
                        bics.append(gmm.bic(g_synth))
                        if bics[-1] < min_bic or min_bic == 0:
                            min_bic = bics[-1]
                            best_gmm = i + 1
                    gmm =  GMM(best_gmm, max_iter = 1000, covariance_type = "full")
                    gmm.fit(g_synth)
                    labels_ = gmm.predict(g)
                    components = {}
                    print("z = [%.1f, %.1f], richness = [%.1f, %.1f]" % (zmin, zmax, rmin, rmax))
                    print("n components: ",gmm.n_components)
                    print(20*"==")
                    counter+=1
                    if gmm.n_components > 1:
                        for j in range(gmm.n_components):
                            Mi = g[labels_ == i]
                            components[j] = Mi
                        means = gmm.means_.flatten()
                        covs = gmm.covariances_.flatten()
                        weights = gmm.weights_.flatten()
                        x = np.linspace(g.min(), g.max(), 1000)
                        colors = ["darkgreen", "darkblue", "darkred", "purple", "peru"]
                        sort_indx = np.argsort(means)
                        if np.all(np.isnan(means)) == False:
                            for j,idxj in enumerate(sort_indx):
                                try:
                                    from scipy.stats import norm
                                    mean, sigma, weight = means[idxj], np.sqrt(covs[idxj]), weights[idxj]
                                    pdf_log = weight * norm.pdf(x, loc=mean, scale=sigma)
                                    pdf_M   = pdf_log / (x**10 * np.log(10))
                                    ci, cf = np.min(components[idxj]), np.max(components[idxj])
                                    hist_i = hist[np.where((bin_centers < 10**cf) & (bin_centers >= 10**ci))]
                                    re_scale_factor = np.max(hist_i)/np.max(pdf_M)
                                    ax.plot(10**x, pdf_M*re_scale_factor, lw=4, alpha=0.7, color = colors[j])
                                    ax.text(0.95, 0.89 - 0.06*(j+1), r"$\log_{10}{M} = %.2f\pm %.2f$" % (mean, sigma), 
                                        transform=ax.transAxes, fontsize=14, ha='right', va='top', color = colors[j])
                                    infered_masses[i,j,...] =  [10**mean, np.log(10)*sigma*10**mean, np.log(10)*sigma*10**mean]
                                except:
                                    continue
                    m, m_upper, m_lower = np.percentile(M, [50, 84, 16])
                    m_lower = m - m_lower
                    m_upper = m_upper - m
                    infered_masses[i, -1,...] =  [m, m_upper, m_lower]
                    y0, y1 = ax.get_ylim()
                    ax.set_ylim((y0, 1.1*y1))
                    ax.set_xscale("log")
                    label = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax)
                    ax.text(0.95, 0.95, label, transform=ax.transAxes, fontsize=14, ha='right', va='top')
                    ax.text(0.95, 0.89, r"$\log_{10}{M} = %.2f^{%.2f}_{%.2f} \mathrm{M}_{\odot}$" % (np.log10(m), m_upper/(np.log(10)*m), m_lower/(np.log(10)*m)), 
                        transform=ax.transAxes, fontsize=14, ha='right', va='top')
                else:
                    ax.axis('off')
            ncols = profiles_per_row
            axs = axes.flatten()
            for idx, ax in enumerate(axs):
                row, col = divmod(idx, ncols)
                if col != 0:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("PDF")
                if row != nrows - 1:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Mass $[\mathrm{M}_{\odot}]$")
            fig.savefig(f"{output_path}/mass_inference_likelihoods.png")
            fig, ax = plt.subplots(figsize = (18,8))
            infered_masses[infered_masses == 0] = np.nan
            richness = np.reshape(richness, (-1, 2))
            richness_errs = np.reshape(richness_errs, (-1, 2))
            for j in range(gmm.n_components + 1):
                M, M_upper, M_lower = infered_masses[:, j, :].T
                if np.all([np.isnan(Mi) for Mi in M]) == False:
                    M, M_upper, M_lower = np.reshape(M, (-1, 2)), np.reshape(M_upper, (-1, 2)), np.reshape(M_lower, (-1,2))
                    color = colors[j] if j < gmm.n_components - 1 else "black"
                    for k in range(2):
                        ls = "--" if k == 0 else "solid"
                        ax.errorbar(richness[:,k]+ 10*j, M[:,k], yerr = [M_upper[:,k], M_lower[:,k]], xerr = richness_errs[:,k],
                            ls = ls, color = color, lw = 3, alpha = 0.5)
            
            ax.set(xlabel = "richness $\lambda$", ylabel = "mass $[\mathrm{M}_{\odot}]$", yscale = "log", xscale = "linear")
            fig.savefig(f"{output_path}/richness_mass_relationship.png")

        if plot_mis_centering == True: 
            Roff = np.linspace(0, 10, 100)
            mis_centering_indxs = params_indxs[1] if len(params_indxs) == 3 else params_indxs[-1]
            params_mis_centering = chain[:, mis_centering_indxs[0] - 1:mis_centering_indxs[1]]
            mis_centering_labels = labels[mis_centering_indxs[0]-1:mis_centering_indxs[1]]
            params_mc, lower_mc, upper_mc = extract_params(params_mis_centering, mis_centering_labels, method = method, verbose = False)
            sigma, sigma_upper, sigma_lower = params_mc[1], upper_mc[1], lower_mc[1]
            sigma_upper = sigma + sigma_upper
            sigma_lower = sigma - sigma_lower
            rho_Roff = Roff**2/(2*sigma**2)*np.exp(-Roff**2 / (2*sigma**2))
            rho_Roff_lower = Roff**2/(2*sigma_upper**2)*np.exp(-Roff**2 / (2*sigma_upper**2))
            rho_Roff_upper = Roff**2/(2*sigma_lower**2)*np.exp(-Roff**2 / (2*sigma_lower**2))
            fig, ax = plt.subplots(figsize = (16,8))
            ax.plot(Roff, rho_Roff, color = 'k')
            ax.fill_between(Roff, rho_Roff_lower, rho_Roff_upper, color = 'k', alpha = 0.5)
            ax.set(xlabel = "$R_{off}$", ylabel = "$\\rho(R_{off})$")
            ax.set_title(r"$\\sigma = %.2f_{.2f}^{.2f}$" % (sigma, sigma_lower, sigma_upper))
            fig.savefig(f"{output_path}/mis_centering.png")
        if plot_hm_relationship == True:
            hm_indxs = params_indxs[-1]
            mass = np.logspace(13, 16, 100)
            redshift = 0.35
            richness_pivot = 3e14/0.7
            redshift_pivot = 0.47
            hm_params = chain[:, hm_indxs[0]-1:hm_indxs[1]]
            hm_labels = labels[hm_indxs[0]-1:hm_indxs[1]]
            hm_params, hm_lower, hm_upper = extract_params(hm_params, hm_labels, method = method, verbose = False)
            hm_lower, hm_upper = hm_params - hm_lower, hm_params + hm_upper
            fig, ax = plt.subplots(figsize = (16,8))
            richness = hm_params[0]*(mass/richness_pivot)**hm_params[1] * ((1 + redshift)/(1 + redshift_pivot))**hm_params[2]
            richness_lower = hm_lower[0]*(mass/richness_pivot)**hm_lower[1] * ((1 + redshift)/(1 + redshift_pivot))**hm_lower[2]
            richness_upper = hm_upper[0]*(mass/richness_pivot)**hm_upper[1] * ((1 + redshift)/(1 + redshift_pivot))**hm_upper[2]
            ax.plot(mass, richness, color = 'darkgreen', lw = 3)
            ax.fill_between(mass, richness_lower, richness_upper, color = 'darkgreen', alpha = 0.5)
            richness_costanzi = 30*(mass/richness_pivot)**(0.75)
            ax.plot(mass, richness_costanzi, color = 'k', lw = 5, ls = "--", label = "Costanzi et al 2019")
            ax.set(ylabel = r"Richness $\lambda$", xlabel = r"Cluster Mass $M_{200}\;[{M_\odot}]$", xscale = "log", yscale = "log")
            ax.set_title("Richness-Mass relationship")
            ax.legend()
            fig.savefig(f"{output_path}/hm_relationship.png")
        if compute_mass == True:
            lensing_convergence = True
            if lensing_convergence == True:
                pass
            pass
        if verbose == True:
            print("Parameter & value")
            for j in range(len(params)):
                print(labels_latex[j] + r"& $%.2f_{%.2f}^{%.2f}$ \\" % (params[j],lower[j],upper[j]))
        if chi2_ == True:
            chi2_values2 = blobs["CHI2"]
            chain2 = backend.get_chain()
            default_chi2_kwargs = (
                ("zoom_in", False),
                ("path", f"{output_path}/chi2"),
                ("zoom", 0.15),
                ("output_file", "chi2.png"),
                ("labels", labels)
            )
            plot_chi2(chi2_values2, chain = chain2, **set_default(kwargs.pop("chi2_kwargs",{}), default_chi2_kwargs)
            )
        if steps:     
            default_steps_kwargs = (
                ("labels", labels_latex),
                ("plot_tracers", True),
                ("output_file", f"{output_path}/parameters_steps.png"),
                ("tracers_aspect", ('k-','k--','k--')),
                ("nwalkers", nwalkers),
                ("nsteps", nsteps),
                ("discard", discard)
            )
            steps_kwargs = set_default(kwargs.pop("steps_kwargs",{}), default_steps_kwargs)
            plot_steps(chain, backend, chi2_values,**steps_kwargs)
        if corner_:
            default_corner_kwargs = (
                ("truths", params),
                ("truths_color","black"),
                ("corner_color", "blue"),
                ("output_file", f"{output_path}/corner.png"),
                ("labels",labels_latex),
                ("fontsize", 16),
                ("range_sigma_ratio", float(args.range_sigma_ratio)),
                ("title_kwargs", {"fontsize":12})
            )
            corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
            plot_corner(chain, priors = priors, priors_args = priors_args, **corner_kwargs)
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
            chi2_values2 = blobs["CHI2"]
            signal = blobs["SIGNAL"]
            flat_idx = np.nanargmin(chi2_values)
            step_idx, walker_idx = np.unravel_index(flat_idx, chi2_values2.shape)
            best_signal = signal[step_idx, walker_idx,:]
            best_signal = np.reshape(best_signal, (len(clusters), len(R)))
            np.savetxt(f"{output_path}/best_signal.txt", best_signal)
            num_profiles = len(clusters)
            if share_plot == False:
                profiles_per_row = 4
                num_rows = (num_profiles + profiles_per_row - 1) // profiles_per_row
                fig = plt.figure(figsize=(10 + 5*num_rows, 5 * num_rows))
                if args.dont_show_results == True:
                    gs = fig.add_gridspec(num_rows, profiles_per_row , wspace=0, hspace=0)
                else:
                    gs = fig.add_gridspec(num_rows, profiles_per_row + 1, wspace=0, hspace=0)
                #fig.subplots_adjust(left=0.25)  # space for text box
                axs = []
                nrows = num_rows
                ncols = profiles_per_row
                sorted_idx_redshift = np.lexsort((bins[:,1], bins[:,2]))
                for i in range(nrows):
                    row = []
                    for j in range(ncols):
                        sharex = axs[0][j] if i > 0 else None  
                        sharey = row[0] if j > 0 else None  
                        if args.dont_show_results == True:
                            ax = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
                        else:
                            ax = fig.add_subplot(gs[i, j+1], sharex=sharex, sharey=sharey)
                        row.append(ax)
                    axs.append(row)
                axs = np.reshape(axs, (nrows, ncols))     
                axes = axs.flatten()
                counter = 0
                for i in range(len(clusters)):
                    counter+=1
                    if counter <= num_profiles:
                        if sort_by_redshift == True:
                            idx = sorted_idx_redshift[i]
                        else:
                            idx = i
                        ax = axes[i]
                        c = clusters[idx]
                        zmin, zmax = redshift_bins[idx]
                        rmin, rmax = richness_bins[idx]

                        default_profiles_kwargs = (
                                ("output_file", None),
                                ("ax_kwargs", dict(xlabel = xlabel, ylabel = xlabel, xscale = xscale, yscale = yscale, ylim = (ymin, ymax),
                                                    title = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax))),
                                ("show_legend", False),
                                ("show_results", False)
                                )  
                        fit = np.nanmedian(signal2bound[:,idx,:], axis = 0)
                        lower_bound, upper_bound = np.nanpercentile(signal2bound[:,idx,:], [16, 84], axis = 0)
                        lower_bound = fit - lower_bound
                        upper_bound = upper_bound - fit 
                        best_fit = best_signal[idx]

                        profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)                
                        plot_profiles(R, c.mean_profile, func, params, c.cov, labels_latex, lower, upper,
                                    np.max(ln_likelihood), np.mean(clusters[0].z), ax = ax, fit = fit, lower_bound = lower_bound, show_invidual_chi2 = True,
                                    upper_bound = upper_bound, signal = signal2bound[:,idx,:], P1halo = P1halo[:,idx,:], P2halo = P2halo[:,idx,:],
                                    best_fit = best_fit, **profiles_kwargs)
                        ax.set_title("")
                        ax.set_ylabel("")
                        ax.set_xlabel("")
                        label = r"$\lambda \in [%.i,%.i]\;,\;z \in [%.2f, %.2f]$" % (rmin, rmax, zmin, zmax)
                        ax.text(0.95, 0.95, label, transform=ax.transAxes, fontsize=12, ha='right', va='top')
                        if args.show_individuals_chi2:
                            current_cov = c.cov
                            diag = np.sqrt(np.diag(current_cov))
                            residual = c.mean_profile - best_fit
                            current_chi2 = np.dot(residual, np.dot(np.linalg.inv(current_cov), residual.T))
                            current_chi2_no_corr = np.sum(residual**2 / diag**2)
                            current_pte = pte(current_chi2, current_cov)
                            ax.text(0.95, 0.88, "$\chi^2 = %.2f\;(%.2f)$ \n $PTE = %.4f$" % (current_chi2, current_chi2_no_corr, current_pte), 
                                transform=ax.transAxes, 
                                fontsize=12, ha='right', va='top')
                    else:
                        ax.axis('off')

                nrows = len(axs)
                ncols = len(axs[0])

                for i in range(nrows):
                    for j in range(ncols):
                        ax = axs[i][j]
                        if j == 0:
                            ax.set_ylabel(ylabel)
                        else:
                            ax.tick_params(labelleft=False)
                        if i == nrows - 1:
                            ax.set_xlabel(xlabel)
                        else:
                            ax.tick_params(labelbottom=False)
                axs = axes.flatten()
                fig.suptitle(model_name, fontsize = 18, fontweight = "bold")
                axs[0].legend(loc = "center right", fontsize = 10)
            elif share_plot == True:
                fig, ax = plt.subplots(figsize = (14,8))
                colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), size  = num_profiles)
                colors = ["darkgreen", "purple","darkblue","darkseagreen","darkred","coral","brown","orange","cyan"]
                profile_labels = []
                show_redshift = True
                val = [np.nanmedian(c.richness) + np.nanmedian(c.z) for c in clusters]
                sorted_idx = np.argsort(val)
                for i, idx in enumerate(sorted_idx):
                    color = colors[i]
                    c = clusters[idx]
                    zmin, zmax = np.nanmin(c.z), np.max(c.z)
                    rmin, rmax = int(np.nanmin(c.richness)), int(np.max(c.richness))
                    rmin = rmin + 1 if abs(int(rmin)) % 10 == 9 else rmin
                    rmax = rmax + 1 if abs(int(rmax)) % 10 == 9 else rmax
                    
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
                    fit = np.median(signal2bound[:,i,:], axis = 0)
                    lower_bound, upper_bound = np.percentile(signal2bound[:,i,:], [16, 84], axis = 0)
                    lower_bound = fit - lower_bound
                    upper_bound = upper_bound - fit 
                    plot_profiles(R + i*0.1, profiles[i], func, params, c.cov, labels_latex, lower, upper,
                                np.max(ln_likelihood), np.mean(c.z), ax = ax, fit = fit, lower_bound = lower_bound,
                                upper_bound = upper_bound, signal = signal2bound_i, show_labels = False, show_error_bars = True, **profiles_kwargs)
                ax.legend(loc = "upper right", fontsize = 10)
                ax.set_title("Best Fitting " + model_name)
                #ax.set_ylim((np.nanmin(np.array(profiles)[np.array(profiles) > 0])*0.1, np.max(profiles)*1.5))
                ax.grid(True)
                
            if args.dont_show_results == False:
                chi2 = np.nanmin(chi2_values)#calculate_chi2(raw_profiles, func(R.value,params), np.linalg.inv(cov_matrix))
                if args.use_obs_chi2 == True:
                    observed_profiles = np.nanmedian(signal, axis = 0)
                    res = raw_profiles - observed_profiles
                    chi2_obs = np.dot(np.dot(res, np.linalg.inv(cov)), res.T)
                    p_value = pte(chi2_obs, cov)
                else:
                    p_value, chi2_mc = pte(chi2, cov, return_samples= True, n_samples=10000)
                    chi2_mc = chi2_mc.flatten()
                    # mu, sigma = np.median(chi2_mc), np.std(chi2_mc)
                    # fig2, ax2 = plt.subplots(figsize = (12,6))
                    # ax2.hist(chi2_mc, bins = 100, histtype = "step", color = "black", density = True, alpha = 0.5, label = r"$\chi^2$ realizations")
                    # chi2_obs = chi2_values[chi2_values < 2000]
                    # mu_obs, sigma_obs = np.median(chi2_obs), np.std(chi2_obs)
                    # ax2.hist(chi2_obs, bins = 100, histtype = "step", color = "darkgreen", density = True, alpha = 0.5, label = r"$\chi^2$ observed")
                    # ax2.plot(np.arange(0, 1.5*np.max(chi2_mc), 0.1), np.exp(-0.5*(np.arange(0, 1.5*np.max(chi2_mc), 0.1) - mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2), color = "black", ls = "--")
                    # ax2.plot(np.arange(0, 1.5*np.max(chi2_obs), 0.1), np.exp(-0.5*(np.arange(0, 1.5*np.max(chi2_obs), 0.1) - mu_obs)**2/sigma_obs**2)/np.sqrt(2*np.pi*sigma_obs**2), color = "darkgreen", ls = "--")
                    # ax2.set_xlabel(r"$\chi^2$")
                    # ax2.set_ylabel("Density")
                    # ax2.set(yscale = "linear", xscale = "linear")
                    # ax2.set_xlim((np.clip(mu - 4*sigma, 0, np.inf), 1.5*np.max(chi2_obs)))
                    # _,ylim = ax2.get_ylim()
                    # ax2.fill_between(np.arange(mu - 3*sigma, mu + 3*sigma, 0.1), 0, ylim, color = "grey", alpha = 0.5)
                    # ax2.axvline(chi2, color = "red", label = r"$\chi^2$ best fit")
                    # ax2.legend()
                    # fig2.tight_layout()
                    # fig2.savefig(output_path + f"chi2_realizations.png")
                bic = BIC(np.size(profiles), len(params), max_ln_likelihood)
                text  = [
                        r'$\chi^{2} = %.4f$' % chi2,
                        r'$PTE = %.6f$' % p_value,
                ]
                for i in range(len(labels)):
                    if labels[i].split('_')[0] == r'$\log':
                        text.append(f'{labels_latex[i]} : {np.round(np.log10(params[i]),2)} $\pm$ {np.round(err[i]/(np.log(10) * paramss[i]),2)}')
                    else:
                        text.append('%s' % labels_latex[i] + ': $%.2f' % params[i] + '^{+%.2f}_{-%.2f}$' % (np.abs(upper[i]),np.abs(lower[i])))   
                s = '\n'.join(text)
                if share_plot:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    #0.15, 0.6
                    ax.text(0.8, 0.8, s, fontsize=13, verticalalignment='top', ha = "left", transform=ax.transAxes, bbox=props, color = 'black')
                else:
                    props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black', alpha = 0.8)
                    fig.text(0.12, 0.9, s, fontsize=16, va='top', ha='right', family='monospace', bbox=props, color = 'black',)     
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
                axes, corner_fig = plot_corner(schain, priors = priors, priors_args = priors_args, **corner_kwargs)
                axes[-1][-1].scatter([],[], color = corner_color, marker = "s", label = f"N steps $=[{d1},{d2}]$")
        if corner_:
            corner_fig.legend(fontsize = 16)
            corner_fig.savefig(f"{output_path}/corner.png", dpi = args.dpi)   
    if make_copy == True:
        os.remove(copy)
def plot_mcmc(source_path, model, labels, ndims = None, nwalkers = None, fil_name = 'mcmc_samples', ext = 'h5', fig_corner = None, fig_profile = None,
              return_cluster = False, plot = False, use_signal = False, discard = 0, steps = False, corner_ = False, make_copy = True,
              method = "median", tau = False, thin = 1, filters = None, use_filters = False, rbins = None, zbins = None, Mbins = None,
              xlabel = None, ylabel = None, model_name = "", return_signal = True, plot_cov = False, plot_corr = False, 
              priors = None, priors_args = None, **kwargs):
    labels_latex = [text2latex(l) for l in labels]
    samples_file = source_path + '/' + fil_name + '.' + ext
    group = grouped_clusters.load_from_path(source_path)
    R = group.R

    func = group.stacked_halo_model_func(model)

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
        blobs_keys = blobs.dtype.base.names
        chi2_values = blobs['CHI2'].flatten()
        signal = blobs['SIGNAL'].reshape((len(chi2_values),-1))
        P1halo = blobs['ONE_HALO'].reshape((len(chi2_values),-1)).reshape((-1, len(R))) if "ONE_HALO" in blobs_keys else np.full(signal.shape, np.nan)
        P2halo = blobs['TWO_HALO'].reshape((len(chi2_values),-1)).reshape((-1, len(R))) if "TWO_HALO" in blobs_keys else np.full(signal.shape, np.nan)
        Masses = blobs['MASS'] if "MASS" in blobs_keys else np.full(signal.shape, np.nan)
        chi2_values = chi2_values[np.where((chi2_values < np.inf) & (chi2_values > -np.inf) & (chi2_values < 1e20))]
        P1halo = np.reshape(P1halo, (-1, len(R)))
        P2halo = np.reshape(P2halo, (-1, len(R)))
        signal2bound = np.reshape(signal,(-1, len(R))) if use_signal else None
        signal2bound = P1halo + P2halo if "ONE_HALO" in blobs_keys and "TWO_HALO" in blobs_keys else signal2bound
        ln_prior = blobs['LN_PRIOR'].flatten()
        ln_likelihood = blobs['LN_LIKELIHOOD'].flatten()
        max_ln_likelihood = np.max(ln_likelihood)
        chain = backend.get_chain(discard = discard, thin = thin, flat = True)
        params, lower, upper = extract_params(chain, labels, method = method)

        chi2_values2 = blobs["CHI2"]
        signal = blobs["SIGNAL"]
        flat_idx = np.nanargmin(chi2_values)
        step_idx, walker_idx = np.unravel_index(flat_idx, chi2_values2.shape)
        best_signal = signal[step_idx, walker_idx,:]
        fit = np.nanmedian(signal2bound, axis = 1)
        lower_bound,upper_bound = np.nanpercentile(signal2bound, [16,84], axis = 1)
        
        chi2 = np.nanmin(chi2_values)
        if args.infere_mass and "MASS" in blobs_keys:
            fig, ax = plt.subplots(figsize = (16,8))
            M = Masses
            bins = np.logspace(13, 17, 50)
            hist, bins_edges = np.histogram(M, bins = bins, density = True)
            hist = np.nan_to_num(hist)
            hist = gaussian_filter1d(hist, sigma = 3, mode = "constant", cval = 0.0)
            hist = hist/np.max(hist)
            ax.stairs(hist, bins, color = "black", lw = 3, alpha = 0.7)
            y0,y1 = ax.get_ylim()
            ax.set_ylim((y0, 1.1*y1))
            ax.set_xscale("log")
            m, m_lower, m_upper = np.nanpercentile(Masses, [50, 16, 84])
            m_lower = m - m_lower
            m_upper = m_upper - m
            ax.text(0.95, 0.89, r"$\log_{10}{M} = %.2f^{%.2f}_{%.2f} \mathrm{M}_{\odot}$" % (np.log10(m), m_upper/(np.log(10)*m), m_lower/(np.log(10)*m)), 
            transform=ax.transAxes, fontsize=14, ha='right', va='top')
            ax.axvline(m, color = "black", lw = 2, alpha = 0.7)
            fig.savefig(f"{source_path}/masses.png")
        if steps:     
            default_steps_kwargs = (
                ("plot_tracers", True),
                ("discard", discard),
                ("nsteps", nsteps),
                ("nwalkers", nwalkers),
                ("output_file", f"{source_path}/parameters_steps.png"),
                ("tracers_aspect", ('k-','k--','k--'))
            )
            steps_kwargs = set_default(kwargs.pop("steps_kwargs",{}), default_steps_kwargs)
            plot_steps(chain, backend, chi2_values, labels_latex,**steps_kwargs)
        if plot:
            extra_text = [
                r'N clusters = $%.i$' % (len(group.richness)),
                r'$\mathrm{richness} = [%.i,%.i]$' %   (np.nanmin(group.richness), np.max(group.richness)),
                r'$\mathrm{redshift} = [%.2f, %.2f]$' % (np.nanmin(group.z),np.max(group.z))
            ]
            color = "black"
            default_profiles_kwargs= (("output_file", None),
                ("show_legend", False),
                ("fit_plot_kwargs", {"color": color, "label" : None}),
                ("bounds_plot_kwargs", {"color": color, "alpha": 0.2, "label" : None}),
                ("data_plot_kwargs", {"color": color}),
                ("show_results", True),
                ("ax_kwargs", dict(xlabel = xlabel, ylabel = ylabel, yscale = yscale, xscale = xscale, 
                    title = r"$\lambda \in [%.i, %.i]\;,\; z\in[%.2f, %.2f]$" % (np.nanmin(group.richness), np.max(group.richness),
                    np.nanmin(group.z), np.max(group.z))))
            )
            profiles_kwargs = set_default(kwargs.pop("profiles_kwargs",{}), default_profiles_kwargs)
            ax,fig, _, bic = plot_profiles(group.R, group.mean_profile, func, params, group.cov, labels_latex, lower, upper, 
                          np.max(ln_likelihood), np.mean(group.z), min_chi2 = chi2, lower_bound = lower_bound, upper_bound = upper_bound, 
                          signal = signal2bound, fit = fit, P1halo = P1halo, P2halo = P2halo, best_fit = best_signal, **profiles_kwargs)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(f"{source_path}/best_fitting.png", dpi = args.dpi)  

            chi2 = np.nanmin(chi2_values)
            p_value, chi2_mc = pte(chi2, group.cov, return_samples= True, n_samples=1000000)
            chi2_mc = chi2_mc.flatten()
            mu, sigma = np.median(chi2_mc), np.std(chi2_mc)
            fig2, ax2 = plt.subplots(figsize = (12,6))
            ax2.hist(chi2_mc, bins = 100, histtype = "step", color = "black", density = True, alpha = 0.5, label = r"$\chi^2$ realizations")
            chi2_obs = chi2_values[ chi2_values < 50]
            mu_obs, sigma_obs = np.median(chi2_obs), np.std(chi2_obs)
            ax2.hist(chi2_obs, bins = 100, histtype = "step", color = "darkgreen", density = True, alpha = 0.5, label = r"$\chi^2$ observed")
            ax2.plot(np.arange(0, 1.5*np.max(chi2_mc), 0.1), np.exp(-0.5*(np.arange(0, 1.5*np.max(chi2_mc), 0.1) - mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2), color = "black", ls = "--")
            ax2.plot(np.arange(0, 1.5*np.max(chi2_obs), 0.1), np.exp(-0.5*(np.arange(0, 1.5*np.max(chi2_obs), 0.1) - mu_obs)**2/sigma_obs**2)/np.sqrt(2*np.pi*sigma_obs**2), color = "darkgreen", ls = "--")
            ax2.set_xlabel(r"$\chi^2$")
            ax2.set_ylabel("Density")
            ax2.set(yscale = "linear", xscale = "linear")
            ax2.set_xlim((np.clip(mu - 4*sigma, 0, np.inf), 1.5*np.max(chi2_obs)))
            _,ylim = ax2.get_ylim()
            ax2.fill_between(np.arange(mu - 3*sigma, mu + 3*sigma, 0.1), 0, ylim, color = "grey", alpha = 0.5)
            ax2.axvline(chi2, color = "red", label = r"$\chi^2$ best fit")
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(group.output_path + f"chi2_realizations.png") 
        if corner_:
            default_corner_kwargs = (
                ("truths", params),
                ("truths_color","black"),
                ("corner_color", "blue"),
                ("output_file", f"{source_path}/corner.png"),
                ("labels",labels_latex),
                ("fontsize", 16),
                ("range_sigma_ratio", float(args.range_sigma_ratio)),
                ("title_kwargs", {"fontsize":12}),
                ("plot_priors", True),
            )
            corner_kwargs = set_default(kwargs.pop("corner_kwargs",{}), default_corner_kwargs)
            plot_corner(chain, priors = priors, priors_args = priors_args, **corner_kwargs)
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
                corner_label = None, corner_color = "blue", other = None, other_label = None, 
                other_color = None, levels = (1-np.exp(-0.5), 1-np.exp(-2) ), bins = 30,  fontsize = 14,
                labels = None, show_labels = False, quantiles = [0.16, 0.5, 0.84] , alpha = 0.5,
                range_sigma_ratio = 4, plot_priors = True, priors = None, priors_args= None, **kwargs):
    default_title_kwargs = (
        ("fontsize", 20),
    )
    default_fig_kwargs = (
        ("figsize", (2*len(truths),2*len(truths))),
    )
    default_bins_kwargs = (
        ('bins',30),
        ('lw', 4),
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
    smooth = 1 if args.smooth_corner == True else None
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
        smooth = smooth,
        smooth1d = smooth,
        fill_contours = True,
        hist_bin_factor= 5,
        n_max_ticks = 5,
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
    if plot_priors == True:
        if priors is not None and priors_args is not None:
            counter = 0
            for i in range(ndims):
                for j in range(i + 1):
                    if i == j:
                        prior, pargs = priors[i], priors_args[i]
                        if prior is None or pargs is None:
                            continue
                        else:
                            if "uniform" in str(prior) or "flat" in str(prior):
                                continue
                            line = axes[i][j].lines[0]
                            xdata,ydata = np.array(line.get_xdata()), np.array(line.get_ydata())
                            xnew = np.linspace(xdata.min(), xdata.max(), 1000)
                            p = np.exp(np.array([prior(xnew_i, *pargs) for xnew_i in xnew]))
                            p = p / p.max() * ydata.max()
                            axes[i][j].fill_between(xnew, p, color = 'darkgreen', lw = 3, alpha = 0.2, edgecolor = "darkgreen")

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

def plot_chi2(chi2, steps=None, zoom=0.05, path=None, output_file=None, chain=None, labels=None,
        interpolate=True, smooth=True, show_scatter=False, thin=1, hist_bins=30, zoom_in = False, **kwargs):
    """
    Plot historical min chi2 and per-parameter χ² vs parameter with glued histograms.
    Zoom regions added as inset axes.
    """
    if interpolate and smooth:
        interpolate = False
        print("smooth must be False if interpolate is True. Setting interpolate to False!")

    figsize = (12, 8)
    default_fig_kwargs = (("figsize", figsize),)
    default_ax_kwargs = (("xlabel", "step"), ("yscale", "linear"), ("xscale", "linear"), ("ylabel", "$\\chi^2_{\\mathrm{min}}$"),)
    default_plot_kwargs = (("color", "black"), ("lw", 3),)
    default_scatter_kwargs = (("s", 30), ("marker", "o"), ("color", "black"), ("alpha", 0.9),)
    default_suptitle_kwargs = (("fontsize", 22), ("t", "$\\chi^2_{\\mathrm{min}}$ history"), ("fontweight", "bold"),)

    if path and not os.path.exists(path):
        os.mkdir(path)

    fig_kwargs = set_default(kwargs.pop("fig_kwargs", {}), default_fig_kwargs)
    ax_kwargs = set_default(kwargs.pop("ax_kwargs", {}), default_ax_kwargs)
    suptitle_kwargs = set_default(kwargs.pop("suptitle_kwargs", {}), default_suptitle_kwargs)
    plot_kwargs = set_default(kwargs.pop("plot_kwargs", {}), default_plot_kwargs)
    scatter_kwargs = set_default(kwargs.pop("scatter_kwargs", {}), default_scatter_kwargs)
    # Main χ² history
    min_chi2 = np.nanmin(chi2, axis=1)
    steps = np.arange(len(min_chi2)) if steps is None else steps
    if thin > 1:
        min_chi2 = min_chi2[::thin]
        steps = steps[::thin]
    fig, ax_main = plt.subplots(**fig_kwargs)
    if not interpolate:
        if smooth:
            min_chi2 = gaussian_filter(min_chi2, sigma=5)

        ax_main.plot(steps, min_chi2, **plot_kwargs)
    else:
        fchi2 = UnivariateSpline(steps, min_chi2)
        new_steps = np.linspace(steps[0], steps[-1], 1000)
        ax_main.plot(new_steps, fchi2(new_steps), **plot_kwargs)
        if show_scatter:
            ax_main.scatter(steps, min_chi2, **scatter_kwargs)
            
    ax_main.set(**ax_kwargs)
    fig.suptitle(**suptitle_kwargs)
    fig.tight_layout()

    # Zoom inset
    if zoom_in:
        zoom_start = steps[-1] * (1 - zoom)
        mask = steps >= zoom_start
        zs, zchi = steps[mask], min_chi2[mask]
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        inset = inset_axes(ax_main, width="20%", height="20%", loc='upper right')
        if not interpolate:
            inset.plot(zs, zchi, **plot_kwargs)
        else:
            new_zs = np.linspace(zs[0], zs[-1], 200)
            inset.plot(new_zs, fchi2(new_zs), **plot_kwargs)
            if show_scatter:
                inset.scatter(zs, zchi, **scatter_kwargs)
        mark_inset(ax_main, inset, loc1=2, loc2=4, fc="none", ec="0.5")

    if output_file:
        savepath = (path + "/" + output_file) if path else output_file
        fig.savefig(savepath, dpi=kwargs.get('dpi', plt.rcParams['savefig.dpi']), transparent=False)

    # Per-parameter χ² vs parameter with glued histogram
    if chain is not None and labels is not None:
        chi2_th = chi2[::thin]
        chain_th = chain[::thin]
        min_idx = np.argmin(chi2_th, axis=1)
        params_at_min = chain_th[np.arange(len(min_idx)), min_idx, :]
        base_title = suptitle_kwargs['t']

        for i, p in enumerate(labels):
            # use GridSpec to glue histogram above main ax
            fig = plt.figure(**fig_kwargs)
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
            ax_hist = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1], sharex=ax_hist)

            x = params_at_min[:, i]
            sort_idx = np.argsort(x)
            xs, ys = x[sort_idx], min_chi2[sort_idx]
            ys = gaussian_filter(ys, sigma=5) if (smooth == True and interpolate == False) else ys
            # histogram on top
            ax_hist.hist(x, bins=hist_bins)
            ax_hist.set_ylabel('count')
            ax_hist.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            # chi2 vs param below
            if not interpolate:
                ax.plot(xs, ys, **plot_kwargs)
            else:
                pars, _ = curve_fit(lambda xx, a, b, c: a*xx**2 + b*xx + c, xs, ys,
                                    p0=[1,1,1], maxfev=int(1e6))
                xp = np.linspace(xs[0], xs[-1], 1000)
                ax.plot(xp, pars[0]*xp**2 + pars[1]*xp + pars[2], **plot_kwargs)
                if show_scatter:
                    ax.scatter(xs, ys, **scatter_kwargs)
            ax.set(**ax_kwargs)
            ax.set_xlabel(text2latex(p))

            # zoom inset
            perc_lo, perc_hi = np.percentile(xs, [100*zoom/2, 100*(1-zoom/2)])
            zmask = (xs >= perc_lo) & (xs <= perc_hi)
            zx, zy = xs[zmask], ys[zmask]
            if zoom_in == True:
                zoom_inset = inset_axes(ax, width="20%", height="20%", loc='upper right')
                zoom_inset.plot(zx, zy, **plot_kwargs)
                zoom_inset.set_xlim(zx.min(), zx.max())
                zoom_inset.set_xscale("linear")
                mark_inset(ax, zoom_inset, loc1=1, loc2=3, fc="none", ec="0.5")

            fig.suptitle(f"{base_title} for {text2latex(p)}", **{'fontsize': suptitle_kwargs['fontsize'], 'fontweight': suptitle_kwargs['fontweight']})
            fig.tight_layout()

            if output_file:
                pname = p.replace('{','').replace('}','').replace('_','').replace('\\','')
                fname = f"chi2_history_{pname}.png"
                save = (path + "/" + fname) if path else fname
                fig.savefig(save, dpi=kwargs.get('dpi', plt.rcParams['savefig.dpi']), transparent=False)

    plt.close('all')


def plot_steps(chain, backend, chi2_values, labels, plot_tracers = False, tracers_aspect = 'k-', 
               output_file = None, nwalkers = 50, cmap = 'viridis', remove_inf = True, log10 = True, 
               discard = None, **kwargs):
    default_fig_kwargs = (
        ("figsize",(6 + len(labels),2*len(labels))),
        ("sharex", True),
        ("constrained_layout", True),
    )
    cbar_label = "$\ln{\chi^2}$ value" if log10 == False else r"$\log_{10}{\chi^2}$ value"
    default_colorbar_kwargs = (
        ("label",cbar_label),
        )
    default_title_kwargs = (
        ("t", "MCMC steps plot"),
        ("fontsize", 24),
    )

    chi2_values[np.logical_not(np.isfinite(chi2_values))] = np.nan if remove_inf == True else chi2_values

    fig_kwargs = set_default(kwargs.pop("fig_kwargs",{}), default_fig_kwargs)
    colorbar_kwargs = set_default(kwargs.pop("colorbar_kwargs", {}),default_colorbar_kwargs)
    title_kwargs = set_default(kwargs.pop("title_kwargs", {}),default_title_kwargs)

    tracers_aspect = np.tile(tracers_aspect, 3) if np.iterable(tracers_aspect) == False else tracers_aspect
    
    tsteps, npar = chain.shape
    stepbin = nwalkers if nwalkers < tsteps else tsteps
    stepbins = np.arange(0, tsteps, stepbin) + stepbin / 2
    runmeb = np.transpose(
    [
        np.nanmedian(chain[k : k + stepbin], axis=0)
        for k in range(0, tsteps, stepbin)
    ]
    )
    runlo, runhi = np.transpose(
    [
        np.nanpercentile(chain[k : k + stepbin], [16, 84], axis=0)
        for k in range(0, tsteps, stepbin)
    ],
    axes=(1, 2, 0),
    )
    discard = 0 if discard is None else discard
    log_prob = backend.get_log_prob(discard = discard, flat = True)
    ndims = len(labels)
    cmap_samples = 5000
    chi2_values = np.log(chi2_values) if log10 == False else np.log10(chi2_values)
    j = np.arange(tsteps)
    colors, cmap = colorscale(
    chi2_values,
    vmin = np.nanmin(chi2_values[-cmap_samples:]),
    vmax = np.nanpercentile(chi2_values[-cmap_samples:], 99),
    cmap = cmap,
    )
    fig, axes = plt.subplots(
        ndims, 1,**fig_kwargs
    )

    for ax, p, m, lo, hi, label in zip(axes, chain.T, runmeb, runlo, runhi, labels):
        ax.scatter(j[::1], p[::1], c=colors[:len(j[::1])], marker=".", s=0.2)
        if plot_tracers == True:
            ax.plot(stepbins, m, tracers_aspect[0], lw=1.5)
            ax.plot(stepbins, lo, tracers_aspect[1], lw=1.2)
            ax.plot(stepbins, hi, tracers_aspect[2], lw=1.2)
        ax.set(ylabel=label)

    fig.tight_layout()

    plt.subplots_adjust(right=0.82, left=0.18)
    bb_top = axes[1].get_position()
    bb_bot = axes[-1].get_position()
    cbar_x = bb_top.x1 + 0.01          
    cbar_y = bb_bot.y0
    cbar_w = 0.02         
    cbar_h = bb_top.y1 - bb_bot.y0

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])

    plt.colorbar(cmap, cax = cax, **colorbar_kwargs)

    fig.suptitle("MCMC step", fontsize = 16, fontweight = "bold")
    #fig.suptitle(**title_kwargs)
    if plot_tracers == True:
        axes[0].plot([],[], tracers_aspect[0], label = "median")
        axes[0].plot([],[], tracers_aspect[1], label = "lower bound")
        axes[0].plot([],[], tracers_aspect[2], label = "upper bound")
        axes[0].legend(
        loc='center right',
        bbox_to_anchor=(-0.3, 0.5),
        frameon=True
        )   
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi)   
    return axes, fig

def plot_profiles(R, data, model, params, cov, labels, lower, upper, max_ln_likelihood = 0 , z = 0, fig = None, ax = None,
                  output_file = None, fit = None, best_fit = None, lower_bound = None, upper_bound = None, show_legend = False,
                  show_results = False, signal = None, plot_bounds = True, min_chi2 = None, show_error_bars = True, 
                  specific_pte = None, specific_chi2 = None, P1halo = None, P2halo = None, **kwargs):
    default_fig_kwargs = (
        ("figsize",(8,8)),    
    )
    default_data_plot_kwargs = (
        ("capsize", 5),
        ("color", "black"),
        ("fmt", 'o'),
        ("label", "data")
    )
    default_fit_plot_kwargs = (
        ("label", "median"),
        ("color", "black"),
        ("ls", "solid"),
        ("lw", 5)
    )
    default_best_fit_plot_kwargs = (
        ("label", "best fit"),
        ("color", "darkgreen"),
        ("ls", "solid"),
        ("lw", 5),
        ("alpha", 0.8)
    )
    default_bounds_plot_kwargs = (
        ("color","grey"),
        ("alpha",0.3),
        ("label",r"$1 \sigma$")
    )
    default_text_kwargs = (
        ("fontsize", 11),
        ("verticalalignment", 'bottom'),
        ("ha", "left"),
        ("color","black"),
        ("extra_text", None),
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
    best_fit_plot_kwargs = set_default(kwargs.pop("best_fit_plot_kwargs",{}), default_best_fit_plot_kwargs)

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
    if best_fit is not None:
        ax.plot(R, best_fit, **best_fit_plot_kwargs)
    if plot_bounds == True:
        if signal is None:
            lower_bound = model(R,np.array(params) - np.array(lower)) if lower_bound is None else np.array(lower_bound)
            upper_bound = model(R,np.array(params) + np.array(upper)) if upper_bound is None else np.array(upper_bound)
        elif signal is not None:
            lower_bound, upper_bound = np.nanpercentile(signal, [16,84], axis = 0)
            fit = np.nanmedian(signal, axis = 0)
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
        if P1halo is not None:
            if np.all(np.isnan(P1halo)) == False:
                P1halo = np.nanmedian(P1halo, axis = 0)
                fit_plot_kwargs["ls"] = "dashed"
                fit_plot_kwargs["label"] = r"1h"
                fit_plot_kwargs["alpha"] = 0.5
                ax.plot(R, P1halo, **fit_plot_kwargs)
        if P2halo is not None:
            if np.all(np.isnan(P2halo)) == False:
                P2halo = np.nanmedian(P2halo, axis = 0)
                fit_plot_kwargs["ls"] = "dotted"
                fit_plot_kwargs["label"] = r"2h"
                fit_plot_kwargs["alpha"] = 0.5
                ax.plot(R, P2halo, **fit_plot_kwargs)
    if show_results == True:
        if specific_chi2 is not None and specific_pte is not None:
            text  = [
                    r'$\chi^{2} = %.3f \; (%.3f)$' % (chi, specific_chi2),
                    r'$PTE = %.4 \; (%.4f)$' % (p_value, specific_p_value)
            ]
        else:
            text  = [
                r'$\chi^{2} = %.4f$' % round(chi2,2),
                r'$PTE = %.4f$' % round(p_value,2)
            ]
        text = text + list(text_kwargs["extra_text"]) if text_kwargs["extra_text"] is not None else text
        for i in range(len(labels)):
            text.append('%s' % labels[i] + ': $%.2f' % params[i] + '^{+%.2f}_{-%.2f}$' % (np.abs(upper[i]),np.abs(lower[i])))   
        text = '\n'.join(text)
        text_kwargs.pop("extra_text",None)
        if args.dont_show_results == False:
            ax.text(0.05, 0.05, text, transform=ax.transAxes, **text_kwargs)
    ax.set(**ax_kwargs)
    if show_legend == True:
        ax.legend(loc = "lower left")
    if output_file is not None:
        fig.savefig(output_file, dpi = args.dpi, transparent = False)   
    return ax, fig, chi2, bic

def plot_cov_matrix(chain, labels, output_file = None, corr = False, norm = "linear", fig = None, 
                    compute_p_values = False, ax = None, **kwargs):
    default_fig_kwargs = (
        ("figsize",(12,12)),    
    )
    default_ax_kwargs = (
        ("xlabel", "Parameter"),
        ("ylabel", "Parameter")
    )
    default_imshow_kwargs = (
        ("cmap", "Reds"),
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
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center", color="grey", fontsize=12)

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


