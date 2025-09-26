import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager
import emcee
import astropy.units as u
import os
import corner
import matplotlib.pyplot as plt
import cProfile
from configparser import ConfigParser
from helpers import *
import profiles
import importlib
import warnings
import emcee
from time import time
from cluster_data import *
from plottery.plotutils import colorscale
import matplotlib.patheffects as path_effects
import re
from matplotlib.colors import LogNorm, SymLogNorm
import sys
import argparse
import multiprocessing as mp
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from icecream import ic


warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path + "/config.ini"
config = ConfigParser()
config.optionxform = str

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

data_path = config["FILES"]["DATA_PATH"]
profile_stacked_model = config["STACKED_HALO_MODEL"]["profile"]
profiles_module = importlib.import_module("profiles")
MCMC_func = importlib.import_module("MCMC_functions")

completeness_file = config["FILES"]["COMPLETENESS"]
cosmological_model = dict(config["COSMOLOGICAL MODEL"])
cosmological_model = {
    key: float(cosmological_model[key]) for key in list(cosmological_model.keys())
}

parser = argparse.ArgumentParser()

parser.add_argument("--all_data","-a", action = "store_true", help = "If true fit all the data given a general function.")
parser.add_argument("--joint_fit", "-j", action = "store_true", help = "Ajust two of more data set at the same time.")
parser.add_argument("--path", "-p", type = str, default = None, help = "An specific path where look the cluster data.")
parser.add_argument("--redshift_bins", "-R", default = None, help = "Redshift bin to general fit.")
parser.add_argument("--PRIOR_CONFIG","-C", type = str, default = "PRIORS", help = "Key in config.ini file that define the priors.")
parser.add_argument("--EMCEE_CONFIG", "-E", type = str, default = "EMCEE CONFIG", help = "Key in config.ini file that determinate parameters about emcee sampler.")
parser.add_argument("--MODEL_CONFIG", "-M", type = str, default = "MODEL CONFIG", help = "Which key in the config.ini file have information about the physical profile model.")
parser.add_argument("--BLOBS_CONFIG", "-B", type = str, default = "BLOBS", help = "This argument specifies which key in config.ini file correspond to emcee blobs")
parser.add_argument("--ncores","-n", type = int, default = 1, help = "How many CPU cores will be use to multiprocessing (1 as predeterminate value)")
parser.add_argument("--plot_cov_matrix", "-PC", action = "store_true", help = "plot general cov matrix")
parser.add_argument("--r_units", "-u", default = "arcmin")
parser.add_argument("--CONFIG_FILE", "-CF", default = None, help = "Configuration file to extract the PRIORS, EMCEE, MODEL and BLOBS config.")
parser.add_argument("--COMPLETENESS_CONFIG", "-CC", type = str, default = "COMPLETENESS CONFIG", help = "Key in config.ini file that determinate the completeness function.")
parser.add_argument("--ask_to_add", "-Y", action = "store_false", help = "If passed wont be asked to add new clusters to the group and will be assumed the existence of a ignore.txt file in the path")
parser.add_argument("--demo-mode", "-D", action = "store_true", help = "If passed, the code will run in demo mode, evaluating the model for few parameters randomly chosen.")
parser.add_argument("--ndemos", "-ND", type = int, default = 5, help = "Number of demos to run in demo mode. Default is 10.")
parser.add_argument("--demo-path", "-DP", type = str, default = "demo_results", help = "Path to save the demo files. Default is current_path/demo_results/.")
parser.add_argument("--use-float32", "-F", action = "store_true", help = "Use float32 to speed up the code.")
parser.add_argument("--compute-correlations", "-CCR", action = "store_true", help = "Compute correlation matrix between observables when the code runs in joint mode. Default is False.")
parser.add_argument("--debug", "-d", action = "store_true", help = "Activate debug mode.")

args = parser.parse_args()

debug = args.debug

float_dtype = np.float64 if not args.use_float32 else np.float32
global compute_chi2
def compute_chi2(y,y1,sigma):
    if len(np.shape(sigma)) == 1:
        y,y1,sigma = np.array(y, dtype = float_dtype),np.array(y1, dype = float_dtype),np.array(sigma, dype = float_dtype)
        cov_inv = sigma
        res = np.array(y - y1, dtype = float_dtype)
        chi = np.dot(np.dot(res.T,cov_inv),res)
        if chi <= 0:
            chi = np.inf
        return chi

global ln_posterior_func 
def ln_posterior_func(theta, x, y, sigma, **kwargs):
    if y.ndim > 1:
        ic(y)
    ln_prior_func = kwargs["ln_prior"]
    ln_likelihood_func = kwargs["ln_likelihood"]
    chi2_func = kwargs["chi2"]
    lp = ln_prior_func(theta)
    infere_mass = kwargs["infere_mass"]
    store_two_halo_term = kwargs["store_two_halo_term"]
    if infere_mass == True and store_two_halo_term == True:
        likelihood, current_chi2, y1 , one_halo, two_halo, mass = ln_likelihood_func(theta, x, y, sigma, **kwargs)
    elif infere_mass == True and store_two_halo_term == False:
        likelihood, current_chi2, y1 , mass = ln_likelihood_func(theta, x, y, sigma, **kwargs)
    elif infere_mass == False and store_two_halo_term == True:
        likelihood, current_chi2, y1 , one_halo, two_halo = ln_likelihood_func(theta, x, y, sigma, **kwargs)
    elif infere_mass == False and store_two_halo_term == False:
        likelihood, current_chi2, y1 = ln_likelihood_func(theta, x, y, sigma, **kwargs)

    posterior = likelihood + lp
    current_chi2 = np.inf if np.all(np.isnan(y1)) or current_chi2 < 0 else current_chi2
    posterior = -np.inf if np.isnan(posterior) or not np.isfinite(posterior) else posterior
    prior = -np.inf if np.isnan(posterior) or not np.isfinite(posterior) else lp
    likelihood = -np.inf if np.isnan(posterior) or not np.isfinite(posterior) else likelihood
    if infere_mass == True:
        if store_two_halo_term == True:
            return posterior, current_chi2, lp, likelihood, y1, one_halo, two_halo, mass
        else:
            return posterior, current_chi2, lp, likelihood, y1, mass
    else:
        if store_two_halo_term == True:
            return posterior, current_chi2, lp, likelihood, y1, one_halo, two_halo
        else:
            return posterior, current_chi2, lp, likelihood, y1


global demo_worker
def demo_worker(params, R, profiles, sampler_kwargs, N_total, counter):
    N_evals = len(params)
    ln_lks, chi2s, mu_s = np.zeros(N_evals), np.zeros(N_evals), np.zeros(N_evals, dtype = object)
    for j,p in enumerate(params):
        ln_lks[j], chi2s[j],  mu_s[j]= ln_likelihood_general(p, R, profiles, None , **sampler_kwargs)
        counter.value+=1
        sys.stdout.write(f"\rEvaluating grid: ({counter.value} / {N_total})")
        sys.stdout.flush()
    return ln_lks, chi2s, mu_s

def main():
    joint_fit = args.joint_fit
    if joint_fit == False:

        specific_path = args.path
        fit_entire_data = args.all_data
        if args.CONFIG_FILE is None:
            fit_entire_data = args.all_data
            prior_config = config[args.PRIOR_CONFIG]
            emcee_config = config[args.EMCEE_CONFIG]
            model_config = config[args.MODEL_CONFIG]
            completeness_config = config[args.COMPLETENESS_CONFIG]
            blobs_config = config[args.BLOBS_CONFIG]
            ncores = None
        else:
            current_path = os.path.dirname(os.path.realpath(__file__))
            config_filepath = current_path +"/"+ str(args.CONFIG_FILE)
            config = ConfigParser()
            config.optionxform = str
            if os.path.exists(config_filepath):
                config.read(config_filepath)
            else:
                raise Found_Error_Config(f"The config file {str(args.CONFIG_FILE)} doesn't exist")
            prior_config = config["PRIORS"]
            emcee_config = config["EMCEE"]
            model_config = config["MODEL"]
            blobs_config = config["BLOBS"]

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


            completeness_config = dict(config["COMPLETENESS"])
            for k in list(completeness_config.keys()):
                if completeness_config[k] in ("True", "False"):
                    completeness_config[k] = str2bool(completeness_config[k])
                elif "," in completeness_config[k]:
                    completeness_config[k] = prop2arr(completeness_config[k], dtype = float)
                elif "dict" in completeness_config[k]:
                    completeness_config[k] = eval(completeness_config[k])

            completeness_func_kwargs = dict(
                Nlambda_true = int(completeness_config["Nlambda_true"]),
                max_lambda_true = float(completeness_config["max_lambda_true"]),
                min_lambda_true = float(completeness_config["min_lambda_true"]),
                Nlambda_obs = int(completeness_config["Nlambda_obs"]),
                Nz_lambda = int(completeness_config["zbins"]),
                function = completeness_config["func"],
                func_kwags = completeness_config.get("func_kwargs", {}),
            )

            completeness_config["function_kwargs"] = completeness_func_kwargs
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
                        arg = list(np.array(arr[:-1], dtype = float_dtype))
                        arg[-1] = int(arg[-1])
                        two_halo_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                    else:
                        two_halo_kwargs[k] = np.array(arr, dtype = float_dtype)
                elif "dict" in two_halo_kwargs[k]:
                    two_halo_kwargs[k] = eval(two_halo_kwargs[k])
            if "two_halo_power_func" in list(two_halo_kwargs.keys()):
                two_halo_kwargs["two_halo_power_func"] = getattr(profiles_module, two_halo_kwargs["two_halo_power_func"])
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
                        arg = list(np.array(arr[:-1], dtype = float_dtype))
                        arg[-1] = int(arg[-1])
                        mis_centering_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                    else:
                        mis_centering_kwargs[k] = np.array(arr, dtype = float_dtype)
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
            n_mis_centering_params = len(prior_parameters_mc)

        two_halo_kwargs["cosmo"] = ccl.CosmologyVanillaLCDM()

        plot_cov_matrix = args.plot_cov_matrix
        ncores = str(args.ncores) if ncores is None else ncores
        r_units = args.r_units

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

        model_id = model_config["name"]
        demo = args.demo_mode
        demo_path = args.demo_path

        off_diag = str2bool(model_config["off_diag"]) if "off_diag" in list(dict(model_config)) else False

        richness_pivot = float(model_config["richness_pivot"])
        redshift_pivot = float(model_config["redshift_pivot"])

        store_two_halo_term = str2bool(emcee_config["store_two_halo_term"])

        profile_stacked_model = model_config["profile"]
        rewrite = str2bool(emcee_config["rewrite"])
        likelihood_func = emcee_config["likelihood"]
        nwalkers = int(emcee_config["nwalkers"])
        nsteps = int(emcee_config["nsteps"])
        del_backend = str2bool(emcee_config["delete"])
        rotate_cov_matrix = str2bool(model_config["rotate_cov_matrix"])
        use_filters = str2bool(model_config["use_filters"])
        filters = eval(model_config["filters"])
        filters_dict = filters
        fil_name, ext = list(prop2arr(emcee_config["output_file"], dtype=str))
        delta = float(model_config["delta"]) if "delta" in list(model_config.keys()) else 500
        background = model_config["background"] if "background" in list(model_config.keys()) else "critical"
        eval_mass = str2bool(model_config["eval_mass"]) if "eval_mass" in list(model_config.keys()) else False
        infere_mass = str2bool(model_config["infere_mass"]) if "infere_mass" in list(model_config.keys()) else False

        subr_grid = str2bool(model_config["subr_grid"]) if "subr_grid" in list(model_config.keys()) else False
        subr_grid_kwargs = dict(model_config["subr_grid_kwargs"]) if "subr_grid_kwargs" in list(model_config.keys()) else {}

        if fit_entire_data == False:

            clusters = grouped_clusters.load_from_path(specific_path, dtype = float_dtype)
            output_path = clusters.output_path
            clusters.mean(from_path = True)
            if rotate_cov_matrix:
                clusters.rotate_cov_matrix()
            R = clusters.R
            rmax, rmin = np.max(clusters.richness), np.min(clusters.richness)
            zmax, zmin = np.max(clusters.z), np.min(clusters.z)
            profile = np.array(clusters.mean_profile, dtype = float_dtype)
            errors = np.array(clusters.error_in_mean, dtype = float_dtype)
            sigma = clusters.cov if hasattr(clusters, "cov") else errors
            rbins, zbins, Mbins = completeness_config.pop("rbins", 25), completeness_config.pop("zbins", 25), completeness_config.pop("Mbins", 25)
            rbins = int(rbins)
            zbins = int(zbins)
            Mbins = int(Mbins)

            debug = True if is_running_via_nohup() == True else args.debug

            func = clusters.stacked_halo_model_func(getattr(profiles_module, profile_stacked_model), 
                                                    rbins = rbins, zbins = zbins, Mbins = Mbins,
                                                    use_filters = use_filters, filters = filters_dict,
                                                    units = r_units, completeness_kwargs = dict(completeness_config),
                                                    use_two_halo_term = use_two_halo_term,
                                                    mis_centering = use_mis_centering, 
                                                    mis_centering_kwargs = mis_centering_kwargs,
                                                    background = background, delta = delta,
                                                    eval_mass = eval_mass, rebinning = use_rebinning,
                                                    rebinning_kwargs = rebinning_kwargs,
                                                    return_1h2h = store_two_halo_term,
                                                    two_halo_kwargs = two_halo_kwargs,
                                                    infere_mass = infere_mass, redshift_pivot = redshift_pivot,
                                                    richness_pivot = richness_pivot,
                                                    verbose = debug, subr_grid = subr_grid,
                                                    subr_grid_kwargs = subr_grid_kwargs)
            
            prior_parameters = dict(prior_config)
            prior_parameters_dict = {
                key: list(prop2arr(prior_parameters[key], dtype=str))
                for key in list(prior_parameters.keys())
            }
            prior_parameters = list(prior_parameters_dict.values())

            if fixed_mis_centering == False and use_mis_centering == True:
                prior_parameters += prior_parameters_mc
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_mc}
            if fixed_halo_model == False:
                prior_parameters += prior_parameters_hm
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_hm}
            if use_two_halo_term == True and two_halo_power:
                prior_parameters += prior_parameters_2h
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_2h}
            
                
            params_labels = list(prior_parameters_dict.keys())

            global ln_prior
            def ln_prior(theta):
                prior = 0.0
                i_theta = 0
                for i in range(len(prior_parameters)):
                    if "free" in prior_parameters[i]:
                        args = np.array(prior_parameters[i][-1].split("|")).astype(
                            float_dtype
                        )
                        prior += getattr(MCMC_func, prior_parameters[i][1])(
                            theta[i_theta], *args
                        )
                        i_theta += 1
                return prior

            global ln_likelihood
            if likelihood_func == 'gaussian':
                def ln_likelihood(theta, x, y, sigma, **kwargs):
                    model = kwargs["model"]
                    in_demo_mode = kwargs["in_demo_mode"] if "in_demo_mode" in list(kwargs.keys()) else False
                    free_params = kwargs["free_params"]
                    fixed_params = kwargs["fixed_params"]
                    fixed_mis_centering = kwargs["fixed_mis_centering"]
                    fixed_halo_model = kwargs["fixed_halo_model"]
                    store_two_halo_term = kwargs["store_two_halo_term"]
                    infere_mass = kwargs["infere_mass"]
                    two_halo_power = kwargs["two_halo_power"]
                    model_kwargs = {}
                    ic("initial theta =",theta)

                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[1] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = theta
                        theta = new_theta
                        ic("theta with fixed parameters =", theta)
                    if two_halo_power == True:
                        n_two_halo_power = kwargs["n_two_halo_params"]
                        two_halo_power = theta[-n_two_halo_power:]
                        theta = theta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                        ic(two_halo_power)
                        ic("theta without two_halo_power =",theta)
                    if fixed_halo_model == False:
                        n_halo_params = kwargs["n_halo_params"]
                        halo_model_params = theta[-n_halo_params:]
                        theta = theta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                        ic(halo_model_params)
                        ic("theta without halo_model_params =",theta)
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = kwargs["n_mis_centering_params"]
                            mis_centering_params = theta[-n_mis_centering_params:]
                            theta = theta[:-2]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                            ic(mis_centering_params)
                            ic("theta without mis_centering_params =",theta)
                        else:
                            model_kwargs["mis_centering_params"] = kwargs["mis_centering_params"]
                    ic(theta)
                    mu = model(x, theta, **model_kwargs)
                    if store_two_halo_term == True:
                        if infere_mass == True:
                           mu, one_halo, two_halo, mass = mu
                        else:
                            mu, one_halo, two_halo = mu
                    elif infere_mass == True and store_two_halo_term == False:
                        mu, mass = mu
                        ic(mu)
                        ic(mass)
                    if "inv_cov_matrix" not in list(kwargs.keys()):
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (y - mu) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum((y - mu)**2 / sigma**2)
                        if store_two_halo_term == False:
                            if infere_mass == True:
                                return np.sum(log_likelihood), current_chi2, mu, mass
                            else:
                                return np.sum(log_likelihood), current_chi2, mu
                        elif store_two_halo_term == True:
                            if infere_mass == True:
                                return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo, mass
                    else:
                        inv_cov_matrix = kwargs["inv_cov_matrix"]
                        log_det_C = kwargs["log_det_C"]
                        residual = y - mu
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        ic(current_chi2)
                        ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        if store_two_halo_term == False:
                            if infere_mass == True:
                                return ln_lk, current_chi2, mu, mass
                            else:
                                return ln_lk, current_chi2, mu
                        elif store_two_halo_term == True:
                            if infere_mass == True:
                                return ln_lk, current_chi2, mu, one_halo, two_halo, mass
                            return ln_lk, current_chi2, mu, one_halo, two_halo
            elif likelihood_func == "chi2":
                def ln_likelihood(theta, x, y, sigma, **kwargs):
                    model = kwargs["model"]
                    in_demo_mode = kwargs["in_demo_mode"] if "in_demo_mode" in list(kwargs.keys()) else False
                    free_params = kwargs["free_params"]
                    fixed_params = kwargs["fixed_params"]
                    fixed_mis_centering = kwargs["fixed_mis_centering"]
                    fixed_halo_model = kwargs["fixed_halo_model"]
                    store_two_halo_term = kwargs["store_two_halo_term"]
                    infere_mass = kwargs["infere_mass"]
                    two_halo_power = kwargs["two_halo_power"]

                    model_kwargs = {}
                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[0] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = theta
                        theta = new_theta

                    if two_halo_power == True:
                        n_two_halo_power = kwargs["n_two_halo_params"]
                        two_halo_power = theta[-n_two_halo_power:]
                        theta = theta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                    if fixed_halo_model == False:
                        n_halo_params = kwargs["n_halo_params"]
                        halo_model_params = theta[-n_halo_params:]
                        theta = theta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = kwargs["n_mis_centering_params"]
                            mis_centering_params = theta[-n_mis_centering_params:]
                            theta = theta[:-2]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                        else:
                            model_kwargs["mis_centering_params"] = kwargs["mis_centering_params"]
                    mu = model(x, theta, **model_kwargs)

                    if store_two_halo_term == True:
                        if infere_mass == True:
                            mu, one_halo, two_halo, mass = mu
                        else:
                            mu, one_halo, two_halo = mu
                    elif store_two_halo_term == False and infere_mass == True:
                        mu, mass = mu
                    if "inv_cov_matrix" not in list(kwargs.keys()):
                        res = np.sum(((y - mu) / sigma)**2)
                        if store_two_halo_term == False:
                            if infere_mass == True:
                                return -0.5 * res, res, mu, mass
                            else:
                                return -0.5 * res, res, mu
                        elif store_two_halo_term == True:
                            if infere_mass == True:
                                return -0.5 * res, res, mu, one_halo, two_halo, mass
                        return -0.5 * res, res, mu
                    else:
                        inv_cov_matrix = kwargs["inv_cov_matrix"]
                        log_det_C = kwargs["log_det_C"]
                        residual = y - mu
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        if store_two_halo_term == False:
                            if infere_mass == True:
                                return ln_lk, current_chi2, mu, mass
                            else:
                                return ln_lk, current_chi2, mu
                        elif store_two_halo_term == True:
                            if infere_mass == True:
                                return ln_lk, current_chi2, mu, one_halo, two_halo, mass
                            return ln_lk, current_chi2, mu, one_halo, two_halo
                #======
            
            fixed_params = [(i,p[1]) for i,p in enumerate(prior_parameters) if "fixed" in p]
            free_params = [(i,p) for i,p in enumerate(prior_parameters) if "free" in p]
            param_limits = [
                np.array(prior_parameters[i][-1].split("|")).astype(float_dtype)[-2::]
                if "free" in prior_parameters[i] else float(prior_parameters[i][-1])
                for i in range(len(prior_parameters))
                
            ]
            prior_distributions = [
                getattr(MCMC_func, prior_parameters[i][1])
                if "free" in prior_parameters[i] else "fixed"
                for i in range(len(prior_parameters))
            ]
            prior_distribution_args = []
            for i in range(len(prior_parameters)):
                if "free" in prior_parameters[i]:
                    if "|" in prior_parameters[i][-1]:
                        prior_distribution_args.append(
                        np.array(prior_parameters[i][-1].split("|")).astype(float_dtype)
                        )
                    else:
                        prior_distribution_args.append(
                        np.array(prior_parameters[i][-2].split("|")).astype(float_dtype)
                        )
                elif "fixed" in prior_parameters[i]:
                    prior_distribution_args.append("fixed")

            params_labels = list(prior_parameters_dict.keys())

            ndims = len(free_params)
            initial_guess = np.zeros((nwalkers, len(prior_parameters)))
            for i in range(len(param_limits)):
                if len(prior_parameters[i]) == 3:
                    initial_guess[:,i] = np.array(
                        random_initial_steps(param_limits[i], nwalkers, 
                        distribution = prior_distributions[i],
                        dist_args = prior_distribution_args[i],
                        nsamples = 1e3)
                        )
                else:
                    initial_guess[:,i] = float(prior_parameters[i][-1])* (1 + np.random.uniform(-0.1, 0.1, size = nwalkers))

            sampler_kwargs = dict(
                model = func,
                ln_prior = ln_prior,
                ln_likelihood = ln_likelihood,
                chi2 = compute_chi2,
                fixed_mis_centering = False,
                fixed_halo_model = fixed_halo_model,
                fixed_params = fixed_params,
                free_params = free_params,
            )

            if use_mis_centering == True:
                sampler_kwargs["mis_centering_params"] = mis_centering_params
                sampler_kwargs["Roff"] = mis_centering_kwargs["Roff"]
                sampler_kwargs["theta2"] = mis_centering_kwargs["theta"]
                sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
                sampler_kwargs["n_mis_centering_params"] = n_mis_centering_params

            sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
            sampler_kwargs["fixed_halo_model"] = fixed_halo_model    
            
            sampler_kwargs["fixed_params"] = fixed_params
            sampler_kwargs["free_params"] = free_params

            ic(fixed_params)
            ic(free_params)

            if use_two_halo_term == True and two_halo_power == True:
                sampler_kwargs["two_halo_power"] = two_halo_power
                sampler_kwargs["n_two_halo_params"] = n_parameters_2h
            else:
                sampler_kwargs["two_halo_power"] = False

            if fixed_halo_model == False:
                sampler_kwargs["n_halo_params"] = len(prior_parameters_hm)
                sampler_kwargs["smooth"] = None
            if np.array(sigma).ndim > 1:
                log_det_C = np.linalg.slogdet(sigma)[1]
                inv_cov_matrix = np.linalg.inv(sigma)        
                sampler_kwargs["log_det_C"] = log_det_C
                sampler_kwargs["inv_cov_matrix"] = inv_cov_matrix

            filename = output_path + "/" + fil_name + "." + ext
            if del_backend == True:
                if os.path.exists(filename):
                    os.remove(filename)
            backend = emcee.backends.HDFBackend(filename)
            dtype = []
            for i,key in enumerate(list(blobs_config.keys())):
                if key == "ONE_HALO" or key == "TWO_HALO" or key == "MASS":
                    continue
                b = blobs_config[key]
                dt, shape = prop2arr(b, dtype = str)
                if dt == "np.float64":
                    dt = np.float64
                if len(shape.split('|'))==1:
                    shape = int(shape.replace('(','').replace(')',''))
                    if shape == 1:
                        dtype.append((key,dt))
                    elif shape > 1:
                        dtype.append((key,np.dtype((dt, shape))))
            if store_two_halo_term == True and use_two_halo_term == True:
                if "TWO_HALO" in list(blobs_config.keys()) and "ONE_HALO" in list(blobs_config.keys()):
                    two_halo_blob = blobs_config["TWO_HALO"]
                    one_halo_blob = blobs_config["ONE_HALO"]
                    dt_1h, shape_1h = prop2arr(one_halo_blob, dtype = str)
                    dt_2h, shape_2h = prop2arr(two_halo_blob, dtype = str)
                    shape_1h = int(shape_1h.replace('(','').replace(')',''))
                    shape_2h = int(shape_2h.replace('(','').replace(')',''))
                    dt_1h = np.float64 if dt_1h == "np.float64" else getattr(np, dt_1h)
                    dt_2h = np.float64 if dt_2h == "np.float64" else getattr(np, dt_2h)
                    dtype.append(("ONE_HALO", np.dtype((dt_1h, shape_1h))))
                    dtype.append(("TWO_HALO", np.dtype((dt_2h, shape_2h))))
                    sampler_kwargs["store_two_halo_term"] = True
                elif "TWO_HALO" not in list(blobs_config.keys()) or "ONE_HALO" not in list(blobs_config.keys()):
                    raise KeyError("Blobs must contain 'TWO_HALO' and 'ONE_HALO' keys")
            elif use_two_halo_term == False:
                print("Store two halo term can't be True if use two halo term is False!")
                sampler_kwargs["store_two_halo_term"] = False
            else:
                sampler_kwargs["store_two_halo_term"] = False

            if infere_mass == True:
                if "MASS" in list(blobs_config.keys()):
                    mass_blob = blobs_config["MASS"]
                    dt_m, shape_m = prop2arr(mass_blob, dtype = str)
                    shape_m = int(shape_m.replace('(','').replace(')',''))
                    dt_m = np.float64 if dt_m == "np.float64" else getattr(np, dt_m)
                    dtype.append(("MASS", np.dtype((dt_m, shape_m))))
                    sampler_kwargs["infere_mass"] = True
                else:
                    raise KeyError("Blobs must contain 'MASS' key")
            else:
                sampler_kwargs["infere_mass"] = False
            sampler_kwargs["fixed_params"] = fixed_params
            sampler_kwargs["free_params"] = free_params
            if use_two_halo_term == True and two_halo_power == True:
                sampler_kwargs["two_halo_power"] = two_halo_power
                sampler_kwargs["n_two_halo_params"] = n_parameters_2h

            sampler_kwargs["fixed_halo_model"] = fixed_halo_model    
            
            if fixed_halo_model == False:
                sampler_kwargs["n_halo_params"] = len(prior_parameters_hm)
                sampler_kwargs["smooth"] = None
            if demo == True:
                run_demo_mode(
                    demo,
                    sampler_kwargs,
                    specific_path,
                    profile_stacked_model,
                    use_two_halo_term,
                    likelihood_func,
                    filename,
                    initial_guess,
                    clusters,
                    xlabel,
                    ylabel,
                    yscale,
                    xscale,
                    model_id,
                    ln_likelihood,
                    demo_path,
                    profile,
                    params_labels,
                    R
                )
            initial_guess = np.delete(initial_guess, np.asarray([p[0] for p in fixed_params], dtype = int), axis = 1)
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(ncores)

            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndims,
                ln_posterior_func,
                args=(
                    R,
                    profile,
                    sigma,
                ),
                kwargs=sampler_kwargs,
                pool = pool,
                backend=backend,
                blobs_dtype = dtype
            )  
            if rewrite == True:
                print("rewriting backend")
                backend.reset(nwalkers, ndims)
            elif rewrite == False and os.path.exists(filename):
                try:
                    initial_guess = None
                    last_chain = backend.get_last_sample()
                    sampler._previous_state = last_chain.coords
                except Exception as e:
                    initial_guess =  np.array(
                        [
                        np.array(
                            [
                                np.random.uniform(*(param_limits[j] * 0.90))
                                for j in range(len(param_limits))
                                ],
                            dtype = float_dtype)
                            for i in range(nwalkers)
                            ],
                        dtype = float_dtype)
                    print("Chain can't open the last sample. return the exception: \n",e)

            print("\033[44m",10*"=","RUNNING MCMC",10*"=" ,"\033[0m")
            print(f"* richness: \033[32m[{int(rmin)},{int(rmax)}]\033[0m")
            print(f"* redshift: \033[32m[{round(zmin,3)},{round(zmax,3)}]\033[0m")
            print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
            print(f"* two halo term: \033[35m{use_two_halo_term}\033[0m")
            print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
            print(f"* output file: \033[35m{filename}\033[0m")
            print(" ",30*"="," ")
            t1 = time()
            sampler.run_mcmc(initial_guess, nsteps, progress=True, store = True)
            t2 = time()
            print(f"The sampler of richness \033[32m[{int(rmin)},{int(rmax)}]\033[0m and redshift \033[32m[{round(zmin,3)},{round(zmax,3)}]\033[0m was finished in {t2 - t1} seconds.")

        if fit_entire_data == True:

            print("Fitting whole data...")
            clusters_list = []
            specific_path = args.path
            specific_path = specific_path + "/" if specific_path [-1] != "/" else specific_path 
            ignore = np.loadtxt(specific_path + "ignore.txt", dtype = str).T if os.path.exists(specific_path + "ignore.txt") else []
            available_paths = [path for path in os.listdir(specific_path) if os.path.isdir(specific_path + path)]
            richness_bins = []
            paths = []
            apply_filter_per_profile = str2bool(model_config["apply_filter_per_profile"])
            for path in available_paths:
                if path in ignore:
                    continue
                current_path = specific_path + path
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

            debug = True if is_running_via_nohup() == True else args.debug

            func,cov, about_clusters, clusters, _, funcs = grouped_clusters.stacked_halo_model_func_by_paths(getattr(profiles_module, profile_stacked_model),
                                                full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths,
                                                verbose_pivots = True, rotate_cov = rotate_cov_matrix, use_filters = use_filters, filters = filters_dict,
                                                completeness_kwargs = dict(completeness_config), use_two_halo_term = use_two_halo_term, off_diag = off_diag,
                                                two_halo_kwargs = two_halo_kwargs, use_mis_centering = use_mis_centering, fixed_RM_relationship = fixed_halo_model
                                                , background = background, delta = delta, eval_mass = eval_mass, apply_filter_per_profile = apply_filter_per_profile
                                                ,rebinning = use_rebinning, rebinning_kwargs = rebinning_kwargs, return_1h2h = store_two_halo_term,
                                                infere_mass = infere_mass, verbose = debug, subr_grid = subr_grid, subr_grid_kwargs = subr_grid_kwargs,)

            R = clusters[-1].R
            profiles = np.zeros(len(clusters)*len(R), dtype = np.float32)
            bins = np.asarray([(*c.richness_bin, *c.redshift_bin) for c in clusters])
            sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
            bins = bins[sorted_idx]
            clusters = [clusters[i] for i in sorted_idx]
            for i in range(len(clusters)):
                profiles[i*len(R) : (i+1)*len(R)] = clusters[i].mean_profile
            cluster, cov = grouped_clusters.compute_joint_cov(off_diag = off_diag, groups = clusters, corr = False)
            
            individuals_covs = [g.cov for g in clusters]
            np.save(f"{specific_path}/individuals_covs.npy", np.array(individuals_covs))
            from plottery.plotutils import update_rcParams
            update_rcParams()
            fig, ax = plt.subplots(2,1, figsize = (12,16), sharex = True, sharey = True)
            colors = ["purple", "darkgreen", "orange","darkred", "darkblue", "peru"]
            for i in range(len(clusters)):
                if i%2 == 0:
                    ax[0].plot(R, clusters[i].mean_profile, color = colors[i//2], 
                    label = r"$\lambda \in [%.i, %.i]$" % (clusters[i].richness_bin[0], 
                    clusters[i].richness_bin[1]))
                    ax[0].fill_between(R, clusters[i].mean_profile - clusters[i].error_in_mean, 
                                        clusters[i].mean_profile + clusters[i].error_in_mean, 
                                        color = colors[i//2], 
                                        alpha = 0.3)
                else:
                    ax[1].plot(R, clusters[i].mean_profile, color = colors[i//2],
                    label = r"$\lambda \in [%.i, %.i]$" % (clusters[i].richness_bin[0],
                    clusters[i].richness_bin[1]))
                    ax[1].fill_between(R, clusters[i].mean_profile - clusters[i].error_in_mean, 
                                        clusters[i].mean_profile + clusters[i].error_in_mean, 
                                        color = colors[i//2], 
                                        alpha = 0.3)
            ax[0].text(0.05, 0.95, r"$z \in [%.1f, %.1f]$" % (bins[0,2],bins[0,3]), ha='left', va='top', 
                        transform=ax[0].transAxes, fontsize = 18, color = "black")
            ax[1].text(0.05, 0.95, r"$z \in [%.1f, %.1f]$" % (bins[1,2],bins[1,3]), ha='left', va='top', 
                        transform=ax[1].transAxes, fontsize = 18, color = "black")
            ax[0].set(xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale)
            fig.suptitle(model_id,fontsize = 24, fontweight = "bold")
            [a.legend(fontsize = 20) for a in ax]
            fig.savefig(specific_path + "stacked_profile.png")

            sigma = np.array(cov, dtype = float_dtype) if cov.ndim > 1 else np.array(clusters.error_in_mean, dtpye = float_dtype)

            prior_parameters = dict(prior_config)
            prior_parameters_dict = {
                key: (
                    list(prop2arr(prior_parameters[key], dtype=str))
                    if "free" in list(prop2arr(prior_parameters[key], dtype=str))
                    else ["none", *list(prop2arr(prior_parameters[key], dtype=str))]
                )
                for key in prior_parameters
            }

            prior_parameters = list(prior_parameters_dict.values())
            if fixed_mis_centering == False and use_mis_centering == True:
                prior_parameters += prior_parameters_mc
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_mc}
            if fixed_halo_model == False:
                prior_parameters += prior_parameters_hm
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_hm}
            if use_two_halo_term == True and two_halo_power:
                prior_parameters += prior_parameters_2h
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_2h}
            
            params_labels = list(prior_parameters_dict.keys())

            fixed_params = [(i,p[1]) for i,p in enumerate(prior_parameters) if "fixed" in p]
            free_params = [(i,p) for i,p in enumerate(prior_parameters) if "free" in p]

            
            param_limits = [
                np.array(prior_parameters[i][2].split("|")).astype(float_dtype)[-2::]
                if "free" in prior_parameters[i] else float(prior_parameters[i][-1])
                for i in range(len(prior_parameters))
                
            ]

            prior_distributions = [
                getattr(MCMC_func, prior_parameters[i][1])
                if "free" in prior_parameters[i] else "fixed"
                for i in range(len(prior_parameters))
            ]
            prior_distribution_args = [
                np.array(prior_parameters[i][2].split("|")).astype(float_dtype)
                if "free" in prior_parameters[i] else "fixed"
                for i in range(len(prior_parameters))
            ]
            ndims = len(free_params)

            initial_guess = np.zeros((nwalkers, len(prior_parameters)))

            for i in range(len(param_limits)):
                if len(prior_parameters[i]) == 3:
                    initial_guess[:,i] = np.array(
                        random_initial_steps(param_limits[i], nwalkers, 
                        distribution = prior_distributions[i],
                        dist_args = prior_distribution_args[i],
                        nsamples = 1e3)
                        )
                else:
                    initial_guess[:,i] = float(prior_parameters[i][-1])* (1 + np.random.uniform(-0.1, 0.1, size = nwalkers))

            global ln_prior_general
            def ln_prior_general(theta):
                prior = 0.0
                i_theta = 0
                for i in range(len(prior_parameters)):
                    if "free" in prior_parameters[i]:
                        args = np.array(prior_distribution_args[i]).astype(
                            float_dtype
                        )
                        prior += prior_distributions[i](
                            theta[i_theta], *args
                        )
                        i_theta += 1
                return prior

            global ln_likelihood_general
            if likelihood_func == 'gaussian':
                def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                    model = kwargs["model"]
                    in_demo_mode = kwargs["in_demo_mode"] if "in_demo_mode" in list(kwargs.keys()) else False
                    free_params = kwargs["free_params"]
                    fixed_params = kwargs["fixed_params"]
                    fixed_mis_centering = kwargs["fixed_mis_centering"]
                    fixed_halo_model = kwargs["fixed_halo_model"]
                    store_two_halo_term = kwargs["store_two_halo_term"]
                    infere_mass = kwargs["infere_mass"]
                    two_halo_power = kwargs["two_halo_power"]
                    bins = kwargs["bins"] if y.ndim == 1 else np.array(y)[:,0:4]
                    y = np.array(y)[:,4:].flatten() if y.ndim > 1 else y
                    ic(bins)
                    ic("initial theta", theta)
                    model_kwargs = {"cbin": bins}
                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[0] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = theta
                        theta = new_theta
                        ic("theta with fixed parameters", theta)
                    if two_halo_power == True:
                        n_two_halo_power = kwargs["n_two_halo_params"]
                        two_halo_power = theta[-n_two_halo_power:]
                        theta = theta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                        ic(two_halo_power)
                        ic("theta without two_halo_power", theta)
                    if fixed_halo_model == False:
                        n_halo_params = kwargs["n_halo_params"]
                        halo_model_params = theta[-n_halo_params:]
                        theta = theta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                        ic(halo_model_params)
                        ic("theta without RM_params", theta)
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = kwargs["n_mis_centering_params"]
                            mis_centering_params = theta[-n_mis_centering_params:]
                            theta = theta[:-n_mis_centering_params]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                            ic(mis_centering_params)
                            ic("theta without mis_centering_params", theta)
                        else:
                            model_kwargs["mis_centering_params"] = kwargs["mis_centering_params"]
                    mu = model(x, theta, **model_kwargs) 
                    ic(mu)
                    if store_two_halo_term == True and infere_mass == True:
                        mu, one_halo, two_halo, mass = mu
                    elif store_two_halo_term == True and infere_mass == False:
                        mu, one_halo, two_halo = mu
                    elif store_two_halo_term == False and infere_mass == True:
                        mu, mass = mu
                    if "inv_cov_matrix" not in list(kwargs.keys()):
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (y - mu) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum((y - mu)**2 / sigma**2)
                        if store_two_halo_term == True and infere_mass == True:
                            return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo, mass
                        elif store_two_halo_term == True and infere_mass == False:
                            return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo
                        elif store_two_halo_term == False and infere_mass == True:
                            return np.sum(log_likelihood), current_chi2, mu, mass
                        return np.sum(log_likelihood), current_chi2, mu
                    else:
                        inv_cov_matrix = kwargs["inv_cov_matrix"]
                        log_det_C = kwargs["log_det_C"]
                        residual = y - mu
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        if store_two_halo_term == True and infere_mass == True:
                            return ln_lk, current_chi2,mu, one_halo, two_halo, mass
                        elif store_two_halo_term == True and infere_mass == False:
                            return ln_lk, current_chi2,mu, one_halo, two_halo
                        elif store_two_halo_term == False and infere_mass == True:
                            return ln_lk, current_chi2,mu, mass
                        return ln_lk, current_chi2 ,mu
            elif likelihood_func == "chi2":
                def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                    model = kwargs["model"]
                    in_demo_mode = kwargs["in_demo_mode"] if "in_demo_mode" in list(kwargs.keys()) else False
                    free_params = kwargs["free_params"]
                    fixed_params = kwargs["fixed_params"]
                    store_two_halo_term = kwargs["store_two_halo_term"]
                    infere_mass = kwargs["infere_mass"]
                    two_halo_power = kwargs["two_halo_power"]
                    bins = kwargs["bins"]

                    model_kwargs = {"cbin": bins}
                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[0] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = theta
                        theta = new_theta
                    if two_halo_power == True:
                        n_two_halo_power = kwargs["n_two_halo_params"]
                        two_halo_power = theta[-n_two_halo_power:]
                        theta = theta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                    if fixed_halo_model == False:
                        n_halo_params = kwargs["n_halo_params"]
                        halo_model_params = theta[-n_halo_params:]
                        theta = theta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = kwargs["n_mis_centering_params"]
                            mis_centering_params = theta[-n_mis_centering_params:]
                            theta = theta[:-2]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                        else:
                            model_kwargs["mis_centering_params"] = kwargs["mis_centering_params"]
                    mu = model(x, theta, **model_kwargs)
                    
                    if store_two_halo_term == True and infere_mass == True:
                        mu, one_halo, two_halo, mass = mu
                    elif store_two_halo_term == True and infere_mass == False:
                        mu, one_halo, two_halo = mu
                    elif store_two_halo_term == False and infere_mass == True:
                        mu, mass = mu
                    if "inv_cov_matrix" not in list(kwargs.keys()):
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (y - mu) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum((y - mu)**2 / sigma**2)
                        if store_two_halo_term == True and infere_mass == True:
                            return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo, mass
                        elif store_two_halo_term == True and infere_mass == False:
                            return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo
                        elif store_two_halo_term == False and infere_mass == True:
                            return np.sum(log_likelihood), current_chi2, mu, mass
                        return np.sum(log_likelihood), current_chi2, mu
                    else:
                        inv_cov_matrix = kwargs["inv_cov_matrix"]
                        log_det_C = kwargs["log_det_C"]
                        residual = y - mu
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        if store_two_halo_term == True and infere_mass == True:
                            return ln_lk, current_chi2, mu, one_halo, two_halo, mass
                        elif store_two_halo_term == True and infere_mass == False:
                            return ln_lk, current_chi2, mu, one_halo, two_halo
                        elif store_two_halo_term == False and infere_mass == True:
                            return ln_lk, current_chi2, mu, mass
                        return ln_lk, current_chi2, mu
                #======

            sampler_kwargs = {
                "ln_likelihood": ln_likelihood_general,
                "ln_posterior": ln_posterior_func,
                "ln_prior": ln_prior_general,
                "model": func,
                "infere_mass": infere_mass,
                "chi2": compute_chi2,
            }

            if use_mis_centering == True:
                sampler_kwargs["mis_centering_params"] = mis_centering_params
                sampler_kwargs["Roff"] = mis_centering_kwargs["Roff"]
                sampler_kwargs["theta2"] = mis_centering_kwargs["theta"]
                sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
                sampler_kwargs["n_mis_centering_params"] = n_mis_centering_params

            sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
            sampler_kwargs["fixed_halo_model"] = fixed_halo_model    
            
            sampler_kwargs["fixed_params"] = fixed_params
            sampler_kwargs["free_params"] = free_params

            if use_two_halo_term == True and two_halo_power == True:
                sampler_kwargs["two_halo_power"] = two_halo_power
                sampler_kwargs["n_two_halo_params"] = n_parameters_2h
            if fixed_halo_model == False:
                sampler_kwargs["n_halo_params"] = len(prior_parameters_hm)
                sampler_kwargs["smooth"] = None

            filename = specific_path + "general_fit_" + fil_name + "." + ext

            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(ncores)
            dtype = []
            for i,key in enumerate(list(blobs_config.keys())):
                if key == "TWO_HALO" or key == "ONE_HALO" or key == "MASS":
                    continue
                b = blobs_config[key]
                dt, shape = prop2arr(b, dtype = str)
                if dt == "np.float64":
                    dt = np.float64
                if len(shape.split('|'))==1:
                    shape = int(shape.replace('(','').replace(')',''))
                    if shape == 1:
                        dtype.append((key,dt))
                    elif shape > 1:
                        dtype.append((key,np.dtype((dt, shape))))
    
            Y = np.array([(*c.richness_bin,*c.redshift_bin,*c.mean_profile)for c in clusters])
            np.savetxt(f"{specific_path}/sigma.txt", sigma)
            np.savetxt(f"{specific_path}/xobs.txt", R)
            np.savetxt(f"{specific_path}/yobs.txt", profiles)
            np.savetxt(f"{specific_path}/cov.txt", cov)
            header = " ".join(["lmin","lmax", "zmin", "zmax"])
            np.savetxt(f"{specific_path}/bins.txt", bins, header = header)

            sampler_kwargs["bins"] = bins

            if store_two_halo_term == True and use_two_halo_term == True:
                if "TWO_HALO" in list(blobs_config.keys()) and "ONE_HALO" in list(blobs_config.keys()):
                    two_halo_blob = blobs_config["TWO_HALO"]
                    one_halo_blob = blobs_config["ONE_HALO"]
                    dt_1h, shape_1h = prop2arr(one_halo_blob, dtype = str)
                    dt_2h, shape_2h = prop2arr(two_halo_blob, dtype = str)
                    shape_1h = int(shape_1h.replace('(','').replace(')',''))
                    shape_2h = int(shape_2h.replace('(','').replace(')',''))
                    dt_1h = np.float64 if dt_1h == "np.float64" else getattr(np, dt_1h)
                    dt_2h = np.float64 if dt_2h == "np.float64" else getattr(np, dt_2h)
                    dtype.append(("ONE_HALO", np.dtype((dt_1h, shape_1h))))
                    dtype.append(("TWO_HALO", np.dtype((dt_2h, shape_2h))))
                    sampler_kwargs["store_two_halo_term"] = True
                elif "TWO_HALO" not in list(blobs_config.keys()) or "ONE_HALO" not in list(blobs_config.keys()):
                    raise KeyError("Blobs must contain 'TWO_HALO' and 'ONE_HALO' keys")
                elif use_two_halo_term == False:
                    print("Store two halo term can't be True if use two halo term is False!")
                    sampler_kwargs["store_two_halo_term"] = False
            else:
                sampler_kwargs["store_two_halo_term"] = False
            if infere_mass == True:
                if "MASS" in list(blobs_config.keys()):
                    mass_blob = blobs_config["MASS"]
                    dt_m, shape_m = prop2arr(mass_blob, dtype = str)
                    shape_m = int(shape_m.replace('(','').replace(')',''))
                    dt_m = np.float64 if dt_m == "np.float64" else getattr(np, dt_m)
                    dtype.append(("MASS", np.dtype((dt_m, shape_m))))
                    sampler_kwargs["infere_mass"] = True
                else:
                    raise KeyError("Blobs must contain 'MASS' key")
            else:
                sampler_kwargs["infere_mass"] = False
            if np.array(sigma).ndim > 1:
                log_det_C = np.linalg.slogdet(sigma)[1]
                inv_cov_matrix = np.linalg.inv(sigma)        
                sampler_kwargs["log_det_C"] = log_det_C
                sampler_kwargs["inv_cov_matrix"] = inv_cov_matrix

            if use_two_halo_term == True and two_halo_power == True:
                sampler_kwargs["two_halo_power"] = two_halo_power
                sampler_kwargs["n_two_halo_params"] = n_parameters_2h
            else:
                sampler_kwargs["two_halo_power"] = False

            cond_number = np.linalg.cond(sigma)
            epsilon = np.finfo(np.float64).eps if dtype == np.float64 else np.finfo(np.float32).eps

            if cond_number >= 1/epsilon:
                print(f"Condition number of covariance matrix is too large: \033[31m{cond_number}\033[0m")
            else:
                print(f"Condition number of covariance matrix: \033[32m{cond_number}\033[0m")
            print("\n")
            if demo == True:
                run_demo_mode_general(
                    demo,
                    sampler_kwargs,
                    specific_path,
                    profile_stacked_model,
                    use_two_halo_term,
                    likelihood_func,
                    filename,
                    initial_guess,
                    clusters_list,
                    xlabel,
                    ylabel,
                    yscale,
                    xscale,
                    model_id,
                    ln_likelihood_general,
                    demo_path,
                    profiles,
                    params_labels,
                    R
                )

            #remove fixed parameters
            initial_guess = np.delete(initial_guess, np.asarray([p[0] for p in fixed_params], dtype = int), axis = 1)   
            if del_backend == True:
                if os.path.exists(filename):
                    os.remove(filename)
            backend = emcee.backends.HDFBackend(filename)

            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndims,
                ln_posterior_func,
                args=(
                    R,
                    Y,
                    sigma,
                ),
                kwargs=sampler_kwargs,
                pool = pool,
                backend = backend,
                blobs_dtype = dtype
            )
            print("\033[44m",10*"=","RUNNING MCMC",10*"=" ,"\033[0m")
            print(f"* entire data available on: \033[35m{specific_path}\033[0m")
            print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
            print(f"* two halo term: \033[35m{use_two_halo_term}\033[0m")
            print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
            print(f"* output file: \033[35m{filename}\033[0m")
            print(" ",30*"="," ")
            t1 = time()
            sampler.run_mcmc(initial_guess, nsteps, progress=True)
            t2 = time()
            print("MCMC was finish in ", t2  - t1,"seconds.")
    elif joint_fit == True:

        specific_paths = args.path.split(",")
        debug = args.debug
        config_files = args.CONFIG_FILE.split(",")
        configs = []
        
        jprior_parameters = []
        jfuncs = []
        jcovs = []
        jprofiles = []
        jclusters_list = []
        jclusters = []
        X = []
        params_indx = []
        jbins = []


        ncores = int(args.ncores) if args.ncores is not None else 35
        compute_correlations = args.compute_correlations

        jylabel = []
        jxlabel = []
        jyscale = []
        jxscale = []

        j_all_clusters = []
        j_paths = []

        local_path = os.path.dirname(os.path.realpath(__file__))
        demo = args.demo_mode
        ndemos = args.ndemos
        demo_path = local_path + "/" + str(args.demo_path)
        sampler_kwargs = {}
        sampler_kwargs["halo_params_added"] = False
        models = []

        jparams_labels = []
        jfixed_params = []
        jfree_params = []

        for (j,c),path in zip(enumerate(config_files), specific_paths):

            config_filepath = local_path +"/"+ str(c)
            print(f"Loading config from :\033[31m{config_filepath}\033[0m")
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


            xlabel_config, ylabel_config = config["MODEL"]["x"], config["MODEL"]["y"]

            ylabel_config = ylabel_config.split(",") if len(ylabel_config.split(",")) == 2 else (ylabel_config,"")
            xlabel_config = xlabel_config.split(",") if len(xlabel_config.split(",")) == 2 else (xlabel_config,"")

            xlabel, xlabel_unit = xlabel_config[0], xlabel_config[1]
            ylabel, ylabel_unit = ylabel_config[0], ylabel_config[1]

            xlabel_unit = xlabel_unit.replace(" ", "") if xlabel_unit is not None else ""
            ylabel_unit = ylabel_unit.replace(" ", "") if ylabel_unit is not None else ""


            ylabel = f"{ylabel} ({ylabel_unit})" if ylabel_unit != "" else ylabel
            xlabel = f"{xlabel} ({xlabel_unit})" if xlabel_unit != "" else xlabel

            jylabel.append(ylabel)
            jxlabel.append(xlabel)

            yscale = model_config["yscale"] if "yscale" in list(model_config.keys()) else "log"
            xscale = model_config["xscale"] if "xscale" in list(model_config.keys()) else "linear"

            jyscale.append(yscale) 
            jxscale.append(xscale)

            apply_filter_per_profile = str2bool(model_config["apply_filter_per_profile"])
            
            dtype = []

            try: 
                model_id = model_config["name"]
            except:
                model_id = None

            models.append(model_id)
            if j == 0:
                rewrite = str2bool(emcee_config["rewrite"])
                likelihood_func = emcee_config["likelihood"]
                nwalkers = int(emcee_config["nwalkers"])
                nsteps = int(emcee_config["nsteps"])
                del_backend = str2bool(emcee_config["delete"])
                off_diag = str2bool(model_config["off_diag"]) if "off_diag" in list(model_config.keys()) else False
                store_two_halo_term = str2bool(model_config["store_two_halo_term"]) if "store_two_halo_term" in list(model_config.keys()) else False
                output_path = emcee_config["joint_output_path"] if "joint_output_path" in list(emcee_config.keys()) else path
            infere_mass = str2bool(model_config["mass_inference"]) if "mass_inference" in list(model_config.keys()) else False
            background = model_config["background"] if "delta" in list(model_config.keys()) else "critical"
            delta = float(model_config["delta"]) if "background" in list(model_config.keys()) else 500
            eval_mass = str2bool(model_config["eval_mass"]) if "eval_mass" in list(model_config.keys()) else False
            rotate_cov_matrix = str2bool(model_config["rotate_cov_matrix"]) if "rotate_cov_matrix" in list(model_config.keys()) else False
            subr_grid = str2bool(model_config["subr_grid"]) if "subr_grid" in list(model_config.keys()) else False
            subr_grid_kwargs = dict(model_config["subr_grid_kwargs"]) if "subr_grid_kwargs" in list(model_config.keys()) else {}

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
                        arg = list(np.array(arr[:-1], dtype = float_dtype))
                        arg[-1] = int(arg[-1])
                        two_halo_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                    else:
                        two_halo_kwargs[k] = np.array(arr, dtype = float_dtype)
                elif "dict" in two_halo_kwargs[k]:
                    two_halo_kwargs[k] = eval(two_halo_kwargs[k])
            if "two_halo_power_func" in list(two_halo_kwargs.keys()):
                two_halo_kwargs["two_halo_power_func"] = getattr(profiles_module, two_halo_kwargs["two_halo_power_func"])
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
                        arg = list(np.array(arr[:-1], dtype = float_dtype))
                        arg[-1] = int(arg[-1])
                        mis_centering_kwargs[k] = np.logspace(*arg) if arr[-1] == "log" else np.linspace(*arg)
                    else:
                        mis_centering_kwargs[k] = np.array(arr, dtype = float_dtype)
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
                n_mis_centering_params = len(prior_parameters_mc)

            two_halo_kwargs["cosmo"] = ccl.CosmologyVanillaLCDM()

            prior_parameters = dict(prior_config)
            prior_parameters_dict = {
                key: list(prop2arr(prior_parameters[key], dtype=str))
                for key in list(prior_parameters.keys())
            }
            prior_parameters = list(prior_parameters_dict.values())

            prior_parameters = list(prior_parameters_dict.values())
            if fixed_mis_centering == False and use_mis_centering == True:
                prior_parameters += prior_parameters_mc
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_mc}
            if fixed_halo_model == False:
                prior_parameters += prior_parameters_hm
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_hm}
            if use_two_halo_term == True and two_halo_power:
                prior_parameters += prior_parameters_2h
                prior_parameters_dict = {**prior_parameters_dict, **prior_parameters_dict_2h}
            
            params_labels = list(prior_parameters_dict.keys())

            fixed_params = [(i,p[1]) for i,p in enumerate(prior_parameters) if "fixed" in p]
            free_params = [(i,p) for i,p in enumerate(prior_parameters) if "free" in p]

            jfixed_params = jfixed_params + fixed_params
            jfree_params = jfree_params + free_params

            if len(params_indx) == 0:
                params_indx.append((0,len(free_params)))
            else:
                params_indx.append((params_indx[-1][1], len(free_params) + params_indx[-1][1]))

            clabels = list(prior_parameters_dict.keys())

            jparams_labels = jparams_labels + clabels

            jprior_parameters = jprior_parameters + prior_parameters
            plot_cov_matrix = args.plot_cov_matrix
            ncores = str(args.ncores) if ncores is None else ncores
            r_units = args.r_units

            profile_stacked_model = model_config["profile"]

            rbins, zbins, Mbins = completeness_config.pop("rbins", 25), completeness_config.pop("zbins", 25), completeness_config.pop("Mbins", 25)
            rbins = int(rbins)
            zbins = int(zbins)
            Mbins = int(Mbins)

            fil_name, ext = list(prop2arr(emcee_config["output_file"], dtype=str))
            apply_filter_per_profile = str2bool(model_config["apply_filter_per_profile"])

            use_filters = str2bool(model_config["use_filters"])
            filters = eval(model_config["filters"])
            filters_dict = filters

            rebinning_config = dict(config["REBINNING"])

            use_rebinning = str2bool(model_config["rebinning"])
            nbins_rebinning = int(rebinning_config["Nbins"])
            pixel_size_rebinning = float(rebinning_config["pixel_size"])
            rebinning_method = rebinning_config["method"]
            interpolation_kwargs = eval(rebinning_config["interpolation_kwargs"])

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

            completeness_config = dict(config["COMPLETENESS"])
            for k in list(completeness_config.keys()):
                if completeness_config[k] in ("True", "False"):
                    completeness_config[k] = str2bool(completeness_config[k])
                elif "," in completeness_config[k]:
                    completeness_config[k] = prop2arr(completeness_config[k], dtype = float)
                elif "dict" in completeness_config[k]:
                    completeness_config[k] = eval(completeness_config[k])

            completeness_func_kwargs = dict(
                Nlambda_true = completeness_config["Nlambda_true"],
                max_lambda_true = completeness_config["max_lambda_true"],
                min_lambda_true = completeness_config["min_lambda_true"],
                Nlambda_obs = completeness_config["Nlambda_obs"],
                Nz_lambda = completeness_config["zbins"],
                function = completeness_config["func"],
                func_kwags = completeness_config.get("func_kwargs", {}),
            )

            print(f'\nExtracting data from:\033[35m{path}\033[0m\n')
            clusters_list = []
            specific_path = path
            specific_path = specific_path + "/" if specific_path [-1] != "/" else specific_path 
            ignore = np.loadtxt(specific_path + "ignore.txt", dtype = str).T if os.path.exists(specific_path + "ignore.txt") else []
            available_paths = [path for path in os.listdir(specific_path) if os.path.isdir(specific_path + path)]
            paths = []
            apply_filter_per_profile = str2bool(model_config["apply_filter_per_profile"])
            for path in available_paths:
                if path in ignore:
                    continue
                current_path = specific_path + path
                try:
                    if is_running_via_nohup() == False and args.ask_to_add == True:
                        cluster = grouped_clusters.load_from_path(current_path)
                        add_cluster = input(f"Do you want to add the next cluster? (Y, yes or enter to add it):\n {cluster}from: \033[35m{path}\033[0m\n").strip().lower()
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

            func,cov, about_clusters, clusters, _, funcs = grouped_clusters.stacked_halo_model_func_by_paths(getattr(profiles_module, profile_stacked_model),
                                                full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths,
                                                verbose_pivots = True, rotate_cov = rotate_cov_matrix, use_filters = use_filters, filters = filters_dict,
                                                completeness_kwargs = dict(completeness_config), use_two_halo_term = use_two_halo_term, off_diag = off_diag,
                                                two_halo_kwargs = two_halo_kwargs, use_mis_centering = use_mis_centering, fixed_RM_relationship = fixed_halo_model
                                                , background = background, delta = delta, eval_mass = eval_mass, apply_filter_per_profile = apply_filter_per_profile
                                                ,rebinning = use_rebinning, rebinning_kwargs = rebinning_kwargs, return_1h2h = store_two_halo_term,
                                                infere_mass = infere_mass, verbose = debug, subr_grid = subr_grid, subr_grid_kwargs = subr_grid_kwargs,)

            R = clusters[-1].R
            profiles = np.zeros(len(clusters)*len(R), dtype = np.float32)
            bins = np.asarray([(*c.richness_bin, *c.redshift_bin) for c in clusters])
            sorted_idx = np.lexsort((bins[:,3], bins[:,2], bins[:,1], bins[:,0]))
            bins = bins[sorted_idx]
            clusters = [clusters[i] for i in sorted_idx]

            jbins.append(bins)
            jclusters = jclusters + clusters

            for i in range(len(clusters)):
                profiles[i*len(R) : (i+1)*len(R)] = clusters[i].mean_profile
            jclusters_list.append(clusters)
            jfuncs.append(func)
            jprofiles.append(profiles)
            X.append(R)

            sub_sampler_kwargs = {}
            sub_sampler_kwargs["model_id"] = model_id

            if use_mis_centering == True:
                sub_sampler_kwargs["mis_centering_params"] = mis_centering_params
                sub_sampler_kwargs["Roff"] = mis_centering_kwargs["Roff"]
                sub_sampler_kwargs["theta2"] = mis_centering_kwargs["theta"]
                sub_sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
                sub_sampler_kwargs["n_mis_centering_params"] = n_mis_centering_params

            sub_sampler_kwargs["fixed_mis_centering"] = fixed_mis_centering
            sub_sampler_kwargs["fixed_halo_model"] = fixed_halo_model    
            
            if fixed_halo_model == False:
                sampler_kwargs["model_with_halo_params"] = model_id
            sub_sampler_kwargs["fixed_params"] = fixed_params
            sub_sampler_kwargs["free_params"] = free_params

            if use_two_halo_term == True and two_halo_power == True:
                sub_sampler_kwargs["two_halo_power"] = two_halo_power
                sub_sampler_kwargs["n_two_halo_params"] = n_parameters_2h
            if fixed_halo_model == False:
                sub_sampler_kwargs["n_halo_params"] = len(prior_parameters_hm)
                sub_sampler_kwargs["smooth"] = None

            sub_sampler_kwargs["bins"] = bins
            if infere_mass == True:
                if "MASS" in list(blobs_config.keys()):
                    mass_blob = blobs_config["MASS"]
                    dt_m, shape_m = prop2arr(mass_blob, dtype = str)
                    shape_m = int(shape_m.replace('(','').replace(')',''))
                    dt_m = np.float64 if dt_m == "np.float64" else getattr(np, dt_m)
                    dtype.append(("MASS", np.dtype((dt_m, shape_m))))
                    sub_sampler_kwargs["infere_mass"] = True
                    sampler_kwargs["infere_mass"] = True
                else:
                    raise KeyError("Blobs must contain 'MASS' key")
            else:
                sub_sampler_kwargs["infere_mass"] = False
                sampler_kwargs["infere_mass"] = False

            if use_two_halo_term == True and two_halo_power == True:
                sub_sampler_kwargs["two_halo_power"] = two_halo_power
                sub_sampler_kwargs["n_two_halo_params"] = n_parameters_2h
            else:
                sub_sampler_kwargs["two_halo_power"] = False
            sub_sampler_kwargs["model"] = func
            sampler_kwargs[model_id] = sub_sampler_kwargs

            if store_two_halo_term == True and use_two_halo_term == True:
                if "TWO_HALO" in list(blobs_config.keys()) and "ONE_HALO" in list(blobs_config.keys()):
                    two_halo_blob = blobs_config["TWO_HALO"]
                    one_halo_blob = blobs_config["ONE_HALO"]
                    dt_1h, shape_1h = prop2arr(one_halo_blob, dtype = str)
                    dt_2h, shape_2h = prop2arr(two_halo_blob, dtype = str)
                    shape_1h = int(shape_1h.replace('(','').replace(')',''))
                    shape_2h = int(shape_2h.replace('(','').replace(')',''))
                    dt_1h = np.float64 if dt_1h == "np.float64" else getattr(np, dt_1h)
                    dt_2h = np.float64 if dt_2h == "np.float64" else getattr(np, dt_2h)
                    dtype.append(("ONE_HALO", np.dtype((dt_1h, shape_1h))))
                    dtype.append(("TWO_HALO", np.dtype((dt_2h, shape_2h))))
                    sampler_kwargs["store_two_halo_term"] = True
                elif "TWO_HALO" not in list(blobs_config.keys()) or "ONE_HALO" not in list(blobs_config.keys()):
                    raise KeyError("Blobs must contain 'TWO_HALO' and 'ONE_HALO' keys")
                elif use_two_halo_term == False:
                    print("Store two halo term can't be True if use two halo term is False!")
                    sampler_kwargs["store_two_halo_term"] = False
            else:
                sampler_kwargs["store_two_halo_term"] = False
            
        sampler_kwargs["models"] = models
        sampler_kwargs["params_idx"] = params_indx

        clusters = jclusters
        clusters, cov = grouped_clusters.compute_joint_cov(off_diag = off_diag, groups = clusters, corr = False)
        Y = np.concatenate([c.mean_profile for c in clusters])
        inv_cov = np.linalg.inv(cov)

        fixed_params = jfixed_params
        free_params = jfree_params

        if np.array(cov).ndim > 1:
            log_det_C = np.linalg.slogdet(cov)[1]
            inv_cov_matrix = np.linalg.inv(cov)        
            sampler_kwargs["log_det_C"] = log_det_C
            sampler_kwargs["inv_cov_matrix"] = inv_cov_matrix
        param_limits = [
            np.array(jprior_parameters[i][2].split("|")).astype(float_dtype)[-2::]
            if "free" in jprior_parameters[i] else float(jprior_parameters[i][-1])
            for i in range(len(jprior_parameters))
            
        ]

        jprior_distributions = [
            getattr(MCMC_func, jprior_parameters[i][1])
            if "free" in jprior_parameters[i] else "fixed"
            for i in range(len(jprior_parameters))
        ]
        jprior_distribution_args = [
            np.array(jprior_parameters[i][2].split("|")).astype(float_dtype)
            if "free" in jprior_parameters[i] else "fixed"
            for i in range(len(jprior_parameters))
        ]
        ndims = len(jfree_params)
        initial_guess = np.zeros((nwalkers, len(jprior_parameters)))
        
        for i in range(len(param_limits)):
            if len(jprior_parameters[i]) == 3:
                initial_guess[:,i] = np.array(
                    random_initial_steps(param_limits[i], nwalkers, 
                    distribution = jprior_distributions[i],
                    dist_args = jprior_distribution_args[i],
                    nsamples = 1e3)
                    )
            else:
                initial_guess[:,i] = float(jprior_parameters[i][-1])* (1 + np.random.uniform(-0.1, 0.1, size = nwalkers))

        global ln_prior_joint
        def ln_prior_joint(theta):
            prior = 0.0
            i_theta = 0
            for i in range(len(jprior_parameters)):
                if "free" in jprior_parameters[i]:
                    args = np.array(jprior_distribution_args[i]).astype(
                        float_dtype
                    )
                    prior += jprior_distributions[i](
                        theta[i_theta], *args
                    )
                    i_theta += 1
            return prior
        global ln_likelihood_joint
        if likelihood_func == 'gaussian':
            def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                print("initial set of parameters:")
                ic(theta)
                models = kwargs["models"]
                params_idx = kwargs["params_idx"]
                store_two_halo_term = kwargs["store_two_halo_term"]
                joint_mu = []
                joint_1halo = []
                joint_2halo = []
                joint_mass = []
                ic(np.shape(sigma))
                y = np.array(y)[:,4:].flatten() if y.ndim > 1 else y
                if "model_with_halo_params" in list(kwargs.keys()):
                    model_with_halo_params = kwargs["model_with_halo_params"]
                    for i,m in enumerate(models):
                        if m == model_with_halo_params:
                            sub_kwargs = kwargs[model_with_halo_params]
                            idx = params_idx[i]
                            stheta = theta[idx[0]:idx[1]]
                            offset = 0
                            free_params = sub_kwargs["free_params"]
                            fixed_params = sub_kwargs["fixed_params"]
                            fixed_mis_centering = sub_kwargs["fixed_mis_centering"]
                            fixed_halo_model = sub_kwargs["fixed_halo_model"]
                            infere_mass = sub_kwargs["infere_mass"]
                            two_halo_power = sub_kwargs["two_halo_power"]
                            if len(fixed_params) > 1 and in_demo_mode == False:
                                free_params_indx = [p[0] for p in free_params]
                                fixed_params_indx = [p[0] for p in fixed_params]
                                fixed_params_values = [p[0] for p in fixed_params]
                                new_theta = np.empty(len(fixed_params) + len(free_params))
                                new_theta[fixed_params_indx] = fixed_params_values
                                new_theta[free_params_indx] = stheta
                                stheta = new_theta   
                            if two_halo_power == True:
                                n_two_halo_power = sub_kwargs["n_two_halo_params"]
                                two_halo_power = stheta[-n_two_halo_power:]
                                stheta = stheta[:-n_two_halo_power]
                            if fixed_halo_model == False:
                                n_halo_params = sub_kwargs["n_halo_params"]
                                halo_model_params = stheta[-n_halo_params:]
                                stheta = stheta[:-n_halo_params]

                ic("halo params = ", halo_model_params)
                for i,m in enumerate(models):
                    xi = x[i]
                    sub_kwargs = kwargs[m]
                    idx = params_idx[i]
                    stheta = theta[idx[0]:idx[1]]
                    model = sub_kwargs["model"]
                    in_demo_mode = sub_kwargs["in_demo_mode"] if "in_demo_mode" in list(sub_kwargs.keys()) else False
                    free_params = sub_kwargs["free_params"]
                    fixed_params = sub_kwargs["fixed_params"]
                    fixed_mis_centering = sub_kwargs["fixed_mis_centering"]
                    fixed_halo_model = sub_kwargs["fixed_halo_model"]
                    infere_mass = sub_kwargs["infere_mass"]
                    two_halo_power = sub_kwargs["two_halo_power"]
                    bins = sub_kwargs["bins"] if y.ndim == 1 else np.array(y)[:,0:4]
                
                    ic(bins)
                    ic("initial theta", stheta)
                    model_kwargs = {"cbin": bins}
                    model_kwargs["RM_params"] = halo_model_params
                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[0] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = stheta
                        stheta = new_theta
                        ic("theta with fixed parameters", stheta)
                    if two_halo_power == True:
                        n_two_halo_power = sub_kwargs["n_two_halo_params"]
                        two_halo_power = stheta[-n_two_halo_power:]
                        stheta = stheta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                        ic(two_halo_power)
                        ic("theta without two_halo_power", stheta)
                    if fixed_halo_model == False:
                        n_halo_params = sub_kwargs["n_halo_params"]
                        halo_model_params = stheta[-n_halo_params:]
                        stheta = stheta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                        ic(halo_model_params)
                        ic("theta without RM_params", stheta)
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = sub_kwargs["n_mis_centering_params"]
                            mis_centering_params = stheta[-n_mis_centering_params:]
                            stheta = stheta[:-n_mis_centering_params]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                            ic(mis_centering_params)
                            ic("theta without mis_centering_params", stheta)
                        else:
                            model_kwargs["mis_centering_params"] = sub_kwargs["mis_centering_params"]
                    mu = model(xi, stheta, **model_kwargs) 
                    ic(mu)
                    if store_two_halo_term == True and infere_mass == True:
                        mu, one_halo, two_halo, mass = mu
                    elif store_two_halo_term == True and infere_mass == False:
                        mu, one_halo, two_halo = mu
                    elif store_two_halo_term == False and infere_mass == True:
                        mu, mass = mu
                        
                    joint_mu = joint_mu + list(mu)
                    if store_two_halo_term == True:
                        joint_1halo = joint_1halo + list(one_halo)
                        joint_2halo = joint_2halo + list(two_halo)
                    if infere_mass == True:
                        joint_mass = joint_mass + list(mass)

                mu = np.array(joint_mu)
                one_halo = np.array(joint_1halo)
                two_halo = np.array(joint_2halo)
                mass = np.array(joint_mass)
                ic(np.shape(y))
                ic(np.shape(mu))
                if "inv_cov_matrix" not in list(kwargs.keys()):
                    log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                        (y - mu) ** 2
                    ) / (sigma**2)
                    current_chi2 = np.sum((y - mu)**2 / sigma**2)
                    if store_two_halo_term == True and infere_mass == True:
                        return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo, mass
                    elif store_two_halo_term == True and infere_mass == False:
                        return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo
                    elif store_two_halo_term == False and infere_mass == True:
                        return np.sum(log_likelihood), current_chi2, mu, mass
                    return np.sum(log_likelihood), current_chi2, mu
                else:
                    inv_cov_matrix = kwargs["inv_cov_matrix"]
                    log_det_C = kwargs["log_det_C"]
                    residual = y - mu
                    current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                    ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                    if store_two_halo_term == True and infere_mass == True:
                        return ln_lk, current_chi2,mu, one_halo, two_halo, mass
                    elif store_two_halo_term == True and infere_mass == False:
                        return ln_lk, current_chi2,mu, one_halo, two_halo
                    elif store_two_halo_term == False and infere_mass == True:
                        return ln_lk, current_chi2,mu, mass
                    return ln_lk, current_chi2 ,mu
        elif likelihood_func == "chi2":
            def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                models = kwargs["models"]
                params_idx = kwargs["params_idx"]
                store_two_halo_term = kwargs["store_two_halo_term"]
                joint_mu = []
                joint_1halo = []
                joint_2halo = []
                joint_mass = []
                y = np.array(y)[:,4:].flatten() if y.ndim > 1 else y
                if "model_with_halo_params" in list(kwargs.keys()):
                    model_with_halo_params = kwargs["model_with_halo_params"]
                    for i,m in enumerate(models):
                        if m == model_with_halo_params:
                            sub_kwargs = kwargs[model_with_halo_params]
                            idx = params_idx[i]
                            theta = theta[idx[0]:idx[1]]
                            theta_full = theta 
                            N = len(theta_full)
                            offset = 0
                            if two_halo_power:
                                n_two_halo = kwargs["n_two_halo_params"]
                                offset += n_two_halo
                            if use_mis_centering and not fixed_mis_centering:
                                n_mis = kwargs["n_mis_centering_params"]
                                offset += n_mis
                            n_halo = kwargs["n_halo_params"]
                            halo_start = N - offset - n_halo
                            halo_end = N - offset
                            halo_model_params = theta_full[halo_start:halo_end]

                for i,m in enumerate(models):
                    halo_model_params = None
                    xi = x[i]
                    sub_kwargs = kwargs[m]
                    idx = params_idx[i]
                    sub_theta = theta[idx[0]:idx[1]]
                    model = sub_kwargs["model"]
                    in_demo_mode = sub_kwargs["in_demo_mode"] if "in_demo_mode" in list(sub_kwargs.keys()) else False
                    free_params = sub_kwargs["free_params"]
                    fixed_params = sub_kwargs["fixed_params"]
                    fixed_mis_centering = sub_kwargs["fixed_mis_centering"]
                    fixed_halo_model = sub_kwargs["fixed_halo_model"]
                    infere_mass = sub_kwargs["infere_mass"]
                    two_halo_power = sub_kwargs["two_halo_power"]
                    bins = sub_kwargs["bins"] if y.ndim == 1 else np.array(y)[:,0:4]
                
                    ic(bins)
                    ic("initial theta", theta)
                    model_kwargs = {"cbin": bins}

                    if len(fixed_params) > 1 and in_demo_mode == False:
                        free_params_indx = [p[0] for p in free_params]
                        fixed_params_indx = [p[0] for p in fixed_params]
                        fixed_params_values = [p[0] for p in fixed_params]
                        new_theta = np.empty(len(fixed_params) + len(free_params))
                        new_theta[fixed_params_indx] = fixed_params_values
                        new_theta[free_params_indx] = theta
                        theta = new_theta
                        ic("theta with fixed parameters", theta)
                    if two_halo_power == True:
                        n_two_halo_power = sub_kwargs["n_two_halo_params"]
                        two_halo_power = theta[-n_two_halo_power:]
                        theta = theta[:-n_two_halo_power]
                        model_kwargs["two_halo_power"] = two_halo_power
                        ic(two_halo_power)
                        ic("theta without two_halo_power", theta)
                    if fixed_halo_model == False:
                        n_halo_params = sub_kwargs["n_halo_params"]
                        halo_model_params = theta[-n_halo_params:]
                        theta = theta[:-n_halo_params]
                        model_kwargs["RM_params"] = halo_model_params
                        ic(halo_model_params)
                        ic("theta without RM_params", theta)
                    if use_mis_centering == True:
                        if fixed_mis_centering == False:
                            n_mis_centering_params = sub_kwargs["n_mis_centering_params"]
                            mis_centering_params = theta[-n_mis_centering_params:]
                            theta = theta[:-n_mis_centering_params]
                            model_kwargs["mis_centering_params"] = mis_centering_params
                            ic(mis_centering_params)
                            ic("theta without mis_centering_params", theta)
                        else:
                            model_kwargs["mis_centering_params"] = sub_kwargs["mis_centering_params"]
                    mu = model(xi, theta, **model_kwargs) 
                    ic(mu)
                    if store_two_halo_term == True and infere_mass == True:
                        mu, one_halo, two_halo, mass = mu
                    elif store_two_halo_term == True and infere_mass == False:
                        mu, one_halo, two_halo = mu
                    elif store_two_halo_term == False and infere_mass == True:
                        mu, mass = mu
                        
                    joint_mu = joint_mu + list(mu)
                    joint_1halo = joint_1halo + list(one_halo)
                    joint_2halo = joint_2halo + list(two_halo)
                    joint_mass = joint_mass + list(mass)

                mu = np.array(joint_mu)
                one_halo = np.array(joint_1halo)
                two_halo = np.array(joint_2halo)
                mass = np.array(joint_mass)

                if "inv_cov_matrix" not in list(kwargs.keys()):
                    log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                        (y - mu) ** 2
                    ) / (sigma**2)
                    current_chi2 = np.sum((y - mu)**2 / sigma**2)
                    if store_two_halo_term == True and infere_mass == True:
                        return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo, mass
                    elif store_two_halo_term == True and infere_mass == False:
                        return np.sum(log_likelihood), current_chi2, mu, one_halo, two_halo
                    elif store_two_halo_term == False and infere_mass == True:
                        return np.sum(log_likelihood), current_chi2, mu, mass
                    return np.sum(log_likelihood), current_chi2, mu
                else:
                    inv_cov_matrix = kwargs["inv_cov_matrix"]
                    log_det_C = kwargs["log_det_C"]
                    residual = y - mu
                    current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                    ln_lk = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                    if store_two_halo_term == True and infere_mass == True:
                        return ln_lk, current_chi2, mu, one_halo, two_halo, mass
                    elif store_two_halo_term == True and infere_mass == False:
                        return ln_lk, current_chi2, mu, one_halo, two_halo
                    elif store_two_halo_term == False and infere_mass == True:
                        return ln_lk, current_chi2, mu, mass
                    return ln_lk, current_chi2, mu
            #======


        sampler_kwargs["ln_likelihood"] = ln_likelihood_joint
        sampler_kwargs["ln_posterior"] = ln_posterior_func
        sampler_kwargs["ln_prior"] = ln_prior_joint
        sampler_kwargs["chi2"] = compute_chi2
        sigma = cov

        cond_number = np.linalg.cond(sigma)
        epsilon = np.finfo(np.float64).eps if float_dtype == np.float64 else np.finfo(np.float32).eps

        if cond_number >= 1/epsilon:
            print(f"Condition number of covariance matrix is too large: \033[31m{cond_number}\033[0m")
        else:
            print(f"Condition number of covariance matrix: \033[32m{cond_number}\033[0m")
        print("\n")

        fil_name = "x".join(models)
        filename = output_path + "joint_fit_" + fil_name + "." + ext
        print("Y = ", Y)
        print("Y shape = ", np.shape(Y))
        from pathos.multiprocessing import ProcessingPool as Pool
        pool = Pool(ncores)
        initial_guess = np.delete(initial_guess, np.asarray([p[0] for p in fixed_params], dtype = int), axis = 1)   
        if del_backend == True:
            if os.path.exists(filename):
                os.remove(filename)
        backend = emcee.backends.HDFBackend(filename)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndims,
            ln_posterior_func,
            args=(
                np.array(X, dtype = object),
                np.array(Y, dtype = object),
                sigma,
            ),
            kwargs=sampler_kwargs,
            pool = pool,
            backend = backend,
            blobs_dtype = dtype
        )
        print("\033[44m",10*"=","RUNNING MCMC",10*"=" ,"\033[0m")
        print(f"* paths:")
        [print(f"\t-\033[35m{p}\033[0m") for p in paths]
        print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
        print(f"* output file: \033[35m{filename}\033[0m")
        print(" ",30*"="," ")
        t1 = time()
        sampler.run_mcmc(initial_guess, nsteps, progress=True)
        t2 = time()
        print("MCMC was finish in ", t2  - t1,"seconds.")

























def run_demo_mode(
    demo,
    sampler_kwargs,
    specific_path,
    profile_stacked_model,
    use_two_halo_term,
    likelihood_func,
    filename,
    initial_guess,
    cluster,
    xlabel,
    ylabel,
    yscale,
    xscale,
    model_id,
    ln_likelihood,
    demo_path,
    profile,
    params_labels,
    R
):
    from plottery.plotutils import update_rcParams
    update_rcParams()
    sampler_kwargs["in_demo_mode"] = True
    print("\033[44m", 10 * "=", "RUNNING DEMO MODE", 10 * "=", "\033[0m")
    print(f"* entire data available on: \033[35m{specific_path}\033[0m")
    print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
    print(f"* two halo term: \033[35m{use_two_halo_term}\033[0m")
    print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
    print(" ",30*"="," ")
    run_random_demo = input("Do you want to run a random demo? (y/n): ")
    if os.path.exists(f"{specific_path}/{demo_path}") == False:
        os.mkdir(f"{specific_path}/{demo_path}")
    if run_random_demo.lower() in ["y", "yes",""]:
        try:
            ndemos = int(input("Number of demos to run: "))
        except ValueError:
            print("Invalid input. Using default value of 10.")
            ndemos = 10
        random_indices = np.random.choice(np.arange(initial_guess.shape[0]), ndemos, replace=False)
        random_initial_guess = initial_guess[random_indices]
        fig, ax = plt.subplots(figsize = (14,8))
        ln_lk = np.zeros(len(random_initial_guess))
        chi2s = np.zeros(len(random_initial_guess))
        profs = np.zeros(len(random_initial_guess))
        P1halos = np.zeros(len(random_initial_guess)) 
        P2halos = np.zeros(len(random_initial_guess))
        Masses = np.zeros(len(random_initial_guess))
        res = []
        for i, guess in enumerate(random_initial_guess):
            res.append(ln_likelihood(guess, R, profile, None, **sampler_kwargs))
            ln_lk[i] = res[0]
            ic(ln_lk[i])
            chi2s[i] = res[1]
            ic(chi2s[i])
            profs[i] = res[2]
            if sampler_kwargs["store_two_halo_term"] == True and sampler_kwargs["infere_mass"] == False:
                P1halos[i] = profs[i][1]
                P2halos[i] = profs[i][2]
                profs[i] = profs[i][0]
            elif sampler_kwargs["store_two_halo_term"] == True and sampler_kwargs["infere_mass"] == True:
                P1halos[i] = profs[i][1]
                P2halos[i] = profs[i][2]
                Masses[i] = profs[i][3]
                profs[i] = profs[i][0]
            elif sampler_kwargs["store_two_halo_term"] == False and sampler_kwargs["infere_mass"] == True:
                Masses = profs[i][1]
                profs[i] = profs[i][0]
        ax.plot(R, profs)
        fig.savefig(f"{specific_path}/{demo_path}/random_demo_profiles.png", dpi=200)
def run_demo_mode_general(
    demo,
    sampler_kwargs,
    specific_path,
    profile_stacked_model,
    use_two_halo_term,
    likelihood_func,
    filename,
    initial_guess,
    clusters_list,
    xlabel,
    ylabel,
    yscale,
    xscale,
    model_id,
    ln_likelihood_general,
    demo_path,
    profiles,
    params_labels,
    R
):
    from plottery.plotutils import update_rcParams
    update_rcParams()
    sampler_kwargs["in_demo_mode"] = True
    print("\033[44m", 10 * "=", "RUNNING DEMO MODE", 10 * "=", "\033[0m")
    print(f"* entire data available on: \033[35m{specific_path}\033[0m")
    print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
    print(f"* two halo term: \033[35m{use_two_halo_term}\033[0m")
    print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
    print(f"* output file: \033[35m{filename}\033[0m")
    print(" ", 30 * "=", " ")
    # Random demos
    run_specific_demo = input("Run an specific demo? (y/n):")
    if run_specific_demo.lower() in ["", "y", "yes"]:
        params = np.array(input("Enter the desired parameters (separated by coma): ").split(","), dtype = float)
        res = ln_likelihood_general(params, R, profiles, None, **sampler_kwargs)
        ln_lk = res[0]
        chi2 = res[1]
        mu = res[2]
        residuals = mu - profiles
        inv_cov = sampler_kwargs["inv_cov_matrix"]
        manual_chi2 = np.dot(residuals.T, np.dot(inv_cov, residuals))
        print("ln likelihood:", ln_lk)
        print("chi2:", chi2)
        print("mu:", mu)
        print("manual chi2:", manual_chi2)

    run_random_demo = input("Run random demo? (y/n): ")
    if run_random_demo.lower() in ["", "y", "yes"]:
        try:
            ndemos = int(input("Enter the desired number of random demos (10 as default): "))
        except ValueError:
            print("An invalid number of random demos were introduced! Setting to 10")
            ndemos = 10

        random_indices = np.random.choice(initial_guess.shape[0], ndemos, replace=False)
        random_initial_guess = initial_guess[random_indices]
        fig, ax = plt.subplots(figsize=(12, 8))
        t1 = time()

        # plot measured data
        for j, cj in enumerate(clusters_list):
            prof = cj.mean_profile
            err = cj.error_in_mean
            R = cj.R
            ax.errorbar(R + 0.5 * j, prof, yerr=err, ls="--", lw=3, color="grey")
        ax.set(xlabel=xlabel, ylabel=ylabel, yscale=yscale, xscale=xscale, title=model_id)

        progress_bar = tqdm(total=len(random_initial_guess), desc="Running Demo")
        ln_lks = np.zeros(ndemos)
        chi2s = np.zeros(ndemos)
        mu_s = np.zeros(ndemos, dtype=object)
        import matplotlib.colors as mcolors
        colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), ndemos)

        for j, p in enumerate(random_initial_guess):
            ln_lks[j], chi2s[j], mu_s[j] = ln_likelihood_general(p, R, profiles, None, **sampler_kwargs)
            progress_bar.update(1)
        progress_bar.close()

        # overlay model curves
        for j, p in enumerate(random_initial_guess):
            if np.isinf(chi2s[j]) or np.isnan(chi2s[j]):
                continue
            if np.isinf(ln_lks[j]) or np.isnan(ln_lks[j]):
                continue
            mu_j = mu_s[j]
            mu_jk = np.reshape(mu_j, (len(clusters_list), len(R)))
            for n, mu_jkn in enumerate(mu_jk):
                ax.plot(R + 0.5 * n, mu_jkn, color=colors[j], alpha=0.6, ls="solid")
            ilabel = (
                r"$\theta$ = " + str(np.round(p, 1)) + f" \n$\chi^2 = $" + str(np.round(chi2s[j],1)) +
                f"\n $\ln{{\mathcal{{L}}}}$ = " + str(np.round(ln_lks[j],1))
            )
            ax.plot([], [], color=colors[j], label=ilabel)

        ymin, ymax = np.min(profiles), np.max(profiles)
        ymin = ymin - 0.5 * ymin if ymin > 0 else ymin + 0.5 * ymin
        ymax = ymax + 0.5 * ymax if ymax > 0 else ymax - 0.5 * ymax
        ax.set_ylim((ymin, ymax))

        print(f"Random demo ended in  {time() - t1} seconds! Saving to {demo_path}!")
        os.makedirs(os.path.join(specific_path, demo_path), exist_ok=True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{specific_path}/{demo_path}/random_demo_profiles.png", dpi=200)

    # Demo per parameter
    run_demo_per_param = input("Run demo for each parameter? (Yes/No): ")
    if run_demo_per_param.lower() in ["", "yes", "y"]:
        random_base = input("Use random set as base? (yes/no): ")
        if random_base.lower() in ["", "yes", "y"]:
            demo_params_base = np.median(initial_guess, axis=0)
        else:
            demo_params_base = np.array(input("Enter parameters separated by comma: ").split(","), dtype=float)

        for i, label in enumerate(params_labels):
            while True:
                pout = label.replace("{","").replace("}","").replace("\\","")
                run_demo = input(f"Run demo to {pout} parameter? (Yes/No): ")
                if run_demo.lower() not in ["", "yes", "y"]:
                    print("="*20)
                    break

                Ndemos_param = int(input(f"Enter N demos: "))
                vmin, vmax = map(float, input("Enter param limits vmin,vmax: ").split(","))
                log_scale = input("Log scale? (Yes/No): ").lower() in ["", "yes", "y"]
                p_evals = (
                    np.linspace(vmin, vmax, Ndemos_param)
                    if not log_scale else np.logspace(vmin, vmax, Ndemos_param)
                )
                params = np.tile(demo_params_base, (Ndemos_param,1))
                params[:, i] = p_evals
                use_pool = input("Use multiprocessing? (Yes/No): ").lower() in ["", "yes", "y"]
                N_evals = len(p_evals)

                if not use_pool:
                    ln_lks_p = np.zeros(N_evals)
                    chi2s_p = np.zeros(N_evals)
                    mu_s_p = np.zeros(N_evals, dtype=object)
                    progress = tqdm(total=N_evals, desc=f"Running demo to {pout}")
                    for j, p in enumerate(params):
                        ln_lks_p[j], chi2s_p[j], mu_s_p[j] = ln_likelihood_general(p, R, profiles, None, **sampler_kwargs)
                        progress.update(1)
                    progress.close()
                else:
                    try:
                        ncores = int(input("Enter number of cores: "))
                    except:
                        print("Invalid cores, using all available.")
                        ncores = cpu_count()
                    pool = Pool(ncores)
                    N_total = len(params)
                    manager = Manager()
                    counter = manager.Value("i", 0)
                    params_split = np.array_split(params, ncores)
                    async_res = [
                        pool.apply_async(demo_worker, args=(chunk, R, profiles, sampler_kwargs, N_total, counter))
                        for chunk in params_split
                    ]
                    results = [r.get() for r in async_res]
                    ln_lks_p = np.concatenate([r[0] for r in results])
                    chi2s_p = np.concatenate([r[1] for r in results])
                    mu_s_p = np.concatenate([r[2] for r in results])
                    pool.close()

                # Plot chi2 vs param
                os.makedirs(f"{specific_path}/{demo_path}/{pout}", exist_ok=True)
                sort_idx = np.argsort(p_evals)
                sort_params = p_evals[sort_idx]
                sort_chi2 = gaussian_filter1d(chi2s_p[sort_idx], 2)
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(sort_params, sort_chi2, lw=3, color="black")
                ax.set(
                    title=f"Demo for {text2latex(label)}",
                    xlabel=text2latex(label),
                    ylabel="$\chi^2$",
                    yscale="log"
                )
                fig.tight_layout()
                fig.savefig(f"{specific_path}/{demo_path}/{pout}/chi2.png")

                # Profile overlays for this param
                fig, ax = plt.subplots(figsize=(12,10))
                cmap = plt.cm.Reds
                norm = plt.Normalize(p_evals.min(), p_evals.max())
                for j, cj in enumerate(clusters_list):
                    ax.errorbar(cj.R + 0.5*j, cj.mean_profile, yerr=cj.error_in_mean,
                                ls="--", lw=3, color="grey")
                ax.set(xlabel=xlabel, ylabel=ylabel, yscale=yscale, xscale=xscale, title=model_id)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                for j, pval in enumerate(p_evals):
                    if np.isinf(chi2s_p[j]) or np.isnan(chi2s_p[j]): continue
                    mu_jk = np.reshape(mu_s_p[j], (len(clusters_list), len(R)))
                    for n, mu_prof in enumerate(mu_jk):
                        style = "dotted" if chi2s_p[j]==np.min(chi2s_p) else "solid"
                        lw = 4 if style=="dotted" else 2
                        ax.plot(R+0.5*n, mu_prof, ls=style, lw=lw,
                                color=cmap(norm(pval)), alpha=0.6)
                plt.colorbar(sm, cax=fig.add_axes([0.92,0.1,0.02,0.8]), label=text2latex(label))
                ymin, ymax = np.min(profiles), np.max(profiles)
                ymin = ymin-0.5*ymin if ymin>0 else ymin+0.5*ymin
                ymax = ymax+0.5*ymax if ymax>0 else ymax-0.5*ymax
                ax.set_ylim((ymin,ymax))
                fig.tight_layout()
                fig.savefig(f"{specific_path}/{demo_path}/{pout}/profiles.png")

                if input(f"Repeat demo for {label}? ").lower() not in ["","y","yes"]:
                    break
                if input("Change base parameters? (yes/no): ").lower() in ["","yes","y"]:
                    demo_params_base = np.array(input("Enter parameters separated by comma: ").split(","), dtype=float)

    # Grid demo
    run_demo_grid = input("Run demo with a param grid? (y/n) ")
    if run_demo_grid.lower() in ["", "y", "yes"]:
        random_base = input("Use random set as base? (yes/no): ")
        if random_base.lower() in ["", "yes", "y"]:
            demo_params_base = np.median(initial_guess, axis=0)
        else:
            demo_params_base = np.array(input("Enter parameters separated by comma: ").split(","), dtype=float)

        grid, indices = [], []
        for i, label in enumerate(params_labels):
            pout = label.replace("{","").replace("}","").replace("\\","")
            if input(f"Add param {pout} to grid? (y/n) ").lower() in ["","y","yes"]:
                vmin, vmax = map(float, input(f"Enter vmin and vmax for {pout}: ").split(","))
                try:
                    Nevals = int(input(f"Enter number of evaluations for {pout} (10 as default): "))
                except:
                    Nevals = 10
                log_space = input("Use log space for {pout}? (y/n) ").lower() in ["","y","yes"]
                p_evals = (
                    np.linspace(vmin, vmax, Nevals) if not log_space else np.logspace(np.log10(vmin), np.log10(vmax), Nevals)
                )
                grid.append(p_evals)
                indices.append(i)
            else:
                grid.append([demo_params_base[i]])

        grid_mesh = np.meshgrid(*grid, indexing="ij")
        flat_grid = np.stack([g.flatten() for g in grid_mesh], axis=-1)
        N_evals = len(flat_grid)
        use_pool = input("Use multiprocessing? (y/n) ").lower() in ["","y","yes"]

        if not use_pool:
            ln_lks_g = np.zeros(N_evals)
            chi2s_g = np.zeros(N_evals)
            mu_s_g = np.zeros(N_evals, dtype=object)
            progress = tqdm(total=N_evals, desc="Running demo in grid mode")
            for j, p in enumerate(flat_grid):
                ln_lks_g[j], chi2s_g[j], mu_s_g[j] = ln_likelihood_general(p, R, profiles, None, **sampler_kwargs)
                progress.update(1)
            progress.close()
        else:
            try:
                ncores = int(input("Enter number of cores: "))
            except:
                ncores = cpu_count()
            pool = Pool(ncores)
            manager = Manager()
            counter = manager.Value("i", 0)
            splits = np.array_split(flat_grid, ncores)
            async_res = [
                pool.apply_async(demo_worker, args=(chunk, R, profiles, sampler_kwargs, N_evals, counter))
                for chunk in splits
            ]
            results = [r.get() for r in async_res]
            ln_lks_g = np.concatenate([r[0] for r in results])
            chi2s_g = np.concatenate([r[1] for r in results])
            mu_s_g = np.concatenate([r[2] for r in results])
            pool.close()

        shape = [len(g) for g in grid]
        chi2_grid = chi2s_g.reshape(shape)
        if input("Plot log chi2? (y/n) ").lower() in ["","y","yes"]:
            chi2_grid = np.log(chi2_grid)
        best_idx = np.unravel_index(np.argmin(chi2_grid), shape)

        d = len(indices)
        fig, axes = plt.subplots(d, d, figsize=(5*d,5*d))
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                idx_i = indices[i]
                idx_j = indices[j]
                if j > i:
                    ax.axis('off')
                    continue
                if i == j:
                    slc = [slice(None) if k==idx_i else best_idx[k] for k in range(len(shape))]
                    chi2_1d = chi2_grid[tuple(slc)]
                    best_val = grid[idx_i][best_idx[idx_i]]
                    ax.plot(grid[idx_i], chi2_1d)
                    if i==d-1:
                        ax.set_xlabel(text2latex(params_labels[idx_i]))
                    else:
                        ax.set_xticklabels([])
                    ax.set_ylabel(r"$\chi^2$")
                    ax.set_title(f"best {text2latex(params_labels[idx_i])} = {best_val:.2f}")
                else:
                    fixed = [best_idx[k] for k in range(len(shape)) if k not in (idx_i, idx_j)]
                    Z = chi2_grid[tuple([slice(None), slice(None)] + fixed)]
                    x0, x1 = grid[idx_j][0], grid[idx_j][-1]
                    y0, y1 = grid[idx_i][0], grid[idx_i][-1]
                    im = ax.imshow(Z, origin='lower', aspect='auto', extent=[x0,x1,y0,y1], interpolation='gaussian')
                    if j==0:
                        ax.set_ylabel(text2latex(params_labels[idx_i]))
                    else:
                        ax.set_yticklabels([])
                    if i==d-1:
                        ax.set_xlabel(text2latex(params_labels[idx_j]))
                    else:
                        ax.set_xticklabels([])

        pos = axes[0,0].get_position()
        cax = fig.add_axes([1-0.02-0.02, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(r"$\chi^2$")
        fig.tight_layout()
        fig.savefig(f"{specific_path}/{demo_path}/demo_grid.png", dpi=200, bbox_inches="tight")
        return


if __name__ == "__main__":
    if args.debug == False:
        ic.disable()
    main()
