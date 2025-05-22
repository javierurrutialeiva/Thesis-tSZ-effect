import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
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

global compute_chi2
def compute_chi2(y,y1,sigma):
    if len(np.shape(sigma)) == 1:
        y,y1,sigma = np.array(y),np.array(y1),np.array(sigma)
        return np.abs(np.sum((y - y1)**2 / sigma **2))
        cov_inv = sigma
        res = np.array(y - y1)
        chi = np.dot(np.dot(res.T,cov_inv),res)
        return chi

global ln_posterior_func 
def ln_posterior_func(theta, x, y, sigma, **kwargs):
    model = kwargs["model"]
    ln_prior_func = kwargs["ln_prior"]
    ln_likelihood_func = kwargs["ln_likelihood"]
    chi2_func = kwargs["chi2"]
    lp = ln_prior_func(theta)
    likelihood, current_chi2, y1 = ln_likelihood_func(theta, x, y, sigma, **kwargs)
    posterior = likelihood + lp
    if not np.isfinite(posterior) or np.isnan(posterior):
        return -np.inf, chi2_func(y,y1,sigma), -np.inf, -np.inf, y1
    return posterior, current_chi2, lp, likelihood, y1


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

parser.add_argument("--ask_to_add", "-Y", action = "store_false", help = "If passed wont be asked to add new clusters to the group and will be assumed the existence of a ignore.txt file in the path")
args = parser.parse_args()



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
            ncores = int(emcee_config["ncores"]) if "ncores" in list(emcee_config.keys()) else None

        plot_cov_matrix = args.plot_cov_matrix
        ncores = str(args.ncores) if ncores is None else ncores
        r_units = args.r_units
        profile_stacked_model = model_config["profile"]
        rewrite = str2bool(emcee_config["rewrite"])
        likelihood_func = emcee_config["likelihood"]
        nwalkers = int(emcee_config["nwalkers"])
        nsteps = int(emcee_config["nsteps"])
        del_backend = str2bool(emcee_config["delete"])
        rotate_cov_matrix = str2bool(model_config["rotate_cov_matrix"])
        use_filters = str2bool(model_config["use_filters"])
        filters = model_config["filters"].split("|")
        fil_name, ext = list(prop2arr(emcee_config["output_file"], dtype=str))
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

        if fit_entire_data == False:

            clusters = grouped_clusters.load_from_path(specific_path)
            output_path = clusters.output_path
            clusters.mean(from_path = True)
            if rotate_cov_matrix:
                clusters.rotate_cov_matrix()
            R = clusters.R
            rmax, rmin = np.max(clusters.richness), np.min(clusters.richness)
            zmax, zmin = np.max(clusters.z), np.min(clusters.z)
            profile = np.array(clusters.mean_profile)
            errors = np.array(clusters.error_in_mean)
            sigma = clusters.cov if hasattr(clusters, "cov") else errors
            rbins, zbins, Mbins = model_config["rbins"], model_config["zbins"], model_config["Mbins"]
            rbins = int(rbins)
            zbins = int(zbins)
            Mbins = int(Mbins)
            func = clusters.stacked_halo_model_func(getattr(profiles_module, profile_stacked_model), 
                                                    rbins = rbins, zbins = zbins, Mbins = Mbins,
                                                    use_filters = use_filters, filters = filters_dict,
                                                    units = r_units)
            global ln_prior

            prior_parameters = dict(prior_config)
            prior_parameters_dict = {
                key: list(prop2arr(prior_parameters[key], dtype=str))
                for key in list(prior_parameters.keys())
            }
            prior_parameters = list(prior_parameters_dict.values())
            def ln_prior(theta):
                prior = 0.0
                i_theta = 0
                for i in range(len(prior_parameters)):
                    if "free" in prior_parameters[i]:
                        args = np.array(prior_parameters[i][-1].split("|")).astype(
                            np.float64
                        )
                        prior += getattr(MCMC_func, prior_parameters[i][1])(
                            theta[i_theta], *args
                        )
                        i_theta += 1
                return prior

            global ln_likelihood
            if likelihood_func == 'gaussian':
                if np.array(sigma).ndim == 1:
                    def ln_likelihood(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        mu = model(x, theta)
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (y - mu) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum((y - mu)**2 / sigma**2)
                        return np.sum(log_likelihood),mu
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2,  y1
            elif likelihood_func == "chi2":
                if np.array(sigma).ndim == 1:
                    def ln_likelihood(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        res = np.sum(((y - y1) / sigma)**2)
                        return -0.5 * res, res, y1
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 = model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2, y1
                #======
            param_limits = [
                np.array(prior_parameters[i][-1].split("|")).astype(float)[-2::]
                for i in range(len(prior_parameters))
                if "free" in prior_parameters[i]
            ]
            ndims = len(param_limits)
            initial_guess = np.array([random_initial_steps(param_limits[i], nwalkers) for i in range(len(param_limits))]).T
            pool = Pool(ncores)
            filename = output_path + "/" + fil_name + "." + ext
            if del_backend == True:
                if os.path.exists(filename):
                    os.remove(filename)
            backend = emcee.backends.HDFBackend(filename)
            dtype = []
            for i,key in enumerate(list(blobs_config.keys())):
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
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndims,
                ln_posterior_func,
                args=(
                    R,
                    profile,
                    sigma,
                ),
                kwargs={
                    "model": func,
                    "ln_prior": ln_prior,
                    "ln_likelihood": ln_likelihood,
                    "chi2": compute_chi2
                },
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
                                np.random.uniform(*param_limits[j] * 0.90)
                                for j in range(len(param_limits))
                                ]
                            )
                            for i in range(nwalkers)
                            ]
                        )
                    print("Chain can't open the last sample. return the exception: \n",e)
                
            print("\033[44m",10*"=","RUNNING MCMC",10*"=" ,"\033[0m")
            print(f"* richness: \033[32m[{int(rmin)},{int(rmax)}]\033[0m")
            print(f"* redshift: \033[32m[{round(zmin,3)},{round(zmax,3)}]\033[0m")
            print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
            print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
            print(f"* output file: \033[35m{filename}\033[0m")
            print(" ",30*"="," ")
            t1 = time()
            sampler.run_mcmc(initial_guess, nsteps, progress=True, store = True)
            t2 = time()
            print(f"The sampler of richness \033[32m[{int(rmin)},{int(rmax)}]\033[0m and redshift \033[32m[{round(zmin,3)},{round(zmax,3)}]\033[0m was finished in {t2 - t1} seconds.")

        if fit_entire_data == True:

            print("Fitting whole data...")
            list_of_clusters = []
            specific_path = args.path
            specific_path = specific_path + "/" if specific_path [-1] != "/" else specific_path 
            ignore = np.loadtxt(specific_path + "ignore.txt", dtype = str).T if os.path.exists(specific_path + "ignore.txt") else []
            available_paths = [path for path in os.listdir(specific_path) if os.path.isdir(specific_path + path)]
            richness_bins = []
            paths = []
            for path in available_paths:
                if path in ignore:
                    continue
                current_path = specific_path + path
                try:
                    clusters = grouped_clusters.load_from_path(current_path)
                    clusters.mean(from_path = True)
                    if is_running_via_nohup() == False and args.ask_to_add == True:
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
            rbins, zbins, Mbins = model_config["rbins"], model_config["zbins"], model_config["Mbins"]
            rbins = int(rbins)
            zbins = int(zbins)
            Mbins = int(Mbins)

            if args.redshift_bins is not None:
                try:
                    redshift_bins = prop2arr(args.redshift_bins, dtype = float)
                    redshift_bins = np.tile(redshift_bins,len(richness_bins)).reshape((len(richness_bins),2))
                except ValueError:
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
            func,cov, about_clusters, clusters, profiles, funcs = clusters.stacked_halo_model_func_by_bins(getattr(profiles_module, profile_stacked_model),
                                                zb = redshift_bins, rb = richness_bins, full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths,
                                                verbose_pivots = True, rotate_cov = rotate_cov_matrix, use_filters = use_filters, filters = filters_dict)
            R = np.round(list_of_clusters[-1].R,2)
            if plot_cov_matrix:
                N = len(funcs)
                Nr = len(R)
                rticks = np.tile(R,N)
                fig, ax = plt.subplots(figsize = (12,12))
                im = ax.imshow(np.abs(cov + np.min(cov)*1e-5), cmap = "winter", norm = LogNorm(vmin = 1e-5, vmax = 1e-3))
                plt.xlim(-0.5, Nr*N - 0.5)
                plt.ylim(-0.5, Nr*N - 0.5)
                plt.locator_params(axis='x', nbins=N*Nr) 
                plt.locator_params(axis='y', nbins=N*Nr) 
                xlabels = [item.get_text() for item in ax.get_xticklabels()]
                ylabels = [item.get_text() for item in ax.get_yticklabels()]
                n = 0
                for i in range(0, len(xlabels)):
                    if i == 0 or i == len(xlabels) - 1:
                        continue
                    else:
                        xlabels[i] = np.round(rticks[n],1)
                        ylabels[i] = np.round(rticks[n],1)
                        n+=1
                ax.set_xticklabels(xlabels, fontsize = 8)
                ax.set_yticklabels(ylabels, fontsize = 8)
                ticks = np.arange(0, N * Nr + 1, Nr) 
                for tick in ticks[:-1]:
                    plt.axhline(tick - 0.5, color='gray', lw=2)  # Horizontal grid lines
                    plt.axvline(tick - 0.5, color='gray', lw=2)  # Vertical grid lines

                for i in range(N):
                    x_corner = i * Nr + Nr/2 - Nr/2
                    y_corner = i * Nr + Nr*0.9
                    current_richness = about_clusters[i]["richness"]
                    current_redshift = about_clusters[i]["redshift"]
                    text = [
                        r"$\lambda \in [%.i,%.i]$" % current_richness,
                        r"$z \in [%.2f,%.2f]$" % current_redshift
                        ]
                    text = '\n'.join(text)
                    text = plt.text(x_corner, y_corner, text, color='yellow', ha='left', va='top', fontsize=10)
                    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),  # Edge color (black)
                                        path_effects.Normal()])  # Fill color (white)
                ax.set_xlabel("R (arcmin)", fontsize = 12)
                ax.set_ylabel("R (arcmin)", fontsize = 12)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax = cax)
                cbar.set_label(r"$\log_{10}{|cov|}$", fontsize=18)
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=12)
                ax.set_xticklabels(xlabels, rotation=90, ha='right', fontsize = 8)
                ax.set_yticklabels(ylabels, fontsize = 8)
                fig.tight_layout()
                fig.savefig(f"{specific_path}cov_general.png", transparent = True)
                np.save(f"{specific_path}cov_general.npy", cov)
            R = list_of_clusters[-1].R
            profiles = np.array(profiles)
            sigma = cov if np.array(cov).ndim > 1 else np.array(clusters.error_in_mean)
            prior_parameters = dict(prior_config)
            prior_parameters_dict = {
                key: list(prop2arr(prior_parameters[key], dtype=str))
                for key in list(prior_parameters.keys())
            }
            prior_parameters = list(prior_parameters_dict.values())
            global ln_prior_general
            def ln_prior_general(theta):
                prior = 0.0
                i_theta = 0
                for i in range(len(prior_parameters)):
                    if "free" in prior_parameters[i]:
                        args = np.array(prior_parameters[i][-1].split("|")).astype(
                            np.float64
                        )
                        prior += getattr(MCMC_func, prior_parameters[i][1])(
                            theta[i_theta], *args
                        )
                        i_theta += 1
                return prior

            global ln_likelihood_general
            if likelihood_func == 'gaussian':
                if np.array(sigma).ndim == 1:
                    def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        mu = model(x, theta)
                        residual = y - mu
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (residual) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum(residual**2/sigma**2)
                        return np.sum(log_likelihood), current_chi2, mu
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2, y1
            elif likelihood_func == "chi2":
                if np.array(sigma).ndim == 1:
                    def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        current_chi2 = np.sum(((y - y1) / sigma)**2)
                        return -0.5 * res, current_chi2, y1
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood_general(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 = model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2, y1
                #======
            param_limits = [
                np.array(prior_parameters[i][-1].split("|")).astype(float)[-2::]
                for i in range(len(prior_parameters))
                if "free" in prior_parameters[i]
            ]
            ndims = len(param_limits)
            initial_guess = np.array(
                [
                    np.array(
                        [
                            np.random.uniform(*param_limits[j] * 0.90)
                            for j in range(len(param_limits))
                        ]
                    )
                    for i in range(nwalkers)
                ]
            )
            filename = specific_path + "general_fit_" + fil_name + "." + ext
            if del_backend == True:
                if os.path.exists(filename):
                    os.remove(filename)
            backend = emcee.backends.HDFBackend(filename)
            pool = Pool(ncores)
            dtype = []
            for i,key in enumerate(list(blobs_config.keys())):
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
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndims,
                ln_posterior_func,
                args=(
                    R,
                    profiles,
                    sigma,
                ),
                kwargs={
                    "model": func,
                    "ln_prior": ln_prior_general,
                    "ln_likelihood": ln_likelihood_general,
                    "chi2": compute_chi2
                },
                pool = pool,
                backend = backend,
                blobs_dtype = dtype
            )
            print("\033[44m",10*"=","RUNNING MCMC",10*"=" ,"\033[0m")
            print(f"* entire data available on: \033[35m{specific_path}\033[0m")
            print(f"* profile model: \033[35m{profile_stacked_model}\033[0m")
            print(f"* likelihood: \033[35m{likelihood_func}\033[0m")
            print(f"* output file: \033[35m{filename}\033[0m")
            print(" ",30*"="," ")
            t1 = time()
            sampler.run_mcmc(initial_guess, nsteps, progress=True)
            t2 = time()
            print("MCMC was finish in ", t2  - t1,"seconds.")
    elif joint_fit == True:
        specific_paths = args.path.split(",")
        if args.CONFIG_FILE is not None:
            current_path = os.path.dirname(os.path.realpath(__file__))
            config_filepath = current_path +"/"+ str(args.CONFIG_FILE)
            config = ConfigParser()
            config.optionxform = str
            if os.path.exists(config_filepath):
                config.read(config_filepath)
            else:
                raise Found_Error_Config(f"The config file {str(args.CONFIG_FILE)} doesn't exist")
            available_keys = list(config.keys())
            priors_config = [dict(config[k]) for k in available_keys if "PRIORS" in list(k.split(" ")) and "HALO" not in list(k.split(" "))]
            params_edges = []
            for i in range(len(priors_config)):
                if len(params_edges) == 0:
                    params_edges.append((0, len(list(priors_config[i].keys())) - 1))
                else:
                    params_edges.append(  
                        ((params_edges[-1][-1] + 1), params_edges[-1][-1] + len(list(priors_config[i].keys())))
                        )
            n_data = len(priors_config)
            model_config = [dict(config[k]) for k in available_keys if "MODEL" in list(k.split(" ")) and "HALO" not in list(k.split(" "))]
            models = [k["profile"] for k in model_config]
            rotate_covs = [str2bool(k["rotate_cov_matrix"]) for k in model_config]
            use_filters = [str2bool(k["use_filters"]) for k in model_config]
            filters = [k["filters"].split("|") for k in model_config]
            off_diag = [str2bool(k["off_diag"]) for k in model_config]
            halo_model_priors_config = dict(config["PRIORS HALO MODEL"])
            emcee_config = config["EMCEE"]
            blobs_config = config["BLOBS"]
            general_config = config["GENERAL CONFIG"]    
            ncores = int(emcee_config["ncores"]) if "ncores" in list(emcee_config.keys()) else None
            rbins, zbins, Mbins = general_config["rbins"], general_config["zbins"], general_config["Mbins"]
            rbins = int(rbins)
            zbins = int(zbins)
            Mbins = int(Mbins)
            join_clusters_list = []
            filters_dicts = []
            joint_covs = []
            joint_profiles = []
            joint_funcs = []
            priors = []
            for p in priors_config:
                prior_parameters = dict(p)
                prior_parameters_dict = {
                    key: list(prop2arr(prior_parameters[key], dtype=str))
                    for key in list(prior_parameters.keys())
                }
                priors.append(list(prior_parameters_dict.values()))
            priors_parameters = np.concatenate(priors, axis = 0)
            def ln_prior_joint(theta):
                prior = 0.0
                i_theta = 0
                for i in range(len(priors_parameters)):
                    if "free" in priors_parameters[i]:
                        args = np.array(priors_parameters[i][-1].split("|")).astype(
                            np.float64
                        )
                        prior += getattr(MCMC_func, priors_parameters[i][1])(
                            theta[i_theta], *args
                        )
                        i_theta += 1
                return prior
            theta = np.random.normal(size = len(priors_parameters))

            for i in range(len(filters)):
                filters_dict = {}
                if use_filters[i] == True:
                    for f in filters[i]:
                        print(f)
                        f = np.array(f.replace('(','').replace(')','').split(','))
                        f = [k for k in f if len(k)>1]
                        func_name = f[0]
                        if len(f) > 1:
                            func_params = {k.split(':')[0]:k.split(':')[1] for k in f[1::]}
                        else:
                            func_params = {}
                        filters_dict[func_name] = func_params
                filters_dicts.append(filters_dict)

            for k,p in enumerate(specific_paths):
                print('Extracting data from', p)
                list_of_clusters = []
                specific_path = p
                specific_path = specific_path + "/" if specific_path [-1] != "/" else specific_path 
                ignore = np.loadtxt(specific_path + "ignore.txt", dtype = str).T if os.path.exists(specific_path + "ignore.txt") else []
                available_paths = [path for path in os.listdir(specific_path) if os.path.isdir(specific_path + path)]
                richness_bins = []
                paths = []
                for path in available_paths:
                    if path in ignore:
                        continue
                    current_path = specific_path + path
                    try:
                        clusters = grouped_clusters.load_from_path(current_path)
                        clusters.mean(from_path = True)
                        if is_running_via_nohup() == False and args.ask_to_add == True:
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
                join_clusters_list.append(all_clusters)
                if args.redshift_bins is not None:
                    try:
                        redshift_bins = prop2arr(args.redshift_bins, dtype = float)
                        redshift_bins = np.tile(redshift_bins,len(richness_bins)).reshape((len(richness_bins),2))
                    except ValueError:
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
                func,cov, about_clusters, clusters, profiles, funcs = clusters.stacked_halo_model_func_by_bins(getattr(profiles_module, models[k]),
                                                    zb = redshift_bins, rb = richness_bins, full = True, Mbins = Mbins, Rbins = rbins, Zbins = zbins, paths = paths,
                                                    verbose_pivots = True, rotate_cov = rotate_covs[k], use_filters = use_filters[k], filters = filters_dict, 
                                                    off_diag = off_diag[k])
                joint_covs.append(cov)
                fig, ax = plt.subplots(figsize = (12,12))
                ax.imshow(cov)
                fig.savefig(f"cov{k}.png")
                joint_funcs.append(func)
                joint_profiles.append(profiles)
            joint_cov = block_diag(*joint_covs)

            global joint_func
            def joint_func(R, params):
                res = np.empty()
                for i in range(len(params_edges)):
                    idx_i, idx_f = params_edges[i]
                    res = np.append(res, joint_funcs[i](R, params[idx_i:idx_f]))
                return res

            R = np.arange(0.5, 15)
            params = np.random.normal(size = len(priors_parameters))
            res = joint_func(R, params)
            print(res)
            global ln_likelihood_joint
            if likelihood_func == 'gaussian':
                if np.array(sigma).ndim == 1:
                    def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        mu = model(x, theta)
                        residual = y - mu
                        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                            (residual) ** 2
                        ) / (sigma**2)
                        current_chi2 = np.sum(residual**2/sigma**2)
                        return np.sum(log_likelihood), current_chi2, mu
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2, y1
            elif likelihood_func == "chi2":
                if np.array(sigma).ndim == 1:
                    def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 =  model(x, theta)
                        current_chi2 = np.sum(((y - y1) / sigma)**2)
                        return -0.5 * res, current_chi2, y1
                else:
                    log_det_C = np.linalg.slogdet(sigma)[1]
                    inv_cov_matrix = np.linalg.inv(sigma)
                    def ln_likelihood_joint(theta, x, y, sigma, **kwargs):
                        model = kwargs["model"]
                        y1 = model(x, theta)
                        residual = y - y1
                        current_chi2 = np.dot(residual.T, np.dot(inv_cov_matrix, residual))
                        X = -0.5 * (current_chi2 + log_det_C + len(y) * np.log(2 * np.pi))
                        return X, current_chi2, y1
if __name__ == "__main__":
    main()
