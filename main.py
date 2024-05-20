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
import re
import sys
import argparse
# load config.ini

# dtype para que cada entrada sea un arreglo: np.dtype((np.float64, shape))

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
prior_parameters = dict(config["PRIORS"])
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
nwalkers = int(config["STACKED_HALO_MODEL"]["nwalkers"])
nsteps = int(config["STACKED_HALO_MODEL"]["nsteps"])
fil_name, ext = list(prop2arr(config["STACKED_HALO_MODEL"]["output_file"], dtype=str))
rewrite = str2bool(config["STACKED_HALO_MODEL"]["rewrite"])
grouped_clusters_path = config["FILES"]["GROUPED_CLUSTERS_PATH"]
individual_clusters_path = config["FILES"]["INDIVIDUAL_CLUSTERS_PATH"]

model_profile = config["STACKED_HALO_MODEL"]["profile"]
R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
R_units = config["CLUSTER PROPERTIES"]["r_units"]
demo = str2bool(config["STACKED_HALO_MODEL"]["DEMO"])
min_richness = int(config["STACKED_HALO_MODEL"]["min richness"])
min_SNR = float(config["STACKED_HALO_MODEL"]["min_SNR"])
skip = str2bool(config["STACKED_HALO_MODEL"]["skip_fitted"])
likelihood_func = config["STACKED_HALO_MODEL"]["likelihood"]
global chi2

def chi2(y,y1,sigma):
    y,y1,sigma = np.array(y),np.array(y1),np.array(sigma)
    return np.abs(np.sum((y - y1)**2 / sigma **2))

def ln_posterior(theta, x, y, sigma, **kwargs):
    model = kwargs["model"]
    ln_prior_func = kwargs["ln_prior"]
    ln_likelihood_func = kwargs["ln_likelihood"]
    chi2 = kwargs["chi2"]
    lp = ln_prior_func(theta)
    likelihood,y1 = ln_likelihood_func(theta, x, y, sigma, **kwargs)
    posterior = likelihood + lp
    if not np.isfinite(posterior) or np.isnan(posterior):
        return -np.inf, chi2(y,y1,sigma), -np.inf, -np.inf, y1
    current_chi2 = chi2(y,y1,sigma)
    return posterior, current_chi2, lp, likelihood, y1

parser = argparse.ArgumentParser()
parser.add_argument('--individual','-i', action = "store_true", help = "If true just fit the specific given interval.")
parser.add_argument('--richness', '-r', type = list_array_or_tuple_type, help = "Richness interval.", default = (20,300))
parser.add_argument('--redshift', '-z', type = list_array_or_tuple_type, help = "Redshift interval.", default = (0,1))

args = parser.parse_args()

individual_bool = args.individual
r_interval = args.richness
z_interval = args.redshift


def main():
    R_profiles = prop2arr(config["CLUSTER PROPERTIES"]["radius"])
    R_units = config["CLUSTER PROPERTIES"]["r_units"]
    whole_data = str2bool(sys.argv[1]) if len(sys.argv) >= 2 else False
    grouped_clusters_list = [
        path
        for path in os.listdir(data_path + grouped_clusters_path)
        if os.path.isdir(data_path + grouped_clusters_path + path) 
        and
        path.split('_')[0] == 'GROUPED'
    ]
    intervals = []
    for g in grouped_clusters_list:
        pattern_richness = r'GROUPED_CLUSTER_RICHNESS=(\d+\.\d+)-(\d+\.\d+)'
        pattern_redshift = r'REDSHIFT=(\d+\.\d+)-(\d+\.\d+)'
        match_richness = re.search(pattern_richness, g)
        match_redshift = re.search(pattern_redshift, g)
        if match_richness:
            intervals.append([[float(match_richness.group(1)),float(match_richness.group(2))],[float(match_redshift.group(1)),float(match_redshift.group(2))]])
    clusters = []
    try:
        R_profiles = R_profiles * getattr(u, R_units)
    except:
        R_profiles = R_profiles
    for n in range(len(grouped_clusters_list)):

        current_interval = intervals[n][0]
        current_redshift_interval = intervals[n][1]
        if (current_interval[0] == 20 and current_interval[1] == 208) or current_interval[0] < r_interval[0] or current_interval[1] > r_interval[1] or current_redshift_interval[0] < z_interval[0] or current_redshift_interval[1] > z_interval[1]:
            continue

        if individual_bool == True and current_interval[0] != r_interval[0] and current_interval[1] != r_interval[1] and current_redshift_interval[0] != z_interval[0] and current_redshift_interval[1] != z_interval[1]:
            continue

        empty_group = grouped_clusters(None)
        empty_group.output_path = (
            data_path + grouped_clusters_path + grouped_clusters_list[n]
        )
        output_path = empty_group.output_path
        try:
            empty_group.load_from_h5()
        except FileNotFoundError:
            continue
        empty_group.mean(from_path = True)
        empty_group.R = np.array([(R_profiles[i + 1] + R_profiles[i]).value/2 for i in range(len(R_profiles) - 1)]) * R_profiles.unit
        profile = np.array(empty_group.mean_profile)
        errors = np.array(empty_group.error_in_mean)
        positive_values = profile[profile > 0]
        SNr_total = np.sqrt(np.sum(profile**2 / errors**2))
#        if len(positive_values) - n_parameters <= 0:
#            continue
        empty_group.completeness_and_halo_func()
        clusters.append(empty_group)
        if whole_data == True:
            continue
        func = empty_group.stacked_halo_model_func(
            getattr(profiles_module, model_profile)
        )

        global ln_prior

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
            def ln_likelihood(theta, x, y, sigma, **kwargs):
                model = kwargs["model"]
                mu = model(x, theta)
                log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
                    (y - mu) ** 2
                ) / (sigma**2)
                return np.sum(log_likelihood),mu
        elif likelihood_func == "chi2":
            def ln_likelihood(theta, x, y, sigma, **kwargs):
                model = kwargs["model"]
                y1 =  model(x, theta)
                res = np.sum(((y - y1) / sigma)**2)
                return -0.5 * res, y1

        param_limits = [
            np.array(prior_parameters[i][-1].split("|")).astype(float)[-2::]
            for i in range(len(prior_parameters))
            if "free" in prior_parameters[i]
        ]
        ndims = len(param_limits)
        initial_guess = np.array([random_initial_steps(param_limits[i], nwalkers) for i in range(len(param_limits))]).T
        """
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
        """
        if demo:
            print(20*"="+"\nRunning demo...\n"+20*"=")
            demo_path = data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + "demo"
            if os.path.exists(demo_path) == False:
                os.mkdir(demo_path)
            #initial guess
            fig,ax = plt.subplots()
            good_parameters = []
            ax.errorbar(
                empty_group.R.value,
                empty_group.mean_profile,
                yerr=empty_group.error_in_mean,
                capsize=4,
                fmt="o",
                label="data",
                color = 'red',
                lw = 1.5
            )
            for sample in initial_guess:
                p = func(empty_group.R,sample) 
                if np.any(p > np.max(empty_group.mean_profile)) or np.any(p <= 1e-10) or p[0] <= p[-1]:
                    continue
                else:
                    good_parameters.append(sample)
                    ax.plot(empty_group.R.value,p,alpha = 0.3, color='black')
            ax.set(yscale = 'log', ylabel = '$\\langle y \\rangle$', xlabel = f'R {empty_group.R.unit}', title = 'DEMO')
            ax.plot([],[]," ",label = f'N of good parameters = {len(good_parameters)}')
            ax.grid(True)
            ax.legend()
            fig.savefig(demo_path + "/initial_sample.png")
            #demo of ln_posterior
            with cProfile.Profile() as pr:
                theta_demo = initial_guess[np.random.randint(0,len(initial_guess))]
                x_demo, y_demo, sigma_demo = empty_group.R.value, empty_group.mean_profile, empty_group.error_in_mean
                N = config["STACKED_HALO_MODEL"]["N demos"]
                for demos in range(len(N)):
                    ln_posterior(theta_demo, x_demo, y_demo, sigma_demo, model = func, ln_prior = ln_prior, ln_likelihood = ln_likelihood, chi2 = chi2)
                    #pr.print_stats()
            continue
        pool = Pool(35)
        filename = output_path + "/" + fil_name + "." + ext
        if skip == True and os.path.exists(filename) == True:
            if os.path.getsize(filename)/1e9 >= 1:
                continue
        print(f"running MCMC in interval richness: {intervals[n][0]} , redhisft {intervals[n][1]}")
        print("saving sample in "+filename)
        backend = emcee.backends.HDFBackend(filename)

        #blobs
        blobs_config = config["BLOBS"]
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
            ln_posterior,
            args=(
                empty_group.R.value,
                empty_group.mean_profile,
                empty_group.error_in_mean,
            ),
            kwargs={
                "model": func,
                "ln_prior": ln_prior,
                "ln_likelihood": ln_likelihood,
                "chi2": chi2
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
        t1 = time()
        sampler.run_mcmc(initial_guess, nsteps, progress=True, store = True)
        t2 = time()
        print(f"The sampler of richess {intervals[i][0]} and redshift {intervals[i][1]} was finished in {t2 - t1} seconds.")
    if whole_data == True:
        print("Fitting whole data...")
        clusters = np.sum(clusters)
        redshift_bins = prop2arr(config["EXTRACT"]["REDSHIFT BINS"],dtype=np.float64)
        func,y_data, err = clusters.stacked_halo_model_func_by_bins(getattr(profiles_module, model_profile),zb = redshift_bins)
        R_profiles = clusters.R
        y_data = np.array(y_data, dtype = 'object')
        err = np.array(err, dtype = 'object')
        global ln_likelihood_general
        def ln_likelihood_general(theta, x, y, sigma, **kwargs):
            model = kwargs["model"]
            mu = model(x, theta)
            log_likelihood = flatten( -0.5 * log_inhomogeneous(2 * np.pi * sigma**2) - 0.5 * (
                (y - mu) ** 2
            ) / (sigma**2))
            return np.sum(log_likelihood),mu

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
        global chi2_general
        def chi2_general(y, y1, sigma):
            y = np.array(flatten(y))
            y1 = np.array(flatten(y1))
            sigma = np.array(flatten(sigma))
            return np.sum((y - y1)**2 / sigma**2)
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
        filename = data_path + config["FILES"]["GROUPED_CLUSTERS_PATH"] + "whole_data.h5"
        backend = emcee.backends.HDFBackend(filename)
        pool = Pool(39)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndims,
            ln_posterior,
            args=(
                R_profiles,
                y_data,
                err,
            ),
            kwargs={
                "model": func,
                "ln_prior": ln_prior_general,
                "ln_likelihood": ln_likelihood_general,
                "chi2": chi2_general
            },
            pool = pool,
#            backend=backend,
        )
        print("running mcmc with whole data")
        t1 = time()
        sampler.run_mcmc(initial_guess, nsteps, progress=True)
        t2 = time()
        samples = sampler.get_gain(flat = True)
        np.save("samples.npy", samples)
        print("MCMC was finish in ", t2  - t1,"seconds.")
if __name__ == "__main__":
    main()
