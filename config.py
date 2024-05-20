#this script read de config.ini file and extract it.
from configparser import ConfigParser
import os
import importlib
from helpers import *
import numpy as np
import astropy.units as u

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
R_profiles = R_profiles * getattr(u,R_units)
demo = str2bool(config["STACKED_HALO_MODEL"]["DEMO"])
only_stacking = str2bool(config["EXTRACT"]["ONLY_STACKING"])
min_richness = int(config["STACKED_HALO_MODEL"]["min richness"])
min_SNR = float(config["STACKED_HALO_MODEL"]["min_SNR"])
skip = str2bool(config["STACKED_HALO_MODEL"]["skip_fitted"])
likelihood_func = config["STACKED_HALO_MODEL"]["likelihood"]

DR6 = config["FILES"]["DR6-ACT-map"]
DR5 = config["FILES"]["DR5-ACT-map"]
DES_Y3 = config["FILES"]["Y3-REDMAPPER"]
MASK_DR6 = config["FILES"]["MASK_DR6-ACT-map"]
MILLIQUAS = config["FILES"]["MILLIQUAS"]
