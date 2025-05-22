from cluster_data import *
from profiles import *
from helpers import *
import argparse 
from multiprocessing import Pool
import pyccl as ccl

profiles_module = importlib.import_module("profiles")

current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path + "/config.ini"
config = ConfigParser()
config.optionxform = str

if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file doesn't exist at {current_path}")

cosmo = ccl.CosmologyVanillaLCDM()

parser = argparse.ArgumentParser()
parser.add_argument("--N_mass_bins", "-MB", type = int, default = 20, help = "Number of mass bins.")
parser.add_argument("--N_redshift_bins", "-zB", type = int, default = 20, help = "Number of redshift bins.")
parser.add_argument("--N_radius_bins", "-RB", type = int, default = 20, help = "Number of radius bins.")

parser.add_argument("--Mass_range", "-MR", type = str, default = "12,16", help = "Limit of Mass in format (Log10(M_min),Log10(M_max)).")
parser.add_argument("--z_range", "-zR", type = str, default = "0.1,1", help = "Limit of redshift in format (z_min,z_max).")
parser.add_argument("--radius_range", "-rR", type = str, default = "1, 20", help = "Limit of radius (physical or angular) in format (r_min, r_max).")
parser.add_argument("--logR", "-L", action = "store_true", help = "If passed the radius array will be log. spaced.")

parser.add_argument("--CONFIG_FILE","-c", type = str, default = "PRIORS", help = "Key in config.ini file that define the priors.")
parser.add_argument("--N_params", "-n", type = int, default = 8, help = "Number of evaluation of each parameter")
parser.add_argument("--params_scale", "-p", default = "lin", help = "Scale of the spacing on each of the paramter, it could be lin and log for linear and base log-10 respectively. The format can be an inividual str or an str with format 'scale, scale, scale...' with the same shape of parameter space.")
parser.add_argument("--output_file", "-f", default = "two_halo_interp.h5", type = str, help = "output name file. As default is saved using h5py. The available format are .h5, .npy and .csv.")
parser.add_argument("--ncores", "-N", default = 0, type = int, help = "Number of cores to computing the 2-halo term grid. As default is 0 (no Multiprocessing)")
parser.add_argument("--r_units", "-U", type = str, default = "arcmin", help = "Unit of radius, it could be physical (Mpc) or angular (arcmin).")


args = parser.parse_args()
N_mass = args.N_mass_bins
N_z = args.N_redshift_bins
N_R = args.N_radius_bins

Log10Mmin, Log10Mmax = np.array(args.Mass_range.split(","), dtype = float)
zmin, zmax = np.array(args.z_range.split(","), dtype = float)
rmin, rmax = np.array(args.radius_range.split(","), dtype = float)

n_cores = args.ncores
print(f"Running two halo term interpolator with {n_cores}")
pool = Pool(n_cores) if n_cores != 0  else None
print(f"Loading configuration from {args.CONFIG_FILE}")
params_scale = [args.params_scale] if len(args.params_scale.split(",")) == 0 else np.array(args.params_scale.split(","))


nparams = args.N_params
M_arr = np.logspace(Log10Mmin, Log10Mmax, N_mass)
z_arr = np.linspace(zmin, zmax, N_z)

R = np.linspace(rmin, rmax, N_R) if args.logR == False else np.logspace(rmin, rmax, N_R)
r_units = args.r_units
assert r_units in ("Mpc", "arcmin"), f"r_units must be Mpc or arcmin. It recived {r_units}!"

current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path +"/"+ str(args.CONFIG_FILE)
config = ConfigParser()
config.optionxform = str
if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file {str(args.CONFIG_FILE)} doesn't exist")

priors_config = config["PRIORS"]
profile_model = getattr(profiles_module, config["MODEL"]["profile"])
warnings.filterwarnings("ignore")

prior_parameters = dict(priors_config)
prior_parameters_dict = {
    key: list(prop2arr(prior_parameters[key], dtype=str))
    for key in list(prior_parameters.keys())
}
prior_parameters = list(prior_parameters_dict.values())
params = []
params_ranges = [np.array(p[-1].split("|"), dtype = float) for p in prior_parameters]

if len(params_scale) < len(params_ranges):
    params_scale = np.full(len(params_ranges), params_scale[0])

for i in range(len(params_ranges)):
    if params_scale[i] == "log":
        params.append(np.logspace(params_ranges[i][-2], params_ranges[i][-1], nparams))
    elif params_scale[i] == "lin":
        params.append(np.linspace(params_ranges[i][-2], params_ranges[i][-1], nparams))
M_arr, z_arr, params, R, evals = make_2halo_term_interpolator(profile_model, M_arr, z_arr, R, cosmo, params, pool = pool, 
                                n_cores = n_cores, overwrite = True, return_samples = True, r_units = r_units)

output_file = args.output_file
file_format = output_file.split(".")[-1]
assert file_format in ["h5", "csv", "npy"], f"{file_format} isn't a supported file format!"

if file_format == "h5":
    print("Saving interpolator to file " + output_file)
    output_file = "interpolator.h5" if output_file == "" else output_file
    with h5py.File(output_file, "w") as f:
        f.create_dataset("Mass", data = M_arr)
        f.create_dataset("z", data = z_arr)
        f.create_dataset("evals", data = evals)
        f.create_dataset("R", data = R)
        for i in range(np.shape(params)[0]):
            p = np.array(params)[i,:]
            f.create_dataset(f"param {i}", data = p)
