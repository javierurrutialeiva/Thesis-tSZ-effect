from cluster_data import *
from profiles import *
from helpers import *
import argparse 
from multiprocessing import Pool, Manager
import pyccl as ccl
from astropy.cosmology import Planck18 as cosmo
import time

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
parser.add_argument("--N_mass_bins", "-MB", type = int, default = 40, help = "Number of mass bins.")
parser.add_argument("--N_redshift_bins", "-zB", type = int, default = 20, help = "Number of redshift bins.")
parser.add_argument("--N_radius_bins", "-RB", type = int, default = 20, help = "Number of radius bins.")

parser.add_argument("--Mass_range", "-MR", type = str, default = "13,16", help = "Limit of Mass in format (Log10(M_min),Log10(M_max)).")
parser.add_argument("--z_range", "-zR", type = str, default = "0.1,1", help = "Limit of redshift in format (z_min,z_max).")
parser.add_argument("--radius_range", "-rR", type = str, default = "1, 20", help = "Limit of radius (physical or angular) in format (r_min, r_max).")
parser.add_argument("--logR", "-L", action = "store_true", help = "If passed the radius array will be log. spaced.")
parser.add_argument("--delta", "-d", type = float, default = 500, help = "Delta for the mass definition. Default is 500.")
parser.add_argument("--CONFIG_FILE","-c", type = str, default = "PRIORS", help = "Key in config.ini file that define the priors.")
parser.add_argument("--N_params", "-n", type = int, default = 2, help = "Number of evaluation of each parameter")
parser.add_argument("--params_scale", "-p", default = "lin", help = "Scale of the spacing on each of the paramter, it could be lin and log for linear and base log-10 respectively. The format can be an inividual str or an str with format 'scale, scale, scale...' with the same shape of parameter space.")
parser.add_argument("--output_file", "-f", default = "two_halo_interp.h5", type = str, help = "output name file. As default is saved using h5py. The available format are .h5, .npy and .csv.")
parser.add_argument("--ncores", "-N", default = 30, type = int, help = "Number of cores to computing the 2-halo term grid. As default is 0 (no Multiprocessing)")
parser.add_argument("--r_units", "-U", type = str, default = "arcmin", help = "Unit of radius, it could be physical (Mpc) or angular (arcmin).")
parser.add_argument("--R_obs_range", "-R", type = str, default = "0, 15", help = "Range of the observed radius in format (r_min, r_max)")
parser.add_argument("--N_R_obs_bins", "-NR", type = int, default = 40, help = "Number of bins for the observed radius.")
parser.add_argument("--logR_obs", "-LR", action = "store_true", help = "If passed the observed radius array will be log. spaced.")
parser.add_argument("--lambda_range", "-Lr", type = str, default = "20, 200", help = "Range of the lambda parameter in format (lambda_min, lambda_max).")
parser.add_argument("--N_lambda_bins", "-NL", type = int, default = 40, help = "Number of bins for the lambda parameter.")
parser.add_argument("--M_obs_range", "-MOR", type = str, default = "13, 16", help = "Range of the observed mass in format (Log10(M_min), Log10(M_max)).")
parser.add_argument("--N_M_obs_bins", "-NMO", type = int, default = 10, help = "Number of bins for the observed mass.")
parser.add_argument("--free-RM-relationship", "-FRM", action = "store_true", help = "If passed the RM relationship is free, otherwise it is fixed to the one defined in the config file.")
parser.add_argument("--N-RM-params", "-NRM", type = int, default = 2, help = "Number of evaluation of each RM parameter. This is only used if --free-RM-relationship is passed.")
parser.add_argument("--demo", "-D", action = "store_true", help = "If passed, the code will run a demo with a reduced number of parameters and save it as demo.png.")
parser.add_argument("--ndemos", "-ND", type = int, default = 4, help = "Number of demos to run. This is only used if --demo is passed.")

args = parser.parse_args()
N_mass = args.N_mass_bins
N_z = args.N_redshift_bins
N_R = args.N_radius_bins
N_R_obs = args.N_R_obs_bins
N_lambda = args.N_lambda_bins
N_M_obs = args.N_M_obs_bins

free_RM_relationship = args.free_RM_relationship


Log10Mmin, Log10Mmax = np.array(args.Mass_range.split(","), dtype = float)
zmin, zmax = np.array(args.z_range.split(","), dtype = float)
rmin, rmax = np.array(args.radius_range.split(","), dtype = float)
R_obs_min, R_obs_max = np.array(args.R_obs_range.split(","), dtype = float)
lambda_min, lambda_max = np.array(args.lambda_range.split(","), dtype = float)
logM_obs_min, logM_obs_max = np.array(args.M_obs_range.split(","), dtype = float)

M = np.logspace(logM_obs_min, logM_obs_max, N_M_obs)
z_arr = np.linspace(zmin, zmax, N_z)


Robs = np.linspace(R_obs_min, R_obs_max, N_R_obs) if args.logR_obs == False else np.logspace(R_obs_min, R_obs_max, N_R_obs)
lambda_obs = np.linspace(lambda_min, lambda_max, N_lambda)

k2halo = np.logspace(-15, 15, 60)
M_arr2halo = np.logspace(Log10Mmin, Log10Mmax, N_mass)
R2halo = np.linspace(rmin, rmax, N_R) if args.logR == False else np.logspace(rmin, rmax, N_R)
delta = args.delta
cosmo2halo = ccl.CosmologyVanillaLCDM()
mdef = ccl.halos.MassDef(delta, "critical")
mfunc = ccl.halos.mass_function_from_name("Tinker10")
mfunc = mfunc(cosmo2halo, mdef)

dndM2halo = np.array([[mfunc(cosmo2halo, Mi, 1/(zi + 1)) for Mi in M_arr2halo] for zi in z_arr])
dndM2halo = dndM2halo * 1/(M_arr2halo * np.log(10))   

bias = ccl.halos.HaloBiasTinker10(cosmo2halo, mass_def=mdef) 
bh = np.array([bias.get_halo_bias(cosmo2halo, M, 1/(1 + zi)) for zi in z_arr])
bM = np.array([[bias.get_halo_bias(cosmo2halo, Mi, 1/(1 + zi)) for Mi in M_arr2halo] for zi in z_arr])
Pk = np.array([ccl.linear_matter_power(cosmo2halo, k2halo, 1/(1+zi)) for zi in z_arr])
Rgrid, M2halo_grid, z2halo_grid, k2halo_grid = np.meshgrid(R2halo, M_arr2halo, z_arr, k2halo, indexing = "ij")
ki_r = Rgrid*k2halo_grid
sin_term = np.sin(ki_r) / np.where(ki_r != 0, ki_r, 1)

lambda_model, _,_  = np.meshgrid(lambda_obs, z_arr, M_arr2halo, indexing = "ij")
lambda2halo_grid = np.meshgrid(lambda_obs, z_arr, M_arr2halo, indexing = "ij")[0]

D_ang2halo = (planck18.angular_diameter_distance(z_arr) * (1 + z_arr))[None,:]
R_Mpc2halo = ((Robs * u.arcmin).to(u.rad)[:,None] * D_ang2halo).value

n_cores = args.ncores
print(f"Running two halo term interpolator with {n_cores}")
pool = Pool(n_cores) if n_cores != 0  else None
print(f"Loading configuration from {args.CONFIG_FILE}")

params_scale = [args.params_scale] if len(args.params_scale.split(",")) == 0 else np.array(args.params_scale.split(","))
nparams = args.N_params
current_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = current_path +"/"+ str(args.CONFIG_FILE)
config = ConfigParser()
config.optionxform = str       
if os.path.exists(config_filepath):
    config.read(config_filepath)
else:
    raise Found_Error_Config(f"The config file {str(args.CONFIG_FILE)} doesn't exist")

priors_config = config["PRIORS"]
print("model profile = ", config["MODEL"]["profile"])
two_halo_profile = getattr(profiles_module, config["MODEL"]["profile"])
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


if free_RM_relationship:
    n_hm_params = args.N_RM_params
    halo_model_priors = config["PRIORS_HALO_MODEL"]
    prior_parameters_hm = dict(halo_model_priors)
    prior_parameters_dict_hm = {
        key: list(prop2arr(prior_parameters_hm[key], dtype=str))
        for key in list(prior_parameters_hm.keys())
    }
    prior_parameters_hm = list(prior_parameters_dict_hm.values())
    params_hm = []
    params_ranges_hm = [np.array(p[-1].split("|"), dtype = float) for p in prior_parameters_hm]
    N_hm_params = len(params_ranges_hm)
    params_hm = []
    for i in range(len(params_ranges_hm)):
        if params_scale[i] == "log":
            params_hm.append(np.logspace(params_ranges_hm[i][-2], params_ranges_hm[i][-1], n_hm_params))
        elif params_scale[i] == "lin":
            params_hm.append(np.linspace(params_ranges_hm[i][-2], params_ranges_hm[i][-1], n_hm_params))
    params = params + params_hm

meshgrid = np.meshgrid(*params, indexing = "ij")
params_arr = np.stack([grid.flatten() for grid in meshgrid], axis = -1)
new_params = np.array_split(np.array(params_arr), n_cores)

def init():
    global Rgrid, lambda2halo_grid, z2halo_grid, R_Mpc2halo, k2halo_grid
    global Pk, bh, dndM2halo, bM, R2halo, M_arr2halo, k2halo
    global sin_term, R, M, z_arr, R_Mpc2halo, k2halo_grid
    global params_range, params_range_hm

@njit(cache=True, fastmath=True)
def compute_two_halo(PRMzk):
    uRMz = 4 * np.pi * ((dndM2halo * bM).T[None, :, :, None] * Rgrid**2 * sin_term * PRMzk)
    uPk = Pk[None, None, :, :] * uRMz
    bhuPk = bh[None, None, :, :, None] * uPk[:, :, :, None, :]
    ki_R2 = R_Mpc2halo[:, :, None] * k2halo[None, None, :]
    sin_term2 = np.sin(ki_R2) / np.where(ki_R2 != 0, ki_R2, 1)
    P2halo = ((sin_term2[None, None, :, :, :] * k2halo_grid[:, :, None, :, :]**2)
              [:, :, :, :, None, :] * bhuPk[:, :, None, :, :, :]) / (2 * np.pi**2)
    return P2halo

if args.demo:
    print("Running demo!")
    init()
    ndemo = args.ndemos
    random_indx = np.random.randint(0, len(params_arr), ndemo)
    random_params = params_arr[random_indx]
    results_demo = np.zeros((len(random_params), len(Robs), len(M), len(z_arr)))
    for i, p in enumerate(random_params):
        if free_RM_relationship:
            params_profile = p[:len(params_ranges)]
            params_halo_model = p[len(params_ranges):] if free_RM_relationship else None
            mass2richness_Norm, mass2richness_Slope, mass2richness_Slope_redshift = params_halo_model 
            mass2richness_Pivot, mass2richness_Pivot_redshift = 3e14/0.7, 0.35
            lambda_eval = mass2richness_Norm * (M2halo_grid / mass2richness_Pivot)**mass2richness_Slope * \
                    ((1 + z2halo_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift  

            p = params_profile
        else:
            lambda_eval = lambda2halo_grid
        PRMzk = two_halo_profile(Rgrid, lambda_eval, z2halo_grid, p)
        P = compute_two_halo(PRMzk)
        two_halo_term = np.reshape(
            np.trapz(
                np.trapz(
                    np.trapz(P, axis=0, x=R2halo),
                axis=0, x=M_arr2halo),
            axis=-1, x=k2halo),
            (len(Robs), len(M), len(z_arr))
        )
        results_demo[i] = two_halo_term
        sys.stdout.write(f"\rProgress: {i+1}/{ndemo} ({(i+1)/ndemo*100:.2f}%)")
        sys.stdout.flush()
    
    fig, ax = plt.subplots(1,ndemo, figsize=(6*ndemo,12))
    nsub = 3
    mass_idx = np.random.choice(len(M), nsub, replace=False)
    z_idx = np.random.choice(len(z_arr), nsub, replace=False)
    Msub = M[mass_idx]
    zsub = z_arr[z_idx]
    i_mass, i_z = np.ix_(mass_idx, z_idx)
    sub_results_demo = results_demo[:,:, i_mass, i_z]
    for i in range(len(random_params)):
        ax[i].set_title(f"Demo {i+1}, parameters: {random_params[i]}")
        [ax[i].plot(Robs, sub_results_demo[i, :, j, k], label=f"z={zsub[k]:.2f}, $\log M$={np.log10(Msub[j]):.2f}") for j in range(len(Msub)) for k in range(len(zsub))]
        ax[i].set(xlabel="R (Mpc)", ylabel="Two-halo term", xscale="linear", yscale="log")
        ax[i].legend()
    fig.tight_layout()
    fig.savefig("demo.png")

AAAA



def worker(params, N_total, counter):
    results = np.zeros((len(params), len(Robs), len(M), len(z_arr)))
    for i, p in enumerate(params):
        t1 = time.time()
        if free_RM_relationship:
            params_profile = p[:len(params_ranges)]
            params_halo_model = p[len(params_ranges):] if free_RM_relationship else None
            mass2richness_Norm, mass2richness_Slope, mass2richness_Slope_redshift = params_halo_model 
            mass2richness_Pivot, mass2richness_Pivot_redshift = 3e14/0.7, 0.35
            lambda_eval = mass2richness_Norm * (M2halo_grid / mass2richness_Pivot)**mass2richness_Slope * \
                    ((1 + z2halo_grid)/(1 + mass2richness_Pivot_redshift)) **mass2richness_Slope_redshift  

            p = params_profile
        else:
            lambda_eval = lambda2halo_grid
        PRMzk = two_halo_profile(Rgrid, lambda_eval, z2halo_grid, p)
        P = compute_two_halo(PRMzk)
        two_halo_term = np.reshape(
            np.trapz(
                np.trapz(
                    np.trapz(P, axis=0, x=R2halo),
                axis=0, x=M_arr2halo),
            axis=-1, x=k2halo),
            (len(Robs), len(M), len(z_arr))
        )
        results[i] = two_halo_term
        counter.value += 1
        sys.stdout.write(f"\rProgress: {counter.value}/{N_total} ({(counter.value/N_total)*100:.2f}% | {time.time() - t1:.2f}s)")
        sys.stdout.flush()
    return results

res_ = []
pool = Pool(n_cores, initializer=init)
N_total = len(params_arr)
manager = Manager()
counter = manager.Value("i", 0)

for p in new_params:
    res_.append(pool.apply_async(worker, args = [p, N_total, counter]))
res = [r.get() for r in res_]
pool.close()
evals = np.concatenate(res, axis=0)

output_file = args.output_file
file_format = output_file.split(".")[-1]
assert file_format in ["h5", "csv", "npy"], f"{file_format} isn't a supported file format!"

if file_format == "h5":
    print("Saving interpolator to file " + output_file)
    output_file = "interpolator.h5" if output_file == "" else output_file
    with h5py.File(output_file, "w") as f:
        f.create_dataset("Mass", data = M)
        f.create_dataset("z", data = z_arr)
        f.create_dataset("evals", data = evals)
        f.create_dataset("R", data = Robs)
        for i, p in enumerate(params):
            f.create_dataset(f"param {i}", data = p)
