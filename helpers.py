import numpy as np
import re
import argparse
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from matplotlib import colors as mcolors
available_colors = list(mcolors.CSS4_COLORS.keys())


global check_none
def check_none(cls):
	class Wrapper(cls):
		def __init__(self,*args,**kwargs):
			if any(arg is None for arg in args) or any(value is None for value in kwargs.values()):
				args = (None,) * (getattr(cls, '__init__').__code__.co_argcount - 1)
				kwargs = {key: None for key in kwargs}
			super(Wrapper, self).__init__(*args, **kwargs)
	return Wrapper

class Found_Error_Config(Exception):
	pass

class Empty_Data(Exception):
	pass

class Attr_error(Exception):
	pass

def prop2arr(prop,delimiter=',',dtype=np.float64):
	"""
	convert a property from a configuration file to a numpy array
	"""
	arr = prop.replace(' ','').split(delimiter);
	return np.array(arr,dtype=dtype)


def power_law(x,a,b):
    return a*(x/45)**b

def double_power_law(x, A1,alpha1, A2, alpha2, x0):
    y = np.zeros_like(x)
    y[x < x0] = A1 * x[x < x0]**alpha1
    y[x >= x0] = A2 * x[x >= x0]**alpha2 - A2*x0**alpha2 + A1 * x0 ** alpha1
    return y

def smooth_power_law(x, A1, alpha1, A2, alpha2, x0, width):
    t = np.tanh((x - x0) / width)
    y = A1 * (x ** alpha1) * (1 - t) + A2 * (x ** alpha2) * t
    return y

def broken_power_law(x, A, alpha1, alpha2, x0):
    y = np.zeros_like(x)
    mask = x <= x0
    y[mask] = A*(x[mask]/x0)**(alpha1)
    y[np.logical_not(mask)] = A*(x[np.logical_not(mask)]/x0)**(alpha2)
    return y

def smoothly_broken_power_law(x, A, alpha1, alpha2, x0, delta):
    return A*(x/x0)**(-alpha1)*(1/2 * (1 + (x/x0)**(1/delta)))**((alpha1 - alpha2)*delta)

def moving_average_func(x,y,size, only_pos = True):
    sort_x = np.sort(x)
    sort_y = np.array(y)[np.argsort(sort_x)]
    padded_y = np.pad(sort_y, (size // 2, size // 2), mode="edge")
    moving_avg_y = np.zeros_like(sort_y, dtype=np.float64)
    moving_err_y = np.zeros_like(sort_y, dtype=np.float64)
    for i in range(len(sort_y)):
        m = np.mean(padded_y[i : i + size])
        if m <= 0 or m < np.min(y):
            m = moving_avg_y[-1] 
        moving_avg_y[i] = m
        moving_err_y[i] = np.std(padded_y[i : i + size], ddof=1) / np.size(padded_y[i : i + size])
    return moving_avg_y, moving_err_y

def str2bool(string):
	return True if string.lower() == "true" else False

def extract_values(s):
    match = re.match(r'GROUPED_CLUSTER_RICHNESS=(\d+\.\d+)-(\d+\.\d+)REDSHIFT=(\d+\.\d+)-(\d+\.\d+)', s)
    if match:
        richness_range = (float(match.group(1)), float(match.group(2)))
        redshift_range = (float(match.group(3)), float(match.group(4)))
        return richness_range, redshift_range
    else:
        return None, None

def closest_path(target, paths):
    target_richness, target_redshift = extract_values(target)
    closest_dist = float('inf')
    closest_str = None
    for s in paths:
        richness_range, redshift_range = extract_values(s)
        if richness_range is not None and redshift_range is not None:
            richness_dist = abs(target_richness[0] - richness_range[0]) + abs(target_richness[1] - richness_range[1])
            redshift_dist = abs(target_redshift[0] - redshift_range[0]) + abs(target_redshift[1] - redshift_range[1])
            total_dist = richness_dist + redshift_dist
            if total_dist < closest_dist:
                closest_dist = total_dist
                closest_str = s

    return closest_str


def log_inhomogeneous(arr):
    log = np.array([np.array([np.log(list(arr[i][j])) for j in range(len(arr[i]))]) for i in range(len(arr))], dtype = 'object')
    return log

def flatten(array):
    flattened = []
    for item in array:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def extract_parameters(result):
    parameters,errors = [], []
    best = list(result.params.values())
    for i in range(len(best)):
        parameters.append(best[i].value)
        errors.append(best[i].stderr)
    return parameters, errors

def group_types(table, names, types):
    grouped_types = np.full_like(table, 'Other', dtype='U10')
    for i in range(len(names)):
        grouped_types[np.char.startswith(table, names[i])] = types[i]
    return grouped_types

def list_array_or_tuple_type(arg):
    try:
        value = eval(arg)
        if isinstance(value, (list, np.ndarray, tuple)):
            return value
        else:
            raise argparse.ArgumentTypeError("Argument must be a list, tuple or numpy array.")
    except:
        raise argparse.ArgumentTypeError("Invalid argument format.")

def random_initial_steps(limits, n):
    params = np.zeros(n)
    for i in range(len(params)):
            params[i] = np.random.uniform(*limits*0.9)
    return params

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    if norm:
        acf /= acf[0]

    return acf

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_time_from_chain(chain):
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    tau = np.empty(len(N))
    for i, n in enumerate(N):
        tau[i] = autocorr_new(chain[:, :n])
    return N,tau

def radial_binning(data, R, patch_size = 1.1, weighted = False, errors = None):
    if np.shape(data)[0] == np.shape(data)[1]:
        center = np.shape(data)[0]//2 , np.shape(data)[1]//2
        patchsize = patch_size*60 * u.arcmin
        pixwidth = patchsize / (data.shape[0] * u.pixel)
        R_pixel = np.array(R / pixwidth).astype(int)
        profile, err, R_cent = [], [], []
        for i in range(0, len(R_pixel) - 1):
            r_in = R_pixel[i]
            r_out = R_pixel[i + 1]
            bin_cent = (r_in + r_out)/2.
            jmin = center[0] - r_out
            jmax = center[0] + r_out + 1
            kmin = center[1] - r_out
            kmax = center[1] + r_out + 1
            target = []
            target_err = []
            for j in np.arange(jmin, jmax):
                for k in np.arange(kmin, kmax):
                    jk = np.array([j,k])
                    dist = np.linalg.norm(jk - np.array(center))
                    if dist > r_in and dist <= r_out:
                        target.append(data[j][k])
                        if weighted == True and errors is not None:
                            target_err.append(errors[j][k])
            if weighted == True and errors is not None:
                w = 1/np.array(target_err)**2
                mean = np.average(target,weights = w)
                error = np.sqrt(1/np.sum(w))
            else:
                mean = np.mean(target)
                error = np.std(target)
            profile.append(mean)
            err.append(error)
            R_cent.append(bin_cent)
        R_cent = R_cent * pixwidth
        return R_cent, profile, err


def split_multimodal(data, min_chi2 = None, nbins = 200, threshold = 0.5, output_path = "param",
                     bins_ratio = 0.25, aim = None, only_aim = False, distance = None, height = None, **kwargs):
    fig_kwargs = kwargs.pop('fig_kwargs', {})
    hist_kwargs = kwargs.pop('hist_kwargs', {})
    scatter_kwargs = kwargs.pop('scatter_kwargs', {})
    axes_kwargs = kwargs.pop('axes_kwargs', {})
    q = np.percentile(data,[16, 50, 84])
    best = q[1]
    lower,upper = np.diff(q)
    fig,ax = plt.subplots(1,**fig_kwargs)
    hist = plt.hist(data, bins = nbins, density = True, **hist_kwargs);
    counts, bins = hist[0],hist[1]
    cut = 1.35 * np.mean(counts) if height is None else height
    distance = len(bins)//20 if distance is None else distance
    aim = min_chi2 if aim is None else aim
    peaks,_ = find_peaks(counts, height = cut, distance = distance)
    ax.axhline(cut, color = 'black')
    ax.axhline(cut/2, color = 'black', ls = '--')
    ax.axvline(best, color = 'green', lw = 1)
    if min_chi2 is not None:
        ax.axvline(min_chi2, color = 'red', lw = 1)
    ax.axvline(best - lower, color = 'green', lw = 1, ls = '--')
    ax.axvline(best + upper, color = 'green', lw = 1, ls = '--')
    ax.scatter(bins[peaks],counts[peaks], **scatter_kwargs)
    ax.grid(True)
    n_components = len(peaks)
    gmm = GaussianMixture(n_components=n_components, means_init=bins[peaks][:, np.newaxis])
    gmm.fit(data[:, np.newaxis])
    probabilities = gmm.predict_proba(data[:,None])
    predicted_labels = np.argmax(probabilities, axis=1)
    selected_data = []
    means,lowers,uppers = [],[],[]
    for i in range(len(peaks)):
        component_index = i
        selected_indices = np.where((predicted_labels == component_index) & (probabilities[:, component_index] > threshold))[0]
        selected_data.append(data[selected_indices])
        s = selected_data[-1]
        if len(s) <= 2:
            continue
        color = np.random.choice(available_colors)
        ax.fill_between((np.min(s),np.max(s)), 0, np.max(counts), 
                        color = color, alpha = 0.3, edgecolors = 'black')
        q = np.percentile(selected_data[i],[16,50,84])
        df = np.diff(q)
        means.append(q[1])
        lowers.append(df[0])
        uppers.append(df[1])
        text = r"$%.2f^{+%.2f}_{-%.2f}$" % (q[1],df[1],df[0])
        ylim = ax.get_ylim()
        y_position = ylim[0] - (0.15 + i%2 * 0.05) * (ylim[1] - ylim[0])
        plt.text(q[1],y_position, text, horizontalalignment='center', verticalalignment='bottom', 
                 fontsize=10, bbox=dict(facecolor=color, alpha=0.5, boxstyle = "round"))
    ax.set(**axes_kwargs)
    fig.tight_layout()
    fig.savefig(f"{output_path}")
    indx = np.argmin(np.abs(np.array(means) - aim))
    if only_aim == True:
        return [means[indx]], [lowers[indx]], [uppers[indx]]
    return means, lowers, uppers
