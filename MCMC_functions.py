
import numpy as np
from scipy.stats import cauchy

def t_student_prior_truncated(theta, scale=1.0, shift=0.0, vmin = -10, vmax = 10):
    slope = theta
    if slope == 0 or slope <= vmin or slope >= vmax:
        return -np.inf
    log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
    return log_prior_slope

def t_student_prior(theta, scale=1.0, shift=0.0,vmin = -10, vmax = 10):
	slope = theta
	log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
	return log_prior_slope

def t_student_prior_damping(theta, scale = 1.0, shift = 0.0, damping = 1.0, vmin = -10, vmax = 10):
    slope = theta
    if slope == 0 or slope <= vmin or slope >= vmax:
        return -np.inf
    log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
    if slope > shift:
        log_prior_slope -=  damping * (slope - shift)
    return log_prior_slope


def flat_prior(theta, vmin, vmax):
	if (vmin <= theta <= vmax):
		return 0.0
	else:
		return -np.inf

def ln_likelihood(theta, x, y, sigma, **kwargs):
	model = kwargs['model']
	mu = model(x,theta)
	log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu)**2) / (sigma**2)
	return np.sum(log_likelihood)

def ln_posterior(theta, x, y, sigma, **kwargs):
	ln_prior = kwargs['ln_prior']
	ln_likelihood = kwargs['ln_likelihood']
	prior = ln_prior(theta)
	if np.isinf(prior):
		return -np.inf
	likelihood = ln_likelihood(theta, x, y, sigma, **kwargs)
	posterior = prior + likelihood
	if np.isnan(posterior) == True:
		return -np.inf
	return posterior
