import argparse
import emcee
from glob import glob
import h5py
from icecream import ic, install
from matplotlib import pyplot as plt, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
from scipy.stats import binned_statistic as binstat

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import corner
from stattools import pte as PTE

update_rcParams()

from kids_ggl_pipeline.helpers.configuration import ConfigFile
from kids_ggl_pipeline.helpers import distributions, functions, io
from kids_ggl_pipeline.sampling.priors import fixed_priors, free_priors


install()


def main():
    args = parse_args()
    cfg = ConfigFile(args.cfg)
    R, signal, signal_err, cov, cov2d, corr = load_data(cfg)
    nbins = len(R)

    observable = cfg.read_observables()[0]
    binvar = {"yc": "y_\mathrm{c}"}
    binvar = binvar.get(observable.name, observable.name)
    print(f"{binvar}")

    model_signal, extra, chi2, params, nwalkers = load_chain(args.cfg, args.h5, nbins)
    # sample_data is None if model is not matched
    model_name, param_names, sample_data = read_param_names(args.cfg)
    ic(model_name, sample_data)
    # for now!
    labels = [f"${p}$" for p in param_names]
    nsteps = params.shape[0]
    best = np.argmin(chi2)
    pte = plot_chi2(args, chi2[best], cov2d)
    chi2min_historical = plot_chi2_history(args, chi2, best, pte, nsteps, nwalkers)
    plot_samples(args, params, chi2, chi2min_historical, nsteps, nwalkers, labels)
    # from now on use burned chains
    if args.burn == -1:
        args.burn = chi2.size - nwalkers
    if 0 < args.burn < chi2.size:
        model_signal = [m[args.burn :] for m in model_signal]
        chi2, params = [i[args.burn :] for i in (chi2, params)]
        best = np.argmin(chi2)
        print(f"Burned {args.burn} samples")
    plot_signal(
        args,
        model_signal,
        best,
        R,
        signal,
        signal_err,
        extra,
        observable.binlows,
        observable.binhighs,
        binvar,
        show_components=True,
    )
    param_info(args, params, best, labels)
    for key in ["Mavg", "Mref_14.5"]:
        try:
            print(f"{key}:")
            mean_mass(args, extra, key, best, nbins, lnlike=-chi2)
        except ValueError:
            pass

    # labels = ('$f_c$', r'$\beta$',)# '$f_\mathrm{mis}$', '$\sigma_\mathrm{mis}$')
    labels = [f"${i}$" for i in param_names]
    if len(labels) > 1:
        output = get_output(args, "corner")
        corner(
            params.T,
            truths=params[best],
            truths_in_1d=True,
            style1d="step",
            bins=10,
            percentiles1d=False,
            output=output,
            labels=labels,
        )

    ic(param_names)
    ic(extra.dtype.names)

    mor_name, *mor_params = cfg.read_section("hod/centrals/mor")
    mor_name = mor_name[1]
    mor_param_names = [p[0] for p in mor_params]
    mor_param_priors = [p[1] for p in mor_params]

    for i, (name, p) in enumerate(zip(mor_param_names, mor_params)):
        if name in param_names:
            mor_params[i] = params[:, param_names.index(name)]
        else:
            mor_params[i] = float(p[-1])
    ic(mor_params)

    if "end2end" in args.root:
        info = cfg.read_section("sampler")
        ic(info)
        # datafile = os.path.join(info.get('path_data'), info.get('data').split()[0])

    if observable.name == "msz":
        # 1 - b
        if "matched" in model_name:
            # the last [0] in sample_data implies that using any of the
            # clusters would give the same answer. This is true as long as
            # there is no scatter in the matched model
            msz = sample_data[0][1][0]
            if mor_name == "powerlaw_linspace":
                one_minus_b = (
                    mor_params[0]
                    / mor_params[1]
                    * (msz / mor_params[0]) ** (1 - mor_params[2])
                )
            elif mor_name == "powerlaw":
                one_minus_b = (
                    10 ** mor_params[0]
                    / 10 ** mor_params[1]
                    * (msz / 10 ** mor_params[0]) ** (1 - mor_params[2])
                )
        elif "predict_mass" in model_name:
            ...
        else:
            if "b_m" not in param_names:
                if mor_name == "powerlaw":
                    one_minus_b = 1 / 10 ** mor_params[1]
        ic(one_minus_b.shape)
        plot_b(args, one_minus_b, best)

    return


def read_config_section(cfg, params, section):
    X = []
    j = 0
    in_section = False
    # read mean relation
    with open(cfg) as f:
        for line in f:
            line = line.strip().split("#")[0]
            if len(line) == 0:
                continue
            # we don't need to read beyond the section so just stop here
            if line[0] == "[" and in_section:
                break
            if line.startswith(f"[{section}]"):
                in_section = True
                continue
            line = line.split()
            if len(line) < 2:
                continue
            if in_section and line[0] == "name":
                func_name = line[1]
                continue
            # repeat and read parameters won't work
            if line[1] in free_priors:
                if in_section:
                    X.append(params[:, j])
                j += 1
            elif in_section:
                X.append(float(line[2]) * np.ones(params.shape[0]))
    return func_name, np.array(X)


def load_chain(cfg, h5, nbins, signame="kappa", burn=0):
    # this so that the file is not blocked for the MCMC
    h5bck = f"{h5}.bck"
    os.system(f"cp {h5} {h5bck}")
    reader = emcee.backends.HDFBackend(h5, read_only=True)
    chain = reader.get_chain(flat=True)
    ic(chain.dtype)
    ic(chain.shape)
    blobs = reader.get_blobs(flat=True)
    ic(blobs.dtype)
    signal = [blobs[f"{signame}:{i+1}"] for i in range(nbins)]
    # create a new structured array
    extra = np.dtype(
        {name: blobs.dtype.fields[name] for name in blobs.dtype.names[nbins:-3]}
    )
    extra = np.ndarray(blobs.shape, extra, blobs, 0, blobs.strides)
    ic(extra)
    chi2 = blobs["chi2"]
    nwalkers = reader.shape[0]
    try:
        os.remove(h5bck)
    except FileNotFoundError:
        pass
    return signal, extra, chi2, chain, nwalkers


def load_chain_old(cfg, h5, nbins, burn=0):
    """Generic function to load data from a chain generated by kids_ggl

    Returns
    -------
    signal : ndarray, shape (nsamples,nbins,npoints)
        the modeled signal
    extra : dict
        all extra outputs as requested in the kids_ggl config
    chi2 : ndarray, shape (nsamples,)
        chi2 samples
    params : ndarray, shape (nsamples,nparams)
        free parameters in the model
    nwalkers : int
        number of walkers used in emcee
    """
    with open(cfg) as cfg:
        while cfg.readline().strip() != "[output]":
            pass
        line = cfg.readline().strip()
        output_cols = [line]
        while line != "[sampler]":
            line = cfg.readline().strip()
            if len(line) == 0 or line[0] == "#" or line == "[sampler]":
                continue
            output_cols.append(line)
    ic(output_cols)
    n_extra = len(output_cols) - 1
    ic(n_extra)
    # this so that the file is not blocked for the MCMC
    # h5bak = h5.replace('.h5', '.bak.h5')
    h5bak = f"{h5}.bak"
    os.system(f"cp {h5} {h5bak}")
    with h5py.File(h5bak, "r") as file:
        mcmc = file["mcmc"]
        nwalkers = mcmc["blobs"].shape[1]
        signal = read_signal(mcmc, nbins, 0)
        if n_extra > 0:
            extra = {}
            for i, key in enumerate(output_cols[1:], 1):
                if "." in key:
                    extra[key] = read_signal(mcmc, nbins, i)
                else:
                    extra[key] = read_scalar(mcmc, nbins, i)
        ic(extra.keys())
        for key, val in extra.items():
            ic(key, val.shape, val[0])
        # this removes the main signal from the blobs
        # and rearranges for easier access of scalars
        # I think I should remove other signals from here too
        blobs = np.array(
            [[list(blob)[nbins:] for blob in blobs] for blobs in mcmc["blobs"]]
        )
        params = np.array([[pars for pars in walker] for walker in mcmc["chain"]])
    try:
        os.remove(h5bak)
    except FileNotFoundError:
        pass
    if burn == -1:
        burn = -nwalkers
    blobs = np.reshape(blobs, (signal.shape[0], blobs.shape[2]))
    ic(blobs.shape)
    # the second of the additional columns contains the chi2
    chi2 = np.array(blobs[:, n_extra * nbins + 1], dtype=float)
    sampled = chi2 > 0
    # mavg = mavg[sampled]
    if n_extra > 0:
        extra = {name: val[sampled][burn:] for name, val in extra.items()}
    chi2 = chi2[sampled]
    signal = signal[sampled]

    params = params.reshape((params.shape[0] * params.shape[1], params.shape[2]))[
        sampled
    ]
    return signal[burn:], extra, chi2[burn:], params[burn:], nwalkers


def load_data(cfg):
    hm_options, sampling_options = cfg.read()
    (
        function,
        function_cov,
        preamble,
        parameters,
        names,
        prior_types,
        nparams,
        repeat,
        join,
        starting,
        output,
    ) = hm_options
    setup = parameters[1][parameters[0].index("setup")]
    R, signal, cov, Nobsbins, Nrbins = io.load_data(sampling_options, setup)
    cov, icov, likenorm, signal_err, cov2d, corr = cov
    return R, signal, signal_err, cov, cov2d, corr


def mean_mass(args, extra, key, best, nbins, lnlike=None):
    logmavg = np.transpose([extra[f"{key}:{i+1}"][args.burn :] for i in range(nbins)])
    ic(logmavg.shape)
    mavg = 10**logmavg / 1e14
    mo = np.median(mavg, axis=0)
    mlo, mhi = np.abs(np.percentile(mavg, [16, 84], axis=0) - mo)
    # this is what Masato is reporting
    logmo = np.median(logmavg, axis=0)
    logmlo, logmhi = np.abs(np.percentile(logmavg, [16, 84], axis=0) - logmo)
    if args.blind:
        # print('dm = {')
        ...
    else:
        for i, (mi, mu, mh) in enumerate(zip(mo, mlo, mhi)):
            print(
                f"m_{i}/1e14 = ${mi:.1f}_{{-{mu:.1f}}}^{{+{mh:.1f}}}$ & ${mavg[best,i]:.2f}$"
            )
        for i, (mi, mu, mh) in enumerate(zip(logmo, logmlo, logmhi)):
            print(
                f"log m_{i} = ${mi:.2f}_{{-{mu:.2f}}}^{{+{mh:.2f}}}$ & ${logmavg[best,i]:.2f}$"
            )
    nbins = mavg.shape[1]
    labels = [f"log $M_{{500,\!{i}}}$" for i in range(nbins)]
    output = get_output(args, "mavg")
    # otherwise make a density plot
    if mavg.shape[1] > 1:
        c = corner(
            logmavg.T,
            truths=logmavg[best],
            truths_in_1d=True,
            style1d="step",
            bins=15,
            labels=labels,
            percentiles1d=False,
            bins1d="doane",
            smooth=0.5,
            output=output,
            lnlike=lnlike,
            color_likelihood="C3",
        )
    return


def param_info(args, params, best, labels):
    xo = np.median(params, axis=0)
    print(f"After {params.shape[0]} samples:")
    if args.blind:
        print("Uncertainties")
    xlo, xhi = np.abs(np.percentile(params, [16, 84], axis=0) - xo)
    for i, (label, x, xl, xh) in enumerate(zip(labels, xo, xlo, xhi)):
        labelname = (
            label.replace("$", "").replace("\mathcal", "").replace("\mathrm", "")
        )
        if args.blind:
            print(f"{labelname:15s} : {xh+xl:.2f}")
            continue
        xbest = params[best, i]
        print(f"{labelname:15s}  {x:6.2f} -{xl:.2f} +{xh:.2f} (best={xbest:5.2f})")
    return


def plot_b(args, one_minus_b, best):
    """NOTE this is only useful if the observable is Msz!"""
    bbest = one_minus_b[best]
    b0 = np.median(one_minus_b)
    blo, bhi = np.abs(b0 - np.percentile(one_minus_b, [16, 84]))
    if args.blind:
        print(f"1-b range = {bhi+blo:.3f}")
    else:
        print(f"1-b = {b0:.2f} -{blo:.2f} +{bhi:.2f} (best={bbest:.2f})")
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.hist(one_minus_b, bins="auto", histtype="stepfilled", color="C9")
    ax.axvline(np.median(one_minus_b), ls="--", color="k")
    ax.axvline(one_minus_b[best], color="C3", ls="-")
    if not args.blind:
        ax.annotate(
            rf"$\langle 1-b\rangle = {b0:.2f}_{{-{blo:.2f}}}^{{+{bhi:.2f}}}$",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=14,
        )
    ax.set(xlabel="$1-b$", yticks=[], xlim=(0, 2))
    ax.set_title(args.root.split("/")[-1], fontsize=14)
    output = get_output(args, "b")
    fig.savefig(output)
    return


def plot_chi2(args, chi2, cov2d):
    pte, chi2_mc = PTE(chi2, cov2d, n_samples=int(1e6), return_samples=True)
    print(f"min chi^2 = {chi2:.1f} --> PTE = {pte:.2f}")
    log = chi2 > 10 * np.median(chi2_mc)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if log:
        bins = np.logspace(
            np.log10(0.5 * chi2_mc.min()), np.log10(2 * chi2_mc.max()), 100
        )
    else:
        bins = 100
    ax.hist(chi2_mc, bins, histtype="step", density=True)
    ax.axvline(chi2, ls="--", color="k")
    ax.set(
        xscale="log" if log else "linear",
        xlabel="$\chi^2$",
        ylabel="$p(\chi^2)$",
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    output = get_output(args, "chi2")
    savefig(output, fig=fig, tight=False)
    return pte


def plot_chi2_history(
    args, chi2, best, pte, nsteps, nwalkers, stepbin=None, mincolor="C3"
):
    if stepbin is None:
        stepbin = nwalkers
    stepbins = np.arange(0, nsteps, stepbin) + stepbin / 2
    chi2min_historical = [np.argmin(chi2[:nwalkers])]
    ic(chi2min_historical)
    for i in range(nwalkers, nsteps, nwalkers):
        x = chi2[i : i + nwalkers]
        if x.min() < chi2[chi2min_historical[-1]]:
            chi2min_historical.append(i + np.argmin(x))
    chi2min_historical = np.array(chi2min_historical)
    runmed = np.array(
        [np.median(chi2[i : i + stepbin]) for i in range(0, nsteps, stepbin)]
    )
    px = [10, 90, 95, 99]
    runlo, runhi, runmax1, runmax2 = np.transpose(
        [np.percentile(chi2[i : i + stepbin], px) for i in range(0, nsteps, stepbin)]
    )
    # plot!
    third_panel = nsteps > 100000
    fig, axes = plt.subplots(2 + third_panel, 1, figsize=(15, 3 * (2 + third_panel)))
    ax = axes[0]
    rng = np.arange(nsteps)
    ax.plot(
        chi2min_historical,
        chi2[chi2min_historical],
        f"{mincolor}o-",
        lw=3,
        ms=6,
    )
    ax.plot(
        [chi2min_historical[-1], chi2.size],
        [chi2[best], chi2[best]],
        f"{mincolor}--",
        lw=2,
    )
    ax.plot(
        best,
        chi2[best],
        f"{mincolor}x",
        mew=3,
        ms=10,
        zorder=100,
        label=rf"$\chi^2_\mathrm{{min}}={chi2[best]:.1f}$ (PTE={pte:.2f})",
    )
    ax.set(ylabel="$\chi^2_\mathrm{min}$")
    ax.legend(loc="upper right")
    ic(chi2[chi2min_historical], chi2[best])
    if chi2[chi2min_historical].max() > 10 * chi2[best]:
        ax.set_yscale("log")
    axes[1].plot(rng, chi2, "-", color="C9", lw=0.5)
    axes[1].plot(rng, chi2, "o", color="C0", ms=1)
    for j, ax in enumerate(axes[1:], 1):
        ax.plot(
            chi2min_historical,
            chi2[chi2min_historical],
            f"{mincolor}o-",
            lw=3,
            ms=4,
            zorder=100,
        )
        ax.plot(
            best,
            chi2[best],
            f"{mincolor}x",
            mew=3,
            ms=10,
            zorder=100,
            label=rf"$\chi^2_\mathrm{{min}}={chi2[best]:.1f}$ (PTE={pte:.2f})",
        )
        ax.plot(stepbins, runmed, "k-")
        ax.plot(stepbins, runlo, "k--")
        ax.plot(stepbins, runhi, "k--")
        ax.plot(
            [chi2min_historical[-1], chi2.size],
            [chi2[best], chi2[best]],
            f"{mincolor}--",
            lw=2,
        )
        ax.axvline(nsteps - nwalkers, color="0.5", ls="--", lw=1)
        ax.annotate(
            f"50%",
            xy=(1 * nsteps, runmed[-1]),
            ha="left",
            va="center",
            fontsize=16,
        )
        ax.annotate(
            f"{px[0]}%",
            xy=(1 * nsteps, runlo[-1]),
            ha="left",
            va="center",
            fontsize=16,
        )
        ax.annotate(
            f"{px[1]}%",
            xy=(1 * nsteps, runhi[-1]),
            ha="left",
            va="center",
            fontsize=16,
        )
        # ax.legend(loc='upper left')
        ax.set(ylabel=r"$\chi^2$")
        # ax.set_yscale('log')
        # ax.set_ylim()
    # axes[0].set(xlim=axes[1].get_xlim())
    if nsteps > 20000:
        x = chi2[-20000:]
        ymin = x.min() - 0.1 * (x.max() - x.min())
        ymax = x.max() + 0.1 * (x.max() - x.min())
        axes[1].set(xlim=(nsteps - 20000, nsteps + 1200), ylim=(ymin, ymax))
    if third_panel:
        nloops = 5
        axes[2].plot(
            rng[-i * nwalkers :],
            chi2[-i * nwalkers :],
            ".",
            color="C0",
            lw=1,
            alpha=1,
        )
        for i in range(nloops):
            ax.axvline(nsteps - i * nwalkers, color="0.5", ls="--", lw=1)
        axes[2].set(xlim=(nsteps - nloops * nwalkers, nsteps))
        axes[2].set(yscale="log")
    axes[-1].set(xlabel="step")
    output = get_output(args, "chi2_history")
    savefig(output, fig=fig, tight=False)
    return chi2min_historical


def plot_samples(
    args,
    params,
    chi2,
    chi2min_historical,
    nsteps,
    nwalkers,
    labels,
    stepbin=None,
    cmap_samples=20000,
):
    if stepbin is None:
        stepbin = nwalkers
    nsteps, npar = params.shape
    stepbins = np.arange(0, nsteps, stepbin) + stepbin / 2
    runmed = np.transpose(
        [np.median(params[i : i + stepbin], axis=0) for i in range(0, nsteps, stepbin)]
    )
    runlo, runhi = np.transpose(
        [
            np.percentile(params[i : i + stepbin], [16, 84], axis=0)
            for i in range(0, nsteps, stepbin)
        ],
        axes=(1, 2, 0),
    )
    fig, axes = plt.subplots(
        npar, 1, figsize=(12, 15), sharex=True, constrained_layout=True
    )
    if not np.iterable(axes):
        axes = [axes]
    i = np.arange(nsteps)
    ms = "." if nsteps < 12000 else ","
    logL = np.log(chi2)
    colors, cmap = colorscale(
        logL,
        vmin=logL[-cmap_samples:].min(),
        vmax=np.percentile(logL[-cmap_samples:], 99),
        cmap="viridis",
    )
    for ax, p, m, lo, hi, label in zip(axes, params.T, runmed, runlo, runhi, labels):
        ax.scatter(i[::1], p[::1], c=colors, marker=ms, s=0.5)
        ax.plot(
            chi2min_historical,
            p[chi2min_historical],
            "C1+-",
            mew=3,
            ms=8,
            lw=2,
            label="historical minimum",
        )
        ax.plot(
            [chi2min_historical[-1], stepbins[-1]],
            2 * [p[chi2min_historical[-1]]],
            "C1--",
            lw=2,
        )
        ax.plot(stepbins, m, "k-", lw=1.5)
        ax.plot(stepbins, lo, "k--", lw=1.2)
        ax.plot(stepbins, hi, "k--", lw=1.2)
        if nsteps < 20000:
            ax.axvline(nsteps - nwalkers, color="0.5", ls="--", lw=1)
        if args.blind:
            ax.set(yticklabels=[])
        ax.set(ylabel=label)
    plt.colorbar(cmap, ax=axes, label="log Likelihood")
    axes[-1].set(xlabel="MCMC step")
    axes[0].legend(loc="upper left", fontsize=12)
    output = get_output(args, "parameters")
    savefig(output, fig=fig, close=False, tight=False)
    return


def plot_signal(
    args,
    model_signal,
    best,
    R,
    signal,
    signal_err,
    extra,
    binlows,
    binhighs,
    binvar=None,
    model_color="k",
    signame="kappa",
    show_components=True,
):
    # remember that ``best`` already accounts for burn-in
    nbins = len(R)
    ncols = min(nbins, args.ncols)
    nrows = nbins // ncols + (nbins % ncols > 0)
    ic(ncols, nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),  # constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(os.path.split(args.root)[1], fontsize=14)
    if nbins == 1:
        axes = [axes]
    else:
        axes = np.reshape(axes, -1)
    annot_kwargs = dict(
        ha="left",
        va="bottom",
        xycoords="axes fraction",
        xy=(0.04, 0.04),
        fontsize=16,
    )
    ic(model_signal)
    good = np.s_[args.burn :]
    for i, (ax, Ri, s, e) in enumerate(zip(axes, R, signal, signal_err)):
        Ri = np.array(Ri, dtype=float)
        predicted = model_signal[i]
        pred_lo, pred_hi = np.percentile(predicted, [2.5, 97.5], axis=0)
        ax.fill_between(Ri, pred_lo, pred_hi, color="C9", lw=0, alpha=0.5, zorder=-2)
        pred_lo, pred_hi = np.percentile(predicted, [16, 84], axis=0)
        ax.fill_between(Ri, pred_lo, pred_hi, color="C9", lw=0, alpha=0.5, zorder=-1)
        ax.errorbar(Ri, s, e, fmt="o", color="C3", ms=8, elinewidth=2, zorder=1)
        ax.plot(Ri, predicted[best], "-", color=model_color, lw=2, zorder=10)
        if i >= nbins - ncols:
            ax.set(xlabel=r"$\theta$ (arcmin)")
        else:
            ax.set(xticklabels=[])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        # ax.set(ylim=(-0.05, 0.16))
        if i % ncols == 0:
            ax.set(ylabel=r"$\kappa(\theta)$")
        # if binning is not None:
        # if args.root.split('_')[-1][:4] == 'zbin':
        #     ibin = int(args.root[-1])
        #     nbins_annot = 3
        # else:
        #     ibin = i
        #     nbins_annot = nbins
        # if ibin == 0:
        #     text = f"${binvar}<{binning[0]}$"
        # elif ibin == nbins_annot - 1:
        #     text = f"${binvar}>{binning[-1]}$"
        # else:
        #     text = f"${binning[ibin-1]}<{binvar}<{binning[ibin]}$"
        # ax.annotate(text, **annot_kwargs)
        # text = f"${binlows[i]}<{binvar}<{binhighs[i]}$"
        # ax.annotate(text, **annot_kwargs)
    for ax in axes[i + 1 :]:
        ax.axis("off")
    #
    # note that extra still contains the burn-in samples
    best = args.burn + best
    names = extra.dtype.names
    ic(names)
    if show_components:
        for i, (ax, Ri) in enumerate(zip(axes, R)):
            # well-centered components
            if f"{signame}.1h.cent:{i+1}" in names:
                key1h = f"{signame}.1h.cent:{i+1}"
            elif f"{signame}.1h:{i+1}" in names:
                key1h = f"{signame}.1h:{i+1}"
            oneh = extra[key1h]
            ax.plot(
                Ri,
                oneh[best],
                ls="--",
                color=model_color,
                lw=2,
                zorder=9,
                label="1-halo",
            )
            key2h = None
            if f"{signame}.2h.cent:{i+1}" in names:
                key2h = f"{signame}.2h.cent:{i+1}"
            elif f"{signame}.2h:{i+1}" in names:
                key2h = f"{signame}.2h:{i+1}"
            if key2h is not None:
                twoh = extra[key2h]
                ax.plot(
                    Ri,
                    twoh[best],
                    ls=":",
                    color=model_color,
                    lw=2,
                    zorder=9,
                    label="2-halo",
                )
            # off-center components
            if f"{signame}.1h.off:{i+1}" in names:
                ax.plot(
                    Ri,
                    extra[f"{signame}.1h.off:{i+1}"][best],
                    ls="--",
                    color="0.5",
                    lw=2,
                    zorder=9,
                    label="1h-off",
                )
            if f"{signame}.2h.off:{i+1}" in names:
                ax.plot(
                    Ri,
                    extra[f"{signame}.2h.off:{i+1}"][best],
                    ls=":",
                    color="0.5",
                    lw=2,
                    zorder=9,
                    label="2h-off",
                )
        # if f'{signame}.1h' in extra or f'{signame}.2h' in extra:
        axes[0].legend(fontsize=12, loc="upper right", ncol=1)
    #
    """
    ic(extra.keys())
    if "kappa.1h" in extra:
        oneh = extra["kappa.1h"]
        ic(oneh[best])
        for i, (ax, Ri) in enumerate(zip(axes, R)):
            ax.plot(
                Ri,
                oneh[best, i],
                ls="--",
                color=model_color,
                lw=2,
                zorder=9,
                label="1-halo",
            )
    if "kappa.2h" in extra:
        twoh = extra["kappa.2h"]
        for i, (ax, Ri) in enumerate(zip(axes, R)):
            ax.plot(
                Ri,
                twoh[best, i],
                ls=":",
                color=model_color,
                lw=2,
                zorder=9,
                label="2-halo",
            )
    """
    names = extra.dtype.names
    if "kappa.1h" in names or "kappa.2h" in names:
        axes[0].legend(fontsize=14, loc="center right")
    # fig.suptitle(args.root.split('/')[-1], fontsize=12)
    tight_kwargs = dict()
    if nbins % ncols > 0:
        tight_kwargs["h_pad"] = -0.5
    for ext in ("pdf", "png"):
        output = get_output(args, "signal", ext=ext)
        # savefig(output, fig=fig, tight=False)
        savefig(output, fig=fig, **tight_kwargs)
    return


def read_param_names(cfg):
    names = []
    model = ""
    sample_data = None
    in_mor = False
    c_mor = []
    with open(cfg) as f:
        for line in f:
            if "#" in line:
                line = line[: line.index("#")]
            line = line.split()
            if len(line) < 2 or line[0] == "name":
                continue
            if line[0] == "model":
                model = line[1]
            if line[1] in free_priors:
                names.append(line[0])
            if "matched" in model and line[0] == "sample_data":
                sample_data = line[1:]
            # mor
            # if in_mor and line[0][0] == '[':
            #     in_mor = False
            # if line[0] == '[hod/centrals/mor]':
            #     in_mor = True
            # if in_mor:
            #     c_mor.append()
    if sample_data is not None:
        files = sorted(glob(sample_data[0]))
        cols = [int(i) for i in sample_data[1].split(",")]
        sample_data = [np.loadtxt(f, usecols=cols, unpack=True) for f in files]
    ic(sample_data)
    return model, names, sample_data


def read_scalar(mcmc, nbins, i):
    scalar = np.array(
        [
            [list(blob)[i * nbins : (i + 1) * nbins] for blob in blobs]
            for blobs in mcmc["blobs"]
        ]
    )
    scalar = scalar.reshape((scalar.shape[0] * scalar.shape[1], scalar.shape[2]))
    return scalar


def read_signal(mcmc, nbins, i):
    signal = np.array(
        [
            [list(blob)[i * nbins : (i + 1) * nbins] for blob in blobs]
            for blobs in mcmc["blobs"]
        ]
    )
    signal = signal.reshape(
        (signal.shape[0] * signal.shape[1], signal.shape[2], signal.shape[3])
    )
    return signal


def get_output(args, name, ext="png"):
    run_name = args.root.split("/")[-1]
    return os.path.join(args.plot_path, f"{run_name}__{name}.{ext}")


def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("root")
    add("--burn", default=-1, type=int)
    add("--debug", action="store_true")
    add("--ncols", default=3, type=int)
    add("--plot-path", default="plots/mcmc")
    add("--blind", dest="blind", action="store_true")
    args = parser.parse_args()
    args.cfg = f"{args.root}.cfg"
    args.h5 = f"{args.root}.h5"
    args.hdr = f"{args.root}.hdr"
    args.plot_path = args.root.replace("halo_model_results", args.plot_path)
    ic(args.plot_path)
    if not args.debug:
        ic.disable()
    os.makedirs(args.plot_path, exist_ok=True)
    return args


if __name__ == "__main__":
    main()
