from cluster_data import *
import ast
from plottery.plotutils import update_rcParams
import healpy as hp
import warnings 

warnings.filterwarnings('ignore') 

global init_extract_data
def init_extract_data(ymap, mask):
    global shared_ymap2extract, shared_mask2extract
    shared_ymap2extract = ymap
    shared_mask2extract = mask

global mask_regions_worker
def mask_regions_worker(ra, dec, ymap, size_rad = np.deg2rad(10/60), mask_shape = "square", wcs = None, using_pool = False, 
                N_total = None, counter = None, worker_id = None):
    if using_pool == True:
        ymap = enmap.ones(ymap.shape, wcs = wcs)
    wcs = ymap.wcs if wcs is None else wcs
    iter = range(len(ra)) if using_pool == True else tqdm(range(len(ra)))
    for i in iter:
        if using_pool == True and counter is not None:
            counter.value +=1
        rai, deci = ra[i], dec[i]
        pos = np.array([np.radians(deci), np.radians(rai)]) 
        pix = enmap.sky2pix(ymap.shape, ymap.wcs, pos)
        if mask_shape == "square":
            pix_size_x = int(np.abs(size_rad / np.deg2rad(ymap.wcs.wcs.cdelt[0]))) 
            pix_size_y = int(np.abs(size_rad / np.deg2rad(ymap.wcs.wcs.cdelt[1])))

            i_min, i_max = max(0, int(pix[0] - pix_size_y//2)), min(ymap.shape[-2], int(pix[0] + pix_size_y//2))
            j_min, j_max = max(0, int(pix[1] - pix_size_x//2)), min(ymap.shape[-1], int(pix[1] + pix_size_x//2))

            ymap[..., i_min:i_max, j_min:j_max] = 0
    
        elif mask_shape == "disk":
            y_pixels, x_pixels = np.indices(ymap.shape[-2:])
            distance = np.sqrt((y_pixels - pix[0])**2 + (x_pixels - pix[1])**2)
            radius_pix = size_rad / np.abs(np.deg2rad(ymap.wcs.wcs.cdelt[0]))
            ymap[..., distance < radius_pix] = 0

        if using_pool == True and counter is not None and N_total is not None:
            sys.stdout.write(f"\rExtracting data: ({counter.value} / {N_total})")
            sys.stdout.flush()

    if using_pool == True and worker_id is not None:
        print("Worker ", worker_id, "was already finished!\n")
    return ymap

global extract_cluter_data
def extract_cluster_data(ymap, ra, dec, z, zerr, richness, richness_err, ID, wcs_info = None, mask = None,
                pix_size = 0.5, using_pool = False, patch_size = 0.8, worker_id = None, counter = None, N_total = None
                ,replace = False, fmask_ratio = 1):
    if os.path.exists(output_path + "individual_clusters") == False: os.mkdir(output_path + "individual_clusters")
    clusters = []
    ID = np.array(ID, dtype = int)
    if ymap is None and mask is None and "shared_ymap2extract" in globals() and "shared_mask2extract" in globals():
        ymap = shared_ymap2extract
        mask = shared_mask2extract
    elif ymap is not None and mask is not None and wcs_info is not None and using_pool == True:
        ymap = enmap.ndmap(ymap, wcs=wcs_info)
        mask = enmap.ndmap(mask, wcs=wcs_info)
    wcs_info = wcs_info if wcs_info is not None else ymap.wcs
    iter = range(len(ra)) if using_pool == True else tqdm(range(len(ra)))
    for i in iter:
        if using_pool == True:
            counter.value+=1
        if using_pool == True and counter is not None and N_total is not None:
            sys.stdout.write(f"\rExtracting data: ({counter.value} / {N_total})")
            sys.stdout.flush()
        rai, deci, zi, zerri = ra[i], dec[i], z[i], zerr[i]
        l, le = richness[i], richness_err[i]
        current_output_path = output_path + "individual_clusters/" + "cluster_ID=" + str(ID[i]) + "/" 
        if replace == False and os.path.exists(current_output_path):
            continue
        box = [
            [np.deg2rad(deci) - np.deg2rad(patch_size) / 2.0, np.deg2rad(rai) - np.deg2rad(patch_size) / 2.0],
            [np.deg2rad(deci) + np.deg2rad(patch_size) / 2.0, np.deg2rad(rai) + np.deg2rad(patch_size) / 2.0],
        ]
        smap = ymap.submap(box) if reproject_maps == False else reproject.thumbnails(ymap, coords = (np.deg2rad(deci),np.deg2rad(rai)), r = np.deg2rad(patch_size)/2.)
        smask = mask.submap(box) if reproject_maps == False else reproject.thumbnails(mask, coords = (np.deg2rad(deci),np.deg2rad(rai)), r = np.deg2rad(patch_size)/2.)
        smask[smask >= 3/4] = 1
        smask[smask <= 3/4] = 0
        fmask = len(smask[smask == 1])/smask.size 
        if fmask <= fmask_ratio:
            continue
        shape = np.shape(smap)
        pcx, pcy = shape[0]//2, shape[1]//2
        pixel_width = np.deg2rad(width) / np.shape(smap)[0]
        x,y = np.indices(np.shape(smap))
        theta = np.sqrt(((x - pcx)*pixel_width)**2 + (((y - pcy))*pixel_width)**2) * u.arcmin
        R = (theta * cosmo.angular_diameter_distance(zi)).value * u.kpc * 1000
        x,y = (x - pcx)*pixel_width, (y - pcy)*pixel_width
        cluster = sz_cluster(
            rai, 
            deci,
            l,
            le,
            R,
            smap,
            smask,
            zi,
            zerri,
            box,
            int(ID[i]),
        )     
        cluster.fmask = fmask 
        cluster.W = np.sum(smask)
        cluster.theta = theta
        cluster.x = x * u.arcmin
        cluster.y = y * u.arcmin
        cluster.generate_profile(r=R_profiles, wcs = wcs_info)
        cluster.output_path = (
            output_path
            + "individual_clusters/"
            + "cluster_ID="
            + str(ID[i])
            + "/" 
        )
        cluster.ID = str(ID[i])
        cluster.save()
        cluster.plot(save = True, plot_signal = True, patchsize = patch_size, pixel_size = pixel_size,
                    show_cluster_information = ["z","richness"], cluster_information_names = ["$z$", r"$\lambda$"])
        clusters.append(cluster)
        plt.close("all")
    if using_pool == True and worker_id is not None:
        print("Worker ", worker_id, "was already finished!\n")
    return clusters



update_rcParams()

parser = argparse.ArgumentParser()

parser.add_argument("--estimate_var", "-V", action = "store_true", help = "Estimate error using the given ymap.")
parser.add_argument("--var_method", "-VM", default= "bootstrap", help = "method to estimate variance.")
parser.add_argument("--compute_corr_matrix", "-CM", action = "store_true", help = "Compute correlation matrix making 5000 random realizations of profiles.")
parser.add_argument("--n_cov_samples", "-NC", default = 1000, type = float)
parser.add_argument("--data_path", "-f", default = None, help = "path that contain data. If it is not passed, the ymap and gal_cat argument wil be considers as global paths.")
parser.add_argument("--ymap", "-M", help = "Path of ymap")
parser.add_argument("--gal_cat", "-G", help = "catalog of galaxy clusters")
parser.add_argument("--patch_size", "-s", default = 0.8, type = float, help = "Size of patches to extract y data in degrees.")
parser.add_argument("--beam_size", "-B", default = 1.6, type = float, help = "Beam size in [arcmin].")
parser.add_argument("--stacking_method", "-S", default = "simple", help = "Stacking method.")
parser.add_argument("--output_path", "-O", default = None, help = "output path to store stacking results.", type = str)
parser.add_argument("--annulus_width", "-W", default = 1, help = "Annulus size in [arcmin].", type = float)
parser.add_argument("--radius", "-r", default = "0,12", help = "rmax and rmin respectively in arcmin units in format rmin,rmax")
parser.add_argument("--N_cores", "-n", default = 1, help = "Number of core to run the respective stacking algorithm.")
parser.add_argument("--coords", "-c", default = "ra,dec,z", help = "Keys in galaxy catalog that corresponds to RA, DEC, Redshift and Redshift error, comma separated (ra,dec,z, z_err) format.")
parser.add_argument("--use_config_file", "-C", default = None, help = "Use a .ini config file instead of passed arguments.")
parser.add_argument("--mask_path", "-m", default = None, help = "Mask associated to ymap. Is None it will not be considered.")
parser.add_argument("--properties", "-p", default = None, help = "A set of properties, with format [('proper_name', 'error_name')...], to store.")
parser.add_argument("--reproject", "-R", action = "store_true", help = "Reproject invidiual maps using Pixell.")
parser.add_argument("--ID", "-I", default = "MEM_MATCH_ID", help = "Name of ID for each indiviual cluster.")
parser.add_argument("--richness", "-l", default = "LAMBDA_CHISQ, LAMBDA_CHISQ_E", help = "Richness keys in format (richness, richness_err).")

args = parser.parse_args()

if args.use_config_file is None:

    data_path = args.data_path
    ymap_path = args.ymap if data_path is None else data_path + args.ymap
    cat_path = args.gal_cat if data_path is None else data_path + args.gal_cat
    mask_path = args.mask_path if data_path is None else data_path + args.mask_path
    patch_size = args.patch_size 
    reproject = args.reproject
    annulus_width = args.annulus_width
    ncores = args.N_cores
    rmin, rmax = np.array(args.radius.strip().split(","), dtype = float)
    R = np.arange(rmin, rmax, annulus_width)
    ID = args.ID
    RA,DEC,Z, ZERR = args.coords.split().split(",")
    LAMDA, LAMDA_E = args.richness.split().split(",")
    output_path = args.output_path

else:
    print("Extracting configuration from " + args.use_config_file)
    config_data = load_config(args.use_config_file)
    estimate_var = config_data.get("estimate_var", False)
    var_method = config_data.get("var_method", "bootstrap")
    compute_corr_matrix = config_data.get("compute_corr_matrix", False)
    n_cov_samples = config_data.get("n_cov_samples", 1000)
    pixel_size = config_data.get("pixel_size", 0.5)
    extract_data = config_data.get("extract_data", False)
    stacking = config_data.get("stacking", True)
    rewrite = config_data.get("rewrite", False)
    fmask_ratio = config_data.get("fmask_ratio", 1)

    data_path = config_data.get("data_path", None)
    ymap_path = config_data.get("ymap", None) if data_path is None else data_path + config_data.get("ymap", None)
    mask_path = config_data.get("mask", None) if data_path is None else data_path + config_data.get("mask", None)
    gal_cat_path = config_data.get("gal_cat", None) if data_path is None else data_path + config_data.get("gal_cat", None)
    output_path = config_data.get("output_path", None)

    patch_size = config_data.get("patch_size", 0.8)
    beam_size = config_data.get("beam_size", 1.6)
    stacking_method = config_data.get("stacking_method", "simple")

    annulus_width = config_data.get("annulus_width", 1.0)

    radius = [float(x) for x in config_data.get("radius", "0,12")]
    rmin, rmax = radius
    R_profiles = np.arange(rmin, rmax, annulus_width)
    N_cores = int(config_data.get("n_cores", 1))
    coords = config_data.get("coords", "ra,dec,z")
    RA, DEC, Z, ZERR = coords
    estimate_background = config_data.get("estimate_background", True)
    use_cov_matrix = config_data.get("use_cov_matrix", True)
    use_corr_matrix = config_data.get("use_corr_matrix", False)

    use_config_file = config_data.get("use_config_file", None)
    properties = config_data.get("properties", None)
    reproject_maps = config_data.get("reproject", False)
    ID = config_data.get("ID", "MEM_MATCH_ID")
    delta_richness = config_data.get("delta_richness", 10)
    snr_threshold = config_data.get("snr_threshold", 8)
    use_bootstrap = config_data.get("bootstrap", False)
    richness = config_data.get("richness", "LAMBDA_CHISQ, LAMBDA_CHISQ_E")
    use_median_redshift = config_data.get("use_median_redshift", False)
    redshift_bins = config_data.get("redshift_bins", None)
    redshift_bins = np.array(redshift_bins, dtype = float) if redshift_bins is not None else None
    zero_level = config_data.get("zero_level", False)
    clusters_mask = config_data.get("clusters_mask","clusters_mask.fits")
    create_mask = config_data.get("create_clusters_mask", True)
    mask_radius = config_data.get("mask_radius", 10)
    mask_shape = config_data.get("mask_shape", "disk")
    estimate_covariance = config_data.get("estimate_covariance", True)
    min_richness = config_data.get("min_richness", None)
    initial_richness = config_data.get("initial_richness", None)
    max_richness = config_data.get("max_richness", None)
    weighted = config_data.get("weighted", True)
    LAMDA, LAMDA_E = richness


if os.path.exists(output_path) == False: 
    os.mkdir(output_path)

global ymap, cat, mask

if __name__ == "__main__":

    ymap = enmap.read_map(ymap_path)
    cat = Table(fits.open(gal_cat_path)[1].data)
    mask = enmap.read_map(mask_path)
    clusters_mask = hp.fitsfunc.read_map(clusters_mask)
    ra, dec, z, zerr = cat[RA], cat[DEC], cat[Z], cat[ZERR]
    lamda, lamda_err = cat[LAMDA], cat[LAMDA_E]
    ra[ra > 180] = ra[ra > 180] - 360
    ID = cat[ID]

    if create_mask == True and os.path.exists(data_path + "clusters_mask.fits") == False:
        print("Building clusters mask")
        clusters_mask = hp.fitsfunc.read_map(data_path + clusters_mask)
        clusters_mask[clusters_mask < 0.5] = 0
        clusters_mask[clusters_mask >= 0.5] = 1
        nside = hp.get_nside(clusters_mask)
        dr = np.deg2rad(mask_radius/60) 
        coords = SkyCoord(ra=ra, dec=dec, unit = "deg", frame='icrs')
        theta = np.radians(90 - coords.dec.degree)
        phi = np.radians(coords.ra.degree)   
        for t,p in zip(theta, phi):
            mask_pix = hp.query_disc(nside, hp.ang2vec(t, p), dr)
            clusters_mask[mask_pix] = 0 
        hp.write_map(data_path + "clusters_mask.fits", clusters_mask, overwrite=True)
        print("saving cluster mask...")



    # if compute_corr_matrix == True:
    #     size_rad = np.deg2rad(10 / 60) #unmask clusters using an square with size of 5 arcmins
    #     Nmax_samples = n_cov_samples
    #     ymap2 = ymap.copy()
    #     print("Computing Correlation matrix using y-map + catalog common area.")
    #     print("*Patch size (rad):",size_rad)
    #     print("*N samples:",Nmax_samples)

    #     new_map_output = ymap_path.replace(".fits", "") + "_unmasked_clusters.fits"
    #     ra, dec = ra[:50], dec[:50]

    #     if os.path.exists(new_map_output) == "OHAA":
    #         ymap2 = enmap.read_map(new_map_output)
    #     else:
    #         if pool is None:
    #             print("Building new mask!")
    #             ymap2 = mask_regions_worker(ra, dec, ymap, size_rad, mask_shape = mask_shape)
    #         elif pool is not None:
    #             N_cores = int(pool._processes)
    #             print(f"Building new mask with {pool._processes} cores!")
    #             manager = Manager()
    #             counter = manager.Value("i", 0 )
    #             pars = np.array_split(np.column_stack((ra, dec)), N_cores)
    #             pars = [(*p.T, np.asarray(ymap), size_rad, mask_shape, ymap.wcs, True, len(ra), counter, i) for i,p in enumerate(pars)]    
    #             new_maps = pool.starmap(mask_regions_worker, pars)
    #             ymap2 = new_maps[0]
    #             for i in range(1,len(new_maps)):
    #                 ymap2*=new_maps[i]
    #             ymap2*=ymap
    #         print(f"saving mask to {new_map_output}")
    #         enmap.write_map(new_map_output, ymap2, allow_modify = True)
        
    #     corr_samples = []
    #     cbar = tqdm(total = Nmax_samples, desc = "Computing Correlation Matrix")
    #     while len(corr_samples) < Nmax_samples:
    #         ra2, dec2 = np.deg2rad(np.random.uniform(ra.min(), ra.max())), np.deg2rad(np.random.uniform(dec.min(), dec.max()))
    #         box = [
    #             [np.deg2rad(dec2) - np.deg2rad(patch_size) / 2.0, np.deg2rad(ra2) - patch_size / 2.0],
    #             [np.deg2rad(dec2) + np.deg2rad(patch_size) / 2.0, np.deg2rad(ra2) + patch_size / 2.0],
    #         ]
    #         smap = ymap2.submap(box) if reproject_maps == False else reproject.thumbnails(ymap2, coords = (dec2,ra2), r = np.deg2rad(patch_size)/2.)
    #         smask = mask.submap(box) if reproject_maps == False else reproject.thumbnails(mask, coords = (np.deg2rad(deci),np.deg2rad(rai)), r = np.deg2rad(patch_size)/2.)
    #         smask[smask >= 3/4] = 1
    #         smask[smask <= 3/4] = 0
    #         fmask = len(smask[smask == 1])/smask.size 

    #         if np.all(smap != 0 ) and fmask >= fmask_ratio:
    #             corr = np.zeros((len(R_profiles) - 1, len(R_profiles) - 1))
    #             Rbins, profile, sigma, data = radial_binning(smap, R_profiles, wcs = ymap.wcs)
    #             num_bins = len(Rbins)
    #             max_length = max([len(d) for d in data])
    #             matrix = []
    #             for d in data:
    #                 matrix.append(np.array(list(d) + [np.nan for i in range(max_length - len(d))]))
    #             masked_matrix = np.ma.masked_array(matrix, np.isnan(matrix))
    #             corr = np.ma.corrcoef(masked_matrix)

    #             corr_samples.append(corr)
    #             cbar.update(1)

    #     corrm = np.mean(corr_samples, axis = 0)
    #     np.save(output_path + "correlation_matrix.npy", np.mean(corr_samples, axis = 0))

    else:
        if os.path.exists(output_path + "correlation_matrix.npy"):
            corrm = np.load(output_path + "correlation_matrix.npy")
            if os.path.exists(output_path + "correlation_matrix.png") == False:
                fig, ax = plt.subplots(figsize = (12,12))
                im = ax.imshow(np.abs(corrm), cmap = "seismic", interpolation = "none", origin = "lower", norm = LogNorm(),
                extent = [R_profiles.min(), R_profiles.max(), R_profiles.min(), R_profiles.max()])
                ax.set_xlabel("R (arcmin)", fontsize = 16)
                ax.set_ylabel("R (arcmin)", fontsize = 16)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax = cax)
                cbar.set_label(r"$\log_{10}{|cov|}$", fontsize=18)  
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=12)
                fig.tight_layout()
                fig.savefig(output_path + "correlation_matrix.png")
    if extract_data == True:
        print("Extracting data...")
        if N_cores > 1:
            print("Running data extractor with", N_cores, "cores!")
            wcs_info = ymap.wcs
            manager = Manager()
            counter = manager.Value("i", 0 )
            N_total = len(ra)
            pars = np.array_split(np.column_stack((ra, dec, z, zerr, lamda, lamda_err, ID)), N_cores)
            
            with Pool(N_cores, initializer=init_extract_data, initargs=(ymap, mask)) as pool:
                pars = [(None, *p.T, wcs_info, None, pixel_size, True, patch_size, i, counter, N_total, rewrite, fmask_ratio) for i,p in enumerate(pars)]
                res = pool.starmap(extract_cluster_data, pars)
                pool.close()
                pool.join()
        else:
            res = extract_cluster_data(ymap, ra, dec, z, zerr, lamda, lamda_err, ID, 
                    mask = mask, pix_size = pixel_size, using_pool = False, patch_size = patch_size, 
                    fmask_ratio = fmask_ratio)

    if stacking == True:
        print("Stacking and spliting data...")
        print(f"Running stacking with {N_cores} cores.") if N_cores > 1 else None
        if rewrite == True:
            available_clusters = [output_path + "individual_clusters/" + p for p in os.listdir(output_path + "individual_clusters")]
            clusters = []
            for i in tqdm(range(len(available_clusters)), desc = "Loading available clusters..."):
                p = available_clusters[i]
                clusters.append(sz_cluster.load_from_path(p))
            g = np.sum(clusters)
            g.output_path = output_path + "/entire_sample"
            if os.path.exists(g.output_path) == False:
                os.mkdir(g.output_path)
            #g.compute_covariance_matrices(R_profiles, width)
            g.save()
            g.plot()
            A
            g.stacking(R_profiles, plot = True, background_err = False, bootstrap = False, compute_cov_matrix = False, ymap = ymap,
             estimate_covariance = False, estimate_background = False, verbose = True, n_pool = N_cores, 
            mask = mask)
            g.plot()
            g.save()
        else:
            g = grouped_clusters.load_from_path(output_path + "/entire_sample")
            g.output_path = output_path + "/entire_sample"
            # if compute_corr_matrix == True:
            #     print("Computing correlation matrix using entire sample")
            #     g.stacking(R_profiles, plot = True, width = patch_size, pool = N_cores, verbose = True,
            #         bootstrap = use_bootstrap, N_realizations = 1000, background_err = estimate_background,
            #         compute_zero_level = zero_level, ymap = ymap, mask = mask, clusters_mask = clusters_mask
            #         )
        subgroups = g.split_optimal_richness(R_profiles = R_profiles, method = "stacking", SNr = snr_threshold, rdistance = delta_richness, 
            width = patch_size, N_realizations = 1000, split_by_median_redshift = use_median_redshift, use_bootstrap = use_bootstrap, n_pool = N_cores,
            redshift_bins = redshift_bins, estimate_background = estimate_background, estimate_covariance = estimate_covariance, compute_zero_level = zero_level, 
            ymap = ymap, mask = mask, clusters_mask = clusters_mask, min_richness = min_richness, initial_richness = initial_richness, weighted = weighted
            , max_richness = max_richness)
        for s in subgroups:
            if os.path.exists(s.output_path) == False:
                os.mkdir(s.output_path)
            s.save()
            s.plot()


    if estimate_var == True:
        if var_method == "bootstrap":
            N = 1000 #typical number of bootstrap resampling
            N_clusters = 100 #number of randomly selected clusters
            bootstrap_profiles = []
            for n in range(N):
                signals = []
                indx = np.random.choice(np.arange(0, len(ra) + 1, 1), N_clusters)
                new_ra, new_dec, new_z = ra[indx], dec[indx], z[indx]
                for i in range(len(new_ra)):
                    rai, deci, zi = new_ra[i], new_dec[i], new_z[i]
                while True:
                    box = [
                        [np.deg2rad(deci) - patch_size / 2.0, np.deg2rad(rai) - patch_size / 2.0],
                        [np.deg2rad(deci) + patch_size / 2.0, np.deg2rad(rai) + patch_size / 2.0],
                    ]

                    smap = ymap.submap(box) if reproject == False else reproject.thumbnails(ymap, coords = (deci,rai), r = np.deg2rad(patch_size)/2.)
                    smask = mask.submap(box) if mask is not None else np.ones(np.shape(smap))
                    if np.all(smask == 1):
                        signals.append(smap)
                        break
                    else:
                        new_indx = np.randon.randint(0, len(ra)) 
                        rai, deci, zi = ra[new_indx], dec[new_indx], z[new_indx]    
                stack = np.mean(signals, axis = 0)
                R_bins, profile, err, _ = radial_binning(stack, R, patch_size = np.rad2deg(width))
                bootstrap_profiles.append(profile)

            mean_profile = np.mean(bootstrap_profiles, axis=0)
            std_profile = np.std(bootstrap_profiles, axis=0)
            one_sigma_bounds = np.percentile(bootstrap_profiles, [16, 84], axis = 0)
            two_sigma_bounds = np.percentile(bootstrap_profiles, [2.5, 97.5], axis = 0)
