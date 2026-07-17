import glob
import nilearn
import nilearn.image
import nibabel
from scipy import ndimage
import numpy as np
import scipy
import json
import os.path
import fire
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
        
def get_stat(data):
    # Only calculate t-stats for non-zero voxels
    non_zero = np.any(data, axis=0) & ~np.any(np.isnan(data),axis=0)
    # Then calculate a t-statistic from this
    stat = np.zeros_like(data[0])
    stat[non_zero] = np.mean(data[:,non_zero], axis=0) / np.std(data[:,non_zero], axis=0) * np.sqrt(len(data))
    return stat

def get_perm_stat(data, random=0):
    # Seed random generator by permutation index to avoid repeats
    rng = np.random.RandomState(random)
    flips = 1-2*rng.randint(0,2,size=len(data))
    # Multiply data by random ones and minus ones
    permuted = data * flips[:,None,None,None]
    # Get stats for this permutation
    return get_stat(permuted)

def get_cluster_mass(stat, clusters, n_clusters, do_sum=True):
    # Calculate the cluster mass for each cluster
    if n_clusters == 0:
        # If there are _no_ clusters, return 0
        return 0
    else:
        # Otherwise return an array of cluster masses
        cluster_stats = [stat[clusters == i + 1] for i in range(n_clusters)]
        cluster_mass = np.array([np.sum(x) if do_sum else np.mean(x) 
                                 for x in cluster_stats])
        return cluster_mass

def get_perm(data, random=0, ref_t=3, clusters=None):
    # Get stats for this permutation
    stat = get_perm_stat(data, random)
    # Extract clusters
    cluster_map, n_clusters = get_clusters(stat, ref_t, clusters)
    # Then find the cluster mass of these clusters
    cluster_mass = get_cluster_mass(stat, cluster_map, n_clusters, do_sum=(clusters is None))
    # And return the max value across clusters
    return np.max(cluster_mass)

def get_clusters(stat, ref_t, clusters=None):
    # Find clusters if not provided, and return their label if they are
    if clusters is None: 
        clusters, n_clusters = ndimage.label(1.0*(stat > ref_t))
    else:
        n_clusters = np.count_nonzero(np.bincount(clusters.flatten())) - 1
    return clusters, n_clusters

def run_group(input_dir='out',
              mask_dir='mask',
              output_dir='group',
              clusters=None,              
              fwhm=5,
              p_thres=0.001,
              n_perm=1000):
    
    # Get input arguments
    args = locals();
    # Store input arguments
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args, f)
    # Load subject level arguments
    sub_args = json.load(open(os.path.join(input_dir,'args.json'), "r"))
    # Load subject level hypotheses, ignoring private keys
    sub_hyp = {k: v for k, v in json.load(open(os.path.join(sub_args['hyp_dir'],sub_args['hyp_file']), "r")).items()
               if k[0] != '_'}
    if 'exclude_nuisance' in sub_args.keys() and sub_args['exclude_nuisance']:
        sub_hyp = {k: v for k, v in sub_hyp.items() if k not in ['pixel', 'size']}
    # Smoothing with a mask is slightly tricky: you don't want to include voxels outside mask
    # But you do want to correct for the fact that voxels near the edge are smoothed with zeros
    # So you smooth, then normalise by the smoothed mask, then mask the result
    # Load the mask and smooth it here, then use it as part of smoothing data later
    mask_img = nilearn.image.load_img(os.path.join(mask_dir,'mask.nii'))
    mask_smooth = nilearn.image.smooth_img(mask_img, fwhm)
    # If a clusters file is provided: run this analysis on *pre-specified* clusters
    # Rather than extracting clusters from maps, we will use the ones in this ROI file
    if clusters is not None:
        clusters = nilearn.image.load_img(os.path.join(mask_dir, clusters)).get_fdata().astype(int)
        if not clusters.shape == mask_img.shape:
            # Annoyingly, all data lives in a slightly cropped standard space
            # Essentially by trial and error I reconstructed the cropping:
            # You can get the transform by fslroi orig.nii orig_cropped.nii 6 79 7 95 1 79
            # For now, I'll simply index the clusters file, but this remains a caveat!
            clusters = clusters[6:85, 7:102, 1:80]
            # This may contain clusters outside the mask, so get rid of those
            clusters = (clusters * mask_img.get_fdata()).astype(int)
    # Collect the null distribution and stats for each each map to make summaray plot at the end
    all_stats = {}

    # Then run through the subject level hypotheses to do stats on each
    for hyp_name in sub_hyp.keys():
        # Get all subject level maps
        files = glob.glob(os.path.join(input_dir,f"s*_cond-*_hyp-{hyp_name}.nii"))
        files.sort()
        # Load the images
        imgs = [nilearn.image.load_img(file) for file in files]
        # Calculate the reference t-value for the requested p threshold
        ref_t = scipy.stats.t.ppf(1-p_thres, len(files) - 1)
        # The following is a nasty list comprehension that relies on nasty eval statements
        # It does exactly the steps above: 1. smooth image, 2. normalise by smoothed mask, 3. mask
        imgs_smooth = [nilearn.image.math_img(
            "mask * (img / np.clip(norm, 1e-12, None))", 
            mask=mask_img, norm=mask_smooth, img=nilearn.image.smooth_img(img, fwhm)) for img in imgs]
        # Then extract the data and continue in numpy format
        maps = np.stack([img.get_fdata() for img in imgs_smooth])

        # Now build the null-distribution of max cluster masses for permutations: random sign flips
        random_mass = np.array(
            Parallel(n_jobs=-1)(delayed(get_perm)(maps, random=i, ref_t=ref_t, clusters=clusters) 
                                for i in range(n_perm)))
        
        # Then calculate the same stats, but for the unpermuted data
        stat_map = get_stat(maps)
        # Extract clusters
        cluster_map, n_clusters = get_clusters(stat_map, ref_t, clusters)
        # If there are _no_ clusters, we're in trouble
        if n_clusters == 0:
            assert False, f"was there not a single cluster for contrast {hyp_name}?!"
        # If there are, find the cluster mass of these clusters
        cluster_mass = get_cluster_mass(stat_map, cluster_map, n_clusters, do_sum=(clusters is None))
        # Then calculate where they fall in the cluster mass null distribution
        p_map = np.zeros_like(stat_map)
        for i, m in enumerate(cluster_mass):
            # Store 1-p in the cluster for easy thresholding in fsleyes
            p_map[cluster_map == (i + 1)] = np.sum(m > np.array(random_mass))/n_perm

        # Create a new image that holds the 1-p map and save it
        p_img = nilearn.image.new_img_like(imgs[0], p_map)
        nibabel.nifti1.save(p_img, os.path.join(output_dir, f"group_rdm-{hyp_name}_clust.nii"))
        # Also store the group t-stats
        stat_img = nilearn.image.new_img_like(imgs[0], stat_map)
        nibabel.nifti1.save(stat_img, os.path.join(output_dir, f"group_rdm-{hyp_name}_t.nii"))
        # And append the null distribution and masses to the all_stats dictionary
        all_stats[hyp_name] = [random_mass, cluster_mass]
    
    # Make one big plot that contains all the null distributions and cluster masses
    plt.figure(figsize=(len(all_stats)*4,4))
    for i, (key, val) in enumerate(all_stats.items()):
        plt.subplot(1, len(all_stats), i+1)
        null_dist, vals = val
        counts, edges = np.histogram(null_dist, bins=int(n_perm/10))
        plt.stairs(counts, edges, fill=True, color=[0.8,0.8,0.8])
        plt.plot(vals, 0.02*np.max(counts)*np.ones_like(vals), 'b*')
        significance = np.percentile(null_dist, 95)
        plt.plot([significance, significance], [0, np.max(counts)], 'r-')
        plt.title(key)
        plt.xlabel('Cluster mass')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'dist.pdf'))

if __name__ == "__main__":
    # Use fire to create command line interface. Example usage:
    # uv run group_stat_bootstrap.py --input_dir="out_05" --output_dir 
    # "group_05" --clusters "YeoBuckner7_maxprob-thr25-2mm.nii" --n_perm=1000
    fire.Fire(run_group)
