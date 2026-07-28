#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 07:36:34 2026

Running the instruction-phase RSA, per timepoint.

@author: Svenja Küchenhoff, 2026
"""


from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from nilearn.image import load_img
from rsatoolbox.util.searchlight import get_volume_searchlight
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
#import pdb; pdb.set_trace() 


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    # Define task pairing relationships
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control  = [], [], []
    # Loop through keys in the *specified order*
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split('_')  # e.g. ['A1', 'forw', 'A', 'reward']
        # Create the pairing suffix (e.g. from '1_forw' → '2_backw')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        # Build the paired key (e.g. 'A2_backw_A_reward')
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        # Only add if both keys exist
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")

    # sanity check: every task-half-1 key in keys_list must have found a partner
    n_th1_expected = sum(1 for k in keys_list if k.split('_')[0].endswith('1'))
    assert len(paired_list_control) == n_th1_expected, (
        f"Expected {n_th1_expected} pairs, got {len(paired_list_control)}. "
        "Some task-half-1 keys did not find their same-execution partner in the dict."
    )
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    # print(paired_list_control)
    return th_1, th_2, paired_list_control


# Names, in canonical order, of the four channels that constitute a "literal split"
# of rewDSR at the A_reward anchor (curr = chunk 0, next = 1, two_next = 2, three_next = 3).
REWDSR_SPLIT_CHANNELS = ('curr_rew', 'next_rew', 'two_next_rew', 'three_next_rew')


# models in .json can be 
# instr = visual instruction similarity
# "selected_models": [
#       "DSR", "rewDSR", "simple",
#       "rewDSR_instr",                                                                                                                  
#       "curr_rew", "curr_rew_instr",
#       "next_rew", "next_rew_instr",
#       "two_next_rew", "two_next_rew_instr",
#       "three_next_rew", "three_next_rew_instr"
#   ]


# Suffix that marks the instruction-similarity variant of a model. Any model in
# `selected_models` ending in this suffix is built by first substituting each
# '_backw_' key's value with its '_forw_' counterpart so that A1_forw and
# A1_backw share the same model vector (they saw the same instruction).
INSTR_SUFFIX = '_instr'


def strip_instr(name):
    """Return (base_name, is_instr). 'rewDSR_instr' -> ('rewDSR', True)."""
    if name.endswith(INSTR_SUFFIX):
        return name[:-len(INSTR_SUFFIX)], True
    return name, False


def instruction_relabel_dict(model_subdict):
    """Replace each '<task>_backw_<state>_<phase>' key's value with the
    corresponding '<task>_forw_<state>_<phase>' value. Forward keys stay
    unchanged. Turns an execution-similarity model dict into an
    instruction-similarity model dict — under the current execution-based
    data pairing, this yields uniform 2x2 sub-blocks per (task_letter_i,
    task_letter_j) in the resulting model RDM."""
    out = dict(model_subdict)
    for k in list(model_subdict.keys()):
        parts = k.split('_')
        if len(parts) >= 2 and parts[1] == 'backw':
            forw_key = k.replace('_backw_', '_forw_', 1)
            if forw_key in model_subdict:
                out[k] = model_subdict[forw_key]
    return out


def verify_instruction_rdm_blocks(rdm, th1_labels, th2_labels, tol=1e-9):
    """Instruction models should be uniform inside each (task_letter_i,
    task_letter_j) 2x2 sub-block. Returns (block_df, all_uniform) — the
    df has one row per (task_i, task_j) with the unique block value and a
    uniformity flag."""
    from collections import defaultdict
    import pandas as _pd
    def _task_id(lbl):
        return lbl.split('_')[0]     # e.g. 'A1_forw_A_reward' -> 'A1'
    g1, g2 = defaultdict(list), defaultdict(list)
    for i, l in enumerate(th1_labels): g1[_task_id(l)].append(i)
    for j, l in enumerate(th2_labels): g2[_task_id(l)].append(j)
    rows = []
    all_uniform = True
    for t1, idx1 in sorted(g1.items()):
        for t2, idx2 in sorted(g2.items()):
            block = np.asarray(rdm)[np.ix_(idx1, idx2)]
            first = float(block.flat[0])
            uniform = np.allclose(block, first, atol=tol)
            if not uniform:
                all_uniform = False
            rows.append({'task1': t1, 'task2': t2,
                          'shape': block.shape,
                          'value': first,
                          'min':  float(block.min()),
                          'max':  float(block.max()),
                          'uniform': uniform})
    return _pd.DataFrame(rows), all_uniform


def _lower_tri_flat(mat):
    """Return the strict lower triangle (k=-1) of a square matrix, flattened.
    Used when ``data_rdm_scope == 'full_no_diag'``. Cosine dissim is symmetric,
    so the upper triangle is redundant; the diagonal (self-pairs) is 0 and
    would be pure autocorrelation, so we drop it too. Row-major ordering
    follows ``np.tril_indices(N, k=-1)`` — the matching model regressor must
    use the same ordering."""
    mat = np.asarray(mat)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], (
        f"_lower_tri_flat expects a square matrix, got {mat.shape}")
    i, j = np.tril_indices(mat.shape[0], k=-1)
    return mat[i, j]


def assemble_full_rdm_from_blocks(W1, A, W2):
    """Build the (n1+n2, n1+n2) block RDM from three (n,n) blocks:

        +----------------+----------------+
        |   W1 (within)  |   A (across)   |
        +----------------+----------------+
        |   A.T          |   W2 (within)  |
        +----------------+----------------+

    The lower-left block is A.T, giving a symmetric assembled matrix
    (cosine/hamming dissim are symmetric)."""
    W1 = np.asarray(W1); A = np.asarray(A); W2 = np.asarray(W2)
    n1, n2 = W1.shape[0], W2.shape[0]
    assert A.shape == (n1, n2), (
        f"A block shape {A.shape} does not match ({n1}, {n2})")
    R = np.empty((n1 + n2, n1 + n2), dtype=float)
    R[:n1, :n1] = W1
    R[:n1, n1:] = A
    R[n1:, :n1] = A.T
    R[n1:, n1:] = W2
    return R


def slice_rewDSR_into_split_channels(model_EVs, EV_keys, use_instruction=False):
    """
    Build (th1, th2) matrices for curr_rew / next_rew / two_next_rew / three_next_rew
    as 12-element chunks of rewDSR at the A_reward anchor.

    Doing it this way makes the four channels a LITERAL split of rewDSR:
      * each chunk is a 12-vec of the raw location value (e.g. [4]*12)
      * hamming dissim then behaves the same way as on rewDSR itself
        (match -> 0, mismatch -> 1), rather than the 9-dim one-hot version
        stored in model_EVs['curr_rew'] etc. (which gives mismatch = 2/9).

    Parameters
    ----------
    use_instruction : bool
        If True, apply ``instruction_relabel_dict`` to the rewDSR sub-dict
        before pairing/slicing — produces the instruction-similarity variant
        of the four split channels.

    Returns
    -------
    dict : {channel_name: (th1_mat, th2_mat)}
        Each matrix has shape (n_pairs, 12).
    """
    rewDSR_sub = {k: v for k, v in model_EVs['rewDSR'].items() if k.endswith('_A_reward')}
    if use_instruction:
        rewDSR_sub = instruction_relabel_dict(rewDSR_sub)
    rewDSR_keys = [k.replace('_instruction_onset', '_A_reward') for k in EV_keys]
    th1_full, th2_full, _ = pair_correct_tasks(rewDSR_sub, rewDSR_keys)

    n_pairs, n_total = th1_full.shape
    assert n_total % 4 == 0, (
        f"rewDSR at A_reward has {n_total} elements, not divisible by 4."
    )
    CHUNK = n_total // 4
    out = {}
    for i, name in enumerate(REWDSR_SPLIT_CHANNELS):
        out[name] = (th1_full[:, i * CHUNK:(i + 1) * CHUNK],
                     th2_full[:, i * CHUNK:(i + 1) * CHUNK])
    return out

#
#
# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

# --- Load configuration ---
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_instruction_full.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")
TR = config.get("TR")
regression_version_full = f"{regression_version}-TR{TR}"


name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}"


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)
PLOTTING = True

# --- Scope of the data/model RDM ------------------------------------------
# 'across_only'   : classic behaviour — n_conds x n_conds cross-block only
#                   (TH1 vs TH2). Each cell is a pure across-runs comparison,
#                   so run-level noise cannot inflate similarity.
# 'full_no_diag'  : symmetric (2n_conds x 2n_conds) full RDM (within-run-1,
#                   across, within-run-2), strict lower triangle only (k=-1),
#                   diagonal dropped. Nearly doubles the number of pairs the
#                   OLS sees. Off-diagonal within-run cells share run-level
#                   noise → within-run pairs tend to look more similar than
#                   across-run pairs at the same true stimulus similarity;
#                   this bias is known but not corrected here.
data_rdm_scope = config.get("data_rdm_scope", "across_only")
assert data_rdm_scope in ("across_only", "full_no_diag"), (
    f"data_rdm_scope must be 'across_only' or 'full_no_diag' "
    f"— got {data_rdm_scope!r}")
print(f"data_rdm_scope = {data_rdm_scope}")

# this should better be: what kind of searchlight_mask do you want?
# make sure to change this in the config files!
#load_searchlights = config.get("load_searchlights", False)
searchlight_mask = config.get("searchlight_mask", None)

print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        only_load_labels = False 
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        only_load_labels = False
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version_full}_{searchlight_mask}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}/results"
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}_smooth{fwhm}/results"
    os.makedirs(results_dir, exist_ok=True)

    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    if searchlight_mask:
        if searchlight_mask == 'no_CSF':
            mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
            mask_name = '_no_CSF' # Found 166.240 searchlights with no CSF mask
        elif searchlight_mask == 'grey_matter':
            mask_file = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
            mask_name = '_grey_matter'  # Found 126.404 searchlights with gm mask 
    else:
        mask_file = ref_img.copy() # full BOLD Found 175.483 searchlights
        mask_name = ''
    mask = mask_file.get_fdata()  
    path_to_searchlight_centers = f"{data_dir}/func/searchlight_centers{mask_name}.pkl"
    path_to_searchlight_neighbours = f"{data_dir}/func/searchlight_neighbors{mask_name}.pkl"
    if os.path.exists(path_to_searchlight_centers):
        with open(path_to_searchlight_centers, "rb") as f:
            centers = pickle.load(f)
        with open(path_to_searchlight_neighbours, "rb") as f:
            neighbors = pickle.load(f)
    else:
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)
        with open(path_to_searchlight_centers, 'wb') as file:
            pickle.dump(centers, file)
            print("stored searchlight centres")
        with open(path_to_searchlight_neighbours, 'wb') as file:
            pickle.dump(neighbors, file)   
            print("stored searchlight neighbors")

    #
    # Step 2: loading conditions for model and data RDMs
    #
    # Model EVs — full dict (used to build DSR / rewDSR / simple below).
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
    # Which models to build + evaluate. Driven by the config so we can swap
    # between the original ['DSR', 'rewDSR', 'simple'] analysis and the new
    # ['curr_rew', 'next_rew', 'two_next_rew', 'three_next_rew'] split_rew_DSR
    # analysis without touching the script.
    selected_models = config.get("selected_models", ['DSR', 'rewDSR', 'simple'])
    # Data EVs — one PE per instruction-phase condition at this TR, per task half.
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs_instr_TRwise(
        data_dir, regression_version=regression_version, TR=TR,
        only_load_labels=only_load_labels,
    )
    EV_keys = sorted(all_EV_keys)
    print(f"including the following EVs in the RDMs: {EV_keys}")

    # Pair task halves by same-execution (A1_forw <-> A2_backw, etc.).
    data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    data_concat = np.concatenate((data_th1, data_th2), axis=0)
    
    #
    # Step 3: compute the model RDMs.
    # Labels aligned with data pairing (row -> TH1 label, col -> TH2 label).
    th1_labels = [p.split(' with ')[0].replace('_instruction_onset', '') for p in paired_labels]
    th2_labels = [p.split(' with ')[1].replace('_instruction_onset', '') for p in paired_labels]

    model_RDM_dir = {}
    # In 'full_no_diag' mode we additionally store the assembled (2n, 2n)
    # block matrix for each model. The OLS then uses its strict lower
    # triangle via _lower_tri_flat(). Keeping model_RDM_dir[model] as the
    # (n, n) across block preserves the existing PLOTTING / verification code.
    model_RDM_full_dir = {}

    # Split-channel sources — one for execution rewDSR, one for instruction
    # rewDSR. Each provides the same four (th1, th2) matrices; the models
    # 'curr_rew' and 'curr_rew_instr' just draw from the different source.
    split_th_by_channel = {}
    split_th_by_channel_instr = {}
    _needs_split_exec  = any(strip_instr(m) == (base, False)[0] and not strip_instr(m)[1]
                              for m in selected_models
                              for base in REWDSR_SPLIT_CHANNELS)
    _needs_split_instr = any(strip_instr(m) == (base, True)[0] and strip_instr(m)[1]
                              for m in selected_models
                              for base in REWDSR_SPLIT_CHANNELS)
    # simpler: check each split channel explicitly
    _base_of = [strip_instr(m) for m in selected_models]
    if any(base in REWDSR_SPLIT_CHANNELS and not is_instr for base, is_instr in _base_of):
        split_th_by_channel = slice_rewDSR_into_split_channels(
            model_EVs, EV_keys, use_instruction=False)
    if any(base in REWDSR_SPLIT_CHANNELS and is_instr for base, is_instr in _base_of):
        split_th_by_channel_instr = slice_rewDSR_into_split_channels(
            model_EVs, EV_keys, use_instruction=True)

    # Every non-'simple' model in `selected_models` is built the same way:
    # hamming dissim over its A_reward vectors, TH1 x TH2, full off-block.
    # Models ending in '_instr' are built with instruction relabelling first.
    for model in selected_models:
        base_name, is_instr = strip_instr(model)
        if base_name == 'simple':
            continue
        if base_name in REWDSR_SPLIT_CHANNELS:
            source = split_th_by_channel_instr if is_instr else split_th_by_channel
            m_th1, m_th2 = source[base_name]
        else:
            # standard path: filter model_EVs[base_name] to its _A_reward keys
            a_rew_sub = {k: v for k, v in model_EVs[base_name].items()
                          if k.endswith('_A_reward')}
            if is_instr:
                a_rew_sub = instruction_relabel_dict(a_rew_sub)
            a_rew_keys = [k.replace('_instruction_onset', '_A_reward') for k in EV_keys]
            m_th1, m_th2, _ = pair_correct_tasks(a_rew_sub, a_rew_keys)
        model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th1, m_th2)
        if data_rdm_scope == 'full_no_diag':
            W1 = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th1, m_th1)
            W2 = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th2, m_th2)
            model_RDM_full_dir[model] = assemble_full_rdm_from_blocks(
                W1, model_RDM_dir[model], W2)

    # Simple — {-1, +1, NaN} based on same/different execution within the same task letter.
    if 'simple' in selected_models:
        model_RDM_dir['simple'] = mc.analyse.my_RSA.build_simple_instruction_RDM(th1_labels, th2_labels)
        if data_rdm_scope == 'full_no_diag':
            all_labels = th1_labels + th2_labels
            model_RDM_full_dir['simple'] = mc.analyse.my_RSA.build_simple_instruction_RDM(
                all_labels, all_labels)

    # ── Instruction-model verification + correlation with execution ──────
    # For every model ending in `_instr`, verify the 2x2 sub-block
    # uniformity property (every (task_letter_i, task_letter_j) block should
    # be internally constant) and print the block table. Then, if the
    # execution counterpart is also present, report Pearson r between the
    # two RDMs so you can see how much variance they share.
    instr_models_present = [m for m in selected_models if m.endswith(INSTR_SUFFIX)]
    for im in instr_models_present:
        rdm = np.asarray(model_RDM_dir[im])
        block_df, all_uniform = verify_instruction_rdm_blocks(
            rdm, th1_labels, th2_labels)
        print(f"\n[instr-model verify] {im}: RDM shape = {rdm.shape}, "
              f"all 2x2 blocks uniform = {all_uniform}")
        with pd.option_context('display.width', 200,
                                'display.max_rows', 100):
            print(block_df.round(4).to_string(index=False))
        # Correlation with execution counterpart. In 'full_no_diag' mode we
        # correlate over the actual OLS regressors (strict lower-tri of the
        # (2n, 2n) block matrix); in 'across_only' mode we correlate over the
        # ravel'd (n, n) A block, which is exactly the regressor the OLS sees.
        base_name = im[:-len(INSTR_SUFFIX)]
        if base_name in model_RDM_dir:
            if data_rdm_scope == 'full_no_diag' and base_name in model_RDM_full_dir:
                exec_flat = _lower_tri_flat(model_RDM_full_dir[base_name])
                instr_flat = _lower_tri_flat(model_RDM_full_dir[im])
                scope_note = "full lower-tri"
            else:
                exec_flat = np.asarray(model_RDM_dir[base_name]).ravel()
                instr_flat = np.asarray(model_RDM_dir[im]).ravel()
                scope_note = "across block"
            m = np.isfinite(exec_flat) & np.isfinite(instr_flat)
            r_pearson = float(np.corrcoef(exec_flat[m], instr_flat[m])[0, 1])
            print(f"[correlation, {scope_note}] {base_name} (execution) vs {im} "
                  f"(instruction): Pearson r = {r_pearson:+.4f}  "
                  f"(n_cells = {int(m.sum())})")
        else:
            print(f"[correlation] {base_name} not in selected_models — "
                  f"skipping execution-vs-instruction correlation.")
    
    if PLOTTING == True:
        # Plot each selected model RDM (plotted from the stored arrays — no
        # recomputation). We always plot the (n, n) across block for readability.
        # If we're in 'full_no_diag' mode, we ALSO plot the assembled (2n, 2n)
        # matrix so you can visually verify the within- and across-run blocks.
        for model in selected_models:
            if model == 'simple':
                vmin, vmax, title = -1, 1, 'simple execution dissim'
            else:
                vmin, vmax, title = 0, 1, f'{model} A_reward hamming dissim'
            mc.analyse.my_RSA.plot_instruction_RDM(model_RDM_dir[model], th1_labels, th2_labels,
                                                   title=title, vmin=vmin, vmax=vmax,
                                                   save_path=f"{results_dir}_{model}")
            if data_rdm_scope == "full_no_diag" and model in model_RDM_full_dir:
                all_labels = th1_labels + th2_labels
                mc.analyse.my_RSA.plot_instruction_RDM(
                    model_RDM_full_dir[model], all_labels, all_labels,
                    title=f'{title} (full 2n x 2n, W1|A|W2)',
                    vmin=vmin, vmax=vmax,
                    save_path=f"{results_dir}_{model}_full")

        # Optional inspection plot: cosine dissim from one random searchlight.
        plot_example_data_RDM = config.get("plot_example_data_RDM", False)
        if plot_example_data_RDM and not only_load_labels:
            rng = np.random.default_rng(42)
            sl_idx = int(rng.integers(0, len(centers)))
            vox_ids = np.asarray(neighbors[sl_idx])
            sl_data = data_concat[:, vox_ids]
            n_conds = sl_data.shape[0] // 2
            example_data_RDM = mc.analyse.my_RSA.compute_cosine_instruction_RDM(sl_data[:n_conds], sl_data[n_conds:])
            mc.analyse.my_RSA.plot_instruction_RDM(
                example_data_RDM, th1_labels, th2_labels,
                title=f'example data RDM (searchlight #{sl_idx}, cosine dissim)', save_path=f"{results_dir}_data"
            )
            if data_rdm_scope == "full_no_diag":
                example_full = mc.analyse.my_RSA.compute_cosine_instruction_RDM(sl_data, sl_data)
                all_labels = th1_labels + th2_labels
                mc.analyse.my_RSA.plot_instruction_RDM(
                    example_full, all_labels, all_labels,
                    title=f'example FULL data RDM (searchlight #{sl_idx})',
                    save_path=f"{results_dir}_data_full")
    
        plt.show(block=False)
    
    #
    # Step 4: compute the data RDM per searchlight (cosine dissim).
    #   'across_only'   : (n_conds, n_conds) off-block, ravel'd -> n_conds**2 cells
    #   'full_no_diag'  : (2n, 2n) symmetric, strict lower tri -> 2n*(2n-1)//2 cells
    # The cache filename carries a '_full' suffix in the second case so the
    # two variants live side-by-side and never overwrite each other.
    os.makedirs(data_rdm_dir, exist_ok=True)
    scope_tag = "" if data_rdm_scope == "across_only" else "_full"
    data_rdm_name = f"data_RDM{scope_tag}"
    data_rdm_npy = f"{data_rdm_dir}/{data_rdm_name}.npy"
    if not os.path.exists(data_rdm_npy):
        if data_rdm_scope == "across_only":
            data_RDMs = mc.analyse.my_RSA.get_instruction_RDM_per_searchlight(
                data_concat, centers, neighbors)
        else:
            data_RDMs = mc.analyse.my_RSA.get_full_instruction_RDM_per_searchlight(
                data_concat, centers, neighbors)
        mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(
            data_RDMs, data_rdm_dir, data_rdm_name, ref_img, centers)
    else:
        data_RDMs = np.load(data_rdm_npy)

    if smoothing == True:
        smooth_name = f"data_RDM_smooth_fwhm{fwhm}{scope_tag}"
        smooth_npy = f"{data_rdm_dir}/{smooth_name}.npy"
        if not os.path.exists(smooth_npy):
            path_to_save_smooth = f"{data_rdm_dir}/{smooth_name}"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs = mc.analyse.handle_MRI_files.smooth_RDMs(
                data_RDMs, ref_img, fwhm, use_rsa_toolbox=False,
                path_to_save=path_to_save_smooth, centers=centers)
        else:
            data_RDMs = np.load(smooth_npy)
    print(f"[data RDM] scope={data_rdm_scope}, cells per searchlight = {data_RDMs.shape[1]}")

    #
    # Step 5: evaluate each single model against every searchlight data RDM.
    # NaN cells in the simple model automatically drop the corresponding data
    # cells from the OLS (see evaluate_model_vec). Both X and Y are z-scored.
    #
    # Helper: pick the correct model matrix and flatten it in a way that
    # matches the current data RDM layout — full ravel of the (n, n) A block
    # in 'across_only' mode, strict lower-tri of the (2n, 2n) full block in
    # 'full_no_diag' mode.
    def _model_regressor(m):
        if data_rdm_scope == "across_only":
            return np.asarray(model_RDM_dir[m]).ravel()
        return _lower_tri_flat(model_RDM_full_dir[m])

    RSA_results = {}
    run_single_models = config.get("run_single_models", True)
    if run_single_models == True:
        for model in selected_models:
            model_flat = _model_regressor(model)
            RSA_results[model] = Parallel(n_jobs=3)(
                delayed(mc.analyse.my_RSA.evaluate_model)(model_flat, d)
                for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}")
            )
            mc.analyse.handle_MRI_files.save_my_RSA_results(
                result_file=RSA_results[model], centers=centers,
                file_path=results_dir, file_name=f"{model}",
                mask=mask, number_regr=0, ref_image_for_affine_path=ref_img,
            )

    # import pdb; pdb.set_trace()
    run_combo_models = config.get("run_combo_models", bool(config.get("combo_models")))
    if run_combo_models:
        combo_list = config["combo_models"]
        for combo in combo_list:
            combo_model_name = combo["name"]
            models_to_combine = combo["regressors"]
            print(f"running combo model {combo_model_name}")
            # check if these models have been computed in model_EVs
            missing = [m for m in models_to_combine if m not in model_RDM_dir]
            if missing:
                for m_int in missing:
                    if m_int.endswith('interaction'):
                        curr_m = m_int.split('_interaction')[0]
                        z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
                        model_RDM_dir[m_int] = [z(model_RDM_dir[curr_m][0]) * z(model_RDM_dir['path_rew'][0])]
                        # model_RDM_dir[m_int] = [model_RDM_dir[curr_m][0]*model_RDM_dir['path_rew'][0]]
                    else:
                        raise ValueError(f"Combo model {combo_model_name} not possible, as {missing} not computed")
  
            # Each model is either a (n_th1, n_th2) across block ('across_only'
            # mode) or a (2n, 2n) block ('full_no_diag'). _model_regressor
            # returns the correctly-shaped 1D vector for whichever mode we're in.
            stacked_model_RDMs = np.stack(
                [_model_regressor(m) for m in models_to_combine],
                axis=1,
            )

            # How correlated are the regressors of this combo model with each other?
            # NaN-safe pearson via pandas; then print a compact upper-triangle summary
            # and save a heatmap alongside the results.
            import pandas as _pd
            corr = _pd.DataFrame(stacked_model_RDMs, columns=models_to_combine).corr().to_numpy()
            print(f"\n[{combo_model_name}] pairwise Pearson r between regressor RDMs:")
            for i in range(len(models_to_combine)):
                for j in range(i + 1, len(models_to_combine)):
                    print(f"    {models_to_combine[i]:>16s} vs {models_to_combine[j]:<16s}: r = {corr[i, j]:+.3f}")
            mc.analyse.my_RSA.plot_model_correlations(
                stacked_model_RDMs, models_to_combine,
                save_path=f"{results_dir}_{combo_model_name}_regressor_corr",
                show=True,
            )
              
            estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
            for i, model in enumerate(models_to_combine):
                # TODO: Change the type of similarity to not throw away half of the matrix.
                mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
            
    

    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "EV_string": EV_string,
        "regression_version": regression_version,
        "TR": TR,
        "regression_version_full": regression_version_full,
        "RDM_version": RDM_version,
        "paired_labels": paired_labels,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "searchlight_mask": searchlight_mask,
        "data_rdm_scope": data_rdm_scope,
        "n_cells_per_searchlight": int(data_RDMs.shape[1]),
        "n_all_EVs": len(all_EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": selected_models,
        "run_combo_models": run_combo_models,
        "combo_models": config.get("combo_models", []),
        "data_dir": data_dir,
        "results_dir": results_dir
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            

