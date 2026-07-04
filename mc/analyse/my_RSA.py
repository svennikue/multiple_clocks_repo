#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:38:27 2025

@author: Svenja Küchenhoff


All things RSA


"""
import mc
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import statsmodels.api as sm
import os
import nibabel as nib


# ---------------------------------------------------------------------------
# Across-task-halves helpers, shared by the model-EV build script and the
# downstream RSA evaluation script so they operate on identical task pairings
# and produce the SAME RDM that the searchlight regression uses.
# ---------------------------------------------------------------------------
HAMMING_MODELS_DEFAULT = frozenset({
    'location', 'DSR', 'prev_buttons', 'buttons_out', 'next_buttons',
    'phys_abstr_space', 'action_DSR', 'state_action_DSR',
    'state_action_glob', 'state_action_loc',
    'rewDSR', 'pathDSR', 'rew_stateactionDSR', 'path_stateactionDSR',
    'DSR_onefut', 'DSR_twofut', 'DSR_threefut', 'DSR_fourfut',
    'DSR_fivefut', 'DSR_sixfut', 'DSR_sevenfut',
    'curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter',
})


def pair_correct_tasks(data_dict, keys_list):
    """Pair ``X1_<dir>_<state>_<phase>`` with its same-goal partner
    ``X2_<flipped_dir>_<state>_<phase>`` (i.e. the other task-half repetition
    of the same goal configuration). The mapping is::

        '1_forw'  ↔ '2_backw'
        '1_backw' ↔ '2_forw'

    Returns ``(th_1, th_2, paired_labels)`` where ``th_1`` and ``th_2`` are
    ``(n_pairs, n_features)`` arrays, in ``keys_list`` order, restricted to
    keys whose pair is also in ``data_dict``. ``paired_labels`` is a list of
    ``"<key> with <pair_key>"`` strings (for inspection / provenance).
    """
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control = [], [], []
    for key in keys_list:
        assert key in data_dict, "Mismatch between model and data RDM keys"
        task, direction, state, phase = key.split('_')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")
    if not th_1:
        return None, None, []
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    return th_1, th_2, paired_list_control


def th1_keys_for(data_dict, keys_list):
    """The list of ``th_1`` keys (in ``keys_list`` order) that survive the
    pairing inside ``pair_correct_tasks`` — i.e. the row/column labels of
    the resulting across-halves RDM.
    """
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    out = []
    for key in keys_list:
        if key not in data_dict:
            continue
        parts = key.split('_')
        if len(parts) != 4:
            continue
        task, direction, state, phase = parts
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        if pair_suffix is None:
            continue
        if f"{task[0]}{pair_suffix}_{state}_{phase}" in data_dict:
            out.append(key)
    return out


def _expand_triu_to_square(vec, include_diagonal=True):
    """Inverse of ``np.triu_indices``: rebuild the symmetric n × n matrix
    from its upper-triangle vector. Use ``include_diagonal`` consistent with
    how the vector was produced.
    """
    vec = np.asarray(vec)
    if include_diagonal:
        n = int(round((-1 + np.sqrt(1 + 8 * len(vec))) / 2))
    else:
        n = int(round((1 + np.sqrt(1 + 8 * len(vec))) / 2))
    rdm = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, k=(0 if include_diagonal else 1))
    rdm[iu] = vec
    rdm[(iu[1], iu[0])] = vec
    return rdm


def build_across_halves_model_RDM(model_name, model_EV, EV_keys,
                                  hamming_models=None,
                                  include_diagonal=True):
    """Build the across-task-halves model RDM exactly the way the downstream
    script ``scripts/fMRI_run_RSA_without_rsatoolbox_clean.py`` does.

    Steps: pair via :func:`pair_correct_tasks`, vstack to get the
    ``(2*n_pairs, n_features)`` matrix, then dispatch to the matching builder
    (``make_categorical_RDM`` for ``path_rew``, ``make_distance_RDM`` for
    ``duration``, ``compute_hamming_distance`` for the order-preserving
    models, ``compute_crosscorr`` for everything else). Returns the
    ``n_pairs × n_pairs`` symmetric across-halves RDM along with the th_1
    keys (= axis labels), the method used and a suggested vmin/vmax/vcenter
    triple for display.

    Returns ``None`` if no pairs could be formed.
    """
    if hamming_models is None:
        hamming_models = HAMMING_MODELS_DEFAULT
    th1, th2, _ = pair_correct_tasks(model_EV, EV_keys)
    if th1 is None:
        return None
    th1_keys = th1_keys_for(model_EV, EV_keys)
    model_concat = np.vstack([th1, th2])

    if model_name == 'path_rew':
        rdm_vec = make_categorical_RDM(model_concat,
                                       include_diagonal=include_diagonal)[0]
        method = 'categorical'
        vrange = {'vmin': -0.5, 'vmax': 0.5, 'vcenter': 0.0}
    elif model_name == 'duration':
        rdm_vec = make_distance_RDM(model_concat,
                                    include_diagonal=include_diagonal)[0]
        method = 'distance'
        vrange = {'vmin': 0.0, 'vmax': 2.0, 'vcenter': 1.0}
    elif model_name in hamming_models:
        rdm_vec = compute_hamming_distance(model_concat,
                                           include_diagonal=include_diagonal,
                                           model_name=model_name)[0]
        method = 'hamming_distance'
        vrange = {'vmin': 0.0, 'vmax': 1.0, 'vcenter': 0.5}
    else:
        rdm_vec = compute_crosscorr(model_concat,
                                    include_diagonal=include_diagonal)[0]
        method = 'crosscorr'
        vrange = {'vmin': 0.5, 'vmax': 1.5, 'vcenter': 1.0}

    rdm_square = _expand_triu_to_square(rdm_vec,
                                        include_diagonal=include_diagonal)
    return rdm_square, th1_keys, method, vrange


def load_data_EVs(data_dir, regression_version, old=False, only_load_labels = False):
    EV_dict = {}
    # import pdb; pdb.set_trace()
    # names need to be 'A1_backw_A_path' etc.
    list_loaded = []
    for th in [1,2]:
        if regression_version.startswith('0'):
            # the old versions are numbers starting with 0. here, only take the first letters.
            pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
            EV_path = f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt"
        else:
            pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{th}.feat/stats"
            EV_path = f"{data_dir}/func/EVs_{regression_version}_pt0{th}/task-to-EV.txt"
        # order from FSL processed EVs to names is stored here:
        
        with open(EV_path, 'r') as file:
        # pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
        # # order from FSL processed EVs to names is stored here:
        # with open(f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                if only_load_labels == False:
                    EV_dict[name] = np.array(nib.load(EV_path).get_fdata()).flatten()
                else:
                    EV_dict[name] = np.zeros((1,1))
                    # reshape data so we have 1 x n_voxels
                    # import pdb; pdb.set_trace()
                if name not in ['press_EV', 'up', 'down', 'left', 'right']:
                    list_loaded.append(name)
    # print(f"loaded the following data EVs in dict: {list_loaded}")
    return EV_dict, list_loaded
    
def load_data_EVs_th(data_dir, regression_version):
    EV_dict = {}
    # names need to be 'A1_backw_A_path' etc.
    list_loaded = []
    for th in [1,2]:
        # NOTE: if you still want to run the old ones, adjust which part of the regression version you take to load.
        pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{th}.feat/stats"
        # order from FSL processed EVs to names is stored here:
        with open(f"{data_dir}/func/EVs_{regression_version}_pt0{th}/task-to-EV.txt", 'r') as file:
        # pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
        # # order from FSL processed EVs to names is stored here:
        # with open(f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                # reshape data so we have 1 x n_voxels
                # import pdb; pdb.set_trace()
                if name not in ['press_EV', 'up', 'down', 'left', 'right']:
                    list_loaded.append(f"th_{th}_{name}")
                    EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                    EV_dict[f"th_{th}_{name}"] = np.array(nib.load(EV_path).get_fdata()).flatten()

    print(f"loaded the following data EVs in dict: {list_loaded}")
    return EV_dict, list_loaded


def compute_hamming_instruction_RDM(th1, th2):
    """
    Full (n1, n2) Hamming DISSIMILARITY between task-half-1 rows and task-half-2 rows.
    RDM[i, j] = 1 - mean(th1[i] == th2[j]). No symmetrization, no diagonal dropping.
    """
    th1 = np.asarray(th1)
    th2 = np.asarray(th2)
    sim = (th1[:, None, :] == th2[None, :, :]).mean(axis=2)
    return 1.0 - sim


def compute_cosine_instruction_RDM(th1, th2):
    """
    Full (n1, n2) cosine DISSIMILARITY between task-half-1 rows and task-half-2 rows.
    Same math as compute_crosscorr, but keeps the entire off-block (no symmetrization,
    no diagonal dropping).
    """
    X1 = np.asarray(th1, dtype=float)
    X2 = np.asarray(th2, dtype=float)
    X1 = X1 - X1.mean(axis=1, keepdims=True)
    X2 = X2 - X2.mean(axis=1, keepdims=True)
    X1 /= np.sqrt(np.einsum('ij,ij->i', X1, X1))[:, None]
    X2 /= np.sqrt(np.einsum('ij,ij->i', X2, X2))[:, None]
    return 1.0 - X1 @ X2.T


def build_simple_instruction_RDM(th1_labels, th2_labels):
    """
    Simple execution-based dissimilarity matrix, shape (n1, n2).
      -1 if same task letter AND same execution (parity match)
      +1 if same task letter AND different execution (parity mismatch)
     NaN if different task letters (i.e. A vs B, etc.) -> excluded from the RSA

    Labels expected to look like 'A1_backw' (may carry extra suffixes, only the
    first two underscore-separated tokens are used). Execution parity encodes
    "canonical (like A1_forw) vs. reversed" — computed as
        parity = (task_half == '2') XOR (direction == 'backw').
    """
    def _parse(label):
        task, direction = label.split('_')[:2]
        letter = task[0]
        half = task[-1]
        parity_reverse = (half == '2') ^ (direction == 'backw')
        return letter, parity_reverse

    n1, n2 = len(th1_labels), len(th2_labels)
    R = np.full((n1, n2), np.nan)
    for i, l1 in enumerate(th1_labels):
        let_i, par_i = _parse(l1)
        for j, l2 in enumerate(th2_labels):
            let_j, par_j = _parse(l2)
            if let_i != let_j:
                continue
            R[i, j] = -1.0 if par_i == par_j else 1.0

    # Balance check: within each same-letter block AND overall, non-NaN entries sum to 0.
    total = np.nansum(R)
    assert np.isclose(total, 0.0), (
        f"Simple dissim RDM not balanced (grand total = {total:.4f}, expected 0)."
    )
    return R


def plot_instruction_RDM(rdm, th1_labels, th2_labels, title, vmin=None, vmax=None, cmap='coolwarm', save_path=None):
    """
    Plot a precomputed (n1, n2) RDM. Does NOT recompute anything.
    NaN cells are rendered transparent (over a white background) so exclusions are visible.
    """
    fig, ax = plt.subplots()
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')
    im = ax.imshow(np.ma.masked_invalid(rdm), aspect='equal', cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(th2_labels)))
    ax.set_yticks(np.arange(len(th1_labels)))
    ax.set_xticklabels(th2_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(th1_labels, fontsize=8)
    ax.set_xlabel('task half 2')
    ax.set_ylabel('task half 1')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def load_data_EVs_instr_TRwise(data_dir, regression_version, TR, only_load_labels=False):
    """
    Loader for per-TR instruction-phase GLMs.
    Reads PEs from   glm_{regression_version}-TR{TR}_pt0{th}.feat/stats
    and EV mapping   EVs_{regression_version}-TR{TR}_pt0{th}/task-to-EV.txt
    for th in {1, 2}. Skips button/press EVs so only the instruction EVs remain.
    """
    EV_dict = {}
    list_loaded = []
    for th in [1, 2]:
        pe_path = f"{data_dir}/func/glm_{regression_version}-TR{TR}_pt0{th}.feat/stats"
        EV_txt  = f"{data_dir}/func/EVs_{regression_version}-TR{TR}_pt0{th}/task-to-EV.txt"
        with open(EV_txt, 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                if name in ['press_EV', 'up', 'down', 'left', 'right']:
                    continue
                EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                if only_load_labels:
                    EV_dict[name] = np.zeros((1, 1))
                else:
                    EV_dict[name] = np.array(nib.load(EV_path).get_fdata()).flatten()
                list_loaded.append(name)
    return EV_dict, list_loaded


def get_instruction_RDM_per_searchlight(fmri_data, centers, neighbors):
    """
    Per-searchlight cosine dissimilarity RDM for the instruction-phase RSA.

    Parameters
    ----------
    fmri_data : ndarray, shape (2 * n_conds, n_voxels)
        TH1 rows stacked on TH2 rows (i.e. the ``data_concat`` from the script).
    centers : sequence of voxel indices for each searchlight.
    neighbors : list of arrays, one per center, giving voxel indices in the
        searchlight neighbourhood.

    Returns
    -------
    sl_rdms : ndarray, shape (n_centers, n_conds * n_conds)
        Each row is a searchlight's full (TH1 x TH2) dissimilarity matrix,
        flattened row-major with ``.ravel()``. No symmetrization, no diagonal
        dropping — same convention as :func:`compute_cosine_instruction_RDM`.
    """
    centers = np.array(centers)
    n_centers = centers.shape[0]
    n_conds = fmri_data.shape[0] // 2

    sl_rdms = np.zeros((n_centers, n_conds * n_conds))

    if n_centers > 1000:
        chunked_center = np.split(
            np.arange(n_centers),
            np.linspace(0, n_centers, 101, dtype=int)[1:-1],
        )
    else:
        chunked_center = [np.arange(n_centers)]

    for chunks in tqdm(chunked_center, desc='Calculating instruction RDMs...'):
        for c in chunks:
            sl_data = fmri_data[:, np.asarray(neighbors[c])]
            th1 = sl_data[:n_conds]
            th2 = sl_data[n_conds:]
            sl_rdms[c, :] = compute_cosine_instruction_RDM(th1, th2).ravel()

    return sl_rdms


def get_RDM_per_searchlight(fmri_data, centers, neighbors, method = 'crosscorr', labels = None, full_mask=None, mask_pairs=None, include_diagonal=True):
    # import pdb; pdb.set_trace()
    centers = np.array(centers)
    #n_conds = fmri_data['1'].shape[0]
    n_conds = int(fmri_data.shape[0]/2)
    
    # first step: parallelise centers/neighbors.
    n_centers = centers.shape[0]
    # For memory reasons, we chunk the data if we have more than 1000 RDMs
    # loop over chunks
    if n_centers > 1000:
        # we can't run all centers at once, that will take too much memory
        # so lets to some chunking
        chunked_center = np.split(np.arange(n_centers),
                                  np.linspace(0, n_centers,
                                              101, dtype=int)[1:-1])
        if include_diagonal == True:
            # output will INCLUDE the diagonal. so triangle number is:
            sl_rdms = np.zeros((n_centers, n_conds * (n_conds + 1) // 2))
        if include_diagonal == False:
            # if excluding the diagonal
            sl_rdms = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
        #for chunks in chunked_center:
        for chunks in tqdm(chunked_center, desc='Calculating RDMs...'):            
            center_data= []
            for c in chunks:
                # grab this centers of this chunk and its and neighbors
                center_neighbors = neighbors[c]
                center_data.append(fmri_data[:, center_neighbors])
            # then compute the RDM per searchlight
            if method == 'crosscorr':
                RDM_corr = mc.analyse.my_RSA.compute_crosscorr(center_data, include_diagonal=include_diagonal)
            elif method == 'crosscorr_and_filter':
                RDM_corr = mc.analyse.my_RSA.compute_crosscorr_and_filter(center_data, labels=labels, full_mask=full_mask, mask_pairs=mask_pairs)
            else:
                assert False, "invalid method"
            sl_rdms[chunks, :] = RDM_corr # then store per voxel and return.
       
    return sl_rdms
        
     
def parse_label_pair(label):
    """
    Example:
        'A1_backw_A_reward with A2_forw_A_reward'
    ->  state = 'A'
        task1 = 'A1_backw'
        task2 = 'A2_forw'
    """
    left, right = label.split(" with ")

    def parse_side(side):
        side = side.replace("_reward", "")
        task, state = side.rsplit("_", 1)
        return task, state

    task1, state1 = parse_side(left)
    task2, state2 = parse_side(right)

    assert state1 == state2, f"State mismatch inside label: {label}"
    return state1, task1, task2

def plot_rdm_with_labels(rdm, labels, group_size=4):
    n = rdm.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(rdm, aspect='auto', cmap='coolwarm', vmin=0, vmax=2)

    # Tick positions and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    # White lines after each task (every `group_size` rows/cols)
    for k in range(group_size, n, group_size):
        ax.axhline(k - 0.5, color='white', linewidth=1)
        ax.axvline(k - 0.5, color='white', linewidth=1)

    plt.tight_layout()
    plt.show()
       
def compute_crosscorr_and_filter(data_chunk, labels = None, full_mask=None, mask_pairs=None, plotting = False, binarise = False):  
    RDM = []
    # import pdb; pdb.set_trace()

    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        # centers the data around zero by subtracting the mean of each row
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        # normalising data
        data_demeaned /= np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))[:, None]    
        # cosine dissimilarity
        rdm_both_halves = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)  

        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
    
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        # import pdb; pdb.set_trace()
        # if you want to mask/filter, do that first
        if full_mask:
            # collect the indices I want to mask
            idx_to_mask = []
            for i, label in enumerate(labels):
                for m in full_mask:
                    if m in label:
                        idx_to_mask.append(i)
            
            rdm[:, idx_to_mask] = np.nan
            rdm[idx_to_mask] = np.nan
            
            if binarise == True:
                # THIS IS ONLY FOR MODEL RDMS!!
                rdm = np.where(np.isnan(rdm), np.nan, (rdm > 0.5).astype(float))
        
        elif mask_pairs:
            # Two cases:
            # 1) mask_pairs is your OLD list of substrings  -> keep old logic
            # 2) mask_pairs is the NEW big dict (state -> loc -> [tasks])
            if isinstance(mask_pairs, dict):
                # import pdb; pdb.set_trace()
                # --- NEW: mask "same state, same location" ---
                # Prepare splitting labels
                parsed = [parse_label_pair(lab) for lab in labels]
                n_cond = len(labels)

                for i in range(n_cond):
                    # loop through all conditions i
                    state_i, t1_i, t2_i = parsed[i] # split the labels
                    # for the respective state of the current condition, call the paired mask.
                    loc_dict_i = mask_pairs.get(state_i, {})
                    if not loc_dict_i:
                        continue
                    # next, check each paired condition j
                    for j in range(i, n_cond):
                        state_j, t1_j, t2_j = parsed[j]
                        if state_j != state_i:
                            continue

                        # Check if there exists a location where BOTH conditions live
                        for loc, tasks in loc_dict_i.items():
                            task_set = set(tasks)
                            if (t1_i in task_set and t2_i in task_set and
                                t1_j in task_set and t2_j in task_set):
                                # same state, same location -> mask
                                rdm[i, j] = np.nan
                                rdm[j, i] = np.nan
                                break  # stop looping over locs for this (i,j)

            else:
                # --- OLD behaviour: substring-based mask_pairs ---
                import pdb; pdb.set_trace()
                for m_l in mask_pairs:
                    idx = [i for i, lab in enumerate(labels) if m_l in lab]
                    idx = np.array(idx, dtype=int)
                    if idx.size > 0:
                        rdm[np.ix_(idx, idx)] = np.nan
   
        #import pdb; pdb.set_trace()
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]    
        RDM.append(rdm[np.triu_indices(n, k=0)]) 
        if plotting == True:
            plot_rdm_with_labels(rdm, labels, group_size=4)
            #plt.figure()
            #plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm')
            #plt.figure()
            #plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')

    return RDM


def make_categorical_RDM(data_chunk, plotting = False, include_diagonal = True):
    RDM = []
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        labels = np.asarray(data).squeeze()  
        same = (labels[:, None] == labels[None, :])
        # dissimilaririty matrix. 
        # 0 = the same; 1 = different
        rdm_both_halves = np.where(same, 0, 1)
        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        # making the matrix symmetric
        rdm = (rdm_small + rdm_small.T) / 2
        
        # vectorize upper triangle
        n = rdm.shape[0]
        k = 0 if include_diagonal else 1
        vec = rdm[np.triu_indices(n, k=k)]
        
        # balance the regressor around 0
        # this will be -0.5 or 0.5
        vec = vec - vec.mean()
        
        RDM.append(vec)
        
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=2, vmin=0)
            plt.figure()
            plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')


    return RDM
                
                
                
                
def make_distance_RDM(data_chunk, plotting = False, include_diagonal = True):
    #import pdb; pdb.set_trace()
    # this computes the z-standardised distance between any 2 datapoints and fills the matrix with it.
    # in the end, it then selects the relevant triangle.
    
    RDM = []
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        # first z-score
        z_vals = (data - data.mean()) / data.std()
        # then take the absolute distance
        rdm_both_halves = np.abs(z_vals - z_vals.T)
        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        
        # scale so max(absdiff) -> 2, min -> 0
        # After scaling, 0 = most similar, 2 = most dissimilar 
        maxd = rdm.max()
        if maxd == 0:
            rdm = np.zeros_like(rdm)
        else:
            rdm = rdm * (2.0 / maxd)
            
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        if include_diagonal:
            RDM.append(rdm[np.triu_indices(n, k=0)]) 
        else:
            RDM.append(rdm[np.triu_indices(n, k=1)]) 
        
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=2, vmin=0)
            plt.figure()
            plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')


    return RDM
    


def make_distance_RDM_cosine_normratio(data_chunk, plotting=False, include_diagonal=True):
    RDM = []
    # penalises pressing different buttons
    # then scales with frequency of pressing the buttons
    #
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]

    for data in data_chunk:
        X = np.asarray(data, dtype=float)
        dot = X @ X.T
        norms = np.linalg.norm(X, axis=1)
        denom = np.outer(norms, norms)
        # safe cosine: when denom == 0 -> 0
        cosine = np.nan_to_num(dot / denom, nan=0.0, posinf=0.0, neginf=0.0)
        # norm ratio: min/max, safe when max == 0 -> 0
        nmin = np.minimum.outer(norms, norms)
        nmax = np.maximum.outer(norms, norms)
        norm_ratio = np.where(nmax > 0, nmin / nmax, 0.0)
        sim = cosine * norm_ratio
        dist = (1.0 - sim) * 2.0   # 0 = most similar, 2 = most dissimilar

        P = dist.shape[0]
        half = P // 2
        rdm_small = dist[half:, :half]
        rdm = (rdm_small + rdm_small.T) / 2.0

        # scale to ensure max -> 2 (keeps original convention)
        maxd = rdm.max()
        rdm = rdm if maxd == 0 else rdm * (2.0 / maxd)

        n = rdm.shape[1]
        k = 0 if include_diagonal else 1
        RDM.append(rdm[np.triu_indices(n, k=k)])

        if plotting:
            plt.figure(); plt.title('Symmetric RDM'); plt.imshow(rdm, aspect='auto', cmap='coolwarm', vmin=0, vmax=2); plt.colorbar()

    return RDM


def make_category_masks(data_chunk, plotting=False, include_diagonal=True, mask_only_path_rew_combos=True):
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]

    outputs = []
    for data in data_chunk:
        labels = np.asarray(data).squeeze()

        # pairwise label comparisons
        same = labels[:, None] == labels[None, :]
        path = labels == 'path'
        reward = labels == 'reward'

        masks_full = {
            'path-path':   path[:, None] & path[None, :],
            'reward-reward': reward[:, None] & reward[None, :],
            'reward-path':  ~same,
            'mask_reward-path': same
        }

        # cut lower-left quadrant and symmetrize (to match your pipeline)
        P = labels.size
        half = P // 2
        masks_sym = {}
        for key, M in masks_full.items():
            M_small = M[half:, :half]
            masks_sym[key] = M_small | M_small.T

        # vectorize upper triangle
        n = next(iter(masks_sym.values())).shape[0]
        k = 0 if include_diagonal else 1
        tri = np.triu_indices(n, k=k)

        masks_vec = {key: M[tri] for key, M in masks_sym.items()}
        outputs.append(masks_vec)

        if plotting:
            plt.figure(figsize=(9,3))
            for i, (key, M) in enumerate(masks_sym.items(), 1):
                plt.subplot(1,3,i)
                plt.title(key)
                plt.imshow(M.astype(int), aspect='auto', cmap='Greys')
                plt.axis('off')
            plt.show()
            
    # import pdb; pdb.set_trace()
    # make sure to make this reversed: only exclude path-reward, inlude everything else.
    if mask_only_path_rew_combos == True:
        # outputs[0].pop('reward-path')
        # outputs[0].pop('path-path')
        for k in ['path-path', 'reward-reward', 'reward-path']:
            outputs[0].pop(k, None)
    
    return outputs[0] if len(outputs) == 1 else outputs

def compute_hamming_distance_weighted(data_chunk, plotting = False, weight= 'now_to_fut', include_diagonal = True, model_name = None): 
    RDM = []
    #
    print(f"weight is {weight}")
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    for data in data_chunk:
        data = np.asarray(data, dtype=object) # data has dimensions (160, 96) -> i want to compare the 160 conditions with each other
        
        len_th = len(data)/2
        if len_th == 24 or len_th == 48:
            no_tasks = 6
        elif len_th == 40:
            no_tasks = 10
        #import pdb; pdb.set_trace()
        # data task_half 1 and task_half 2 concatenated.
        overlap = np.equal(data[:, None, :], data[None, :,:])
        len_rep = 8
        if model_name.startswith('rew') or model_name.startswith('path'):
            len_rep = 4
        
        # ----- CREATE WEIGHTS -----
        if weight == 'now_to_fut':
            # 8 futures decreasing by decay = .75
            base = 0.5 ** np.arange(len_rep)
            weights = np.repeat(base, 12)   # 8 × 12 = 96
            
        elif weight == 'fut_to_now':
            # future is weighted the strongest, now the least
            base = np.flip(0.5 ** np.arange(len_rep))
            weights = np.repeat(base, 12)   # 8 × 12 = 96
            
        elif weight == 'close_to_far_fut':
            # highest weight near center, lowest at extremes
            dist = np.minimum(np.arange(len_rep), len_rep - np.arange(len_rep))
            base = 0.5 ** dist
            weights = np.repeat(base, 12)
            
        elif weight == 'far_to_close_fut':
            # lowest weight near center, highest at extremes
            dist = np.minimum(np.arange(len_rep), len_rep - np.arange(len_rep))
            base = np.flip(0.5 ** dist)
            weights = np.repeat(base, 12)
            
        else:
            # no weighting (standard Hamming similarity)
            weights = np.ones(data.shape[1])
            
        # normalize so weights sum to 1
        weights = weights / weights.sum()

        # ----- WEIGHTED SIMILARITY -----
        hamming_sim_matrix = (overlap * weights).sum(axis=2)
        
        # convert to distance
        rdm_both_halves = 1 - hamming_sim_matrix
        
        # rdm = 1 - hamming_sim_matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + rdm_small.T)/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        if include_diagonal:
            RDM.append(rdm[np.triu_indices(n, k=0)]) 
        else:
            RDM.append(rdm[np.triu_indices(n, k=1)]) 
         
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=1, vmin=0)
            plt.title(model_name)
            for i in range(0,n,int(n/no_tasks)):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,int(n/no_tasks)):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            # for i in range(0,n,8):
            #     plt.axvline(i-0.5, color='white', ls = 'dashed')
            # for i in range(0,n,8):
            #     plt.axhline(i-0.5, color='white', ls = 'dashed')
            if no_tasks == 10:
                labels = ['A1_backw', 'A1_forw', 'B1_backw', 'B1_forw', 'C1_backw', 'C1_forw', 'D1_back', 'D1_forw', 'E1_backw', 'E1_forw']
            elif no_tasks == 6:
                labels = ['A1_backw', 'A1_forw', 'C1_backw', 'C1_forw', 'E1_backw', 'E1_forw']
            plt.yticks(np.arange(2, rdm.shape[1], int(n/no_tasks)), labels)
            plt.colorbar()
            
    return RDM
    
    
def compute_hamming_distance(data_chunk, plotting = False, include_diagonal = True, model_name = None, no_tasks = 10): 
    RDM = []
    #
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    for data in data_chunk:
        # import pdb; pdb.set_trace()
        data = np.asarray(data, dtype=object)
        # data task_half 1 and task_half 2 concatenated.
        # overlap: are values the same if you stack rows vertically vs horizontally?
        # overlap = data[:,None,:] == data[None, :, :]
        overlap = np.equal(data[:, None, :], data[None, :,:])
        # axis 0 = row A, axis 1 = row B, axis 2 = element-wise overlap
        # mean of axis 2 = fraction of positions where row i and row j are identical
        hamming_sim_matrix = overlap.mean(axis = 2)
        rdm_both_halves = 1 - hamming_sim_matrix
        # rdm = 1 - hamming_sim_matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + rdm_small.T)/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        if include_diagonal:
            RDM.append(rdm[np.triu_indices(n, k=0)]) 
        else:
            RDM.append(rdm[np.triu_indices(n, k=1)]) 
        len_th = len(data)/2
        # import pdb; pdb.set_trace()
        if len_th == 24 or len_th == 48:
            no_tasks = 6
            len_task = 8
        elif len_th == 40:
            no_tasks = 10
            len_task = 8
        elif len_th == 2880:
            no_tasks = 8
            len_task = 360
            
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=1, vmin=0)
            if model_name:
                plt.title(f'across half RDM for {model_name} based on hamming distance, random threshold.')
            else:
                plt.title('across half RDM based on hamming distance, random threshold.')
            len_task = int(n/no_tasks)
            for i in range(0,n,int(n/no_tasks)):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,int(n/no_tasks)):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,len_task):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,len_task):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            if no_tasks == 10:
                labels = ['A1_backw', 'A1_forw', 'B1_backw', 'B1_forw', 'C1_backw', 'C1_forw', 'D1_back', 'D1_forw', 'E1_backw', 'E1_forw']
            elif no_tasks == 6:
                labels = ['A1_backw', 'A1_forw', 'C1_backw', 'C1_forw', 'E1_backw', 'E1_forw']
            elif no_tasks == 8:
                labels = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3', '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
                
            plt.yticks(np.arange(2, rdm.shape[1], int(n/no_tasks)), labels)
            # plt.xticks(np.arange(2, rdm.shape[1], int(n/no_tasks)), labels)
            plt.colorbar()
            
    return RDM
        
def compute_hamming_distance_within(data_chunk, plotting=False, include_diagonal=True, model_name=None, no_tasks=10, block_size = None):
    """Within-run Hamming distance. Input is (N, features), not (2N, features)."""
    RDM_within, RDM_between = [], []
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    for data in data_chunk:
        data = np.asarray(data, dtype=object)
        overlap = np.equal(data[:, None, :], data[None, :, :])
        hamming_sim_matrix = overlap.mean(axis=2)
        rdm = 1 - hamming_sim_matrix

        n = rdm.shape[0]
        k = 0 if include_diagonal else 1
        i, j = np.triu_indices(n, k=k)
        
        rdm_flat = rdm[i, j]
        within_block_idx = (i // block_size) == (j // block_size)
        
        RDM_within.append(rdm_flat[within_block_idx])
        RDM_between.append(rdm_flat[~within_block_idx])
    
        if plotting:
            plt.figure()
            plt.imshow(rdm, aspect='auto', cmap='coolwarm', vmax=1, vmin=0)
            if model_name:
                plt.title(f'within-run RDM for {model_name} (Hamming)')
            else:
                plt.title('within-run RDM (Hamming)')
            len_task = int(n / no_tasks)
            for i in range(0, n, len_task):
                plt.axvline(i - 0.5, color='white', ls='dashed')
                plt.axhline(i - 0.5, color='white', ls='dashed')
            if no_tasks == 8:
                labels = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3', '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
                plt.yticks(np.arange(2, n, int(n / no_tasks)), labels)
            plt.colorbar()

    return RDM_within, RDM_between, rdm


def compute_hamming_difference(data_chunk, combination, plotting = False, include_diagonal = True, model_name = None):
    RDM = []
    #
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    for data in data_chunk:
        
        data = np.asarray(data, dtype=object)
        # data task_half 1 and task_half 2 concatenated.
        # overlap: are values the same if you stack rows vertically vs horizontally?
        # overlap = data[:,None,:] == data[None, :, :]
        # overlap = np.equal(data[:, None, :], data[None, :,:])
        # split into state and action
        states  = np.char.partition(data.astype(str), '-')[..., 0]
        actions = np.char.partition(data.astype(str), '-')[..., 2]
        
        # state similarity
        state_sim = states[:, None, :] == states[None, :, :]
        # state dissimilarity
        state_dissim = states[:, None, :] != states[None, :, :]
        
        # action similarity
        action_sim = actions[:, None, :] == actions[None, :, :]
        # action dissimilarity
        action_dissim = actions[:, None, :] != actions[None, :, :]
        
        # combine (both must be true)
        # import pdb; pdb.set_trace()
        if combination.startswith('sa_ss'):
            overlap = action_sim & state_sim 
        elif combination.startswith('sa_ds'):
            overlap = action_sim & state_dissim
        elif combination.startswith('da_ss'):
            overlap = action_dissim & state_sim
        elif combination.startswith('da_ds'):
            overlap = action_dissim & state_dissim

        
        # axis 0 = row A, axis 1 = row B, axis 2 = element-wise overlap
        # mean of axis 2 = fraction of positions where row i and row j are identical
        hamming_sim_matrix = overlap.mean(axis = 2)
        rdm_both_halves = 1 - hamming_sim_matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + rdm_small.T)/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        if include_diagonal:
            RDM.append(rdm[np.triu_indices(n, k=0)]) 
        else:
            RDM.append(rdm[np.triu_indices(n, k=1)]) 
            
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=1, vmin=0)
            plt.title(model_name)
            for i in range(0,n,int(n/10)):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,int(n/10)):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            labels = ['A1_backw', 'A1_forw', 'B1_backw', 'B1_forw', 'C1_backw', 'C1_forw', 'D1_back', 'D1_forw', 'E1_backw', 'E1_forw']
            plt.yticks(np.arange(2, rdm.shape[1], int(n/10)), labels)
            plt.colorbar()
            
            #plt.figure()
            #plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')
            
    return RDM



def compute_crosscorr(data_chunk, plotting = False, include_diagonal = True, no_tasks = 10, model = None):  
    RDM = []
    # import pdb; pdb.set_trace()
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        # centers the data around zero by subtracting the mean of each row
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        # normalising data
        data_demeaned /= np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))[:, None]    
        # cosine dissimilarity
        rdm_both_halves = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)  
        
        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        if include_diagonal:
            RDM.append(rdm[np.triu_indices(n, k=0)]) 
        else:
            RDM.append(rdm[np.triu_indices(n, k=1)]) 
        
        if plotting == True:
        
            # # import pdb; pdb.set_trace()
            # plt.figure()
            # plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=2, vmin=0)
            # plt.title('RDM, threshold at 2 and 0.')
            # then z-score
            rdm_z = (rdm - np.nanmean(rdm))/ np.nanstd(rdm)
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm')
            if model:
                plt.title(f'across th z-scored RDM for {model}, random threshold.')
            else:
                plt.title('across half z-scored RDM, random threshold.')
            plt.colorbar()
            
        
            # plt.figure()
            # plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm')
            # if model:
            #     plt.title(f'across th RDM for {model}, random threshold.')
            # else:
            #     plt.title('across half RDM, random threshold.')
            # plt.colorbar()

            len_task = int(n/no_tasks)
            for i in range(0,n,int(n/no_tasks)):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,int(n/no_tasks)):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,len_task):
                plt.axvline(i-0.5, color='white', ls = 'dashed')
            for i in range(0,n,len_task):
                plt.axhline(i-0.5, color='white', ls = 'dashed')
            if no_tasks == 10:
                labels = ['A1_backw', 'A1_forw', 'B1_backw', 'B1_forw', 'C1_backw', 'C1_forw', 'D1_back', 'D1_forw', 'E1_backw', 'E1_forw']
            elif no_tasks == 6:
                labels = ['A1_backw', 'A1_forw', 'C1_backw', 'C1_forw', 'E1_backw', 'E1_forw']
            elif no_tasks == 8:
                labels = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3', '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
            plt.yticks(np.arange(2, rdm.shape[1], int(n/no_tasks)), labels)
            
    return RDM


def compute_crosscorr_within(data_chunk, plotting=False, include_diagonal=True, no_tasks=10, model=None, block_size=None):
    """Within-run cosine dissimilarity. Input is (N, features), not (2N, features)."""
    RDM_within, RDM_between = [], []
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]

    for data in data_chunk:
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))
        norms[norms == 0] = 1
        data_demeaned /= norms[:, None]
        rdm = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)

        n = rdm.shape[0]

        k = 0 if include_diagonal else 1
        i, j = np.triu_indices(n, k=k)
        
        rdm_flat = rdm[i, j]
        within_block_idx = (i // block_size) == (j // block_size)
        
        RDM_within.append(rdm_flat[within_block_idx])
        RDM_between.append(rdm_flat[~within_block_idx])

        if plotting:
            plt.figure()
            plt.imshow(rdm, aspect='auto', cmap='coolwarm')
            if model:
                plt.title(f'RDM within and across tasks for {model}')
            else:
                plt.title('RDM within and across tasks RDM')
            plt.colorbar()
            len_task = int(n / no_tasks)
            for i in range(0, n, len_task):
                plt.axvline(i - 0.5, color='white', ls='dashed')
                plt.axhline(i - 0.5, color='white', ls='dashed')
            if no_tasks == 8:
                labels = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
                           '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
                plt.yticks(np.arange(2, n, int(n / no_tasks)), labels)

    return RDM_within, RDM_between, rdm


def mask_RDM(lower_tri, n, labels, mask=None, binarise = False, plotting = False):
    # import pdb; pdb.set_trace()
    # this puts it to the upper triangle 
    masked_RDM = np.full((n, n), np.nan, dtype = float) 
    iu = np.triu_indices(n, 0) 
    masked_RDM[iu] = lower_tri 

    # collect the indicesI want to mask
    idx_to_mask = []
    for i, label in enumerate(labels):
        for m in mask:
            if m in label:
                idx_to_mask.append(i)
    
    masked_RDM[:, idx_to_mask] = np.nan
    masked_RDM[idx_to_mask] = np.nan
    masked_vector = masked_RDM[np.triu_indices(n, 0)]
    
    if binarise == True:
        # THIS IS ONLY FOR MODEL RDMS!!
        masked_vector = np.where(np.isnan(masked_vector), np.nan, (masked_vector > 0.5).astype(float))
    if plotting == True:
        plt.figure()
        plt.imshow(masked_RDM, aspect = 'auto', cmap = 'coolwarm')

    return masked_vector


def evaluate_model_vec(X, Y):
    """Vectorised standardised OLS — same numerical convention as
    :func:`evaluate_model`, but accepts either a single target or a
    stack of targets in one call.

    Use this for both empirical RSA fits and permutation-null OLS so
    the perm β distribution is on the same scale as the empirical β
    (CLAUDE.md rule #4: same function, empirical and perm).

    Parameters
    ----------
    X : ndarray
        Regressor RDM(s). Shape ``(n_pairs,)`` for a single regressor,
        or ``(n_pairs, n_features)`` for a combo design.
    Y : ndarray
        Target data RDM(s). Shape ``(n_pairs,)`` for one target, or
        ``(n_targets, n_pairs)`` for a batch (e.g. permutation RDMs).

    Returns
    -------
    t, beta, p : ndarray
        Three arrays of matching shape. If ``Y`` was 1-D the returned
        arrays are 1-D of length ``n_features``. If ``Y`` was 2-D the
        returned arrays are 2-D of shape ``(n_targets, n_features)``.

    Numerical convention (matches `evaluate_model`):
      * Add an intercept column to X (`sm.add_constant`-style).
      * Drop rows where X or Y has any NaN. Same row mask applied to
        Y; for batched Y the mask is computed jointly across all
        targets so the design is identical for every perm.
      * Z-score each non-intercept X column (mean 0, std 1, ddof=0).
      * Z-score Y row-by-row (each target separately; mean 0, std 1).
      * Fit OLS via `np.linalg.solve` on the normal equations.
      * t, p use df = n_finite_rows − (n_features + 1) (intercept counted).
      * Returns only the non-intercept columns of t, beta, p.
    """
    # Normalise input shapes
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        # Caller passed a (n_pairs,) regressor as a row vector.
        X = X.reshape(-1, 1)
    elif X.ndim == 2 and X.shape[1] == 1 and X.shape[0] == 1:
        # Single scalar — degenerate.
        X = X.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    Y_raw = np.asarray(Y, dtype=float)
    y_was_1d = Y_raw.ndim == 1
    if y_was_1d:
        Y = Y_raw[None, :]
    else:
        Y = Y_raw

    n_pairs, n_feat = X.shape
    n_targets, n_pairs_y = Y.shape
    if n_pairs_y != n_pairs:
        raise ValueError(f'X has {n_pairs} pairs but Y has {n_pairs_y}')

    # Append intercept column (matches sm.add_constant).
    X_aug = np.column_stack([np.ones(n_pairs, dtype=float), X])
    n_aug = n_feat + 1

    # Joint NaN-row filter — applied identically to every target row so
    # the design X is the same across perms.
    fin = np.isfinite(X_aug).all(axis=1) & np.isfinite(Y).all(axis=0)

    NAN_OUT = np.full((n_targets, n_feat), np.nan)
    if fin.sum() < n_aug + 1:
        if y_was_1d:
            return NAN_OUT[0].copy(), NAN_OUT[0].copy(), NAN_OUT[0].copy()
        return NAN_OUT.copy(), NAN_OUT.copy(), NAN_OUT.copy()

    Xk = X_aug[fin]
    Yk = Y[:, fin]

    # Z-score the non-intercept regressor columns (col 0 = intercept stays 1).
    for i in range(1, n_aug):
        mu = Xk[:, i].mean()
        sd = Xk[:, i].std()
        if sd > 0:
            Xk[:, i] = (Xk[:, i] - mu) / sd
        else:
            Xk[:, i] = 0.0
    # Z-score each Y row (each target / each perm).
    mu_y = Yk.mean(axis=1, keepdims=True)
    sd_y = Yk.std(axis=1, keepdims=True)
    sd_y = np.where(sd_y > 0, sd_y, 1.0)
    Yz = (Yk - mu_y) / sd_y

    # OLS via normal equations. Detect rank deficiency.
    XtX = Xk.T @ Xk
    if np.linalg.matrix_rank(XtX) < n_aug:
        if y_was_1d:
            return NAN_OUT[0].copy(), NAN_OUT[0].copy(), NAN_OUT[0].copy()
        return NAN_OUT.copy(), NAN_OUT.copy(), NAN_OUT.copy()
    XtX_inv = np.linalg.inv(XtX)
    XtY = Xk.T @ Yz.T               # (n_aug, n_targets)
    BETA = XtX_inv @ XtY            # (n_aug, n_targets)

    preds = Xk @ BETA               # (n_fin, n_targets)
    resid = Yz.T - preds            # (n_fin, n_targets)
    n_fin = Xk.shape[0]
    df = max(n_fin - n_aug, 1)
    sigma2 = (resid ** 2).sum(axis=0) / df              # (n_targets,)
    var_diag = np.diag(XtX_inv)                          # (n_aug,)
    se = np.sqrt(np.outer(sigma2, var_diag))             # (n_targets, n_aug)
    BETA_T = BETA.T                                      # (n_targets, n_aug)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.where(se > 0, BETA_T / se, np.nan)
    # two-sided parametric p from the t-distribution
    from scipy import stats as _scipy_stats
    P = 2 * (1 - _scipy_stats.t.cdf(np.abs(T), df=df))

    # Drop the intercept column from the outputs.
    t_out    = T[:, 1:]
    beta_out = BETA_T[:, 1:]
    p_out    = P[:, 1:]

    if y_was_1d:
        return t_out[0], beta_out[0], p_out[0]
    return t_out, beta_out, p_out


def evaluate_model(model_rdm, data_rdm):
    """Thin backward-compatible wrapper around :func:`evaluate_model_vec`.

    Accepts the same arguments as the previous statsmodels-based
    implementation and returns ``(t, beta, p)`` for the non-intercept
    regressors. Now uses the same vectorised standardised OLS as the
    permutation-null code path — see CLAUDE.md rule #4.
    """
    return evaluate_model_vec(np.asarray(model_rdm), np.asarray(data_rdm))




def plot_model_correlations(stacked_model_RDMs, model_names,
                            figsize=(8, 6), cmap='coolwarm', annot=True,
                            fmt='.2f', vmin=-1, vmax=1, cmap_center=0,
                            show=True, save_path=None, conditions_masking = None):
    """
    Plot Pearson correlations between model RDMs.

    Parameters
    ----------
    stacked_model_RDMs : array-like, shape (n_entries, n_models)
        Each column should be a vectorized model RDM (e.g. upper-triangle).
    model_names : list of str, length n_models
        Labels for the models (used on x/y ticks).
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name (diverging recommended, e.g. 'bwr' or 'coolwarm').
    annot : bool
        Whether to annotate cells with correlation numbers.
    fmt : str
        Format string for annotations.
    vmin, vmax : float
        Value range for colormap (defaults to -1..1).
    show : bool
        Whether to call plt.show().
    save_path : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    corr : ndarray, shape (n_models, n_models)
        Pearson correlation matrix between model columns.
    fig, ax : matplotlib objects
        Figure and axes (for further customization).
    """
    if conditions_masking:
        for cond in conditions_masking:
            X = np.asarray(stacked_model_RDMs)
            X = X[conditions_masking[cond]]
    
            if X.ndim != 2:
                raise ValueError("stacked_model_RDMs must be 2D (n_entries, n_models).")
            if X.shape[1] != len(model_names):
                raise ValueError("Length of model_names must match number of model columns.")
        
            # correlation matrix (columns are variables)
            corr = np.corrcoef(X, rowvar=False)
        
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(corr, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        
            # ticks / labels
            ax.set_xticks(np.arange(len(model_names)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_yticklabels(model_names)
        
            # annotations
            if annot:
                # choose contrasting text color depending on background brightness
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        val = corr[i, j]
                        txt = format(val, fmt)
                        # white text for strong colors, black otherwise
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        ax.text(j, i, txt, ha='center', va='center', color=text_color, fontsize=9)
        
            # colorbar and layout
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Pearson r', rotation=270, labelpad=12)
        
            ax.set_title(f"Model RDM correlations (Pearson r), only {cond}")
            plt.tight_layout()
        
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
            if show:
                plt.show()
    else:
        X = np.asarray(stacked_model_RDMs)
        if X.ndim != 2:
            raise ValueError("stacked_model_RDMs must be 2D (n_entries, n_models).")
        if X.shape[1] != len(model_names):
            raise ValueError("Length of model_names must match number of model columns.")
    
        # correlation matrix (columns are variables)
        corr = np.corrcoef(X, rowvar=False)
    
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
        # ticks / labels
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticklabels(model_names)
    
        # annotations
        if annot:
            # choose contrasting text color depending on background brightness
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    val = corr[i, j]
                    txt = format(val, fmt)
                    # white text for strong colors, black otherwise
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, txt, ha='center', va='center', color=text_color, fontsize=4)
    
        # colorbar and layout
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson r', rotation=270, labelpad=12)
    
        ax.set_title('Model RDM correlations (Pearson r)')
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        if show:
            plt.show() 

    return corr, fig, ax
