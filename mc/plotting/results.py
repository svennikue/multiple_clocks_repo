#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:13:01 2024

this script offers several specific functions to plot my results.

@author: Svenja Küchenhoff
"""

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import os
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
import scipy.stats as st 
import math
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import mc


# --- Helpers for stats ---
def one_tailed_ttest_greater_than_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan  # t, p, mean
    t_stat, p_two = st.ttest_1samp(x, 0.0, nan_policy='omit')
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
    return float(t_stat), float(p_one), float(np.mean(x))

def stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def plot_model_rdm_half(
    ev_array,
    labels=None,
    method="crosscorr",
    label_half="first",
    group_size="auto",
    title=None,
    cmap="RdBu_r",
    vmin=0.5,
    vmax=1.5,
    vcenter=1.0,
    show=True,
):
    """
    Plot a model RDM with only the upper triangle visible (lower half masked white).

    Parameters
    ----------
    ev_array : array-like, shape (n_conditions, n_features)
        Condition-by-feature matrix used to compute the RDM.
    labels : list[str] or None
        Labels for conditions (same length as n_conditions).
    method : {"crosscorr", "corrcoef"}
        "crosscorr" uses mc.analyse.my_RSA.compute_crosscorr (half-size cross-half RDM).
        "corrcoef" uses 1 - corrcoef across conditions.
    label_half : {"first", "second", "full", None}
        Which half of labels to show (used for concatenated conditions).
    group_size : int
        Interval for dashed grid lines. Set to 0/None to disable.
    title : str or None
        Title for the plot.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax, vcenter : float
        Color scale parameters; vcenter=1 matches dissimilarity (1 - r).
    show : bool
        Whether to call plt.show().
    """
    
    if labels is None:
        labels = [str(i) for i in range(ev_array.shape[0])]
    labels = list(labels)

    if method == "crosscorr":
        ev_array = np.asarray(ev_array, dtype=float)
        if ev_array.shape[0] % 2 != 0:
            raise ValueError("crosscorr expects an even number of conditions.")
        vec = mc.analyse.my_RSA.compute_crosscorr(ev_array, include_diagonal=True)[0]
        n = int((-1 + np.sqrt(1 + 8 * len(vec))) / 2)
        rdm = np.zeros((n, n), dtype=float)
        iu = np.triu_indices(n, k=0)
        rdm[iu] = vec
        rdm[(iu[1], iu[0])] = vec
        half_n = ev_array.shape[0] // 2
        if label_half == "second":
            labels = labels[half_n:half_n + n]
        else:
            labels = labels[:n]
    elif method == "corrcoef":
        ev_array = np.asarray(ev_array, dtype=float)
        corr = np.corrcoef(ev_array)
        rdm = 1.0 - corr
        n_total = rdm.shape[0]
        half_n = n_total // 2
        if label_half == "first":
            rdm = rdm[:half_n, :half_n]
            labels = labels[:half_n]
        elif label_half == "second":
            rdm = rdm[half_n:half_n * 2, half_n:half_n * 2]
            labels = labels[half_n:half_n * 2]
        elif label_half in ("full", None):
            pass
        else:
            raise ValueError(f"Unknown label_half: {label_half}")
    elif method == 'hamming_distance':
        data = np.asarray(ev_array, dtype=object)
        overlap = np.equal(data[:, None, :], data[None, :,:])
        hamming_sim_matrix = overlap.mean(axis = 2)
        rdm = 1.0 - hamming_sim_matrix
        n_total = rdm.shape[0]
        half_n = n_total // 2
        if label_half == "first":
            rdm = rdm[:half_n, :half_n]
            labels = labels[:half_n]
        elif label_half == "second":
            rdm = rdm[half_n:half_n * 2, half_n:half_n * 2]
            labels = labels[half_n:half_n * 2]
        elif label_half in ("full", None):
            pass
        else:
            raise ValueError(f"Unknown label_half: {label_half}")
        vmin = 0
        vmax = 1
        vcenter = 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    n = rdm.shape[0]
    rdm_masked = rdm.copy()
    rdm_masked[np.tril_indices(n, k=-1)] = np.nan

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(
        rdm_masked,
        cmap=cmap_obj,
        norm=norm,
        interpolation="none",
        aspect="equal",
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=12)

    if group_size:
        if group_size == "auto":
            boundaries = []
            prev_task = None
            for i, lab in enumerate(labels):
                task = lab.split("_", 1)[0] if isinstance(lab, str) else str(lab)
                if prev_task is None:
                    prev_task = task
                    continue
                if task != prev_task:
                    boundaries.append(i)
                    prev_task = task
            for k in boundaries:
                ax.axhline(k - 0.5, color="black", ls="dashed", linewidth=0.8)
                ax.axvline(k - 0.5, color="black", ls="dashed", linewidth=0.8)
        else:
            for k in range(group_size, n, group_size):
                ax.axhline(k - 0.5, color="black", ls="dashed", linewidth=0.8)
                ax.axvline(k - 0.5, color="black", ls="dashed", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dissimilarity (1 - r)", rotation=270, labelpad=12)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax, rdm


# ---------------------------------------------------------------------------
# Publication-ready model-RDM helpers
# Used by scripts/create_fMRI_model_RDMs_on_clean_beh.py. Both helpers consume
# matrices that have ALREADY been computed by the main script — they never
# recompute an RDM and never re-derive an EV. They only handle layout, label
# grouping and saving at the requested A4 size.
# ---------------------------------------------------------------------------
def _ev_task_code(ev_label):
    """``'A1_forw_A_reward'`` → ``'A1_forw'`` (task code + direction)."""
    parts = ev_label.split('_')
    return '_'.join(parts[:2])


def _group_labels_by_task(ev_labels, task_label_lookup):
    """Walk ``ev_labels`` and return per-task-block centres + boundaries +
    display labels (resolved via ``task_label_lookup``)."""
    task_per = [_ev_task_code(ev) for ev in ev_labels]
    unique_tasks = []
    for t in task_per:
        if not unique_tasks or unique_tasks[-1] != t:
            unique_tasks.append(t)
    starts = {}
    for i, t in enumerate(task_per):
        starts.setdefault(t, i)
    starts_list = [starts[t] for t in unique_tasks]
    sizes = [starts_list[i+1] - starts_list[i]
             for i in range(len(starts_list)-1)]
    sizes.append(len(task_per) - starts_list[-1])
    centres = [starts_list[i] + sizes[i] / 2 - 0.5
               for i in range(len(unique_tasks))]
    boundaries = [starts_list[i] - 0.5 for i in range(1, len(starts_list))]
    display = [task_label_lookup.get(t, t) for t in unique_tasks]
    return centres, boundaries, display


def plot_model_rdm_pub(rdm, ev_labels, task_label_lookup, *,
                       save_stem=None, title=None,
                       vmin=0.5, vmax=1.5, vcenter=1.0,
                       cmap='RdBu_r',
                       fig_width_cm=4.0, fig_height_cm=4.0, font_pt=8,
                       mask_lower=True, show=True):
    """Publication-ready model RDM. Takes a **precomputed** RDM matrix (the
    main script must compute it once and pass it in).

    Parameters
    ----------
    rdm              : (n, n) RDM matrix matching ``ev_labels``.
    ev_labels        : EV strings (length n) like ``'A1_forw_A_reward'``.
    task_label_lookup: dict ``{'A1_forw': '1-7-5-3', ...}`` — the goal
                       configuration as actually executed (direction-aware).
    vmin/vmax/vcenter: colorbar range. Defaults match crosscorr dissimilarity;
                       use ``vmin=0, vmax=1, vcenter=0.5`` for hamming.
    fig_width_cm,
    fig_height_cm    : printed size of the saved figure. Fonts render at
                       ``font_pt`` exactly when dropped into an A4 page at 100 %.

    Saves to ``<save_stem>.pdf`` and ``<save_stem>.png`` when ``save_stem``
    is given (no recomputation, no return-value side effects).
    """
    rdm = np.asarray(rdm, dtype=float)
    if rdm.shape[0] != rdm.shape[1] or rdm.shape[0] != len(ev_labels):
        raise ValueError(f"RDM shape {rdm.shape} doesn't match {len(ev_labels)} labels")

    centres, boundaries, display = _group_labels_by_task(ev_labels, task_label_lookup)

    rdm_disp = rdm.copy()
    if mask_lower:
        rdm_disp[np.tril_indices(rdm.shape[0], k=-1)] = np.nan

    cm_per_in = 2.54
    figsize = (fig_width_cm / cm_per_in, fig_height_cm / cm_per_in)
    rc = {
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_pt,
        'axes.labelsize':  font_pt,
        'axes.titlesize':  font_pt + 1,
        'xtick.labelsize': font_pt,
        'ytick.labelsize': font_pt,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='white')
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        ax.imshow(rdm_disp, cmap=cmap_obj, norm=norm,
                  interpolation='none', aspect='equal')
        ax.set_xticks(centres)
        ax.set_yticks(centres)
        ax.set_xticklabels(display, rotation=90)
        ax.set_yticklabels(display)
        for k in boundaries:
            ax.axhline(k, color='black', ls='-', linewidth=0.4)
            ax.axvline(k, color='black', ls='-', linewidth=0.4)
        if title:
            ax.set_title(title, loc='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if save_stem:
            fig.savefig(save_stem + '.pdf', bbox_inches='tight')
            fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    return fig


def plot_model_activation_examples(EVs, model_names, example_task,
                                   task_label_lookup, *,
                                   temp_order=None, save_stem=None,
                                   panel_width_cm=2.0, panel_height_cm=3.0,
                                   font_pt=9, show=True):
    """One-task "schematic" of every model's activation pattern. Pulls
    directly from the already-built ``EVs`` dict — no recomputation.

    For each model in ``model_names``, stacks the 8 ``temp_order`` bins of
    ``example_task`` into an (n_features, 8) panel. Object-dtype EVs (e.g.
    button strings) are integer-encoded per panel just for display; the
    underlying values in ``EVs`` are NOT modified.
    """
    if temp_order is None:
        temp_order = ['A_path', 'A_reward', 'B_path', 'B_reward',
                      'C_path', 'C_reward', 'D_path', 'D_reward']

    panels = []
    for m in model_names:
        if m not in EVs:
            continue
        cols = []
        for tb in temp_order:
            v = EVs[m].get(f'{example_task}_{tb}')
            if v is None:
                cols.append(None); continue
            arr_v = np.asarray(v)
            if arr_v.ndim == 0:
                arr_v = arr_v.reshape(1)
            cols.append(arr_v)
        valid = [c for c in cols if c is not None]
        if not valid:
            continue
        n_feat = max(c.size for c in valid)
        is_string = any(c.dtype == object or c.dtype.kind in ('U', 'S')
                        for c in valid)
        if is_string:
            all_strs = set()
            for c in valid:
                all_strs.update(str(x) for x in c.tolist())
            sorted_strs = sorted(s for s in all_strs if s not in ('nan', 'None'))
            s2i = {s: i for i, s in enumerate(sorted_strs)}
            arr = np.full((n_feat, len(cols)), np.nan)
            for j, c in enumerate(cols):
                if c is None: continue
                for i, x in enumerate(c[:n_feat]):
                    arr[i, j] = s2i.get(str(x), np.nan)
            cmap = 'tab10'
        else:
            arr = np.full((n_feat, len(cols)), np.nan)
            for j, c in enumerate(cols):
                if c is None: continue
                arr[:c.size, j] = c.astype(float)
            cmap = 'Greys'
        panels.append((m, arr, cmap))

    if not panels:
        return None

    cm_per_in = 2.54
    fig_width_cm  = panel_width_cm * len(panels) + 0.5
    fig_height_cm = panel_height_cm
    figsize = (fig_width_cm / cm_per_in, fig_height_cm / cm_per_in)

    short_xticks = [t.replace('_path', '-p').replace('_reward', '-r')
                    for t in temp_order]
    rc = {
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_pt,
        'axes.titlesize':  font_pt + 1,
        'xtick.labelsize': max(font_pt - 2, 6),
        'ytick.labelsize': max(font_pt - 2, 6),
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize,
                                 constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, (m, arr, cmap) in zip(axes, panels):
            ax.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest')
            ax.set_title(m, loc='left')
            ax.set_xticks(range(len(temp_order)))
            ax.set_xticklabels(short_xticks, rotation=90)
            ax.set_yticks([])
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        ex_label = task_label_lookup.get(example_task, example_task)
        fig.suptitle(f'Example task: {ex_label}  ({example_task})',
                     fontsize=font_pt + 1)
        if save_stem:
            fig.savefig(save_stem + '.pdf', bbox_inches='tight')
            fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    return fig


def plot_dsr_task_matrices(
    EVs,
    tasks,
    temp_order=None,
    location_colors=None,
    rotation_bins=None,
    include_one_hot=True,
    include_rotations=True,
    include_dsr_ev_panels=True,
    show=True,
    save_dir=None,
):
    """
    Plot one-hot location matrices and DSR-rotated matrices for one or more tasks.

    Parameters
    ----------
    EVs : dict
        EV dictionary containing "location" and "DSR" entries.
    tasks : list[str] or str
        Task names to plot (one or two recommended).
    temp_order : list[str] or None
        Order of bins within a task. Defaults to A_path..D_reward.
    location_colors : dict[int, str] or None
        Mapping of location index (1-9) to hex color.
    rotation_bins : list[str] or None
        Subset of bins to plot rotations for. Defaults to temp_order.
    include_one_hot : bool
        Whether to plot the base one-hot location matrix.
    include_rotations : bool
        Whether to plot the DSR-rotated matrices.
    show : bool
        Whether to call plt.show().
    save_dir : str or None
        Optional directory to save figures as PNG.
    """
    if temp_order is None:
        temp_order = [
            "A_path", "A_reward",
            "B_path", "B_reward",
            "C_path", "C_reward",
            "D_path", "D_reward",
        ]
    if location_colors is None:
        location_colors = {
            1: "#008080",
            2: "#99E6E6",
            3: "#C9F2F2",
            4: "#008F64",
            5: "#66B88F",
            6: "#D9FFD9",
            7: "#00331F",
            8: "#146633",
            9: "#7CD973",
        }
    if rotation_bins is None:
        rotation_bins = list(temp_order)

    if isinstance(tasks, str):
        tasks = [tasks]
    tasks = list(tasks)
    if not tasks:
        raise ValueError("tasks must include at least one task name.")

    def _to_label_matrix(one_hot):
        one_hot = np.asarray(one_hot, dtype=float)
        if one_hot.ndim != 2:
            raise ValueError("Expected a 2D one-hot matrix.")
        n_rows, n_cols = one_hot.shape
        labels = np.zeros((n_rows, n_cols), dtype=int)
        for i in range(n_rows):
            row = one_hot[i]
            if not np.isfinite(row).any() or np.nanmax(row) <= 0:
                continue
            j = int(np.nanargmax(row))
            labels[i, j] = j + 1
        return labels

    def _vector_to_label_row(vec, n_locations):
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.size % n_locations != 0:
            raise ValueError("Vector length must be divisible by n_locations.")
        n_bins_local = vec.size // n_locations
        labels_row = np.zeros(vec.size, dtype=int)
        for b in range(n_bins_local):
            start = b * n_locations
            end = start + n_locations
            block = vec[start:end]
            if not np.isfinite(block).any() or np.nanmax(block) <= 0:
                continue
            j = int(np.nanargmax(block))
            labels_row[start + j] = j + 1
        return labels_row

    cmap_list = ["#FFFFFF"] + [location_colors[i] for i in range(1, 10)]
    cmap = mcolors.ListedColormap(cmap_list)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    figs = {}
    for task in tasks:
        bins_curr_task = [f"{task}_{temp_bin}" for temp_bin in temp_order]
        missing = [b for b in bins_curr_task if b not in EVs.get("location", {})]
        if missing:
            raise KeyError(f"Missing EVs['location'] entries for {task}: {missing}")

        one_hot_mat = np.vstack([EVs["location"][b] for b in bins_curr_task])
        n_bins, n_locations = one_hot_mat.shape
        if n_locations != 9:
            raise ValueError(f"Expected 9 locations, got {n_locations}.")

        panels = []
        if include_one_hot:
            panels.append(("one_hot", None))
        if include_rotations:
            for rot_bin in rotation_bins:
                panels.append(("rotation", rot_bin))
        if not panels:
            raise ValueError("No panels selected to plot.")

        n_panels = len(panels)
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (kind, rot_bin) in zip(axes, panels):
            if kind == "one_hot":
                labels_mat = _to_label_matrix(one_hot_mat)
                title = f"{task} one-hot"
            else:
                key = f"{task}_{rot_bin}"
                if key not in EVs.get("DSR", {}):
                    raise KeyError(f"Missing EVs['DSR'] entry: {key}")
                rotated = np.asarray(EVs["DSR"][key], dtype=float).reshape(n_bins, n_locations)
                labels_mat = _to_label_matrix(rotated)
                title = f"{task} DSR @ {rot_bin}"

            im = ax.imshow(labels_mat, cmap=cmap, norm=norm, interpolation="none", aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.set_xticks(np.arange(n_locations))
            ax.set_xticklabels([str(i) for i in range(1, n_locations + 1)], fontsize=7)
            ax.set_yticks(np.arange(n_bins))
            ax.set_yticklabels(list(temp_order), fontsize=7)
            ax.tick_params(length=0)

        for ax in axes[len(panels):]:
            ax.axis("off")

        legend_handles = [
            Patch(facecolor=location_colors[i], edgecolor="none", label=str(i))
            for i in range(1, 10)
        ]
        fig.legend(
            handles=legend_handles,
            title="Location",
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

        fig.tight_layout(rect=[0, 0, 0.95, 1])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{task}_DSR_matrices.png"), dpi=200)
        if show:
            plt.show()

        figs[task] = fig

        if include_dsr_ev_panels:
            dsr_missing = [f"{task}_{b}" for b in temp_order if f"{task}_{b}" not in EVs.get("DSR", {})]
            if dsr_missing:
                raise KeyError(f"Missing EVs['DSR'] entries for {task}: {dsr_missing}")

            dsr_vectors = [np.asarray(EVs["DSR"][f"{task}_{b}"], dtype=float).ravel() for b in temp_order]
            dsr_rows = [_vector_to_label_row(vec, n_locations) for vec in dsr_vectors]
            dsr_matrix = np.vstack(dsr_rows)

            fig_panels, axes = plt.subplots(len(temp_order), 1, figsize=(12, 0.7 * len(temp_order)))
            axes = np.atleast_1d(axes).ravel()
            for ax, row, label in zip(axes, dsr_rows, temp_order):
                ax.imshow(row.reshape(1, -1), cmap=cmap, norm=norm, interpolation="none", aspect="auto")
                ax.set_yticks([0])
                ax.set_yticklabels([label], fontsize=8)
                ax.set_xticks([])
                ax.tick_params(length=0)
            fig_panels.suptitle(f"{task} DSR EVs (1x72 each)", fontsize=10)
            fig_panels.tight_layout(rect=[0, 0, 1, 0.96])
            if save_dir:
                fig_panels.savefig(os.path.join(save_dir, f"{task}_DSR_ev_panels.png"), dpi=200)
            if show:
                plt.show()

            fig_concat, axc = plt.subplots(figsize=(12, 3.2))
            axc.imshow(dsr_matrix, cmap=cmap, norm=norm, interpolation="none", aspect="auto")
            axc.set_yticks(np.arange(len(temp_order)))
            axc.set_yticklabels(list(temp_order), fontsize=8)
            axc.set_xticks([])
            axc.tick_params(length=0)
            axc.set_title(f"{task} DSR EVs concatenated (8x72)")
            fig_concat.tight_layout()
            if save_dir:
                fig_concat.savefig(os.path.join(save_dir, f"{task}_DSR_ev_concat.png"), dpi=200)
            if show:
                plt.show()

    return figs
    


def plot_results_per_roi_and_prefstate(
    df,
    title_string_add,
    plot_by_pfc=False,
    plot_by_cingulate_and_MTL=False,
    metric_col='state_cv_consistency',
    p_col='p_perm',
    alpha_sig=0.05,
    bins=20):
    """
    Plot histograms of `metric_col` split by ROI (columns) and pref_state (rows).
    Rows: pref_state A, B, C, D (only those present in df are plotted, in A-D order).
    Columns: ROIs after renaming/collapsing per provided flags.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --- prepare dataframe & ROI labels ---
    df = df.copy()
    df['roi'] = mc.analyse.helpers_human_cells.rename_rois(
        df,
        collapse_pfc=plot_by_pfc,
        plot_by_cingulate_and_MTL=plot_by_cingulate_and_MTL
    )

    # Order ROI columns as they appear (or customize as needed)
    rois = [r for r in df['roi'].dropna().unique().tolist() if isinstance(r, (str, int, float))]
    n_cols = max(1, len(rois))

    # Which pref_states to show (A-D order, but include only those present)
    desired_states = ['A', 'B', 'C', 'D']
    states_present = [s for s in desired_states if s in df.get('pref_state', []).unique().tolist()] \
                     if 'pref_state' in df.columns else []
    if not states_present:
        # Fallback: single row if no pref_state column or empty
        states_present = ['All']
        df['pref_state'] = 'All'
    n_rows = len(states_present)

    # --- common bin edges across all data for comparability ---
    all_vals = df[metric_col].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError(f"No finite values found in column '{metric_col}'.")
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    # --- precompute counts to get a global y-limit ---
    ylim_max = 0
    precomp = {}  # (state, roi) -> (vals_all, vals_sig, vals_nonsig)
    has_p = (p_col in df.columns)

    for s in states_present:
        df_s = df.loc[df['pref_state'] == s]
        for roi in rois:
            sub = df_s.loc[df_s['roi'] == roi]
            vals = sub[metric_col].to_numpy(dtype=float)
            mask_valid = np.isfinite(vals)

            if has_p:
                pvals = sub[p_col].to_numpy(dtype=float)
                mask_valid &= np.isfinite(pvals)

                vals = vals[mask_valid]
                pvals = pvals[mask_valid]

                sig_mask = pvals < alpha_sig
                vals_sig = vals[sig_mask]
                vals_nonsig = vals[~sig_mask]

                c_sig, _ = np.histogram(vals_sig, bins=bin_edges)
                c_nonsig, _ = np.histogram(vals_nonsig, bins=bin_edges)
                counts = c_sig + c_nonsig

                precomp[(s, roi)] = (vals, vals_sig, vals_nonsig)
            else:
                vals = vals[mask_valid]
                counts, _ = np.histogram(vals, bins=bin_edges)
                precomp[(s, roi)] = (vals, None, None)

            if counts.size:
                ylim_max = max(ylim_max, int(counts.max()))

    # --- figure/axes ---
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6.5, 2.2 * n_cols), max(4.5, 2.1 * n_rows + 2.0)),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.3, 'hspace': 0.35}
    )

    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # --- plotting ---
    for ri, s in enumerate(states_present):
        for ci, roi in enumerate(rois):
            ax = axes[ri, ci]
            vals_all, vals_sig, vals_nonsig = precomp.get((s, roi), (np.array([]), None, None))

            # Plot histograms
            if vals_sig is not None:  # with p-values split
                if vals_nonsig.size:
                    ax.hist(vals_nonsig, bins=bin_edges,
                            color='lightgray', edgecolor='black', alpha=1.0, label='n.s.')
                if vals_sig.size:
                    ax.hist(vals_sig, bins=bin_edges,
                            color='salmon', edgecolor='black', alpha=0.95, label=f'p<{alpha_sig:.2f}')
            else:  # no p-values available
                if vals_all.size:
                    ax.hist(vals_all, bins=bin_edges,
                            color='lightgray', edgecolor='black')

            # zero line
            ax.axvline(0, color='k', linestyle='dashed', linewidth=1.2)

            # Stats box (if any data)
            if vals_all.size:
                try:
                    t_stat, p_one, mval = one_tailed_ttest_greater_than_zero(vals_all)
                    sig = stars(p_one)
                    txt = f"n={vals_all.size}\nmean={mval:.2f}\n{sig} (p={p_one:.1e})"
                except Exception:
                    txt = f"n={vals_all.size}\nmean={np.nanmean(vals_all):.2f}"
            else:
                txt = "n=0\nNo data"

            ax.text(
                0.98, 0.96, txt,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                fontsize=9.5
            )

            # Column titles: ROI names on the top row
            if ri == 0:
                ax.set_title(str(roi), pad=6)

            # Row labels on the leftmost column
            if ci == 0:
                ax.set_ylabel(f"pref_state = {s}\nFrequency", fontsize=10.5)

            ax.tick_params(axis='both', labelsize=10, width=1.0, length=4)

    # --- consistent y-axis ---
    for ri in range(n_rows):
        axes[ri, 0].set_ylim(0, max(1, int(ylim_max * 1.04)))

    # --- shared labels + title ---
    fig.supxlabel(metric_col)
    fig.suptitle(
        f"Cross-validated state consistency per cell, split by ROI (columns) and pref_state (rows)\n{title_string_add}",
        fontsize=12, fontweight='bold', y=0.99
    )

    # --- legend: only if p-values are present; put it in the last axis ---
    if has_p:
        axes[-1, -1].legend(frameon=False, fontsize=9, loc='upper left')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.3, hspace=0.35)
    plt.show()


def plot_perm_spatial_consistency(perm_df, true_df, path_to_pval_table, group_results_path):

    out_dir    = f"{group_results_path}/figs"   # where to save the PNGs
    alpha_fdr  = 0.05                           # FDR level
    

    NOW_SET = {330, 0, 30}
    #FUTURE_REWARD_SET = {90, 180, 270}

    
    def to_int_safe(x):
        try: return int(x)
        except Exception:
            try: return int(float(x))
            except Exception: return np.nan
    
    def bh_reject(pvals, alpha=0.05):
        p = np.asarray(pvals, float)
        mask = np.isfinite(p); m = mask.sum()
        sig = np.zeros_like(p, dtype=bool)
        if m == 0: return sig
        order = np.argsort(p[mask])
        p_sorted = p[mask][order]
        thresh = alpha * (np.arange(1, m+1) / m)
        passed = p_sorted <= thresh
        if passed.any():
            kmax = np.nonzero(passed)[0].max()
            cutoff = p_sorted[kmax]
            sig[mask] = p[mask] <= cutoff
        return sig
    
    def roi_from_label(cell_label: str) -> str:
        # tweak if your labels differ
        if any(tag in cell_label for tag in ["ACC","vCC","AMC"]): return "ACC"
        if "PCC" in cell_label: return "PCC"
        if "OFC" in cell_label: return "OFC"
        if any(tag in cell_label for tag in ["MCC","HC"]): return "hippocampal"
        if "EC"  in cell_label: return "entorhinal"
        if "AMYG" in cell_label: return "amygdala"
        return "mixed"
    
    def ensure_columns(df, alpha_fdr):
        # keys as strings
        for c in ["session_id","neuron_id"]:
            if c in df: df[c] = df[c].astype(str)
        # ROI (compute if missing)
        if "roi" not in df.columns:
            df["roi"] = df["neuron_id"].astype(str).apply(roi_from_label)
        # significance (use existing if present; else compute from p_perm)
        if "sig_FDR_all" not in df.columns:
            if "p_perm" not in df.columns:
                raise ValueError("Table must have 'sig_FDR_all' or 'p_perm'.")
            df["sig_FDR_all"] = bh_reject(df["p_perm"].values, alpha=alpha_fdr)
        return df
    
    def beeswarm_by_roi(df, title, outpath=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        if df.empty:
            print(f"[warn] no data for: {title}")
            return
        rois = sorted(df["roi"].unique())
        xpos = {roi: i+1 for i, roi in enumerate(rois)}
    
        fig, ax = plt.subplots(figsize=(max(6, 1.2*len(rois)), 5))
    
        # background violin (blue)
        data_by_roi = [df.loc[df["roi"]==roi, "avg_consistency_at_peak"].to_numpy() for roi in rois]
        parts = ax.violinplot(data_by_roi, positions=list(xpos.values()),
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor('C0')     # blue
            body.set_alpha(0.25)
            body.set_edgecolor('none')
    
        # beeswarm points
        def jitter(n, scale=0.08): return rng.normal(0, scale, size=n)
        for roi in rois:
            sub = df[df["roi"] == roi]
            x0  = xpos[roi]
            y   = sub["avg_consistency_at_peak"].to_numpy()
            j   = jitter(len(y))
            #sig = sub["sig_FDR_all"].to_numpy()
            #sig = sub[sub["p_perm"]<0.05].to_numpy()
            sig = (sub["p_perm"] < 0.05).to_numpy()
    
            # non-significant (grey)
            ax.scatter(x0 + j[~sig], y[~sig], s=18, alpha=0.7, linewidths=0, c="#B0B0B0")
            # significant (orange, on top)
            ax.scatter(x0 + j[sig],  y[sig],  s=24, alpha=0.9, edgecolors='k', linewidths=0.3, c="#FF8C00")
    
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(rois, rotation=20)
        ax.set_xlim(0.5, len(rois)+0.5)
        ax.set_ylabel("avg_consistency_at_peak")
        ax.set_title(title)
    
        handles = [
            Line2D([0],[0], marker='o', linestyle='', color='#B0B0B0', label='non-significant'),
            Line2D([0],[0], marker='o', linestyle='', markeredgecolor='k', markeredgewidth=0.3,
                   color='#FF8C00', label='significant'),
            Line2D([0],[0], linestyle='-', color='C0', alpha=0.25, label='background distribution')
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False)
        plt.tight_layout()
        # fig.savefig(outpath, dpi=200, bbox_inches='tight')
        # plt.close(fig)
        # print(f"saved: {outpath}")
    
    # --- load table + ensure needed cols ---
    df = pd.read_csv(path_to_pval_table)
    df = ensure_columns(df, alpha_fdr)
    
    # integer version of shift for filtering
    shift_int = df["mode_peak_shift"].apply(to_int_safe)
    
    # --- make 4 plots ---
    beeswarm_by_roi(df,
        title=f"Beeswarm by ROI — ALL (q<{alpha_fdr})",
        outpath=os.path.join(out_dir, f"beeswarm_all_q{alpha_fdr:.2f}.png"))
    
    beeswarm_by_roi(df[shift_int.isin(NOW_SET)],
        title=f"Beeswarm by ROI — CURRENT {sorted(NOW_SET)} (q<{alpha_fdr})",
        outpath=os.path.join(out_dir, f"beeswarm_current_q{alpha_fdr:.2f}.png"))
    
    beeswarm_by_roi(df[~shift_int.isin(NOW_SET)],
        title=f"Beeswarm by ROI — FUTURE (q<{alpha_fdr})",
        outpath=os.path.join(out_dir, f"beeswarm_future_q{alpha_fdr:.2f}.png"))
    
    # beeswarm_by_roi(df[shift_int.isin(FUTURE_REWARD_SET)],
    #     title=f"Beeswarm by ROI — FUTURE REWARDS {sorted(FUTURE_REWARD_SET)} (q<{alpha_fdr})",
    #     outpath=os.path.join(out_dir, f"beeswarm_future_rewards_q{alpha_fdr:.2f}.png"))

    
    
    


def slope_plot_early_late_per_roi(df_early, df_late, title_string_add):
    
    # import pdb; pdb.set_trace()
    # Define your colors
    early_color = '#00BFC4'      # turquoise-blue
    late_color = '#E07B39'       # terracotta-orange
    
    # Merge the two DataFrames on cell and roi
    merged_df = pd.merge(
        df_early[['cell', 'roi', 'average_corr', 'model']],
        df_late[['cell', 'roi', 'average_corr', 'model']],
        on=['cell', 'roi', 'model'],
        suffixes=('_before', '_after')
    ).reset_index(drop=True)
    
    models = df_early['model'].unique().tolist()
    
    # only plot subset for now
    # import pdb; pdb.set_trace()
    models = ['complete_musicbox_reg', 'clo_model', 'curr_rings_split_clock_model', 'one_fut_rings_split_clock_model', 'two_fut_rings_split_clock_model', 'three_fut_rings_split_clock_model', 'phas_model', 'state_reg']
    
    
    # List of unique ROIs
    rois = merged_df['roi'].unique()
    n_roi = len(rois)
    
    for model in models:
        merged_df_model = merged_df[merged_df['model'] == model]
        
        # Plot
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 5, 5), sharey=True)
        if n_roi == 1:
            axes = [axes]
        
        for ax, roi in zip(axes, rois):
            df_roi = merged_df_model[merged_df_model['roi'] == roi]
        
            for _, row in df_roi.iterrows():
                # Grey line connecting before and after
                ax.plot([0, 1], [row['average_corr_before'], row['average_corr_after']], color='gray', linewidth=0.5)
        
            # Scatter points
            ax.scatter([0]*len(df_roi), df_roi['average_corr_before'], color=early_color, label='before', zorder=3)
            ax.scatter([1]*len(df_roi), df_roi['average_corr_after'], color=late_color, label='after', zorder=3)
        
            # Aesthetics
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Early', 'Late'])
            ax.set_title(roi)
            ax.set_ylabel('Average Correlation')
            ax.set_xlim(-0.5, 1.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            ax.tick_params(axis='both', labelsize=12)
            ax.axhline(0, linestyle='solid', color='black', linewidth=1)
    
        # Only add legend to the first axis
        axes[0].legend()
        
        fig.suptitle(f"{model}\n — {title_string_add}", fontsize=12, y=0.97)
    
        
        plt.tight_layout()
        plt.show()

    
    

def plotting_two_df_corr_perm_histogram_by_ROIs(df_early, df_late, title_string_add):
    # import pdb; pdb.set_trace()
    
    # Define colors
    early_color = '#00BFC4'      # turquoise-blue
    late_color = '#E07B39'       # terracotta-orange
    
    line_thickness = 2
    
    # Function to get significance stars
    def get_significance(corrs):
        t_stat, p_two = st.ttest_1samp(corrs, 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
        if p_one < 0.001:
            return '***'
        elif p_one < 0.01:
            return '**'
        elif p_one < 0.05:
            return '*'
        else:
            return ''
    
    models = df_early['model'].unique().tolist()
    
    # only plot subset for now
    # import pdb; pdb.set_trace()
    # models = ['complete_musicbox_reg', 'location_reg', 'musicbox_onlynowand3future_complete_reg', 'musicbox_onlynextand2future_complete_reg', 'midn_model', 'phas_model', 'stat_model', 'phas_stat_model', 'clo_model', 'curr_rings_split_clock_model', 'one_fut_rings_split_clock_model', 'two_fut_rings_split_clock_model', 'three_fut_rings_split_clock_model']
    
    
    for model in models:
        df_early_model = df_early[df_early['model'] == model]
        df_late_model = df_late[df_late['model'] == model]
        
        rois = sorted(set(df_early_model['roi'].unique()).union(df_late_model['roi'].unique()))
        n_roi = len(rois)
        
        # fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 5, 5), sharey=True)
        
        fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 6, 3), sharey=True)

        if n_roi == 1:
            axes = [axes]
    
        for ax, roi in zip(axes, rois):
            # Get early and late data
            corrs_early = df_early_model[df_early_model['roi'] == roi]['average_corr'].dropna()
            corrs_late = df_late_model[df_late_model['roi'] == roi]['average_corr'].dropna()
    
            # Get stars for significance vs zero
            early_sig = get_significance(corrs_early)
            late_sig = get_significance(corrs_late)
    
            # KDE plots
            
            # # KDE plots with custom bandwidth and curve height
            # sns.kdeplot(
            #     corrs_early, ax=ax, color=early_color, fill=True, alpha=0.4,
            #     linewidth=line_thickness, bw_adjust=0.2, label=f"early {early_sig}"
            # )
            # sns.kdeplot(
            #     corrs_late, ax=ax, color=late_color, fill=True, alpha=0.4,
            #     linewidth=line_thickness, bw_adjust=0.2, label=f"late {late_sig}"
            # )
        

            # Vertical lines at means
            mean_early = corrs_early.mean()
            mean_late = corrs_late.mean()
            ax.axvline(mean_early, color=early_color, linestyle='solid', linewidth=line_thickness)
            ax.axvline(mean_late, color=late_color, linestyle='solid', linewidth=line_thickness)
    
            # Zero reference line
            ax.axvline(0, color='black', linestyle='dashed', linewidth=line_thickness)
    
            # Plot overlapping histograms with transparency (true frequency)
            ax.hist(
                corrs_early, bins=20, color=early_color, alpha=0.5,
                label=f"early {early_sig}", edgecolor='black'
            )
            ax.hist(
                corrs_late, bins=20, color=late_color, alpha=0.5,
                label=f"late {late_sig}", edgecolor='black'
            )
            
            # Add vertical lines for means
            ax.axvline(mean_early, color=early_color, linestyle='solid', linewidth=2)
            ax.axvline(mean_late, color=late_color, linestyle='solid', linewidth=2)
            
            # Add zero reference
            ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Y-axis now shows count (no need to change scale)
            ax.set_ylabel("Frequency", fontsize=10)


            # Labels and formatting
            ax.set_title(f"{roi}\n{len(corrs_early)} early / {len(corrs_late)} late neurons", fontsize=10)
            ax.set_xlabel("Correlation coefficient", fontsize=12)
            ax.tick_params(axis='both', labelsize=10, width=2, length=6)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend()
    
        # Move model name to top of entire figure
        fig.suptitle(f"{model}\n — {title_string_add}", fontsize=12, y=0.93)

        plt.tight_layout()
        plt.show()
        





def plot_overlap_in_cells(df1, df2, top_x_percent):
    # import pdb; pdb.set_trace()

    # Define your ROI order (top to bottom)
    # first get all rois
    rois = df1['roi'].unique().tolist()
    roi_rank = {roi: i for i, roi in enumerate(rois)}
    
    # --- Setup filtering ---
    def get_top_cells(df, model_name='stat_model', top_percent=top_x_percent):
        df_filtered = df[df['model'] == model_name]
        cutoff = df_filtered['average_corr'].quantile(1 - top_percent / 100)
        return df_filtered[df_filtered['average_corr'] >= cutoff]
    
    # --- Filter ---
    df1_top = get_top_cells(df1)
    df2_top = get_top_cells(df2)
    
    # --- Sets of cell IDs ---
    cells1 = set(df1_top['cell'])
    cells2 = set(df2_top['cell'])
    
    only1 = cells1 - cells2
    only2 = cells2 - cells1
    both = cells1 & cells2

    # --- Create plot data ---
    plot_data = []
    
    def add_points(df, cells, label, x_center):
        for cell in cells:
            row = df[df['cell'] == cell].iloc[0]
            roi = row['roi']
            corr = row['average_corr']
            if roi not in roi_rank:
                continue  # Skip unknown ROIs
            y_base = -roi_rank[roi]  # invert for top-to-bottom
            x = np.random.normal(loc=x_center, scale=0.2)
            y = np.random.normal(loc=y_base, scale=0.2)
            size = corr * 800  # adjust scaling more aggressively
            plot_data.append({'x': x, 'y': y, 'group': label, 'size': size, 'roi': roi})
    
    add_points(df1_top, only1, 'df1 only', -1)
    add_points(df2_top, only2, 'df2 only', 1)
    # Average the corr from both dfs for overlap
    for cell in both:
        row1 = df1_top[df1_top['cell'] == cell].iloc[0]
        row2 = df2_top[df2_top['cell'] == cell].iloc[0]
        roi = row1['roi']
        if roi not in roi_rank:
            continue
        avg_corr = (row1['average_corr'] + row2['average_corr']) / 2
        y_base = -roi_rank[roi]
        x = np.random.normal(loc=0, scale=0.2)
        y = np.random.normal(loc=y_base, scale=0.2)
        size = (avg_corr - 0.3) * 800
        plot_data.append({'x': x, 'y': y, 'group': 'overlap', 'size': size, 'roi': roi})
    
    plot_df = pd.DataFrame(plot_data)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for group, alpha in zip(['df1 only', 'df2 only', 'overlap'], [0.5, 0.5, 0.9]):
        subset = plot_df[plot_df['group'] == group]
        plt.scatter(subset['x'], subset['y'], s=subset['size'], alpha=alpha, label=group)
    
    # Add ROI labels on y-axis
    y_ticks = [-roi_rank[roi] for roi in rois]
    plt.yticks(y_ticks, rois)
    plt.xticks([])  # Remove x-axis ticks (groups are implicit)
    plt.xlabel('')
    plt.ylabel('ROI')
    plt.title(f'Overlapping Structured Representations by ROI\n(Top {top_x_percent}% average_corr, Model = state_model)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    

def plot_perms_per_cell_and_roi(df_results, n_perms, corr_thresh=0.05, save=False, model_name_string=None):
    if save==True:
        res_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/elastic_net_reg/corrs"
        if not os.path.isdir(res_folder):
            res_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/corrs"
        
    # import pdb; pdb.set_trace()
    models = df_results['model'].unique().tolist()
    cells = df_results['cell'].unique().tolist()
    rois = df_results['roi'].unique().tolist()
    # Custom colors
    color_task_perms = '#214066'   # dark turquoise blue
    color_time_perms = '#7A9DB1'   # blue-grey
    true_val_color = '#E2725B'   # terracotta/salmon
    
    # Always plotting the top 25 cells.
    # n_rows = int(np.ceil(np.sqrt(len(df_strong_curr_model))))
    # n_cols = int(np.ceil(len(df_strong_curr_model) / n_rows))
    n_rows = 5
    n_cols = 5

    # plot those cells that are strong for the respective model (corr higher than 0.05)
    # df_strong_cells = df_results[df_results['average_corr'] > corr_thresh]
    for curr_model in models:
        df_curr_model = df_results[df_results['model'] == curr_model].copy()
        
        # 1: COMPUTE SOME PERM STATS PER MODEL/CELL
        
        # If 'time_perm_0' exists, compute p_val_time for each row
        if 'time_perm_0' in df_curr_model.columns:
            p_val_times = []
            for _, row in df_curr_model.iterrows():
                perm_values = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                if not math.isnan(row['average_corr']):
                    p_val_time = np.mean(perm_values >= row['average_corr'])
                    p_val_times.append(p_val_time)
                else:
                    p_val_times.append(np.nan)
            print(f"there were n = {np.sum(np.isnan(p_val_times))} nans in the average corr for {curr_model}!")
            df_curr_model['p_val_time'] = p_val_times

        # also store p vals for task perms
        if 'task_perm_0' in df_curr_model.columns:
            p_val_tasks = []
            for _, row in df_curr_model.iterrows():
                perm_values = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                if not math.isnan(row['average_corr']):
                    p_val_task = np.mean(perm_values >= row['average_corr'])
                    p_val_tasks.append(p_val_task)
                else:
                    p_val_tasks.append(np.nan)
            df_curr_model['p_val_task'] = p_val_tasks
        

        # and store the difference between both p val perms
        for idx, row in df_curr_model.iterrows():
            if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
                perm_values_task = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                perm_values_time = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                _, p_value_diff_perms = ttest_ind(list(perm_values_task), list(perm_values_time))
                df_curr_model.loc[idx, 'p_val_perm_diff'] = p_value_diff_perms
         
        if save==True:
            # save the entire df for cells only for this model.
            os.makedirs(f"{res_folder}/cells_per_model", exist_ok=True)         
            df_curr_model.to_csv(f"{res_folder}/cells_per_model/{curr_model}_{model_name_string}.csv", index=False)
            
        
        # 2:  PRINTING STATS
        # then print some stats: percentage of cells, overall and per ROI for
        # each of the permutation ps

        results_file = []
        
        # first: overall
        n_cells = len(df_curr_model)
        # import pdb; pdb.set_trace()
        mean_avg_corr = np.mean(df_curr_model['average_corr'])
        
        print(f"for {curr_model}, for n = {n_cells} cells all over the brain, the mean corr is {mean_avg_corr:.3f}")
        results_file.append(f"for {curr_model}, for n = {n_cells} cells all over the brain, the mean corr is {mean_avg_corr:.3f}")
        
        
        if 'task_perm_0' in df_curr_model.columns:
            n_p_val_task_sig = len(df_curr_model[df_curr_model['p_val_task'] < 0.05])
            
            if n_p_val_task_sig > 0:
                mean_corr_sig_task = np.mean(df_curr_model[df_curr_model['p_val_task'] < 0.05])
            else:
                mean_corr_sig_task = 0
                   
            print(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells)*100:.3f} % cells are sig. for task config shuffles,")
            results_file.append(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells)*100:.3f} % cells are sig. for task config shuffles,")
        
        
        # this is the one I want to keep.
        if 'time_perm_0' in df_curr_model.columns:
            n_p_val_time_sig = len(df_curr_model[df_curr_model['p_val_time'] < 0.05])
            
            # and compute the mean for the significant cells.
            df_curr_model_sig = df_curr_model[df_curr_model['p_val_time'] < 0.05]
            n_sig_cells = len(df_curr_model_sig)
            if n_sig_cells > 0:
                mean_avg_corr_sig = np.mean(df_curr_model_sig['average_corr'])
            else:
                mean_avg_corr_sig = 0
            print(f"for {curr_model}, for n = {n_sig_cells} cells or or {(n_sig_cells/n_cells)*100:.3f} % of cells all over the brain are sig., their mean corr being r = {mean_avg_corr_sig:.3f}")
            results_file.append(f"for {curr_model}, for n = {n_sig_cells} cells or or {(n_sig_cells/n_cells)*100:.3f} % of cells all over the brain are sig., their mean corr being r = {mean_avg_corr_sig:.3f}")
            
            # also save the signficant cells as .csv
            if save == True:
                df_curr_model_sig.to_csv(f"{res_folder}/{curr_model}_{model_name_string}_sig_after_temp_perms.csv", index=False)
            

        if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
            n_p_val_perm_diff_sig = len(df_curr_model[df_curr_model['p_val_perm_diff'] < 0.05])
            
            if n_p_val_perm_diff_sig > 0:
                mean_corr_sig_diff = np.mean(df_curr_model[df_curr_model['p_val_perm_diff'] < 0.05])
            else:
                mean_corr_sig_diff = 0
            
            print(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells)*100} % have sig. different perm distributions.")
            results_file.append(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells)*100:.1f} % have sig. different perm distributions.")
        
        
        # second: per roi
        for roi in rois:
            df_curr_model_curr_roi = df_curr_model[df_curr_model['roi'] == roi].copy().reset_index(drop=True)
            n_cells_in_roi = len(df_curr_model_curr_roi)
            mean_avg_corr = np.mean(df_curr_model_curr_roi['average_corr'])
            print(f"for {curr_model}, for n = {n_cells_in_roi} cells in {roi}, mean corr is {mean_avg_corr:.3f}")
            results_file.append(f"for {curr_model}, for n = {n_cells_in_roi} cells in {roi}, mean corr is {mean_avg_corr:.3f}")
            
            if 'time_perm_0' in df_curr_model.columns:
                n_p_val_time_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_time'] < 0.05])
                print(f"n = {n_p_val_time_sig} or {(n_p_val_time_sig/n_cells_in_roi)*100:.1f} % cells are sig. for time shuffles,")
                results_file.append(f"n = {n_p_val_time_sig} or {(n_p_val_time_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task config shuffles,")
                
            
                # and compute the mean for the significant cells.
                df_curr_model_curr_roi_sig = df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_time'] < 0.05]
                n_sig_cells_curr_model_curr_roi = len(df_curr_model_curr_roi_sig)
                if n_sig_cells_curr_model_curr_roi > 0:
                    mean_avg_corr_sig_curr_model_curr_roi = np.mean(df_curr_model_curr_roi_sig['average_corr'])
                else:
                    mean_avg_corr_sig_curr_model_curr_roi = 0
                print(f"for {curr_model}, for n = {n_sig_cells_curr_model_curr_roi} cells or {(n_sig_cells_curr_model_curr_roi/n_cells_in_roi)*100:.1f} % of cells in {roi} are sig., their mean corr being r = {mean_avg_corr_sig_curr_model_curr_roi:.3f}")
                results_file.append(f"for {curr_model}, for n = {n_sig_cells_curr_model_curr_roi} cells or {(n_sig_cells_curr_model_curr_roi/n_cells_in_roi)*100:.1f} % of cells in {roi} are sig., their mean corr being r = {mean_avg_corr_sig_curr_model_curr_roi:.3f}")
                
                
                
            if 'task_perm_0' in df_curr_model.columns:
                n_p_val_task_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_task'] < 0.05])
                print(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task config shuffles,")
                results_file.append(f"n = {n_p_val_task_sig} or {(n_p_val_task_sig/n_cells_in_roi)*100:.1f} % cells are sig. for task shuffles,")
                
            if 'task_perm_0' in df_curr_model.columns and 'time_perm_0' in df_curr_model.columns:
                n_p_val_perm_diff_sig = len(df_curr_model_curr_roi[df_curr_model_curr_roi['p_val_perm_diff'] < 0.05])
                print(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells_in_roi)*100:.1f} % have sig. different perm distributions.")
                results_file.append(f"n = {n_p_val_perm_diff_sig} or {(n_p_val_perm_diff_sig/n_cells_in_roi)*100:.1f} % have sig. different perm distributions.")

            
        # Write everything to a .txt file at the end
        if save==True:
            with open(f"{res_folder}/{curr_model}_{model_name_string}_stats.txt", 'w') as f:
                f.write('\n'.join(results_file))
    
    

        # 3. PLOTTING
        # plot the distributions for the nicest 25 cells per model, filtered for significant cells if possible
        
        if 'time_perm_0' in df_curr_model.columns and n_sig_cells > 0:
            if len(df_curr_model_sig) > 25:
                df_strong_curr_model = df_curr_model_sig.sort_values('average_corr', ascending=False).head(25).reset_index(drop=True)
            else:
                df_strong_curr_model = df_curr_model_sig.sort_values('average_corr', ascending=False).reset_index(drop=True) 
        else:
            # if there are no significant cells after temporal permutation testing
            # subset to only plot the top 25 cells
            if len(df_curr_model) > 25: 
                df_strong_curr_model = df_curr_model.sort_values('average_corr', ascending=False).head(25).reset_index(drop=True)
            else:
               df_strong_curr_model = df_curr_model.sort_values('average_corr', ascending=False).reset_index(drop=True) 
            
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        fig.suptitle(f"{curr_model}", fontsize=15, y=0.99)  # Title slightly above the top
        axs = axs.flatten()
        
        # Determine common x-axis range for centering
        if 'task_perm_0' in df_strong_curr_model.columns:
            all_values = df_strong_curr_model[[f'task_perm_{i}' for i in range(n_perms)]].values.flatten()
            
        if 'task_perm_0' in df_strong_curr_model.columns and 'time_perm_0' in df_strong_curr_model.columns:
            values_task = df_strong_curr_model[[f'task_perm_{i}' for i in range(n_perms)]].values.flatten()
            values_time = df_strong_curr_model[[f'time_perm_{i}' for i in range(n_perms)]].values.flatten()
            all_values = np.concatenate((values_task, values_time))

        else:
            all_values = df_strong_curr_model['average_corr'].values.flatten()

        xlim = max(abs(np.nanmin(all_values)), abs(np.nanmax(all_values)))  # Symmetric about 0
    
        for idx, row in df_strong_curr_model.iterrows():
            avg_corr = row['average_corr']
            
            ax = axs[idx]
            if 'task_perm_0' in df_strong_curr_model.columns:
                perm_values_task = row[[f'task_perm_{i}' for i in range(n_perms)]].values
                if pd.Series(perm_values_task).isna().all() == True:
                    perm_values_task = np.zeros(1)
                else:
                    ax.hist(perm_values_task, bins=30, color=color_task_perms, alpha=0.5, label='Task perm.', edgecolor=None)
                    # Calculate one-tailed p-value
                    p_val_task = np.mean(perm_values_task >= avg_corr)
                    ax.text(0.95, 0.70, f"p_task = {p_val_task:.3f}", ha='right', va='top', transform=ax.transAxes)
                
            if 'time_perm_0' in df_strong_curr_model.columns:
                perm_values_time = row[[f'time_perm_{i}' for i in range(n_perms)]].values
                ax.hist(perm_values_time, bins=30, color=color_time_perms, alpha=0.5, label='Time perm.', edgecolor=None)
                # Calculate one-tailed p-value
                p_val_time = np.mean(perm_values_time >= avg_corr)
                ax.text(0.95, 0.95, f"p_time = {p_val_time:.3f}", ha='right', va='top', transform=ax.transAxes)
                
            # true corr   
            ax.axvline(avg_corr, color=true_val_color, linestyle='--', linewidth=2)
            
            # 0 lin e
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            # Center x-axis around 0
            ax.set_xlim(-xlim, xlim)
            ax.set_title(f"{row['roi']} | {row['cell']}", fontsize=10)
            ax.set_xlabel("Correlation", fontsize = 9)
            ax.set_ylabel("Count")
        
        # Hide any unused subplots
        for ax in axs[len(df_strong_curr_model):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 1.02])  # Adjust layout to make room for the title

        # then store these figures if on cluster. 
        if save == True:
            os.makedirs(f"{res_folder}/figures", exist_ok=True)
            plt.savefig(f"{res_folder}/figures/{curr_model}_{model_name_string}_perms_best_cells.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    
        
        
    
    



def overview_regression(results, rois, models, combo = False, only_reward_times = None, no_bins_per_state = None):
    # import pdb; pdb.set_trace()
    # Prepare data arrays for t-values and annotations for significance
    t_values = np.zeros((len(rois), len(models)))
    p_values = np.zeros_like(t_values)
    
    if combo == True:
        # first figure out the correct model order.
        for r, roi in enumerate(rois):
            models = results[roi]['label_regs']
        for r, roi in enumerate(rois):
            for m, model in enumerate(models):
                # import pdb; pdb.set_trace()
                t_values[r, m] = results[roi]['t_vals'][m]
                p_values[r, m] = results[roi]['p_vals'][m]
                
                
    else:                 
        for r, roi in enumerate(rois):
            for m, model in enumerate(models):
                if model in results[roi][model]['label_regs']:
                    t_values[r, m] = results[roi][model]['t_vals'][0]
                    p_values[r, m] = results[roi][model]['p_vals'][0]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))  # Adjust size as needed
    cax = ax.matshow(t_values, cmap='viridis')  # Choose a colormap that fits your preferences
    
    # Add a color bar
    fig.colorbar(cax, label='T-value magnitude')
    
    # Set up axes
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(rois)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(rois)
    
    # Rotate the tick labels for better readability
    plt.xticks(rotation=45)
    
    # Adding significance annotations
    for i in range(len(rois)):
        for j in range(len(models)):
            signif = '*' if p_values[i, j] < 0.05 else ''
            signif += '*' if p_values[i, j] < 0.01 else ''
            signif += '*' if p_values[i, j] < 0.005 else ''
            ax.text(j, i, signif, color='black', ha='center', va='center', fontsize=12)
    
    # Title and labels
    ax.set_title('T-values and Significance of Models Across ROIs')
    if only_reward_times == True:
        ax.set_title(f"T-values and Significance of State Across ROIs, only rew times, {no_bins_per_state} bins per state")
    if only_reward_times == False:
        ax.set_title(f"T-values and Significance of State Across ROIs, {no_bins_per_state} bins per state")
    if combo == True:
        ax.set_xlabel('Each model as regressor in combined GLM')
    else:     
        ax.set_xlabel('Models, in separate GLMs as single regressors')
    ax.set_ylabel('ROIs')
    
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_model_rdm_correlation(
    rdm_dict,
    title=None,
    corr_method='pearson',
    save_path=None,
    show=True,
    figsize=None,
    cmap='RdBu_r',
    annot_fontsize=8,
    model_order=None,
):
    """Heatmap of pairwise correlations between model RDMs.

    Each model RDM is reduced to a 1-D vector (upper triangle if it is a
    symmetric square matrix, else flattened in full) and all pairs are
    correlated. Intended as a regressor-collinearity control, independent
    of any ROI/data RDM.

    Parameters
    ----------
    rdm_dict : dict[str, ndarray | tuple]
        Mapping model_name -> RDM. If the value is a tuple/list (as returned
        by `compute_crosscorr` / `compute_hamming_distance`), its first element
        is taken as the RDM.
    title : str, optional
        Figure title.
    corr_method : {'pearson', 'spearman'}
    save_path : str, optional
        If given, the figure is saved to this path with dpi=150.
    show : bool
        If True, calls plt.show() before returning.
    figsize : tuple, optional
    cmap : str
    annot_fontsize : int
        Font size of the per-cell correlation annotations.
    model_order : list[str], optional
        Force a specific row/column order. Defaults to insertion order.

    Returns
    -------
    fig, ax, corr_matrix
    """
    model_names = list(model_order) if model_order is not None else list(rdm_dict.keys())

    vectors = []
    for name in model_names:
        rdm = rdm_dict[name]
        if isinstance(rdm, (tuple, list)):
            rdm = rdm[0]
        rdm = np.asarray(rdm, dtype=float)

        # symmetric square: take strict upper triangle; else flatten
        if (rdm.ndim == 2
                and rdm.shape[0] == rdm.shape[1]
                and np.allclose(rdm, rdm.T, equal_nan=True)):
            iu = np.triu_indices_from(rdm, k=1)
            vec = rdm[iu]
        else:
            vec = rdm.ravel()
        vectors.append(vec)

    lens = [len(v) for v in vectors]
    if len(set(lens)) != 1:
        raise ValueError(
            f"RDMs have mismatching vectorized lengths: "
            f"{dict(zip(model_names, lens))}"
        )

    M = np.stack(vectors, axis=0)  # (n_models, n_features)

    if corr_method == 'pearson':
        corr = np.corrcoef(M)
    elif corr_method == 'spearman':
        n_m = len(model_names)
        corr = np.eye(n_m)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                r = st.spearmanr(M[i], M[j], nan_policy='omit').correlation
                corr[i, j] = r
                corr[j, i] = r
    else:
        raise ValueError(f"unknown corr_method: {corr_method}")

    n = len(model_names)
    if figsize is None:
        figsize = (1.0 * n + 2, 1.0 * n + 1)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(model_names, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=annot_fontsize, color='black',
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{corr_method} r', fontsize=9)

    if title is not None:
        ax.set_title(title, fontsize=11)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax, corr


# ─────────────────────────────────────────────────────────────────────
# Shared ROI × stat plotting helpers — used by encoding_state_sustained_cv
# (figs 5/6/11) and by DSR-RSA add-on plotting. Same "feel" everywhere.
# ─────────────────────────────────────────────────────────────────────
def _stars(q):
    if q is None or not np.isfinite(q):
        return ''
    if q < .001: return '***'
    if q < .01:  return '**'
    if q < .05:  return '*'
    return ''


def plot_roi_tstat_heatmap(
    t_matrix, rois, col_labels, *,
    q_matrix=None,
    panel_groups=None,
    cmaps=None,
    title=None,
    cbar_label='t vs 0',
    fig_size_cm=(10.0, 13.0),
    font_tick=10, font_axis=10, font_big=11,
    save_path=None,
):
    """ROI × stat heatmap of t-stats, with FDR-driven bold stars in cells.

    Mirrors the style of `encoding_state_sustained_cv.fig11_wilcoxon_heatmap`
    so that DSR-RSA, sustained-state, etc. all use the same look.

    Parameters
    ----------
    t_matrix : (n_rois, n_cols) array of t-statistics.
    rois     : list of ROI names (length n_rois).
    col_labels : list of column labels (length n_cols).
    q_matrix : optional (n_rois, n_cols) of BH-FDR q-values used to draw
        the bold stars (* < .05, ** < .01, *** < .001).
    panel_groups : optional list of (slice_or_indices, cmap, panel_title)
        triples to split the heatmap into multiple panels side-by-side
        (one panel per group). When None, a single panel is drawn.
        Example: [((0,1,2), 'RdBu_r', 'controls'), ((3,), 'PiYG_r', 'DSR')]
    cmaps : str or list — used only if `panel_groups` is None.
    fig_size_cm : (width_cm, height_cm) for the overall figure.

    Returns the matplotlib Figure.
    """
    cm = 1.0 / 2.54
    rcparams = {
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_tick,
        'pdf.fonttype':    42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    }
    n_rows = len(rois)

    if panel_groups is None:
        panel_groups = [(slice(None), cmaps or 'RdBu_r', '')]

    width_ratios = []
    for grp in panel_groups:
        idxs = grp[0]
        if isinstance(idxs, slice):
            n = len(range(*idxs.indices(t_matrix.shape[1])))
        else:
            n = len(list(idxs))
        width_ratios.append(max(n, 1) * 1.0)

    plt.rcParams.update(rcparams)
    fig = plt.figure(figsize=(fig_size_cm[0] * cm, fig_size_cm[1] * cm),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, len(panel_groups), width_ratios=width_ratios,
                          wspace=0.10)

    for pi, (idxs, cmap_name, panel_title) in enumerate(panel_groups):
        ax = fig.add_subplot(gs[0, pi])
        if isinstance(idxs, slice):
            cols = list(range(*idxs.indices(t_matrix.shape[1])))
        else:
            cols = list(idxs)
        sub_t = t_matrix[:, cols]
        sub_q = q_matrix[:, cols] if q_matrix is not None else None
        sub_labels = [col_labels[c] for c in cols]
        if np.isfinite(sub_t).any():
            vmax = max(1.0, float(np.nanmax(np.abs(sub_t))))
        else:
            vmax = 1.0
        im = ax.imshow(sub_t, cmap=cmap_name, vmin=-vmax, vmax=vmax,
                       aspect='auto')
        if sub_q is not None:
            img = ax.images[0].get_array()
            for i in range(n_rows):
                for j in range(len(cols)):
                    s = _stars(sub_q[i, j])
                    if not s:
                        continue
                    intensity = abs(float(img[i, j])) / max(vmax, 1e-9)
                    col = 'white' if intensity > 0.55 else 'black'
                    ax.text(j, i, s, ha='center', va='center',
                            fontsize=font_big + 1, fontweight='bold',
                            color=col, zorder=5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(sub_labels, rotation=30, ha='right',
                           fontsize=font_axis)
        ax.set_yticks(range(n_rows))
        if pi == 0:
            ax.set_yticklabels(rois, fontsize=font_axis)
        else:
            ax.set_yticklabels([])
        if panel_title:
            ax.set_title(panel_title, fontsize=font_big, pad=2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='both', length=1.5, pad=1)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          location='bottom', fraction=0.12, pad=0.55,
                          shrink=0.9, aspect=5)
        cb.set_label(cbar_label, fontsize=font_axis)
        cb.ax.tick_params(labelsize=font_axis)

    if title:
        fig.suptitle(title, fontsize=font_big)
    if save_path is not None:
        for ext in ('.pdf', '.png'):
            base = os.path.splitext(save_path)[0]
            fig.savefig(base + ext, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig



def _tick_no_trailing_zeros(x, pos):
    if np.isclose(x, 0):
        return '0'
    return f'{x:g}'

def plot_per_roi_stat_histograms(
    results_df, *,
    stat_col, stat_label,
    roi_col='roi',
    roi_order=None, roi_colours=None,
    q_per_roi=None,
    empirical_per_roi=None,
    mark_sig_neurons=None,
    bar_color='lightgray',
    n_cols=4, panel_w_cm=3.0, panel_h_cm=1.5,
    font_small=9,
    save_path=None):
    
    """Per-ROI histograms of `stat_col`."""
    mark_sig_neurons = mark_sig_neurons or {}
    cm = 1.0 / 2.54
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_small,
        'pdf.fonttype':    42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    if roi_order is None:
        roi_order = [r for r in results_df[roi_col].unique() if pd.notna(r)]
    else:
        roi_order = [r for r in roi_order if (results_df[roi_col] == r).any()]

    n_rois = len(roi_order)
    if n_rois == 0:
        return None

    n_cols = min(n_cols, n_rois)
    n_rows = int(np.ceil(n_rois / n_cols))
    fig_w = (panel_w_cm * n_cols + 2.2) * cm
    fig_h = (panel_h_cm * n_rows + 0.7) * cm

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        constrained_layout=True, squeeze=False
    )

    all_vals = results_df[stat_col].dropna().to_numpy()
    if all_vals.size:
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        bins = np.linspace(lo, hi, 22)
    else:
        bins = 20

    for i, r in enumerate(roi_order):
        ax = axes[i // n_cols, i % n_cols]
        rdf = results_df.loc[results_df[roi_col] == r]
        v = rdf[stat_col].dropna().to_numpy()

        if v.size:
            ax.hist(v, bins=bins, color=bar_color, edgecolor='black',
                    linewidth=0.2, alpha=0.55)

            if r in mark_sig_neurons:
                v_sig = np.asarray(mark_sig_neurons[r], float)
                v_sig = v_sig[np.isfinite(v_sig)]
                if v_sig.size:
                    col = (roi_colours or {}).get(r, '#444')
                    ax.hist(v_sig, bins=bins, color=col,
                            edgecolor='black', linewidth=0.2, alpha=0.95)

        ax.axvline(0, color='gray', ls='--', lw=1.5)   # a) 0-line thickness

        if empirical_per_roi and r in empirical_per_roi:
            emp = empirical_per_roi[r]
            if emp is not None and np.isfinite(emp):
                col = (roi_colours or {}).get(r, '#444')
                ax.axvline(float(emp), color=col, lw=2.0)  # a) empirical line thickness

        title_str = r
        if q_per_roi and r in q_per_roi:
            s = _stars(q_per_roi[r])
            if s:
                title_str = f'{r} {s}'
        ax.set_title(title_str, fontsize=font_small, pad=2)

        ax.tick_params(axis='both', labelsize=font_small, length=1.5, pad=1)
        ax.xaxis.set_major_formatter(FuncFormatter(_tick_no_trailing_zeros))  # b)

        # c) only once for the whole figure if multiple rows
        if i == 0 and n_rows == 1:
            ax.set_ylabel('# cells / units', fontsize=font_small)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel(stat_label, fontsize=font_small)

        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    for k in range(n_rois, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis('off')

    if n_rows > 1:
        fig.supylabel('# cells / units', fontsize=font_small)

    # d) caption instead of legend
    caption = 'Gray bars show the distribution; black line shows the mean; dashed gray line shows 0; colored line shows the empirical value.'
    if mark_sig_neurons:
        caption = 'Gray bars show the distribution; colored bars show significant cells; black line shows the mean; dashed gray line shows 0; colored line shows the empirical value.'

    fig.subplots_adjust(bottom=0.5)
    fig.text(0.5, 0.00, caption, ha='center', va='bottom', fontsize=font_small)

    if save_path is not None:
        for ext in ('.pdf', '.png'):
            base = os.path.splitext(save_path)[0]
            fig.savefig(base + ext, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig

