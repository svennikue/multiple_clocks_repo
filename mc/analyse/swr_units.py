#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple-triggered single-unit firing: does a hippocampal ripple coincide with a
change in mPFC firing?

The prediction behind the whole project is that hippocampus informs mPFC -- that
the action plan represented in mPFC is loaded with hippocampal help. If that is
true at all, the cheapest observable signature is that mPFC units change their
firing around hippocampal ripples.

What makes this testable here without touching the LFP again:

  * ripple times are in session seconds (the bundle), and
  * `all_cells_firing_rate_grid{N}_sub{XX}.csv` is a cells x 25 ms-bin matrix per
    grid, on the SAME clock (`session_s = new_grid_onset + bin * 0.025`) that the
    behavioural reconstruction was validated on.

Two things keep this honest:

1. **Row order comes from `all_cells_region_labels_sub{XX}.txt`**, never from
   `neurons_with_ROI_labels.csv`. The label file has exactly one line per row of
   the firing matrix in every session checked; the ROI table has fewer rows
   (cells that failed atlas assignment were dropped) and its 1-based `cell idx`
   cannot be assumed to index the original rows. Using it risks silently calling
   a hippocampal cell an mPFC one.
2. **Hippocampal units are the positive control.** HC units must show a
   peri-ripple firing increase -- that is the best-established fact about
   ripples there is. If they do not, the spike and LFP clocks are not aligned
   and no mPFC result means anything.

Inference is by circular shift of the ripple times within each grid, so the null
carries the same firing autocorrelation and the same number of ripples.

@author: Svenja Kuchenhoff
"""

import os
import re
import glob

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io

BIN_S = 0.025                 # the derivatives' bin, same as the behaviour
DEDUP_S = 0.05                # ripples on two derivations closer than this are one event
MPFC_LABELS = ("ACC",)        # native label for medial prefrontal cortex here
HC_LABELS = ("HC",)


def _beh_dir(session, data_root=None):
    return os.path.join(swr_io.session_deriv_dir(int(session), data_root),
                        "cells_and_beh")


def unit_labels(session, data_root=None):
    """Region label per ROW of the firing matrix, or None if absent."""
    p = os.path.join(_beh_dir(session, data_root),
                     f"all_cells_region_labels_sub{int(session)}.txt")
    if not os.path.isfile(p):
        return None
    return [ln.strip() for ln in open(p) if ln.strip() != ""]


def grid_firing(session, grid, data_root=None):
    """(cells x bins) firing matrix for one grid, or None."""
    hits = glob.glob(os.path.join(
        _beh_dir(session, data_root),
        f"all_cells_firing_rate_grid{int(grid)}_sub{int(session)}.csv"))
    if not hits:
        return None
    return pd.read_csv(hits[0], header=None).to_numpy(float)


def dedup_ripples(t, tol_s=DEDUP_S):
    """One ripple detected on several derivations is one event, not several."""
    t = np.sort(np.asarray(t, float))
    if not t.size:
        return t
    keep = np.concatenate([[True], np.diff(t) > tol_s])
    return t[keep]


def peri_ripple_matrix(session, ripple_t, beh, half_s=1.0, n_shift=200,
                       data_root=None, seed=42):
    """Peri-ripple firing per unit, with a circular-shift null.

    Returns (offsets_s, observed, null_mean, null_sd, labels, n_ripples) where
    `observed` is (n_units, n_offsets) in the matrix's own firing units, and the
    null summaries are over `n_shift` within-grid circular shifts.

    The shift is applied to every grid and the totals are pooled BEFORE the
    statistic is taken, so the null has the same structure as the observed
    value: one number per shift, over the whole session. Accumulating variance
    per grid instead would mix per-grid sums with per-window means and give a
    meaningless spread.
    """
    session = int(session)
    labels = unit_labels(session, data_root)
    if labels is None:
        return None
    half = int(round(half_s / BIN_S))
    offs = np.arange(-half, half + 1)
    rng = np.random.default_rng(seed)

    # collect the usable grids first, so the shift loop can pool across them
    blocks = []
    n_used = 0
    n_units = None
    for grid, g in beh.groupby("grid_no"):
        M = grid_firing(session, grid, data_root)
        if M is None or M.shape[0] != len(labels):
            continue
        onset = float(g.new_grid_onset.iloc[0])
        nb = M.shape[1]
        b = np.round((np.asarray(ripple_t, float) - onset) / BIN_S).astype(int)
        b = b[(b >= half) & (b < nb - half)]
        if not b.size or nb - 2 * half <= 1:
            continue
        blocks.append((M, b, half, nb))
        n_used += b.size
        n_units = M.shape[0]
    if not blocks or not n_used:
        return None

    obs = np.zeros((n_units, offs.size))
    for M, b, h, nb in blocks:
        obs += M[:, b[:, None] + offs[None, :]].sum(axis=1)
    obs /= n_used

    null = np.empty((n_shift, n_units, offs.size))
    for k in range(n_shift):
        tot = np.zeros((n_units, offs.size))
        for M, b, h, nb in blocks:
            lo, span = h, nb - 2 * h
            sh = int(rng.integers(1, span))
            bs = lo + np.mod(b - lo + sh, span)
            tot += M[:, bs[:, None] + offs[None, :]].sum(axis=1)
        null[k] = tot / n_used
    return (offs * BIN_S, obs, null.mean(axis=0), null.std(axis=0),
            labels, n_used)


def sessions_with_units(sessions, data_root=None, want=MPFC_LABELS):
    """Sessions that have a firing matrix and at least one unit in `want`."""
    out = []
    for s in sessions:
        lab = unit_labels(s, data_root)
        if not lab:
            continue
        if not glob.glob(os.path.join(_beh_dir(s, data_root),
                                      f"all_cells_firing_rate_grid*_sub{int(s)}.csv")):
            continue
        if any(x.upper() in want for x in lab):
            out.append(int(s))
    return out
