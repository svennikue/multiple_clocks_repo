"""Lag-wise group statistics at two units of analysis: CELL and SUBJECT.

Why this exists
---------------
Cells recorded in the same session are not independent observations — they
share a task history, an electrode, and a subject. Testing across cells
therefore inflates the effective N. Testing across sessions is the
conservative option, but throws away within-session information. Neither is
"the" right answer, so every lag-wise result should be reported at BOTH
units and the reader told which one a quoted t and df refer to.

This module is the single implementation of that computation, shared by
`scripts/per_lag_encoding.py`, `scripts/wrapper_future_spatial_peaks.py`
and `scripts/overlay_double_dissociation.py`, so the numbers in the stats
CSVs and the numbers under the figures cannot drift apart.

Note both estimators show the SAME aggregation effect in mPFC: the
cell-level peak sits at 30 deg and the subject-level peak at 60 deg. That is
a property of the aggregation, not of either pipeline, which is why the
manuscript reports the effect as spanning 30-60 deg rather than committing
to one lag.

All tests are one-sample t-tests of Fisher-z-transformed cross-validated
correlations against zero. Fisher-z is the inferential scale (it stabilises
the variance of r); raw r is carried alongside as the descriptive
effect size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

FISHER_CLIP = 0.9999999


def fisher_z(curves):
    """Fisher transform with clipping so |r| = 1 does not become inf."""
    return np.arctanh(np.clip(np.asarray(curves, dtype=float),
                              -FISHER_CLIP, FISHER_CLIP))


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values. NaNs stay NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    pg = p[good]
    order = np.argsort(pg)
    ranked = pg[order] * len(pg) / np.arange(1, len(pg) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.clip(ranked, 0, 1)
    out[good] = restored
    return out


def aggregate_curves(curves, subjects, unit='cell', fisher=False):
    """Return per-analysis-unit lag curves.

    Parameters
    ----------
    curves   : (n_cells, n_lags) cross-validated r per cell per lag.
    subjects : (n_cells,) session/subject id per cell.
    unit     : 'cell'    -> one row per cell (rows are NOT independent).
               'subject' -> average across that session's cells, one row
                            per session, every session weighted equally.
    fisher   : transform before averaging. Averaging in z-space is the
               correct order of operations for the t-test.
    """
    curves = np.asarray(curves, dtype=float)
    if fisher:
        curves = fisher_z(curves)
    if unit == 'cell':
        return curves
    if unit != 'subject':
        raise ValueError(f"unit must be 'cell' or 'subject', got {unit!r}")
    subjects = np.asarray(subjects)
    return np.stack([np.nanmean(curves[subjects == s], axis=0)
                     for s in np.unique(subjects)])


def _t_gt0(values):
    """One-sample t (>0) per column, plus the two-sided p."""
    n_lags = values.shape[1]
    t = np.full(n_lags, np.nan)
    p_one = np.full(n_lags, np.nan)
    p_two = np.full(n_lags, np.nan)
    n_val = np.zeros(n_lags, dtype=int)
    for j in range(n_lags):
        x = values[:, j]
        x = x[np.isfinite(x)]
        n_val[j] = x.size
        if x.size >= 2:
            res = ttest_1samp(x, 0.0, alternative='greater')
            t[j] = float(res.statistic)
            p_one[j] = float(res.pvalue)
            p_two[j] = float(ttest_1samp(x, 0.0).pvalue)
    return t, p_one, p_two, n_val


def lagwise_tests_both_units(curves_by_roi, lags_deg, analysis_label=''):
    """Lag-wise one-sample tests at BOTH units of analysis, for every ROI.

    Parameters
    ----------
    curves_by_roi : {roi: (curves, subjects)} where
        curves   is (n_cells, n_lags) cross-validated r, and
        subjects is (n_cells,) session ids.
    lags_deg      : sequence of lag values, len == n_lags.
    analysis_label: free-text tag written into the `analysis` column
        (e.g. 'per_lag_encoding_noctrl', 'spatial_peaks_paired').

    Returns
    -------
    DataFrame, one row per (roi x unit x lag), with raw p plus two FDR
    corrections: across the lags within an ROI+unit, and across every
    ROI x lag within a unit.
    """
    lags_deg = list(lags_deg)
    rows = []
    for roi, (curves, subjects) in curves_by_roi.items():
        curves = np.asarray(curves, dtype=float)
        if curves.size == 0:
            continue
        subjects = np.asarray(subjects)
        for unit in ('cell', 'subject'):
            z = aggregate_curves(curves, subjects, unit=unit, fisher=True)
            raw = aggregate_curves(curves, subjects, unit=unit, fisher=False)
            t, p_one, p_two, n_val = _t_gt0(z)
            for j, lag in enumerate(lags_deg):
                rows.append({
                    'analysis':      analysis_label,
                    'roi':           roi,
                    'analysis_unit': unit,
                    'lag_deg':       lag,
                    'n_cells':       int(curves.shape[0]),
                    'n_subjects':    int(np.unique(subjects).size),
                    'n_units_valid': int(n_val[j]),
                    'df':            int(n_val[j] - 1) if n_val[j] else np.nan,
                    'mean_raw_r':    float(np.nanmean(raw[:, j])),
                    'mean_fisher_z': float(np.nanmean(z[:, j])),
                    't_fisher_z':    t[j],
                    'p_one_sided':   p_one[j],
                    'p_two_sided':   p_two[j],
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # FDR across the lags of one ROI x unit.
    out['p_one_sided_fdr_lags'] = np.nan
    out['p_two_sided_fdr_lags'] = np.nan
    for (_roi, _unit), idx in out.groupby(['roi', 'analysis_unit']).groups.items():
        idx = list(idx)
        out.loc[idx, 'p_one_sided_fdr_lags'] = bh_fdr(
            out.loc[idx, 'p_one_sided'].to_numpy())
        out.loc[idx, 'p_two_sided_fdr_lags'] = bh_fdr(
            out.loc[idx, 'p_two_sided'].to_numpy())
    # FDR across every ROI x lag within a unit.
    out['p_one_sided_fdr_rois_lags'] = np.nan
    out['p_two_sided_fdr_rois_lags'] = np.nan
    for _unit, idx in out.groupby('analysis_unit').groups.items():
        idx = list(idx)
        out.loc[idx, 'p_one_sided_fdr_rois_lags'] = bh_fdr(
            out.loc[idx, 'p_one_sided'].to_numpy())
        out.loc[idx, 'p_two_sided_fdr_rois_lags'] = bh_fdr(
            out.loc[idx, 'p_two_sided'].to_numpy())
    return out.sort_values(['roi', 'analysis_unit', 'lag_deg']).reset_index(drop=True)


def target_window_tests_both_units(curves_by_roi, lags_deg,
                                   target_lags_by_roi, analysis_label=''):
    """Predicted-lag tests at BOTH units.

    Two tests per (roi x unit), both on Fisher-z values:
      * target_mean_vs_zero  — mean over the ROI's predicted lags > 0.
      * target_vs_other_lags — within each unit, the predicted-lag mean
        beats that same unit's mean over the remaining lags (paired).

    `target_lags_by_roi` maps roi -> iterable of predicted lags; ROIs
    absent from it are skipped (no a-priori prediction).
    """
    lags_arr = np.asarray(list(lags_deg))
    rows = []
    for roi, (curves, subjects) in curves_by_roi.items():
        targets = target_lags_by_roi.get(roi)
        if not targets:
            continue
        curves = np.asarray(curves, dtype=float)
        if curves.size == 0:
            continue
        subjects = np.asarray(subjects)
        tmask = np.isin(lags_arr, list(targets))
        if tmask.sum() == 0 or (~tmask).sum() == 0:
            continue
        for unit in ('cell', 'subject'):
            z = aggregate_curves(curves, subjects, unit=unit, fisher=True)
            r_t = np.nanmean(z[:, tmask], axis=1)
            r_o = np.nanmean(z[:, ~tmask], axis=1)
            for test_name, vals in (('target_mean_vs_zero', r_t),
                                    ('target_vs_other_lags', r_t - r_o)):
                v = vals[np.isfinite(vals)]
                if v.size < 2:
                    continue
                res = ttest_1samp(v, 0.0, alternative='greater')
                rows.append({
                    'analysis':        analysis_label,
                    'roi':             roi,
                    'analysis_unit':   unit,
                    'target_lags_deg': '+'.join(str(int(t)) for t in targets),
                    'test':            test_name,
                    'n_cells':         int(curves.shape[0]),
                    'n_subjects':      int(np.unique(subjects).size),
                    'n_units_valid':   int(v.size),
                    'df':              int(v.size - 1),
                    'mean_fisher_z':   float(np.mean(v)),
                    't':               float(res.statistic),
                    'p_one_sided':     float(res.pvalue),
                    'p_two_sided':     float(ttest_1samp(v, 0.0).pvalue),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['p_one_sided_fdr_rois'] = np.nan
    out['p_two_sided_fdr_rois'] = np.nan
    for (_unit, _test), idx in out.groupby(['analysis_unit', 'test']).groups.items():
        idx = list(idx)
        out.loc[idx, 'p_one_sided_fdr_rois'] = bh_fdr(
            out.loc[idx, 'p_one_sided'].to_numpy())
        out.loc[idx, 'p_two_sided_fdr_rois'] = bh_fdr(
            out.loc[idx, 'p_two_sided'].to_numpy())
    return out.reset_index(drop=True)
