#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 statistics: count GLM with cluster-robust SEs, and a circular-shift
permutation test on the artifact-free time axis.

Model. Ripple counts are counts, so a Poisson (or negative-binomial) GLM with
`log(artifact-free exposure)` as an **offset** is the right likelihood --
not a t-test on rate ratios, which is unstable when windows are short and
throws away the count structure.

Clustering. The unit of observation is the session, but the unit of
independence is the **patient**: two sessions from one patient are the same
electrode in the same tissue with the same pathology. That differs from the
cell analyses, where different neurons are recorded each day and treating days
as independent is defensible. `is_first_session` and
`session_index_within_subject` are carried as covariates so the task-familiarity
question is still asked, while the SEs cluster on `subject_key`.

Inference. Cluster-robust SEs are reported, but the **primary** inference is a
circular-shift permutation, because:
  * it makes no distributional assumption about ripple timing;
  * shifting on the artifact-free axis means a shifted null can never place an
    event inside an artifact -- which would otherwise make the null easier to
    beat than the data;
  * `shift = 0` is row 0 of the shift table, so observed and permuted values go
    through an identical code path (CLAUDE.md rule 4).

The shift-table idiom (row 0 = observed) follows
`mc/analyse/future_spatial_peaks.py:569`.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

import mc.analyse.swr_windows as win

SEED = 42


# =============================================================================
# GLM
# =============================================================================

def fit_count_glm(tbl, formula, cluster_col="subject_key", family="poisson",
                  exposure_col="exposure_s"):
    """Poisson/NB GLM on counts with log-exposure offset and cluster-robust SEs.

    Returns a dict with the coefficient table, the Pearson dispersion, and a
    flag telling the caller whether to refit as negative binomial.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = tbl[tbl[exposure_col] > 0].copy()
    d["_offset"] = np.log(d[exposure_col].to_numpy(float))
    d = d.dropna(subset=[cluster_col])

    fam = sm.families.Poisson() if family == "poisson" else \
        sm.families.NegativeBinomial()
    m = smf.glm(formula, data=d, family=fam, offset=d["_offset"])
    res = m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})

    resid_p = res.resid_pearson
    dispersion = float(np.sum(resid_p ** 2) / res.df_resid) if res.df_resid else np.nan

    out = pd.DataFrame({
        "coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues,
        "ci_lo": res.conf_int()[0], "ci_hi": res.conf_int()[1],
    })
    out["rate_ratio"] = np.exp(out.coef)
    return {
        "table": out, "dispersion": dispersion, "n_obs": int(res.nobs),
        "n_clusters": int(d[cluster_col].nunique()),
        "overdispersed": bool(np.isfinite(dispersion) and dispersion > 1.5),
        "family": family, "formula": formula, "result": res,
    }


# =============================================================================
# CONTRASTS
# =============================================================================

def rate_by_condition(counts, groupby=("state", "discovery")):
    """Per-derivation rates averaged across derivations.

    Each derivation contributes equally; pooling raw events would let long or
    ripple-rich derivations dominate.
    """
    per = (counts.groupby(["session", "pair_id", *groupby])
                 .agg(n=("n_ripples", "sum"), exp=("exposure_s", "sum"))
                 .reset_index())
    per = per[per.exp > 0]
    per["rate"] = per.n / per.exp
    return per


def interaction_contrast(counts, state="D", groupby=("state", "discovery")):
    """(first - later) at `state`, minus the mean (first - later) at the others.

    This is the statistic the hypothesis actually predicts: an elevation
    specific to the first uncovering of D, over and above any general
    first-vs-later difference. A plain D_first - D_later would also be produced
    by a global novelty effect.
    """
    per = rate_by_condition(counts, groupby)
    w = per.pivot_table(index=["session", "pair_id", "state"],
                        columns="discovery", values="rate")
    if "first" not in w or "later" not in w:
        return np.nan
    w = w.dropna(subset=["first", "later"])
    diff = (w["first"] - w["later"]).groupby("state").mean()
    if state not in diff.index:
        return np.nan
    others = diff.drop(index=state)
    return float(diff[state] - (others.mean() if len(others) else 0.0))


def simple_contrast(counts, state="D", groupby=("state", "discovery")):
    """Plain (first - later) at `state`."""
    per = rate_by_condition(counts, groupby)
    w = per.pivot_table(index=["session", "pair_id", "state"],
                        columns="discovery", values="rate")
    if "first" not in w or "later" not in w:
        return np.nan
    w = w.dropna(subset=["first", "later"])
    diff = (w["first"] - w["later"]).groupby("state").mean()
    return float(diff.get(state, np.nan))


# =============================================================================
# CIRCULAR-SHIFT PERMUTATION
# =============================================================================

def shift_table(n_perms, n_units, rng, max_shift_s):
    """(1 + n_perms, n_units) shifts, **row 0 = zeros = observed**.

    Row 0 being the observed case is what guarantees empirical and null values
    are computed by the identical code path.
    """
    s = np.zeros((n_perms + 1, n_units))
    s[1:] = rng.uniform(0, 1, size=(n_perms, n_units)) * max_shift_s
    return s


def circular_shift_test(per_pair, windows_by_session, contrast_fn,
                        n_perms=1000, seed=SEED, scope="session", verbose=False):
    """Permutation test by circularly shifting event trains on the clean axis.

    `per_pair` : list of dicts with keys
        session, pair_id, events (1-D array of peak times), intervals (n,2),
        meta (dict of columns to attach)
    `windows_by_session` : {session: windows DataFrame}
    `contrast_fn` : counts DataFrame -> scalar statistic

    scope='session' shifts every derivation of a session by the same amount,
    preserving within-session co-occurrence between derivations (primary).
    scope='contact' shifts each derivation independently (secondary).
    """
    rng = np.random.default_rng(seed)
    units = sorted({p["session"] for p in per_pair}) if scope == "session" \
        else [(p["session"], p["pair_id"]) for p in per_pair]
    idx_of = {u: i for i, u in enumerate(units)}

    # shift range = the clean duration of each unit, so a shift wraps within
    # that recording's own artifact-free timeline
    max_shift = float(max(
        (np.diff(np.asarray(p["intervals"], float).reshape(-1, 2), axis=1).sum()
         for p in per_pair), default=0.0))

    shifts = shift_table(n_perms, len(units), rng, max_shift)

    stats = np.full(n_perms + 1, np.nan)
    for r in range(n_perms + 1):
        parts = []
        for p in per_pair:
            u = p["session"] if scope == "session" else (p["session"], p["pair_id"])
            sh = shifts[r, idx_of[u]]
            w = windows_by_session.get(p["session"])
            if w is None or not len(w):
                continue
            a = win.assign_events_to_windows(p["events"], w, p["intervals"],
                                             shift_s=sh)
            a["session"] = p["session"]
            a["pair_id"] = p["pair_id"]
            for k, v in p.get("meta", {}).items():
                a[k] = v
            parts.append(a)
        if not parts:
            continue
        stats[r] = contrast_fn(pd.concat(parts, ignore_index=True))
        if verbose and r % 100 == 0:
            print(f"    shift {r}/{n_perms}", flush=True)

    observed = stats[0]
    null = stats[1:]
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(observed):
        return {"observed": observed, "p_one_tailed": np.nan, "n_perms": 0}

    p_hi = (1 + np.sum(null >= observed)) / (len(null) + 1)
    p_two = (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
        "z_vs_null": float((observed - np.mean(null)) / np.std(null))
        if np.std(null) > 0 else np.nan,
        "p_one_tailed": float(p_hi), "p_two_tailed": float(p_two),
        "n_perms": int(len(null)), "scope": scope, "null": null,
    }
