#!/usr/bin/env python3
"""
Cross-phase state generalisation in human single units.
=======================================================

NEW SCRIPT — does not modify or import from `encoding_state_sustained_cv.py`.
Reads only:
    <DATA_DIR>/all_neurons_avg_per_config.csv       (neuron x config x 360 bins)
    <DATA_DIR>/all_location_snippets.csv            (behavioural events)
    <RSA_RESULTS_CSV>                               (ROI assignment only)
Writes only into OUT_BASE/<timestamp>/ (nothing else in the data tree is touched).

--------------------------------------------------------------------------
WHY THIS ANALYSIS
--------------------------------------------------------------------------
`encoding_state_sustained_cv.py` defines state as `bin // 90`, i.e. the
quarter of the (time-warped) trial. That makes "state" almost collinear with
elapsed time in the trial: for 73.5% of cells a smooth cubic in trial time
explains >= 80% of the variance that the 4-state step model explains. Any cell
with slow trial-locked structure therefore reads as state-selective, which is a
plausible reason "state" appears strong in every region.

The fMRI analysis solved the equivalent problem by forcing the state model to
account for two behaviourally and visually different periods that share only
their sequence index ('path'/walking and 'reward'/consuming). This script is the
single-unit analogue of exactly that contrast.

For each neuron we ask:

    Does the cell prefer the same position in the sequence while WALKING
    as it does while CONSUMING REWARD?

Operationally, per neuron:
  1. Split its configs into two halves.
  2. From half 1, estimate a 4-vector of state tuning using PATH bins only.
  3. From half 2, estimate a 4-vector of state tuning using REWARD bins only.
  4. r_cross = Pearson r between those two 4-vectors (z-scored).
     Both split directions and many random splits are averaged.
  5. r_within = same, but path-with-path and reward-with-reward. This is the
     split-half reliability, i.e. the ceiling r_cross could attain.

r_cross is cross-validated over configs AND across the behavioural phase, so it
cannot be produced by within-period noise correlations.

Why this defeats the time-in-trial confound: a cell with a smooth bump somewhere
in the trial has high apparent state selectivity, but its bump does not hold the
same *rank ordering over the four states* when measured in the reward period as
in the path period, beyond what the temporal autocorrelation alone predicts. The
circular-shift null preserves each cell's full autocorrelation (so drift, bumps
and adaptation are all retained) while destroying alignment to the state
boundaries, so it is the correct reference here.

ABSTRACTNESS INDEX
    a = r_cross / r_within
An abstract state code generalises across the path/reward boundary at close to
its own reliability (a -> 1). A code driven by period-specific sensory or motor
correlates has r_within > 0 but r_cross -> 0.

--------------------------------------------------------------------------
IMPORTANT DATA NOTE (verified before writing this script)
--------------------------------------------------------------------------
In `all_location_snippets.csv`, `onset_bin` IS in the warped 360-bin coordinate
(every reward onset falls at within-state bin 30-79, median 61), but
`run_length` is NOT in that coordinate (event onsets never overrun the next
onset, yet run_lengths sum to ~395 per trial rather than 360). This script
therefore derives the path/reward boundary from `onset_bin` ONLY and never uses
`run_length`. Two independent checks confirm the 360-bin trace is warped to
exactly 90 bins per state: the 4-cycles/trial Fourier component is the strongest
of all 180 harmonics in the population profile, and a state label map built from
run_length predicts firing significantly WORSE than `bin // 90`.

--------------------------------------------------------------------------
ROBUSTNESS FACTORS (all four combinations are run)
--------------------------------------------------------------------------
LABEL_MODE  'per_config' : boundary = that config's own median reward onset.
            'fixed'      : boundary = within-state bin 60 for every config.
DETREND     'none'       : raw per-config profiles.
            'cubic'      : a cubic polynomial in trial time is regressed out of
                           each config profile first. This removes the smooth
                           temporal component explicitly rather than relying on
                           the null to discount it.
"""
import os, json, argparse, datetime
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUT_BASE = os.path.join(DATA_DIR, "group", "state_crossphase_generalisation")
RSA_RESULTS_CSV = os.path.join(
    DATA_DIR, "group", "encoding_state_sustained_cv",
    "2026-06-25_14-38-13_relabelled_2026-07-29_15-05-10",
    "state_sustained_cv_results.csv")

N_BINS = 360
N_STATES = 4
STATE_LEN = N_BINS // N_STATES          # 90
FIXED_BOUNDARY = 60                     # within-state bin, 'fixed' mode
MIN_CONFIGS = 4                         # need >=2 per half
MIN_BINS_PER_CELL = 4                   # per state x phase cell of the design
N_SPLITS = 40                           # random config half-splits, averaged
N_PERM = 1000
SIG_ALPHA = 0.05
ROI_ORDER = ["EC", "mOFC", "mPFC", "PHC", "PCC", "HC_anterior", "HC_mid"]

STATE_IDX = np.arange(N_BINS) // STATE_LEN
WS_IDX = np.arange(N_BINS) % STATE_LEN
ONEHOT = np.stack([(STATE_IDX == s).astype(float) for s in range(N_STATES)], 1)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_inputs(data_dir=DATA_DIR, rsa_csv=RSA_RESULTS_CSV):
    bins = [f"bin_{i:03d}" for i in range(N_BINS)]
    fr = pd.read_csv(os.path.join(data_dir, "all_neurons_avg_per_config.csv"))
    sn = pd.read_csv(os.path.join(data_dir, "all_location_snippets.csv"))
    roi_map = dict(zip(*(lambda d: (d.neuron, d.roi))(pd.read_csv(rsa_csv))))
    return fr, sn, bins, roi_map


def reward_onsets(sn):
    """Median reward onset per (session, config, slot) in within-state bins.

    Uses onset_bin only (see module docstring). Returns
    {(session, config): array of 4 boundaries} for configs with all 4 slots.
    """
    slot_i = {c: i for i, c in enumerate("ABCD")}
    rw = sn.drop_duplicates(["session", "config", "grid_no", "slot",
                             "phase", "onset_bin"])
    rw = rw[rw.phase == "reward"].copy()
    rw["ws"] = rw.onset_bin - STATE_LEN * rw.slot.map(slot_i)
    bad = ((rw.ws < 0) | (rw.ws >= STATE_LEN)).sum()
    if bad:
        raise ValueError(f"{bad} reward onsets outside their own state block — "
                         "onset_bin is not in the warped coordinate.")
    out = {}
    for (se, cf), d in rw.groupby(["session", "config"]):
        v = d.groupby("slot")["ws"].median()
        o = np.array([v.get(s, np.nan) for s in "ABCD"], float)
        if np.isfinite(o).all():
            out[(se, cf)] = np.clip(o, 5.0, STATE_LEN - 5.0)
    return out


# ---------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------
def phase_masks(boundary):
    """(path_mask, reward_mask) over 360 bins for a 4-vector of boundaries."""
    b = np.asarray(boundary, float)[STATE_IDX]
    rew = WS_IDX >= b
    return ~rew, rew


def cubic_detrend(Y):
    """Regress a cubic in trial time out of each row of Y (n, 360)."""
    t = (np.arange(N_BINS) - (N_BINS - 1) / 2) / ((N_BINS - 1) / 2)
    X = np.c_[np.ones(N_BINS), t, t ** 2, t ** 3]
    Yf = np.where(np.isfinite(Y), Y, 0.0)
    B = np.linalg.lstsq(X, Yf.T, rcond=None)[0]
    return np.where(np.isfinite(Y), Y - (X @ B).T, np.nan)


def _tuning(Y, mask):
    """State tuning per config.  Y (..., K, 360), mask (K, 360) -> (..., K, 4)."""
    Yv = np.where(np.isfinite(Y), Y, 0.0)
    ok = np.isfinite(Y) & mask
    num = np.einsum("...kb,bs->...ks", Yv * ok, ONEHOT)
    den = np.einsum("...kb,bs->...ks", ok.astype(float), ONEHOT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den >= MIN_BINS_PER_CELL, out, np.nan)


def _zc(V):
    """Z-score along the last axis (the 4 states)."""
    m = np.nanmean(V, -1, keepdims=True)
    s = np.nanstd(V, -1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 1e-12, (V - m) / s, np.nan)


def _corr4(A, B):
    """Pearson r between z-scored 4-vectors along the last axis."""
    return np.nanmean(_zc(A) * _zc(B), -1)


def crossphase_for_neuron(Y, boundaries, splits, n_perm, rng, detrend):
    """Return (r_cross, r_within, perm_r_cross) for one neuron.

    Y          (K, 360) per-config profiles
    boundaries (K, 4)   reward-onset boundary per config
    splits     list of (idx_half1, idx_half2)
    """
    K = Y.shape[0]
    if detrend == "cubic":
        Y = cubic_detrend(Y)
    Pm = np.zeros((K, N_BINS), bool)
    Rm = np.zeros((K, N_BINS), bool)
    for k in range(K):
        Pm[k], Rm[k] = phase_masks(boundaries[k])

    # observed
    Tp, Tr = _tuning(Y, Pm), _tuning(Y, Rm)

    def agg(T, idx):
        return np.nanmean(T[..., idx, :], -2)

    def stat(Tp_, Tr_):
        xs, ws = [], []
        for h1, h2 in splits:
            xs += [_corr4(agg(Tp_, h1), agg(Tr_, h2)),
                   _corr4(agg(Tp_, h2), agg(Tr_, h1))]
            ws += [_corr4(agg(Tp_, h1), agg(Tp_, h2)),
                   _corr4(agg(Tr_, h1), agg(Tr_, h2))]
        return np.nanmean(xs, 0), np.nanmean(ws, 0)

    r_cross, r_within = stat(Tp, Tr)

    # permutations: circular shift each config profile independently
    perm = np.full(n_perm, np.nan)
    if n_perm > 0:
        CH = 200
        got = []
        for s0 in range(0, n_perm, CH):
            n = min(CH, n_perm - s0)
            sh = rng.integers(0, N_BINS, size=(n, K))
            idx = (np.arange(N_BINS)[None, None, :] - sh[:, :, None]) % N_BINS
            Ysh = np.take_along_axis(np.broadcast_to(Y, (n, K, N_BINS)), idx, axis=2)
            c, _ = stat(_tuning(Ysh, Pm), _tuning(Ysh, Rm))
            got.append(np.atleast_1d(c))
        perm = np.concatenate(got)[:n_perm]
    return float(r_cross), float(r_within), perm


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def run(label_mode, detrend, fr, sn, bins, roi_map, rmap,
        n_perm=N_PERM, n_splits=N_SPLITS, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    Y_all = fr[bins].to_numpy(float)
    keys = list(zip(fr.session, fr.config))
    labels = fr.neuron_label.to_numpy()

    by_neuron = {}
    for i, (n, k) in enumerate(zip(labels, keys)):
        if k in rmap and n in roi_map and np.isfinite(Y_all[i]).sum() > 300:
            by_neuron.setdefault(n, []).append(i)

    rows = []
    for ni, (n, idxs) in enumerate(sorted(by_neuron.items())):
        if len(idxs) < MIN_CONFIGS:
            continue
        ii = np.array(idxs)
        Y = Y_all[ii]
        if label_mode == "per_config":
            B = np.array([rmap[keys[j]] for j in ii])
        else:
            B = np.full((len(ii), N_STATES), float(FIXED_BOUNDARY))
        K = len(ii)
        splits = []
        for _ in range(n_splits):
            p = rng.permutation(K)
            splits.append((p[:K // 2], p[K // 2:]))
        rc, rw, perm = crossphase_for_neuron(Y, B, splits, n_perm, rng, detrend)
        v = np.isfinite(perm)
        p = ((np.sum(perm[v] >= rc) + 1) / (v.sum() + 1)
             if v.any() and np.isfinite(rc) else np.nan)
        rows.append(dict(
            neuron=n, roi=roi_map[n], n_configs=K,
            r_cross=rc, r_within=rw,
            abstractness=(rc / rw if np.isfinite(rw) and rw > 0.02 else np.nan),
            p_perm=p, sig=bool(np.isfinite(p) and p < SIG_ALPHA),
            perm_mean=float(np.nanmean(perm)) if v.any() else np.nan,
            perm_sd=float(np.nanstd(perm)) if v.any() else np.nan,
            z_vs_null=((rc - np.nanmean(perm)) / np.nanstd(perm)
                       if v.any() and np.nanstd(perm) > 1e-12 else np.nan)))
        if verbose and (ni + 1) % 100 == 0:
            print(f"    {ni+1}/{len(by_neuron)}", flush=True)
    return pd.DataFrame(rows)


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pk = p[ok]
    o = np.argsort(pk)
    n = pk.size
    r = pk[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(n)
    qq[o] = np.clip(r, 0, 1)
    q[ok] = qq
    return q


def summarise(df):
    df = df.copy()
    df["q_fdr"] = bh_fdr(df.p_perm)
    df["sig_fdr"] = df.q_fdr < SIG_ALPHA
    out = []
    for roi, d in df.groupby("roi"):
        rc = d.r_cross.dropna()
        t = stats.ttest_1samp(rc, 0) if len(rc) > 2 else None
        z = d.z_vs_null.dropna()
        tz = stats.ttest_1samp(z, 0) if len(z) > 2 else None
        out.append(dict(
            roi=roi, n_cells=len(d),
            mean_r_cross=rc.mean(), sem_r_cross=rc.sem(),
            t_r_cross=t.statistic if t else np.nan,
            p_r_cross=t.pvalue if t else np.nan,
            mean_r_within=d.r_within.mean(),
            mean_abstractness=d.abstractness.median(),
            mean_z_vs_null=z.mean(),
            t_z_vs_null=tz.statistic if tz else np.nan,
            p_z_vs_null=tz.pvalue if tz else np.nan,
            frac_sig=d.sig.mean(), n_sig=int(d.sig.sum()),
            n_sig_fdr=int(d.sig_fdr.sum())))
    s = pd.DataFrame(out)
    s["p_r_cross_fdr"] = bh_fdr(s.p_r_cross)
    order = {r: i for i, r in enumerate(ROI_ORDER)}
    return df, s.sort_values("roi", key=lambda c: c.map(lambda r: order.get(r, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--modes", default="per_config,fixed")
    ap.add_argument("--detrend", default="none,cubic")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = a.out or os.path.join(OUT_BASE, stamp)
    os.makedirs(out_dir, exist_ok=True)

    fr, sn, bins, roi_map = load_inputs()
    rmap = reward_onsets(sn)
    print(f"neurons with ROI: {len(roi_map)} | configs with reward onsets: {len(rmap)}")

    summaries = {}
    for lm in a.modes.split(","):
        for dt in a.detrend.split(","):
            tag = f"{lm}_{dt}"
            print(f"\n=== {tag} ===", flush=True)
            df = run(lm, dt, fr, sn, bins, roi_map, rmap,
                     n_perm=a.n_perm, n_splits=a.n_splits)
            df, s = summarise(df)
            df.to_csv(os.path.join(out_dir, f"crossphase_cells_{tag}.csv"), index=False)
            s.to_csv(os.path.join(out_dir, f"crossphase_roi_{tag}.csv"), index=False)
            summaries[tag] = s
            print(s[["roi", "n_cells", "mean_r_cross", "t_r_cross", "p_r_cross",
                     "mean_abstractness", "frac_sig", "n_sig_fdr"]]
                  .round(4).to_string(index=False))

    json.dump(dict(n_perm=a.n_perm, n_splits=a.n_splits, modes=a.modes,
                   detrend=a.detrend, fixed_boundary=FIXED_BOUNDARY,
                   min_configs=MIN_CONFIGS, sig_alpha=SIG_ALPHA,
                   rsa_results_csv=RSA_RESULTS_CSV),
              open(os.path.join(out_dir, "settings.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
