#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What do the ripples mean? Seven hypotheses, one script.

Replaces `swr_build_windows.py` + `swr_h1_stats.py`, which split window building
from testing across two scripts and one intermediate file for no benefit.

The hypotheses, in the order of how directly they follow from the fMRI result
(mPFC action-plan representation peaking exactly when the fourth reward D is
revealed) and how well the confounds can be controlled:

  H1  PRIMARY, pre-declared. Ripple rate is elevated at the FIRST arrival at D
      in a grid -- the moment the route first becomes knowable -- relative to
      every later arrival at D in the same grid.
      Contrast: first_D vs later_D.
      Why this one is primary: it is a single contrast fixed in advance, it
      matches the fMRI timing, and first and later D are the same event
      (uncovering the fourth reward, positive feedback, end of the traversal)
      differing only in whether the plan was already known.

  H2  Planning vs execution. Ripples are elevated in the quiet gaps during the
      phase before the grid is first solved, relative to after.
      Contrast: phase (discovery/later) on PAUSE windows.
      Weaker: the two phases differ in movement, error rate and duration. Pause
      windows and the movement covariate control the first two; the exposure
      offset controls the third.

  H3  Ripples predict subsequent performance. The ripple rate in the pause after
      first-D predicts how well the subject then executes the grid.
      Test: rate -> errors in the following repeats. Unit = grid.
      This is the only directional test here: a neural measure predicting later
      behaviour cannot be explained by the behaviour causing the neural measure.

  H4  Feedback. Ripple rate differs after incorrect vs correct uncovering.
      Contrast: error vs correct, 1 s windows truncated at the next uncovering.
      Now testable per press: `swr_behaviour.uncover_events` reconstructs every
      uncovering and its outcome in session seconds (validated at 99.8% of
      correct uncoverings against the stimulus PC's own log).

  H5  Explore vs plan vs execute, as three phases rather than two. "Exploration"
      in `add_phase` merges finding rewards whose identity is unknown with
      knowing all four and not yet executing them reliably; `add_phase3`
      separates them.
      Contrast: plan vs execute, on pause windows.

  H6  Is D special because it completes the plan, or just because it is D?
      Contrast: the INTERACTION (D - mean(A,B,C)) on the discovery traversal
      minus the same at later traversals, on the 4x2 state x discovery table.
      This is the one that discriminates the hypothesis from its alternative:
      the hypothesis predicts an interaction, "D is special for another reason"
      predicts a main effect of state with no interaction.

  H7  Feedback that could change behaviour vs feedback that could not. While the
      rewards are being discovered a CORRECT uncovering is informative; once the
      route is known it is an ERROR that carries information. The diagonal of
      the same feedback x phase table as H4, asked separately because the main
      effect and this interaction are different questions.

Multiple comparisons: H1 is pre-declared primary and is NOT corrected. H2-H4 are
secondary and are FDR-corrected across themselves, in line with the rest of the
project. Every design is also reported without correction so nothing is hidden.

Confound control, applied to every design:
  - counts, not rates, with log(artifact-free seconds) as a GLM offset, so
    windows of different length are compared correctly;
  - the movement-key count inside each window as a covariate, because the
    conditions differ in how much the subject was moving and ripple rate is
    movement-sensitive;
  - inference by circular shift of the event train on the artifact-free time
    axis, which preserves each session's ripple autocorrelation and the window
    structure. The observed value goes through the identical code path as the
    permutations (row 0 of the shift table).

Usage
-----
    conda activate env_multiple_clocks
    python scripts/swr_hypotheses.py run                     # all seven
    python scripts/swr_hypotheses.py run --which=H6          # one
    python scripts/swr_hypotheses.py run --n_perms=5000      # publication run
    python scripts/swr_hypotheses.py export                  # portable bundle

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_windows as win
import mc.analyse.swr_stats as sst
import mc.analyse.swr_behaviour as swb
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"
N_PERMS = 1000
SEED = 42

# Window lengths. Short and equal within a design, so the contrast is never
# between windows of different length or different behavioural content.
LOCK_D_S = 2.0        # H1: after arrival at D
LOCK_FB_S = 1.0       # H4: after feedback, truncated at the next uncovering
MIN_PAUSE_S = 0.5     # H2/H3: a pause shorter than this is not a rest period

PRIMARY = "H1"
SECONDARY = ("H2", "H3", "H4", "H5", "H6", "H7")


def _settings(which, n_perms, analysis_name):
    return {"analysis": "swr_hypotheses", "which": which, "n_perms": n_perms,
            "seed": SEED, "analysis_name": analysis_name,
            "lock_D_s": LOCK_D_S, "lock_feedback_s": LOCK_FB_S,
            "min_pause_s": MIN_PAUSE_S, "primary": PRIMARY,
            "secondary_fdr_over": list(SECONDARY),
            "created": datetime.now().isoformat(timespec="seconds")}


# --------------------------------------------------------------- loading ----
def _sessions_with_events(analysis_name, R):
    out = []
    for p in sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*",
                                           "LFP-ripples", analysis_name,
                                           "ripple_events.csv"))):
        out.append((int(p.split(os.sep)[-4][1:]), os.path.dirname(p)))
    return out


def _subject_map(R):
    m = pd.read_csv(os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                                 "session_manifest.csv"))
    return m.set_index("session")[["subject_key", "recording_site"]].to_dict("index")


def _count_windows(sess, rip_dir, windows, beh, R, subj):
    """Ripple counts per (derivation x window), with exposure and movement."""
    ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
    ev = ev[ev.passed.fillna(False)]
    iv_all = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
    qc = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv")).set_index("pair_id")
    if not len(windows) or not len(ev):
        return pd.DataFrame()

    mv = swb.presses_in_windows(sess, beh, windows, data_root=R)
    rows = []
    for pair_id, e in ev.groupby("pair_id"):
        if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
            continue
        iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
        a = win.assign_events_to_windows(e.t_peak_s.to_numpy(float), windows, iv)
        a = a.reset_index(drop=True)
        a["session"] = sess
        a["pair_id"] = pair_id
        a["pair_roi"] = qc.loc[pair_id, "pair_roi"] if pair_id in qc.index else None
        a["n_moves"] = mv
        meta = subj.get(sess, {})
        a["subject_key"] = meta.get("subject_key")
        a["recording_site"] = meta.get("recording_site")
        rows.append(a)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _gather(design_fn, analysis_name, R, subj, needs_uncover=False):
    """Build one design across every session that has detection output."""
    parts, per_session = [], {}
    for sess, rip_dir in _sessions_with_events(analysis_name, R):
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: behaviour unreadable ({e})"); continue
        try:
            if needs_uncover:
                u = swb.uncover_events(sess, beh=beh, data_root=R)
                w = design_fn(u) if len(u) else pd.DataFrame()
            else:
                w = design_fn(beh)
        except Exception as e:
            print(f"  s{sess:02d}: window build failed ({type(e).__name__}: {e})")
            continue
        if not len(w):
            continue
        c = _count_windows(sess, rip_dir, w, beh, R, subj)
        if len(c):
            parts.append(c)
            per_session[sess] = w
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
            per_session)


# ------------------------------------------------------------- contrasts ----
def _interaction_D_vs_ABC(counts):
    """(D - mean(A,B,C)) during discovery, minus the same at later traversals.

    The 4x2 table separates SK's hypothesis from its obvious alternative:
      hypothesis   -> D is special BECAUSE it completes the plan, so the D
                      advantage exists on the first traversal and not later
                      => interaction
      alternative  -> D is special for another reason (it is last, it ends the
                      traversal), so the D advantage is there every time
                      => main effect of state, no interaction
    """
    need = {"state", "discovery"}
    if not need.issubset(counts.columns):
        return np.nan
    g = (counts.groupby(["discovery", "state"])
         .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
    g["r"] = g["n"] / g["e"].replace(0, np.nan)
    out = {}
    for disc in ("first", "later"):
        if disc not in g.index.get_level_values(0):
            return np.nan
        sub = g.loc[disc]
        if "D" not in sub.index:
            return np.nan
        abc = [x for x in ("A", "B", "C") if x in sub.index]
        if not abc:
            return np.nan
        rd, ra = sub.loc["D", "r"], float(np.mean(sub.loc[abc, "r"]))
        if not (np.isfinite(rd) and np.isfinite(ra)) or rd <= 0 or ra <= 0:
            return np.nan
        out[disc] = np.log(rd) - np.log(ra)
    return float(out["first"] - out["later"])


def _log_rate_diff(counts, col, hi, lo):
    """log rate ratio between two levels of `col`, pooled over derivations.

    Pooled as summed counts over summed exposure rather than a mean of per-window
    rates: a mean of rates weights a 0.5 s window as heavily as a 5 s one and is
    dominated by the short ones.
    """
    g = counts.groupby(col).agg(n=("n_ripples", "sum"), e=("exposure_s", "sum"))
    if hi not in g.index or lo not in g.index:
        return np.nan
    if g.loc[lo, "e"] <= 0 or g.loc[hi, "e"] <= 0:
        return np.nan
    r_hi = g.loc[hi, "n"] / g.loc[hi, "e"]
    r_lo = g.loc[lo, "n"] / g.loc[lo, "e"]
    if r_lo <= 0 or r_hi <= 0:
        return np.nan
    return float(np.log(r_hi) - np.log(r_lo))


def _glm(counts, formula, label):
    try:
        res = sst.fit_count_glm(counts, formula)
    except Exception as e:
        print(f"    GLM failed for {label}: {type(e).__name__}: {e}")
        return None
    return res


def _perm_p(observed, null, one_sided=True):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not null.size or not np.isfinite(observed):
        return np.nan, np.nan, np.nan
    if one_sided:
        p = (1 + np.sum(null >= observed)) / (1 + null.size)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + null.size)
    z = (observed - null.mean()) / (null.std() + 1e-12)
    return float(p), float(z), float(null.std())


def _bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.sum():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv)); out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ------------------------------------------------------------------ H1-H4 ---
def _shifted_counts_factory(analysis_name, R, subj, per_session, max_shift_frac=1.0):
    """Return a function(rng) -> counts table with each session's events shifted.

    The shift is circular on the artifact-free axis, so a shifted event can never
    land inside an artifact and every session keeps its own ripple
    autocorrelation and its own number of events.
    """
    cache = {}
    for sess, w in per_session.items():
        rip_dir = os.path.join(swr_io.session_deriv_dir(sess, R), "LFP-ripples",
                               analysis_name)
        ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
        ev = ev[ev.passed.fillna(False)]
        iv_all = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv")).set_index("pair_id")
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception:
            continue
        mv = swb.presses_in_windows(sess, beh, w, data_root=R)
        cache[sess] = (ev, iv_all, qc, w, mv)

    def make(rng):
        rows = []
        for sess, (ev, iv_all, qc, w, mv) in cache.items():
            span = float(w.end_s.max() - w.start_s.min()) or 1.0
            for pair_id, e in ev.groupby("pair_id"):
                if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                    continue
                iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                if not len(iv):
                    continue
                t = e.t_peak_s.to_numpy(float)
                lo, hi = iv[:, 0].min(), iv[:, 1].max()
                L = hi - lo
                if L <= 0:
                    continue
                sh = rng.uniform(0.05 * L, 0.95 * L)
                t_sh = lo + np.mod(t - lo + sh, L)
                a = win.assign_events_to_windows(t_sh, w, iv).reset_index(drop=True)
                a["session"] = sess
                a["pair_id"] = pair_id
                a["n_moves"] = mv
                meta = subj.get(sess, {})
                a["subject_key"] = meta.get("subject_key")
                rows.append(a)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return make


def _run_window_hypothesis(name, question, design_fn, contrast, formula,
                           analysis_name, R, subj, n_perms, needs_uncover=False,
                           out_dir=None):
    print("\n" + "=" * 74)
    print(f" {name}: {question}")
    print("=" * 74)
    counts, per_session = _gather(design_fn, analysis_name, R, subj,
                                  needs_uncover=needs_uncover)
    if not len(counts):
        print("  no windows built"); return None
    print(f"  {len(counts)} window-derivation rows, "
          f"{counts.session.nunique()} sessions, "
          f"{counts.subject_key.nunique()} subjects")

    obs = contrast(counts)
    print(f"  observed log rate ratio: {obs:+.4f}"
          f"  ({100*(np.exp(obs)-1):+.1f}% rate change)" if np.isfinite(obs)
          else "  observed: undefined")

    res = _glm(counts, formula, name)
    if res is not None and "table" in res:
        tb = res["table"]
        print(f"  GLM ({'NB' if res.get('use_nb') else 'Poisson'}, "
              f"dispersion {res.get('dispersion', np.nan):.2f}, "
              f"cluster-robust by subject):")
        print("   " + tb.round(4).to_string().replace("\n", "\n   "))

    make = _shifted_counts_factory(analysis_name, R, subj, per_session)
    null = np.array([contrast(make(np.random.RandomState(SEED + k)))
                     for k in range(n_perms)])
    p, z, sd = _perm_p(obs, null, one_sided=True)
    # The null is not centred on zero: a log ratio of small counts is biased
    # downward (Jensen), and first-D windows are few. That is precisely why
    # inference is against the shifted null and not against zero -- the null
    # carries the identical bias because it is built by the identical code.
    print(f"  permutation ({n_perms} circular shifts): null {np.nanmean(null):+.4f}"
          f" +- {sd:.4f}, z = {z:+.2f}, p_one_sided = {p:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        counts.to_csv(os.path.join(out_dir, f"{name}_counts.csv"), index=False)
        np.save(os.path.join(out_dir, f"{name}_null.npy"), null)
    return {"hypothesis": name, "question": question, "observed_log_rr": obs,
            "null": null,
            "rate_change_pct": 100 * (np.exp(obs) - 1) if np.isfinite(obs) else np.nan,
            "null_mean": float(np.nanmean(null)), "null_sd": sd, "z": z,
            "p_perm": p, "n_perms": n_perms,
            "n_rows": int(len(counts)), "n_sessions": int(counts.session.nunique()),
            "n_subjects": int(counts.subject_key.nunique()),
            "counts": counts}


def _run_H3(analysis_name, R, subj, n_perms, out_dir=None):
    """Does the ripple rate after first-D predict subsequent performance?"""
    name = "H3"
    question = ("ripple rate in the pause after first-D predicts errors in the "
                "following repeats")
    print("\n" + "=" * 74); print(f" {name}: {question}"); print("=" * 74)

    rows = []
    for sess, rip_dir in _sessions_with_events(analysis_name, R):
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
            rt = swb.repeat_table(sess, beh=beh, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: {type(e).__name__}: {e}"); continue
        if not len(rt):
            continue
        ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
        ev = ev[ev.passed.fillna(False)]
        iv_all = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv")).set_index("pair_id")

        for grid, g in rt.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            disc = g[g.is_discovery == 1]
            later = g[g.is_discovery == 0]
            if not len(disc) or not len(later):
                continue
            t0 = float(disc.t_D.iloc[0])
            t1 = float(later.t_start.iloc[0])
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < MIN_PAUSE_S:
                continue
            # through _finalise so it carries duration_s like every other design
            w = win._finalise(pd.DataFrame([{
                "start_s": t0, "end_s": t1, "grid_no": int(grid),
                "condition": "post_first_D", "is_test": True}]))
            for pair_id, e in ev.groupby("pair_id"):
                if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                    continue
                iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                a = win.assign_events_to_windows(e.t_peak_s.to_numpy(float), w, iv)
                if not len(a) or float(a.exposure_s.iloc[0]) <= 0:
                    continue
                rows.append({
                    "session": sess, "pair_id": pair_id, "grid_no": int(grid),
                    "subject_key": subj.get(sess, {}).get("subject_key"),
                    "n_ripples": float(a.n_ripples.iloc[0]),
                    "exposure_s": float(a.exposure_s.iloc[0]),
                    "rate_hz": float(a.n_ripples.iloc[0]) / float(a.exposure_s.iloc[0]),
                    "pause_s": t1 - t0,
                    "errors_after": float(later.n_errors.sum()),
                    "errors_per_repeat_after": float(later.n_errors.mean()),
                    "n_repeats_after": int(len(later)),
                    "errors_in_discovery": float(disc.n_errors.iloc[0]),
                })
    tab = pd.DataFrame(rows)
    if len(tab) < 10:
        print(f"  only {len(tab)} grids with a usable post-first-D pause"); return None
    print(f"  {len(tab)} grid-derivations, {tab.session.nunique()} sessions, "
          f"{tab.subject_key.nunique()} subjects")

    # Negative binomial on the OUTCOME: errors after, offset by repeats after,
    # predicted by the ripple rate. Ripple rate is the predictor here, not the
    # outcome -- this is the directional test.
    tab["_off"] = np.log(tab.n_repeats_after.clip(lower=1))
    formula = "errors_after ~ rate_hz + errors_in_discovery + np.log(pause_s)"
    obs = np.nan
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        m = smf.glm(formula, data=tab, family=sm.families.NegativeBinomial(),
                    offset=tab["_off"]).fit(cov_type="cluster",
                    cov_kwds={"groups": tab.subject_key.astype("category").cat.codes})
        print("  NB GLM (cluster-robust by subject), outcome = errors after first-D:")
        print("   " + pd.DataFrame({"coef": m.params, "se": m.bse, "z": m.tvalues,
                                    "p": m.pvalues}).round(4).to_string()
              .replace("\n", "\n   "))
        obs = float(m.params.get("rate_hz", np.nan))
        p_glm = float(m.pvalues.get("rate_hz", np.nan))
    except Exception as e:
        print(f"  GLM failed: {type(e).__name__}: {e}")
        p_glm = np.nan

    # Permutation: shuffle the ripple rate across grids WITHIN subject, so the
    # null keeps each subject's rate distribution and each grid's difficulty.
    rng = np.random.RandomState(SEED)
    null = np.full(n_perms, np.nan)
    for k in range(n_perms):
        sh = tab.groupby("subject_key").rate_hz.transform(
            lambda v: rng.permutation(v.to_numpy()))
        d = tab.assign(rate_hz=sh)
        try:
            mm = smf.glm(formula, data=d, family=sm.families.NegativeBinomial(),
                         offset=d["_off"]).fit()
            null[k] = float(mm.params.get("rate_hz", np.nan))
        except Exception:
            null[k] = np.nan
    p, z, sd = _perm_p(-obs if np.isfinite(obs) else np.nan,
                       -null, one_sided=True)   # prediction: MORE ripples -> FEWER errors
    print(f"  permutation ({n_perms} within-subject shuffles of rate): "
          f"z = {z:+.2f}, p_one_sided = {p:.4f}  (H: more ripples -> fewer errors)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tab.to_csv(os.path.join(out_dir, "H3_grids.csv"), index=False)
        np.save(os.path.join(out_dir, "H3_null.npy"), null)
    return {"hypothesis": name, "question": question, "observed_log_rr": obs,
            "null": null,
            "rate_change_pct": np.nan, "null_mean": float(np.nanmean(null)),
            "null_sd": sd, "z": z, "p_perm": p, "p_glm": p_glm,
            "n_perms": n_perms, "n_rows": int(len(tab)),
            "n_sessions": int(tab.session.nunique()),
            "n_subjects": int(tab.subject_key.nunique()), "counts": tab}


# ------------------------------------------------------------------- main ---
def run(which=None, analysis_name=ANALYSIS_NAME, n_perms=N_PERMS, save_all=True):
    R = swr_io.get_data_root()
    out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr", "hypotheses")
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(R), "group", "swr"),
                     "swr_hypotheses")
    np.random.seed(SEED)
    subj = _subject_map(R)
    wanted = [which] if isinstance(which, str) else (
        list(which) if which else ["H1", "H2", "H3", "H4", "H5", "H6", "H7"])

    results = []
    if "H1" in wanted:
        results.append(_run_window_hypothesis(
            "H1", "ripples elevated at the FIRST arrival at D vs later arrivals",
            lambda b: win.windows_first_D(b, post_s=LOCK_D_S),
            lambda c: _log_rate_diff(c, "condition", "first_D", "later_D"),
            "n_ripples ~ condition + n_moves",
            analysis_name, R, subj, n_perms, out_dir=out_dir if save_all else None))
    if "H2" in wanted:
        results.append(_run_window_hypothesis(
            "H2", "ripples elevated in pauses before the grid is first solved",
            # `windows_pauses` labels the pause by the phase of the repeat that
            # FOLLOWS it (`phase_after`): exploration / first_correct /
            # later_repeats. The contrast is the two ends of that.
            lambda b: win.windows_pauses(b).query(
                f"duration_s >= {MIN_PAUSE_S}"),
            lambda c: _log_rate_diff(c, "phase_after", "exploration",
                                     "later_repeats"),
            "n_ripples ~ phase_after + n_moves",
            analysis_name, R, subj, n_perms, out_dir=out_dir if save_all else None))
    if "H3" in wanted:
        results.append(_run_H3(analysis_name, R, subj, n_perms,
                               out_dir=out_dir if save_all else None))
    if "H5" in wanted:
        results.append(_run_window_hypothesis(
            "H5", "ripples across explore / plan / execute pauses",
            lambda b: win.windows_phase3(b, min_pause_s=MIN_PAUSE_S),
            lambda c: _log_rate_diff(c, "phase3", "plan", "execute"),
            "n_ripples ~ C(phase3, Treatment('execute')) + n_moves",
            analysis_name, R, subj, n_perms, out_dir=out_dir if save_all else None))
    if "H6" in wanted:
        results.append(_run_window_hypothesis(
            "H6", "within the discovery traversal, is D special? (D vs A/B/C, "
                  "first vs later)",
            lambda b: win.windows_discovery(b, lock_s=LOCK_D_S),
            _interaction_D_vs_ABC,
            "n_ripples ~ C(state) * C(discovery) + n_moves",
            analysis_name, R, subj, n_perms, out_dir=out_dir if save_all else None))
    if "H7" in wanted:
        results.append(_run_window_hypothesis(
            "H7", "feedback that could change behaviour vs feedback that could not",
            lambda u: win.windows_informative(u, lock_s=LOCK_FB_S),
            lambda c: _log_rate_diff(c, "informative", "informative",
                                     "uninformative"),
            "n_ripples ~ C(feedback) * C(phase) + n_moves",
            analysis_name, R, subj, n_perms, needs_uncover=True,
            out_dir=out_dir if save_all else None))
    if "H4" in wanted:
        results.append(_run_window_hypothesis(
            "H4", "ripple rate differs after error vs correct feedback",
            lambda u: win.windows_feedback(u, lock_s=LOCK_FB_S),
            lambda c: _log_rate_diff(c, "feedback", "error", "correct"),
            "n_ripples ~ feedback * phase + n_moves",
            analysis_name, R, subj, n_perms, needs_uncover=True,
            out_dir=out_dir if save_all else None))

    results = [r for r in results if r is not None]
    if not results:
        print("\nnothing ran"); return None

    summary = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("counts", "null")} for r in results])
    # H1 is pre-declared primary and is not corrected; H2-H4 are corrected
    # across themselves. Both are reported.
    sec = summary.hypothesis.isin(SECONDARY)
    summary["q_fdr_secondary"] = np.nan
    if sec.any():
        summary.loc[sec, "q_fdr_secondary"] = _bh_fdr(summary.loc[sec, "p_perm"])
    n_sec = int(sec.sum())
    summary["role"] = np.where(summary.hypothesis == PRIMARY,
                               "primary (pre-declared, uncorrected)",
                               f"secondary (FDR across the {n_sec} run here)")

    print("\n" + "=" * 78); print(" SUMMARY"); print("=" * 78)
    cols = ["hypothesis", "observed_log_rr", "rate_change_pct", "z", "p_perm",
            "q_fdr_secondary", "n_subjects", "n_sessions", "role"]
    print(summary[[c for c in cols if c in summary.columns]]
          .round(4).to_string(index=False))

    if save_all:
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, "hypothesis_summary.csv"), index=False)
        swr_io.write_settings(out_dir, _settings(wanted, n_perms, analysis_name))
        try:
            rfig.hypothesis_figure(
                {r["hypothesis"]: r for r in results},
                out_stem=os.path.join(out_dir, "figures", "hypotheses"),
                title="Hippocampal ripple rate by condition")
        except Exception as e:
            print(f"  figure failed: {type(e).__name__}: {e}")
        print(f"\n saved -> {out_dir}")
    return None


def export_bundle(analysis_name=ANALYSIS_NAME, out_name="swr_bundle"):
    """Everything needed to redo any of these statistics WITHOUT the LFP.

    The cluster holds the raw recordings; the analysis of what the ripples mean
    does not need them. This writes one pickle (and the same tables as CSVs, so
    it is readable without this repo) containing, for every session:

        ripples    one row per accepted ripple: session, subject, pair, ROI,
                   MNI, t_peak_s, duration, peak frequency, amplitude
        intervals  the artifact-free intervals per derivation -- REQUIRED, since
                   a rate is ripples per artifact-free second and any window
                   analysis is wrong without it
        pairs      the bipolar derivations with their coordinates
        behaviour  all_trial_times per session, with phase labels
        uncover    every uncovering attempt with its outcome, in session seconds
        channel_qc per-derivation counts, clean time and exclusion flags

    Typical size is a few MB, against tens of GB of LFP.
    """
    import pickle
    R = swr_io.get_data_root()
    out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr", "bundle")
    os.makedirs(out_dir, exist_ok=True)
    subj = _subject_map(R)

    rip, iv, pr, beh_all, unc_all, qc_all = [], [], [], [], [], []
    for sess, rip_dir in _sessions_with_events(analysis_name, R):
        meta = subj.get(sess, {})
        clean_dir = os.path.join(swr_io.session_deriv_dir(sess, R), "LFP-clean",
                                 analysis_name)
        try:
            e = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
            e = e[e.passed.fillna(False)]
            keep = [c for c in ("pair_id", "t_peak_s", "duration_s",
                                "peak_freq_hz", "amp_peak_uv", "rms_peak_z",
                                "spectral_passed_strict", "spectral_passed_relaxed")
                    if c in e.columns]
            e = e[keep].copy()
            e["session"] = sess
            e["subject_key"] = meta.get("subject_key")
            e["recording_site"] = meta.get("recording_site")
            rip.append(e)

            i = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
            i["session"] = sess; iv.append(i)

            q = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv"))
            q["session"] = sess; qc_all.append(q)

            p_ = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
            pr.append(p_)
        except FileNotFoundError as err:
            print(f"  s{sess:02d}: {err}"); continue

        try:
            b = swr_io.load_behaviour(sess, data_root=R)
            b = win.add_phase3(win.add_phase(b))
            b["session"] = sess
            b["subject_key"] = meta.get("subject_key")
            beh_all.append(b)
            u = swb.uncover_events(sess, data_root=R)
            if len(u):
                u["subject_key"] = meta.get("subject_key")
                unc_all.append(u)
        except Exception as err:
            print(f"  s{sess:02d}: behaviour skipped ({type(err).__name__}: {err})")

    def cat(x):
        return pd.concat(x, ignore_index=True) if x else pd.DataFrame()

    bundle = {"ripples": cat(rip), "intervals": cat(iv), "pairs": cat(pr),
              "behaviour": cat(beh_all), "uncover": cat(unc_all),
              "channel_qc": cat(qc_all),
              "meta": {"analysis_name": analysis_name,
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "data_root": R,
                       "note": "rates must use intervals for exposure; "
                               "a ripple rate is events per ARTIFACT-FREE second"}}

    pkl = os.path.join(out_dir, f"{out_name}.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(bundle["meta"], f, indent=2)

    print("\n" + "=" * 74); print(" EXPORT BUNDLE"); print("=" * 74)
    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k:12s} {len(v):7d} rows"
                  + (f", {v.session.nunique()} sessions" if "session" in v else ""))
    print(f"\n  {pkl}  ({os.path.getsize(pkl)/1e6:.1f} MB)")
    print("  CSVs alongside it, so the bundle is readable without this repo.")
    print("\n  rsync it back with:")
    print(f"    rsync -avz <user>@ssh.swc.ucl.ac.uk:"
          f"/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/swr/bundle/ \\")
    print("      ~/Documents/projects/multiple_clocks/data/ephys_humans/"
          "derivatives/group/swr/bundle/")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({'run': run, 'export': export_bundle})
    else:
        run()
