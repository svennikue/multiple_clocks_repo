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

# CORRECTION (2026-09-04). These seven were written by Claude from SK's verbal
# description of the idea. They were NEVER pre-registered and were never SK's
# own declared predictions, so labelling H1 "primary, pre-declared" and
# FDR-correcting H2-H7 as a registered family was wrong: it dressed a set of
# exploratory probes up as a confirmatory study, and then reported them as a
# failed one. They are exploratory, all seven, and the q-values below should be
# read as descriptive only. The real question -- when, over what window, and how
# ripples interact with movement -- is still open and is being worked out in
# swr_explore.py / swr_findings.py.
PRIMARY = "H1"                      # kept only so old outputs stay readable
SECONDARY = ("H2", "H3", "H4", "H5", "H6", "H7")


def _settings(which, n_perms, analysis_name):
    return {"analysis": "swr_hypotheses", "which": which, "n_perms": n_perms,
            "seed": SEED, "analysis_name": analysis_name,
            "lock_D_s": LOCK_D_S, "lock_feedback_s": LOCK_FB_S,
            "min_pause_s": MIN_PAUSE_S, "primary": PRIMARY,
            "secondary_fdr_over": list(SECONDARY),
            "created": datetime.now().isoformat(timespec="seconds")}


# --------------------------------------------------------------- loading ----
class RippleStore:
    """Where the detected ripples come from, behind one interface.

    Two sources:
      `sessions`  the per-session detection output on this machine. What the
                  cluster has.
      `bundle`    a bundle downloaded from the cluster, carrying the same three
                  tables for every session in a few MB. What the laptop has.

    The bundle exists so these statistics can be redone without moving the LFP.
    Everything that reads ripples goes through here, so both paths run the
    IDENTICAL analysis code -- the source cannot change a result.
    """

    def __init__(self, analysis_name=ANALYSIS_NAME, data_root=None, bundle=None):
        self.analysis_name = analysis_name
        self.R = data_root or swr_io.get_data_root()
        self.bundle_path = None
        if bundle is None:
            self.source = "sessions"
            paths = sorted(glob.glob(os.path.join(
                swr_io.derivatives_dir(self.R), "s*", "LFP-ripples",
                analysis_name, "ripple_events.csv")))
            self._sessions = [int(p.split(os.sep)[-4][1:]) for p in paths]
            self._dirs = {s: os.path.dirname(p)
                          for s, p in zip(self._sessions, paths)}
        else:
            self.source = "bundle"
            self._load_bundle(bundle)

    def _load_bundle(self, bundle):
        """`bundle` is the bundle directory, or the .pkl inside it."""
        if os.path.isdir(bundle):
            d = bundle
            tabs = {k: pd.read_csv(os.path.join(d, f"{k}.csv"))
                    for k in ("ripples", "intervals", "channel_qc")}
        else:
            import pickle
            with open(bundle, "rb") as f:
                b = pickle.load(f)
            d = os.path.dirname(bundle)
            tabs = {k: b[k] for k in ("ripples", "intervals", "channel_qc")}
        self.bundle_path = d
        self._rip = {s: g for s, g in tabs["ripples"].groupby("session")}
        self._iv = {s: g for s, g in tabs["intervals"].groupby("session")}
        self._qc = {s: g.set_index("pair_id")
                    for s, g in tabs["channel_qc"].groupby("session")}
        self._sessions = sorted(self._rip)
        # the bundle carries its own subject key, so a stale local manifest
        # cannot silently re-cluster the robust standard errors
        r = tabs["ripples"]
        self._subj = (r[["session", "subject_key", "recording_site"]]
                      .drop_duplicates("session").set_index("session")
                      .to_dict("index"))

    def sessions(self):
        return list(self._sessions)

    def get(self, sess):
        """(accepted events, artifact-free intervals, channel QC) for a session.

        Events are already filtered to those that passed detection, in both
        sources -- the bundle stores only accepted ripples.
        """
        sess = int(sess)
        if self.source == "bundle":
            return (self._rip[sess], self._iv[sess], self._qc[sess])
        d = self._dirs[sess]
        ev = pd.read_csv(os.path.join(d, "ripple_events.csv"))
        ev = ev[ev.passed.fillna(False)]
        iv = pd.read_csv(os.path.join(d, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(d, "channel_qc.csv")).set_index("pair_id")
        return ev, iv, qc

    def subject_map(self):
        if self.source == "bundle":
            return self._subj
        m = pd.read_csv(os.path.join(swr_io.derivatives_dir(self.R), "group",
                                     "swr", "session_manifest.csv"))
        return m.set_index("session")[["subject_key",
                                       "recording_site"]].to_dict("index")

    def describe(self):
        n = sum(len(self._rip[s]) for s in self._sessions) \
            if self.source == "bundle" else None
        where = self.bundle_path or f"{swr_io.derivatives_dir(self.R)}/s*/LFP-ripples"
        print(f"  source: {self.source}  ({len(self._sessions)} sessions"
              + (f", {n} accepted ripples" if n is not None else "") + ")")
        print(f"          {where}")


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


def _count_windows(sess, store, windows, beh, R, subj):
    """Ripple counts per (derivation x window), with exposure and movement."""
    ev, iv_all, qc = store.get(sess)
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


def _gather(design_fn, store, R, subj, needs_uncover=False):
    """Build one design across every session that has detection output."""
    parts, per_session = [], {}
    for sess in store.sessions():
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
            # designs that need the session number (the ones reading the 25 ms
            # location series) get it from the frame rather than a closure
            beh = beh.assign(session=int(sess))
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
        c = _count_windows(sess, store, w, beh, R, subj)
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
def _shifted_counts_factory(store, R, subj, per_session, max_shift_frac=1.0):
    """Return a function(rng) -> counts table with each session's events shifted.

    The shift is circular on the artifact-free axis, so a shifted event can never
    land inside an artifact and every session keeps its own ripple
    autocorrelation and its own number of events.
    """
    from mc.analyse.swr_artifact import CleanAxis

    # One entry per (session, derivation): the events, the artifact-free
    # intervals, the windows and the movement covariate. Built once; the
    # permutation loop only redraws the shift.
    cache = []
    for sess, w in per_session.items():
        ev, iv_all, qc = store.get(sess)
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception:
            continue
        mv = swb.presses_in_windows(sess, beh, w, data_root=R)
        meta = subj.get(sess, {})
        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            clean_total = float(CleanAxis(iv).total)
            if clean_total <= 0:
                continue
            cache.append((sess, pair_id, e.t_peak_s.to_numpy(float), iv, w, mv,
                          clean_total, meta.get("subject_key")))

    def make(rng):
        rows = []
        for sess, pair_id, t, iv, w, mv, clean_total, skey in cache:
            # The shift is drawn on the ARTIFACT-FREE axis and applied by
            # `assign_events_to_windows` itself, so a shifted event can never
            # land inside an artifact -- where the detector could not have
            # found it, and where it would make the null easier to beat.
            # shift_s=0 is the observed case through the identical function.
            sh = rng.uniform(0.05 * clean_total, 0.95 * clean_total)
            a = win.assign_events_to_windows(t, w, iv, shift_s=sh)
            a = a.reset_index(drop=True)
            a["session"] = sess
            a["pair_id"] = pair_id
            a["n_moves"] = mv
            a["subject_key"] = skey
            rows.append(a)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return make


def _run_window_hypothesis(name, question, design_fn, contrast, formula,
                           store, R, subj, n_perms, needs_uncover=False,
                           out_dir=None, cond_col=None, forest=None):
    print("\n" + "=" * 74)
    print(f" {name}: {question}")
    print("=" * 74)
    counts, per_session = _gather(design_fn, store, R, subj,
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

    if n_perms and n_perms > 0:
        make = _shifted_counts_factory(store, R, subj, per_session)
        null = np.array([contrast(make(np.random.RandomState(SEED + k)))
                         for k in range(n_perms)])
        p, z, sd = _perm_p(obs, null, one_sided=True)
    else:
        # exploration mode: the GLM is the whole output. No permutation means
        # no p_perm -- do not quietly substitute the GLM p for it.
        null, p, z, sd = np.zeros(0), np.nan, np.nan, np.nan
    # The null is not centred on zero: a log ratio of small counts is biased
    # downward (Jensen), and first-D windows are few. That is precisely why
    # inference is against the shifted null and not against zero -- the null
    # carries the identical bias because it is built by the identical code.
    if null.size:
        print(f"  permutation ({n_perms} circular shifts): null "
              f"{np.nanmean(null):+.4f} +- {sd:.4f}, z = {z:+.2f}, "
              f"p_one_sided = {p:.4f}")
    else:
        print("  permutation: SKIPPED (n_perms=0) -- GLM only, exploratory")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        counts.to_csv(os.path.join(out_dir, f"{name}_counts.csv"), index=False)
        np.save(os.path.join(out_dir, f"{name}_null.npy"), null)
        # the data behind the number: spread across subjects and derivations,
        # the two confounds, the power imbalance, and the null
        try:
            rfig.condition_distributions(
                name, counts, cond_col or "condition", null=null, observed=obs,
                question=question,
                out_stem=os.path.join(out_dir, "figures", f"{name}_distributions"))
        except Exception as e:
            print(f"    distribution figure failed: {type(e).__name__}: {e}")
        # per-session effect, so a pooled number cannot hide being carried by
        # a handful of sessions
        if forest:
            try:
                rfig.effect_forest(
                    name, counts, *forest, question=question,
                    out_stem=os.path.join(out_dir, "figures", f"{name}_forest"))
            except Exception as e:
                print(f"    forest figure failed: {type(e).__name__}: {e}")
    return {"hypothesis": name, "question": question, "observed_log_rr": obs,
            "null": null,
            "rate_change_pct": 100 * (np.exp(obs) - 1) if np.isfinite(obs) else np.nan,
            "null_mean": float(np.nanmean(null)), "null_sd": sd, "z": z,
            "p_perm": p, "n_perms": n_perms,
            "n_rows": int(len(counts)), "n_sessions": int(counts.session.nunique()),
            "n_subjects": int(counts.subject_key.nunique()),
            "counts": counts}


def _run_H3(store, R, subj, n_perms, out_dir=None):
    """Does the ripple rate after first-D predict subsequent performance?"""
    name = "H3"
    question = ("ripple rate in the pause after first-D predicts errors in the "
                "following repeats")
    print("\n" + "=" * 74); print(f" {name}: {question}"); print("=" * 74)

    rows = []
    for sess in store.sessions():
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
            rt = swb.repeat_table(sess, beh=beh, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: {type(e).__name__}: {e}"); continue
        if not len(rt):
            continue
        ev, iv_all, qc = store.get(sess)

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
        try:
            rfig.regression_distributions(
                "H3", tab, "rate_hz", "errors_after", null=null, observed=obs,
                question=question,
                out_stem=os.path.join(out_dir, "figures", "H3_distributions"))
        except Exception as e:
            print(f"    distribution figure failed: {type(e).__name__}: {e}")
        # per-session effect, so a pooled number cannot hide being carried by
        # a handful of sessions
        if forest:
            try:
                rfig.effect_forest(
                    name, counts, *forest, question=question,
                    out_stem=os.path.join(out_dir, "figures", f"{name}_forest"))
            except Exception as e:
                print(f"    forest figure failed: {type(e).__name__}: {e}")
    return {"hypothesis": name, "question": question, "observed_log_rr": obs,
            "null": null,
            "rate_change_pct": np.nan, "null_mean": float(np.nanmean(null)),
            "null_sd": sd, "z": z, "p_perm": p, "p_glm": p_glm,
            "n_perms": n_perms, "n_rows": int(len(tab)),
            "n_sessions": int(tab.session.nunique()),
            "n_subjects": int(tab.subject_key.nunique()), "counts": tab}


# ------------------------------------------------------------------- main ---
def run(which=None, analysis_name=ANALYSIS_NAME, n_perms=N_PERMS, save_all=True,
        bundle=None, out_dir=None):
    """Run the hypothesis tests.

    `bundle` points at a bundle directory (or its .pkl) downloaded from the
    cluster; without it the per-session detection output on this machine is
    used. Both go through `RippleStore`, so the analysis is identical.
    """
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "hypotheses")
    swr_io.start_log(os.path.dirname(out_dir), "swr_hypotheses")
    np.random.seed(SEED)
    store = RippleStore(analysis_name, R, bundle=bundle)
    store.describe()
    subj = store.subject_map()
    wanted = [which] if isinstance(which, str) else (
        list(which) if which else ["H1", "H2", "H3", "H4", "H5", "H6", "H7"])

    results = []
    if "H1" in wanted:
        results.append(_run_window_hypothesis(
            "H1", "ripples elevated at the FIRST arrival at D vs later arrivals",
            lambda b: win.windows_first_D(b, post_s=LOCK_D_S),
            lambda c: _log_rate_diff(c, "condition", "first_D", "later_D"),
            "n_ripples ~ condition + n_moves",
            store, R, subj, n_perms, out_dir=out_dir if save_all else None,
            cond_col="condition",
            forest=("condition", "first_D", "later_D")))
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
            store, R, subj, n_perms, out_dir=out_dir if save_all else None,
            cond_col="phase_after",
            forest=("phase_after", "exploration", "later_repeats")))
    if "H3" in wanted:
        results.append(_run_H3(store, R, subj, n_perms,
                               out_dir=out_dir if save_all else None))
    if "H5" in wanted:
        results.append(_run_window_hypothesis(
            "H5", "ripples across explore / plan / execute pauses",
            lambda b: win.windows_phase3(b, min_pause_s=MIN_PAUSE_S),
            lambda c: _log_rate_diff(c, "phase3", "plan", "execute"),
            "n_ripples ~ C(phase3, Treatment('execute')) + n_moves",
            store, R, subj, n_perms, out_dir=out_dir if save_all else None,
            cond_col="phase3", forest=("phase3", "plan", "execute")))
    if "H6" in wanted:
        results.append(_run_window_hypothesis(
            "H6", "within the discovery traversal, is D special? (D vs A/B/C, "
                  "first vs later)",
            lambda b: win.windows_discovery(b, lock_s=LOCK_D_S),
            _interaction_D_vs_ABC,
            "n_ripples ~ C(state) * C(discovery) + n_moves",
            store, R, subj, n_perms, out_dir=out_dir if save_all else None,
            cond_col=["state", "discovery"]))
    if "H7" in wanted:
        results.append(_run_window_hypothesis(
            "H7", "feedback that could change behaviour vs feedback that could not",
            lambda u: win.windows_informative(u, lock_s=LOCK_FB_S),
            lambda c: _log_rate_diff(c, "informative", "informative",
                                     "uninformative"),
            "n_ripples ~ C(feedback) * C(phase) + n_moves",
            store, R, subj, n_perms, needs_uncover=True,
            out_dir=out_dir if save_all else None, cond_col="informative",
            forest=("informative", "informative", "uninformative")))
    if "H4" in wanted:
        results.append(_run_window_hypothesis(
            "H4", "ripple rate differs after error vs correct feedback",
            lambda u: win.windows_feedback(u, lock_s=LOCK_FB_S),
            lambda c: _log_rate_diff(c, "feedback", "error", "correct"),
            "n_ripples ~ feedback * phase + n_moves",
            store, R, subj, n_perms, needs_uncover=True,
            out_dir=out_dir if save_all else None,
            cond_col=["feedback", "phase"],
            forest=("feedback", "error", "correct")))

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
    # not "primary"/"secondary": none of these was pre-registered (see the
    # correction at the top of this file)
    summary["role"] = np.where(summary.hypothesis == PRIMARY,
                               "exploratory (uncorrected)",
                               f"exploratory (FDR shown across the {n_sec} run "
                               "here, descriptive only)")

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


# ============================================================================
# SWEEP -- the definitional choices, laid out side by side
# ============================================================================
# Every hypothesis here rests on choices that were made once and then never
# revisited: how long a window is, what counts as a phase, what "errors after"
# means. None of them is obviously right. This runs the alternatives against
# each other with the GLM only (n_perms=0 by default), so the cost of looking
# is seconds rather than an hour.
#
# It is an exploration tool. A p from here is not an inference -- the variant
# has to be chosen on grounds other than its p, and then tested with the
# permutation on data that did not pick it.

def _pool_later_per_grid(counts):
    """One later-D row per (grid, derivation), so the contrast is 1:1.

    As run, every later arrival at D is its own row: ~11 later-Ds against 1
    first-D per grid, unpaired and dominated by the later condition. Pooling
    counts and exposure within grid makes it a matched comparison at the cost
    of statistical power.
    """
    keys = [k for k in ("session", "pair_id", "grid_no", "condition",
                        "subject_key", "recording_site") if k in counts.columns]
    agg = {"n_ripples": "sum", "exposure_s": "sum", "duration_s": "sum"}
    if "n_moves" in counts.columns:
        agg["n_moves"] = "sum"
    out = counts.groupby(keys, as_index=False).agg(agg)
    out["rate_hz"] = out.n_ripples / out.exposure_s.replace(0, np.nan)
    return out


def _glm_row(counts, formula, term, label, extra=None):
    res = _glm(counts, formula, label)
    row = {"variant": label, "n_rows": len(counts),
           "n_sessions": int(counts.session.nunique()),
           "n_windows": int(counts.groupby(["session", "start_s"]).ngroups)
           if "start_s" in counts.columns else np.nan}
    row.update(extra or {})
    if res is not None and "table" in res:
        tb = res["table"]
        hit = [i for i in tb.index if term in str(i)]
        if hit:
            r = tb.loc[hit[0]]
            row.update({"term": hit[0], "coef": float(r["coef"]),
                        "se": float(r["se"]), "z": float(r["z"]),
                        "p_glm": float(r["p"]),
                        "rate_ratio": float(r.get("rate_ratio", np.nan))})
    return row


def sweep(which="H1", bundle=None, analysis_name=ANALYSIS_NAME, n_perms=0,
          out_dir=None):
    """Test the definitional alternatives for one hypothesis, side by side."""
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, f"swr_sweep_{which}")
    np.random.seed(SEED)
    store = RippleStore(analysis_name, R, bundle=bundle)
    store.describe()
    subj = store.subject_map()
    print(f"\n  sweeping {which}   (n_perms={n_perms}"
          + ("  -- GLM only, exploratory)" if not n_perms else ")"))

    rows, keep = [], {}

    if which == "H1":
        # What does "around D" mean? t_D in all_trial_times is the moment D is
        # UNCOVERED (the press). Arrival, dwell and the deliberation period
        # before the press are different events and ask different questions.
        # Dwell is ~3x longer on the discovery traversal, so any window running
        # past the press compares a subject still standing on the reward with
        # one who has already moved on. `deliberation` is the only variant that
        # is matched by construction.
        variants = []
        for w in (0.5, 1.0, 2.0):
            variants.append((f"uncover-locked, {w:g} s", "uncover_locked", w))
            variants.append((f"arrival-locked, {w:g} s", "arrival_locked", w))
        variants += [
            ("deliberation (arrive -> press)", "deliberation", None),
            ("dwell (press -> leave)", "dwell", None),
            ("1 s BEFORE arriving at D", "pre_arrival", 1.0),
            ("1 s after leaving D", "post_leave", 1.0),
        ]
        for lab, kind, w in variants:
            def build(b, kind=kind, w=w):
                dev = swb.d_events(int(b.session.iloc[0]), beh=b, data_root=R)
                return win.windows_from_d_events(dev, kind=kind,
                                                 w_s=w if w else 1.0)
            c, _ = _gather(build, store, R, subj)
            if not len(c):
                print(f"    {lab}: no windows"); continue
            for pairing in ("unique later-Ds", "later-Ds pooled per grid"):
                cc = c if pairing == "unique later-Ds" else _pool_later_per_grid(c)
                d = cc.groupby("condition").duration_s.median()
                rows.append(_glm_row(
                    cc, "n_ripples ~ condition + n_moves", "condition",
                    f"{lab}  [{pairing}]",
                    extra={"obs_log_rr": _log_rate_diff(cc, "condition",
                                                        "first_D", "later_D"),
                           "median_dur_first": float(d.get("first_D", np.nan)),
                           "median_dur_later": float(d.get("later_D", np.nan)),
                           "kind": kind, "pairing": pairing}))
            keep[lab] = c
            print(f"    {lab:34s} done ({len(c)} rows)")

    elif which in ("H2", "H5"):
        variants = [
            ("pauses >= 0.5 s  (as run)", dict(min_pause_s=0.5)),
            ("pauses >= 1 s", dict(min_pause_s=1.0)),
            ("pauses >= 2 s", dict(min_pause_s=2.0)),
            ("pauses >= 5 s", dict(min_pause_s=5.0)),
        ]
        for lab, kw in variants:
            mp = kw["min_pause_s"]
            c2, _ = _gather(lambda b, mp=mp: win.windows_pauses(b).query(
                f"duration_s >= {mp}"), store, R, subj)
            if len(c2):
                d = c2.groupby("phase_after").duration_s.median()
                rows.append(_glm_row(
                    c2, "n_ripples ~ phase_after + n_moves", "later_repeats",
                    f"2-phase, {lab}",
                    extra={"obs_log_rr": _log_rate_diff(
                        c2, "phase_after", "exploration", "later_repeats"),
                        "median_dur_first": float(d.get("exploration", np.nan)),
                        "median_dur_later": float(d.get("later_repeats", np.nan))}))
                keep[f"2-phase, {lab}"] = c2
            c3, _ = _gather(lambda b, mp=mp: win.windows_phase3(b, min_pause_s=mp),
                            store, R, subj)
            if len(c3):
                d = c3.groupby("phase3").duration_s.median()
                rows.append(_glm_row(
                    c3, "n_ripples ~ C(phase3, Treatment('execute')) + n_moves",
                    "plan", f"3-phase, {lab}",
                    extra={"obs_log_rr": _log_rate_diff(c3, "phase3", "plan",
                                                        "execute"),
                           "median_dur_first": float(d.get("plan", np.nan)),
                           "median_dur_later": float(d.get("execute", np.nan))}))
                keep[f"3-phase, {lab}"] = c3
            print(f"    {lab:36s} done")

    elif which == "H3":
        rows = _sweep_H3(store, R, subj, out_dir)

    tab = pd.DataFrame(rows)
    if len(tab):
        print("\n" + "=" * 100)
        print(f" {which}: definitional variants  (GLM only -- NOT an inference)")
        print("=" * 100)
        cols = [c for c in ("variant", "n_rows", "obs_log_rr", "coef", "se",
                            "z", "p_glm", "median_dur_first", "median_dur_later",
                            "n_grids", "unit", "outcome") if c in tab.columns]
        print(tab[cols].round(4).to_string(index=False))
        tab.to_csv(os.path.join(out_dir, f"sweep_{which}.csv"), index=False)
        try:
            rfig.sweep_figure(tab, which,
                              out_stem=os.path.join(out_dir, "figures",
                                                    f"sweep_{which}"))
        except Exception as e:
            print(f"  sweep figure failed: {type(e).__name__}: {e}")
    if keep:
        try:
            rfig.window_definition_figure(
                keep, which,
                out_stem=os.path.join(out_dir, "figures", f"windows_{which}"))
        except Exception as e:
            print(f"  window figure failed: {type(e).__name__}: {e}")
    print(f"\n saved -> {out_dir}")
    return None


def _sweep_H3(store, R, subj, out_dir):
    """H3's three separate choices: the window, the outcome, and the unit."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    windows = [("pause after first-D (as run)", None), ("fixed 1 s after D", 1.0),
               ("fixed 2 s after D", 2.0), ("fixed 5 s after D", 5.0)]
    rows = []
    for wlab, fixed in windows:
        recs = []
        for sess in store.sessions():
            try:
                beh = swr_io.load_behaviour(sess, data_root=R)
                rt = swb.repeat_table(sess, beh=beh, data_root=R)
            except Exception:
                continue
            if not len(rt):
                continue
            ev, iv_all, qc = store.get(sess)
            for grid, g in rt.groupby("grid_no"):
                g = g.sort_values("rep_overall")
                disc, later = g[g.is_discovery == 1], g[g.is_discovery == 0]
                if not len(disc) or not len(later):
                    continue
                t0 = float(disc.t_D.iloc[0])
                t1 = float(later.t_start.iloc[0])
                if not (np.isfinite(t0) and np.isfinite(t1)):
                    continue
                end = t0 + fixed if fixed else t1
                if end - t0 < MIN_PAUSE_S:
                    continue
                w = win._finalise(pd.DataFrame([{
                    "start_s": t0, "end_s": end, "grid_no": int(grid),
                    "condition": "post_first_D", "is_test": True}]))
                if not len(w):
                    continue
                for pair_id, e in ev.groupby("pair_id"):
                    if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                        continue
                    iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                    a = win.assign_events_to_windows(e.t_peak_s.to_numpy(float), w, iv)
                    if not len(a) or float(a.exposure_s.iloc[0]) <= 0:
                        continue
                    nxt = later.iloc[0]
                    recs.append({
                        "session": sess, "grid_no": int(grid),
                        "pair_id": pair_id,
                        "subject_key": subj.get(sess, {}).get("subject_key"),
                        "n_ripples": float(a.n_ripples.iloc[0]),
                        "exposure_s": float(a.exposure_s.iloc[0]),
                        "rate_hz": float(a.n_ripples.iloc[0]) / float(a.exposure_s.iloc[0]),
                        "pause_s": end - t0,
                        "errors_all_later": float(later.n_errors.sum()),
                        "errors_next_repeat": float(nxt.n_errors),
                        "n_repeats_after": int(len(later)),
                        "errors_in_discovery": float(disc.n_errors.iloc[0]),
                        "reps_to_solve": float(nxt.get("reps_to_solve", np.nan)),
                    })
        t = pd.DataFrame(recs)
        if len(t) < 10:
            continue
        grid_avg = (t.groupby(["session", "subject_key", "grid_no"], as_index=False)
                    .agg(rate_hz=("rate_hz", "mean"),
                         errors_all_later=("errors_all_later", "first"),
                         errors_next_repeat=("errors_next_repeat", "first"),
                         n_repeats_after=("n_repeats_after", "first"),
                         pause_s=("pause_s", "first"),
                         errors_in_discovery=("errors_in_discovery", "first")))
        for unit, d in (("per derivation", t), ("per grid", grid_avg)):
            for outcome, off in (("errors_all_later", "n_repeats_after"),
                                 ("errors_next_repeat", None)):
                dd = d.copy()
                dd["_off"] = (np.log(dd[off].clip(lower=1)) if off
                              else np.zeros(len(dd)))
                f = (f"{outcome} ~ rate_hz + errors_in_discovery "
                     "+ np.log(pause_s)")
                try:
                    m = smf.glm(f, data=dd,
                                family=sm.families.NegativeBinomial(),
                                offset=dd["_off"]).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": dd.subject_key.astype("category").cat.codes})
                    rows.append({
                        "variant": f"{wlab} | {unit} | {outcome}",
                        "window": wlab, "unit": unit, "outcome": outcome,
                        "n_rows": len(dd),
                        "n_grids": int(grid_avg.groupby(["session", "grid_no"]).ngroups),
                        "coef": float(m.params["rate_hz"]),
                        "se": float(m.bse["rate_hz"]),
                        "z": float(m.tvalues["rate_hz"]),
                        "p_glm": float(m.pvalues["rate_hz"]),
                        "frac_zero_rate": float((dd.rate_hz == 0).mean()),
                        "median_pause_s": float(dd.pause_s.median())})
                except Exception as e:
                    print(f"    {wlab}|{unit}|{outcome} failed: {e}")
        print(f"    {wlab:32s} done ({len(t)} derivation-rows)")
    return rows


# ============================================================================
# DESCRIPTIVE -- no permutations, no GLM, just what the data look like
# ============================================================================

def _peth_across(store, align_by_session, half_s, bin_s):
    """PETH averaged over derivations within subject, then across subjects.

    The subject is the unit of inference everywhere else in this script, so it
    is the unit here too: a subject with five derivations must not count five
    times in the error band.
    """
    per_subj = {}
    centres = None
    for sess, aligns in align_by_session.items():
        if not len(aligns):
            continue
        try:
            ev, iv_all, qc = store.get(sess)
        except KeyError:
            continue
        skey = store.subject_map().get(sess, {}).get("subject_key", f"s{sess}")
        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            c, r, n, _ = win.peri_event_rate(e.t_peak_s.to_numpy(float), aligns,
                                             iv, half_s=half_s, bin_s=bin_s)
            centres = c
            per_subj.setdefault(skey, []).append(r)
    if centres is None or not per_subj:
        return None
    M = np.vstack([np.nanmean(np.vstack(v), axis=0) for v in per_subj.values()])
    n_sub = M.shape[0]
    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0) / max(np.sqrt(n_sub), 1)
    return centres, mean, sem, n_sub


def _align_reward(beh, state, first):
    """t_<state> on the first traversal of each grid (or on later ones)."""
    out = {}
    for sess, b in beh.groupby("session"):
        ts = []
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            sel = g.iloc[:1] if first else g.iloc[1:]
            ts += [float(v) for v in sel[f"t_{state}"] if np.isfinite(v)]
        out[int(sess)] = np.asarray(ts, float)
    return out


def _align_uncover(unc, query):
    out = {}
    for sess, u in unc.groupby("session"):
        q = u.query(query) if query else u
        out[int(sess)] = q.t_s.to_numpy(float)
    return out


def describe(bundle=None, analysis_name=ANALYSIS_NAME, out_dir=None,
             counts_dir=None, half_s=5.0, bin_s=0.25):
    """Every descriptive view of the data, with no inference at all.

    Fast on purpose: the per-hypothesis figures used to be written only after
    that hypothesis's 1000 permutations, which made looking at the data cost an
    hour. Nothing here permutes.

    `counts_dir` regenerates the per-hypothesis panels from `H*_counts.csv`
    already written by `run`, so no design has to be rebuilt.
    """
    import pickle
    R = swr_io.get_data_root()
    gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    if out_dir is None:
        out_dir = os.path.join(gdir, "descriptive")
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    swr_io.start_log(out_dir, "swr_describe")

    store = RippleStore(analysis_name, R, bundle=bundle)
    store.describe()

    if bundle and os.path.isdir(bundle):
        rip = pd.read_csv(os.path.join(bundle, "ripples.csv"))
        beh = pd.read_csv(os.path.join(bundle, "behaviour.csv"))
        unc = pd.read_csv(os.path.join(bundle, "uncover.csv"))
    else:
        with open(bundle, "rb") as f:
            b = pickle.load(f)
        rip, beh, unc = b["ripples"], b["behaviour"], b["uncover"]

    print(f"\n  {len(rip)} ripples | {len(beh)} repeats | {len(unc)} uncoverings")

    # ---- 1. what the ripples themselves look like
    print("\n  [1/5] ripple attributes")
    rfig.ripple_property_figure(
        rip, out_stem=os.path.join(figs, "ripple_properties"),
        title="Accepted ripples: the four Chen attributes")
    rfig.ripple_property_figure(
        rip, split="recording_site",
        out_stem=os.path.join(figs, "ripple_properties_by_site"),
        title="Ripple attributes by recording site")

    # ---- 2. rate per session and per subject
    print("  [2/5] rate distributions")
    qc = pd.concat([store.get(s)[2].assign(session=s) for s in store.sessions()])
    qc = qc[~qc.excluded.fillna(False)]
    rfig.rate_distribution_figure(
        qc, out_stem=os.path.join(figs, "rate_distributions"),
        title="Ripple rate per derivation, session and subject")

    # ---- 3. peri-event rate: the time course behind every contrast
    print("  [3/5] peri-event time histograms (the slow one)")
    panels = []
    tr = {}
    for lab, first in (("first traversal", True), ("later traversals", False)):
        got = _peth_across(store, _align_reward(beh, "D", first), half_s, bin_s)
        if got:
            tr[lab] = got
    if tr:
        panels.append(("Aligned to arrival at D  (H1)", tr))

    tr = {}
    for st in ("A", "B", "C", "D"):
        got = _peth_across(store, _align_reward(beh, st, True), half_s, bin_s)
        if got:
            tr[st] = got
    if tr:
        panels.append(("Discovery traversal, each reward  (H6)", tr))

    tr = {}
    for lab, q in (("correct", "correct == 1"), ("error", "correct == 0")):
        got = _peth_across(store, _align_uncover(unc, q), half_s, bin_s)
        if got:
            tr[lab] = got
    if tr:
        panels.append(("Aligned to feedback  (H4)", tr))

    tr = {}
    for lab, q in (("correct, discovery", "correct == 1 and is_discovery == 1"),
                   ("error, discovery", "correct == 0 and is_discovery == 1"),
                   ("correct, later", "correct == 1 and is_discovery == 0"),
                   ("error, later", "correct == 0 and is_discovery == 0")):
        got = _peth_across(store, _align_uncover(unc, q), half_s, bin_s)
        if got:
            tr[lab] = got
    if tr:
        panels.append(("Feedback x phase  (H7)", tr))

    if panels:
        rfig.peth_figure(panels, out_stem=os.path.join(figs, "peri_event_rate"),
                         title="Ripple rate around task events "
                               "(exposure-corrected, mean +- SEM across subjects)")

    # ---- 4/5. per-hypothesis panels, from counts already written by `run`
    if counts_dir and os.path.isdir(counts_dir):
        print(f"  [4/5] per-hypothesis panels from {counts_dir}")
        for f in sorted(glob.glob(os.path.join(counts_dir, "H*_counts.csv"))):
            name = os.path.basename(f).split("_")[0]
            c = pd.read_csv(f)
            nl = os.path.join(counts_dir, f"{name}_null.npy")
            null = np.load(nl) if os.path.isfile(nl) else None
            col = rfig.HYP_COND_COL.get(name, "condition")
            try:
                rfig.condition_distributions(
                    name, c, col, null=null,
                    out_stem=os.path.join(figs, f"{name}_distributions"))
            except Exception as e:
                print(f"    {name} distributions failed: {type(e).__name__}: {e}")
            fo = {"H1": ("condition", "first_D", "later_D"),
                  "H2": ("phase_after", "exploration", "later_repeats"),
                  "H5": ("phase3", "plan", "execute"),
                  "H7": ("informative", "informative", "uninformative"),
                  "H4": ("feedback", "error", "correct")}.get(name)
            if fo and fo[0] in c.columns:
                try:
                    rfig.effect_forest(name, c, *fo,
                                       out_stem=os.path.join(figs, f"{name}_forest"))
                except Exception as e:
                    print(f"    {name} forest failed: {type(e).__name__}: {e}")
            if "repeat_number" in c.columns:
                rfig.rate_by_ordinal(
                    c, "repeat_number",
                    out_stem=os.path.join(figs, f"{name}_by_repeat"),
                    title=f"{name}: rate across repeats", split="phase_after")
            if "solve_index" in c.columns and "repeat_number" not in c.columns:
                rfig.rate_by_ordinal(
                    c, "solve_index",
                    out_stem=os.path.join(figs, f"{name}_by_solve"),
                    title=f"{name}: rate across solves", xlabel="Solve index")
    print(f"\n saved -> {figs}")
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
        fire.Fire({'run': run, 'export': export_bundle,
                   'describe': describe,
                   'sweep': sweep})
    else:
        run()
