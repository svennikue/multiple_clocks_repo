#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the methods-illustration figure that contrasts the UNFOLDING code with
the CONCURRENT (DSR) action-plan code, for one example task configuration.

The figure explains, panel by panel, what
``scripts/create_fMRI_model_RDMs_on_clean_beh.py`` simulates and what
``scripts/fMRI_run_RSA_without_rsatoolbox_clean.py`` then regresses:

  a) the ABCD loop cut into equal bins == equal angles into the future
  b) the example configuration and the trajectory the subject actually walked
  c) the encodings across bins (position in sequence; physical location)
  d) the concurrent code read out from two different bins
  e) the unfolding code read out from the same two bins
  f) similarity = counting overlapping elements, for a within-task pair and
     for an across-task pair where the two formats disagree
  g) the resulting RDMs: single example task, and across all tasks

Two example tasks are produced, each in the framework it actually belongs to:

  '5-9-4-3'  fMRI  — 8 bins/loop, 12 resampled steps per bin (96-element DSR),
                     built from one subject's cleaned behaviour.
  '3-7-9-5'  sEEG  — 12 bins/loop (4 states x 3 subgoal phases), 1 location per
                     bin (12-element DSR), built from the modal trajectory over
                     all correct trials of that configuration.

Nothing is written into the repo: all output lands in
``data/derivatives/group/method_schematic_<date>/``.

@author: Svenja Kuechenhoff
"""

import json
import os
from datetime import date

import numpy as np

import mc
from mc.plotting import method_schematic as msch

SOURCE_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks"
EPHYS_DIR = f"{SOURCE_DIR}/data/ephys_humans/derivatives"
OUT_DIR = (f"{SOURCE_DIR}/data/derivatives/group/"
           f"method_schematic_{date.today().strftime('%d-%m-%Y')}")

# fMRI example. The example subject is chosen automatically (see
# `_pick_fmri_subject`): the lowest-numbered subject whose two task halves
# followed the SAME modal route through the example configuration, so the
# single-task RDM in panel g is exactly the diagonal block of the all-tasks
# RDM. Set FMRI_SUB to a subject id to override.
FMRI_SUB = 'sub-02'        # its two halves took different routes through
                           # B path, which panel g2 uses to show that the same
                           # configuration can be solved more than one way
FMRI_EV_STRING = 'DSR_loc-fut-rews-state-dur-type'
FMRI_TASK = '5-9-4-3'
# Panel f: the two comparison types the RSA actually uses — reward-reward and
# path-path. Bin order is A_path, A_reward, B_path, B_reward, ...
FMRI_PAIRS = ((1, 3), (2, 4))       # A_reward vs B_reward; B_path vs C_path

# sEEG example
SEEG_TASK = '3-7-9-5'
SEEG_CONFIGS = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
                '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
SEEG_PAIRS = ((0, 3), (1, 4))       # A_early vs B_early; A_middle vs B_middle

SHOW = False


def _pick_fmri_subject(task=None, prefer_pair=(2, 4)):
    """Pick the example subject by a fixed rule.

    1. The two task halves must have followed the SAME modal route through the
       example configuration. Otherwise the across-halves RDM compares two
       different routes, and its diagonal block stops looking like the
       single-task matrix the figure uses to explain it.
    2. Prefer the route that visits the most distinct grid locations — the
       concurrent-code strips then show the whole trajectory rather than the
       same two colours over and over.
    3. Then: no non-adjacent steps, the most commonly walked route among the
       qualifying subjects, and finally the lowest subject number.
    """
    task = task or FMRI_TASK
    base = f"{SOURCE_DIR}/data/derivatives"
    cands, families = [], {}
    for sub in sorted(os.listdir(base)):
        if not os.path.exists(f"{base}/{sub}/beh/{sub}_beh_fmri_clean.csv"):
            continue
        if not os.path.exists(f"{base}/{sub}/beh/modelled_EVs/"
                              f"{sub}_modelled_EVs_{FMRI_EV_STRING}.pkl"):
            continue
        try:
            examples, by_key = msch.build_fmri_examples(sub, str(SOURCE_DIR))
        except Exception:
            continue
        if task not in examples:
            continue
        ex = examples[task]
        half, direction = ex['task_key'][1], ex['task_key'].split('_')[1]
        partner = {'1_forw': '2_backw', '1_backw': '2_forw'}.get(
            f"{half}_{direction}")
        other = by_key.get(f"{ex['task_key'][0]}{partner}") if partner else None
        if other is None or not np.array_equal(ex['bin_locs'],
                                               other['bin_locs']):
            continue
        route = tuple(map(tuple, ex['bin_locs']))
        families[route] = families.get(route, 0) + 1
        _, bad = msch.non_adjacent_steps(ex)
        cands.append(dict(
            sub=sub, route=route,
            n_distinct=len(set(ex['traj'].tolist())), n_bad=len(bad),
            overlap=float((ex['bin_locs'][prefer_pair[0]]
                           == ex['bin_locs'][prefer_pair[1]]).mean())))
    if not cands:
        raise RuntimeError(f"no subject with matching task halves for {task}")
    cands.sort(key=lambda c: (-c['n_distinct'], c['n_bad'],
                              -families[c['route']], c['sub']))
    best = cands[0]
    print(f"  example subject {best['sub']}: halves walked the same route, "
          f"{best['n_distinct']} distinct locations, {best['n_bad']} "
          f"non-adjacent steps, route shared by {families[best['route']]} "
          f"subjects ({len(cands)} of the cohort qualified)")
    print(f"  panel-f bins {prefer_pair} share "
          f"{best['overlap']:.0%} of their locations")
    return best['sub'], len(cands)


def _pair_log(pairs):
    out = []
    for p in pairs:
        a, b = p['ex_a'], p['ex_b']
        dsr = float((a['dsr'][p['bin_a']] == b['dsr'][p['bin_b']]).mean())
        loc = float((a['bin_locs'][p['bin_a']] == b['bin_locs'][p['bin_b']]).mean())
        st = float(a['bin_states'][p['bin_a']] == b['bin_states'][p['bin_b']])
        out.append(dict(kind=p['kind'],
                        task_a=a['name'], bin_a=a['bin_labels'][p['bin_a']],
                        task_b=b['name'], bin_b=b['bin_labels'][p['bin_b']],
                        concurrent_similarity=dsr,
                        location_similarity=loc,
                        position_similarity=st))
    return out


def _report_pairs(pairs):
    for d in _pair_log(pairs):
        print(f"  [{d['kind']:>7}] {d['task_a']} {d['bin_a']:<9} vs "
              f"{d['task_b']} {d['bin_b']:<9} | concurrent "
              f"{d['concurrent_similarity']:.3f}  location "
              f"{d['location_similarity']:.3f}  position "
              f"{d['position_similarity']:.0f}")


def _log(settings, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2, default=str)
    print(f"  settings -> {path}")


def build_fmri_figure():
    print(f"\n=== fMRI example {FMRI_TASK} ===")
    sub = FMRI_SUB
    n_ok = None
    if sub is None:
        sub, n_ok = _pick_fmri_subject()
    else:
        print(f"  example subject {sub} (fixed)")
    examples, by_key = msch.build_fmri_examples(sub, SOURCE_DIR)
    if FMRI_TASK not in examples:
        raise KeyError(f"{FMRI_TASK} not found for {sub}: {sorted(examples)}")
    ex = examples[FMRI_TASK]
    others = [e for k, e in examples.items() if k != FMRI_TASK]
    label_map = {k: e['name'] for k, e in by_key.items()}

    across = msch.across_task_rdms_fmri(sub, SOURCE_DIR, FMRI_EV_STRING,
                                        label_map=label_map,
                                        n_bins=ex['n_bins'])
    ex_partner = msch.partner_example(ex, by_key)
    if ex_partner is None:
        print("  [warn] no partner task half found — skipping panel g2")
    else:
        diff = msch.route_differences(ex, ex_partner)
        print(f"  task halves: {ex['task_key']} vs {ex_partner['task_key']}; "
              f"routes differ in {len(diff)}/{ex['n_bins']} bins"
              + (f" ({', '.join(ex['bin_labels'][k] for k in diff)})"
                 if diff else ""))
        blk = msch.single_task_rdms_from_across(ex, across)
        ah = msch.across_half_rdms(ex, ex_partner)
        for (na, A), (_, B) in zip(blk.items(), ah.items()):
            print(f"  panel g2 == the {ex['name']} block of the all-tasks RDM "
                  f"for {na}: {np.allclose(A, B)}")
    stem = os.path.join(OUT_DIR, f"method_schematic_fmri_{FMRI_TASK}")
    panel_dir = os.path.join(OUT_DIR, f"panels_fmri_{FMRI_TASK}")
    pairs, page_cm = msch.make_method_figure(
        ex, others, across, label_map=label_map, within_pair=(1, 3),
        within_pairs=FMRI_PAIRS, save_stem=stem, show=SHOW,
        ex_partner=ex_partner)
    _, panel_sizes = msch.save_panels(
        ex, others, across, label_map=label_map, within_pair=(1, 3),
        within_pairs=FMRI_PAIRS, out_dir=panel_dir,
        prefix=f'fmri_{FMRI_TASK}', show=SHOW, ex_partner=ex_partner)
    print(f"  overview page: {page_cm[0]:.1f} x {page_cm[1]:.1f} cm")
    for k, v in panel_sizes.items():
        print(f"    {k:<20} {v}")

    seq, bad = msch.non_adjacent_steps(ex)
    print(f"  trajectory: {seq}")
    print(f"  rewards: {ex['rewards']}")
    if bad:
        print(f"  NOTE non-adjacent steps after binning: {bad}")
    _report_pairs(pairs)
    _log(dict(framework='fmri', subject=sub,
              subject_rule=('fixed example subject; its two halves solved the '
                            'configuration by different routes, which panel g2 '
                            'illustrates. _pick_fmri_subject() offers the '
                            'matching-route alternative.'),
              task_halves=[ex['task_key'],
                           ex_partner['task_key'] if ex_partner else None],
              route_diff_bins=[ex['bin_labels'][k] for k in
                               (msch.route_differences(ex, ex_partner)
                                if ex_partner else [])],
              partner_trajectory=(ex_partner['traj'].tolist()
                                  if ex_partner else None),
              task=FMRI_TASK,
              ev_string=FMRI_EV_STRING, n_bins=ex['n_bins'],
              len_per_bin=ex['len_per_bin'],
              bin_labels=ex['bin_labels'],
              modal_trajectory=ex['traj'].tolist(),
              trajectory_locations=seq,
              non_adjacent_steps=bad,
              rewards=ex['rewards'],
              within_pair=[1, 3],
              panel_f_pairs=[list(p) for p in FMRI_PAIRS],
              comparisons=_pair_log(pairs),
              across_task_models=list(across),
              panel_dir=panel_dir,
              panel_sizes_cm=panel_sizes,
              overview_page_cm=page_cm),
         stem + '_settings.json')
    return ex


def build_seeg_figure():
    print(f"\n=== sEEG example {SEEG_TASK} ===")
    examples = {}
    for cfg in SEEG_CONFIGS:
        e = msch.build_seeg_example(cfg, EPHYS_DIR)
        if e is None:
            print(f"  (skip) no correct trials found for {cfg}")
            continue
        examples[cfg] = e
    if SEEG_TASK not in examples:
        raise KeyError(f"{SEEG_TASK} not available in {EPHYS_DIR}")
    ex = examples[SEEG_TASK]
    others = [e for k, e in examples.items() if k != SEEG_TASK]

    across = msch.across_task_rdms_seeg(examples)
    stem = os.path.join(OUT_DIR, f"method_schematic_seeg_{SEEG_TASK}")
    panel_dir = os.path.join(OUT_DIR, f"panels_seeg_{SEEG_TASK}")
    pairs, page_cm = msch.make_method_figure(
        ex, others, across, label_map=None, within_pair=(0, 3),
        within_pairs=SEEG_PAIRS, save_stem=stem, show=SHOW)
    _, panel_sizes = msch.save_panels(
        ex, others, across, label_map=None, within_pair=(0, 3),
        within_pairs=SEEG_PAIRS, out_dir=panel_dir,
        prefix=f'seeg_{SEEG_TASK}', show=SHOW)
    print(f"  overview page: {page_cm[0]:.1f} x {page_cm[1]:.1f} cm")
    for k, v in panel_sizes.items():
        print(f"    {k:<20} {v}")

    seq, bad = msch.non_adjacent_steps(ex)
    print(f"  trajectory (12 bins): {ex['traj'].tolist()} -> {seq}")
    if bad:
        print(f"  NOTE non-adjacent steps after binning: {bad}")
    print(f"  rewards: {ex['rewards']}  (loop rotated by {ex['bin_shift']} bins "
          f"so bin 0 sits on reward A; {ex['state_onsets_matched']}/4 state "
          f"onsets matched)")
    _report_pairs(pairs)
    _log(dict(framework='seeg', task=SEEG_TASK, configs=list(examples),
              n_bins=ex['n_bins'], len_per_bin=ex['len_per_bin'],
              bin_labels=ex['bin_labels'],
              modal_trajectory=ex['traj'].tolist(),
              trajectory_locations=seq,
              non_adjacent_steps=bad,
              rewards=ex['rewards'], within_pair=[0, 3],
              panel_f_pairs=[list(p) for p in SEEG_PAIRS],
              bin_shift_to_align_A=ex['bin_shift'],
              state_onsets_matched=ex['state_onsets_matched'],
              comparisons=_pair_log(pairs),
              across_task_models=list(across),
              panel_dir=panel_dir,
              panel_sizes_cm=panel_sizes,
              overview_page_cm=page_cm),
         stem + '_settings.json')
    return ex


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    build_fmri_figure()
    build_seeg_figure()
    print(f"\nfigures written to {OUT_DIR}")
