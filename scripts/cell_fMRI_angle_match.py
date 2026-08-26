#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONE script for everything in final_splits/: brain overlays + profile overlays
+ the per-cell assignment CSV, for the three chosen cell-averaging splits.

Splits (all IN-MASK cells only; outside-mask cells shown grey):
  1. z_tercile_consist : z-axis, 3 terciles, consistency-weighted
                         -> ventral 0deg, mid 30deg, dorsal 60deg (fMRI 63/76/80)
  2. pc1_ventral_dorsal: PC1 axis, median split, unweighted -> 30deg / 60deg
  3. y_antpost_subj    : y-axis, median split, subject-weighted -> 60deg / 90deg

Brain figures: OPAQUE fMRI-gradient backdrop (no transparency); cells drawn with
a black halo behind each coloured sphere so they pop with dark contrast.
Profile overlays: 4.5 x 3 cm, Arial >= 9 pt, bootstrap 16-84% bands, grey
outside-mask line. Each cell/line coloured by its group's preferred angle on the
cyclic wheel (same wheel as the backdrop).

Each brain is saved twice: plain, and with thin "t-bars" showing the split axis
(grey) and the division boundary/boundaries (black).

Outputs (in <master>/final_splits/):
  {scheme}_{lh,rh}_medial.{png,pdf}       -- brain overlay (no bars)
  {scheme}_{lh,rh}_medial_bars.{png,pdf}  -- brain overlay + split axis/boundary
  {scheme}_profiles.{png,pdf}             -- profile overlay
  final_splits_per_cell.csv               -- per-cell group + angle
  final_splits_summary.csv                -- per-group summary
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

import gradient_brain_cells_by_lag as gb
from mne.datasets import fetch_fsaverage
from mne.viz import Brain

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9
mpl.rcParams['pdf.fonttype'] = 42

OUT = gb.MASTER_DIR / 'final_splits'
L = np.arange(0, 360, 30)
CM, FS = 1 / 2.54, 9
GREY = (0.7, 0.7, 0.7)

BACKDROP_ALPHA = 1.0        # no transparency on the rainbow gradient
SURF_ALPHA = 0.25
CELL_SCALE = 0.58           # coloured sphere (bigger = more pop)
CELL_DARKEN = 0.85          # richer/darker hue for contrast vs bright backdrop
JITTER_MM = 3.0             # spread co-located cells so they read individually
BAR_SCALE = 0.18            # split-line thickness (a bit fatter, still not fat)
BAR_X = 2.0                 # |x| the bars are drawn at: small = toward the
                            # (medial-view) viewer, so bars sit ON TOP of cells
AXIS_BAR_COLOR = (0.5, 0.5, 0.5)   # grey = split axis
BOUND_BAR_COLOR = (0.0, 0.0, 0.0)  # black = division boundary

# scheme: (tag, axis_column, n_groups, weighting, group_names ventral->dorsal)
SCHEMES = [
    ('z_tercile_consist', 'MNI_z', 3, 'consist', ['ventral', 'mid', 'dorsal']),
    ('pc1_ventral_dorsal', 'grad_axis_coord', 2, 'unw', ['ventral', 'dorsal']),
    ('y_antpost_subj', 'MNI_y', 2, 'subj', ['posterior', 'anterior']),
]


# ── shared computation ───────────────────────────────────────────────
def cmean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 \
        if a.size else np.nan


def _prof_from(Rg, sg, cg, weighting):
    if weighting == 'unw':
        return np.nanmean(Rg, 0)
    if weighting == 'consist':
        w = np.clip(cg, 0, None)
        return np.nansum(w[:, None] * Rg, 0) / (w.sum() + 1e-12)
    return np.nanmean(np.vstack([np.nanmean(Rg[sg == s], 0)
                                 for s in np.unique(sg)]), 0)


def profile(R, sel, subj, consist, weighting):
    return _prof_from(R[sel], subj[sel], consist[sel], weighting)


def group_angle(R, sel, subj, consist, weighting):
    p = profile(R, sel, subj, consist, weighting)
    return int(L[np.nanargmax(p)]) if np.isfinite(p).any() else np.nan


def boot_band(R, sel, subj, consist, weighting, nboot=1000):
    Rg, sg, cg = R[sel], subj[sel], consist[sel]
    rng = np.random.default_rng(0)
    boots = np.empty((nboot, len(L)))
    subs = np.unique(sg)
    for b in range(nboot):
        if weighting == 'subj':
            take = rng.choice(subs, subs.size, replace=True)
            rows = np.concatenate([np.where(sg == s)[0] for s in take])
            sb = np.concatenate([np.full((sg == s).sum(), i)
                                 for i, s in enumerate(take)])
            boots[b] = _prof_from(Rg[rows], sb, cg[rows], 'subj')
        else:
            i = rng.integers(0, len(Rg), len(Rg))
            boots[b] = _prof_from(Rg[i], sg[i], cg[i], weighting)
    return np.nanpercentile(boots, 16, 0), np.nanpercentile(boots, 84, 0)


def make_labels(x, inm, n):
    lab = np.array(['outside'] * len(x), dtype=object)
    if n == 2:
        med = np.median(x[inm])
        lab[inm & (x <= med)] = 'g0'; lab[inm & (x > med)] = 'g1'
    else:
        e = np.quantile(x[inm], [1/3, 2/3])
        names = ['g0', 'g1', 'g2']
        lab[inm] = [names[i] for i in np.digitize(x[inm], e)]
    return lab


def build_scheme(c, R, subj, consist, fmri, axis, n, weighting):
    inm = c['in_gradient_mask'].to_numpy(bool)
    lab = make_labels(axis, inm, n)
    order = sorted([g for g in set(lab) if g != 'outside'],
                   key=lambda g: np.nanmean(axis[lab == g]))
    ang = {g: group_angle(R, lab == g, subj, consist, weighting) for g in order}
    fm = {g: cmean(fmri[lab == g]) for g in order}
    return lab, order, ang, fm


def boundary_rule(axis_col, axis, inm, n):
    unit = 'PC1 mm' if axis_col == 'grad_axis_coord' else f'{axis_col} mm'
    direction = {'MNI_z': 'ventral->dorsal', 'MNI_y': 'posterior->anterior',
                 'grad_axis_coord': 'ventral->dorsal'}[axis_col]
    if n == 2:
        med = np.median(axis[inm])
        return (f'{unit} median split at {med:.2f} ({direction}); '
                f'low = axis<=median, high = axis>median')
    e = np.quantile(axis[inm], [1/3, 2/3])
    return (f'{unit} terciles at {e[0]:.2f} and {e[1]:.2f} ({direction})')


def summarize_scheme(tag, axis_col, axis, lab, order, rename, weighting, n,
                     R, subj, consist, fmri, inm):
    rule = boundary_rule(axis_col, axis, inm, n)
    rows = []
    for g in list(order) + ['outside']:
        sel = lab == g
        if sel.sum() == 0:
            continue
        prof = profile(R, sel, subj, consist, weighting)
        fv = fmri[sel]; av = axis[sel]
        rows.append(dict(
            scheme=tag, axis=axis_col, weighting=weighting,
            group=rename.get(g, 'outside'), in_mask=g != 'outside',
            n_in_group=int(sel.sum()), n_in_mask_total=int(inm.sum()),
            n_cells_total=int(len(lab)), boundary_rule=rule,
            group_axis_min=round(float(np.nanmin(av)), 2),
            group_axis_max=round(float(np.nanmax(av)), 2),
            group_axis_mean=round(float(np.nanmean(av)), 2),
            pooled_lag_deg=int(L[np.nanargmax(prof)]),
            pooled_peak_r=round(float(np.nanmax(prof)), 3),
            fmri_mean_deg=round(cmean(fv), 0),
            fmri_min_deg=round(float(np.nanmin(fv)), 0),
            fmri_max_deg=round(float(np.nanmax(fv)), 0)))
    return rows


# ── split-axis + boundary "t-bar" geometry ───────────────────────────
def gradient_pc1():
    """Gradient-mask centroid + PC1 unit vector (folded x), oriented +z."""
    m = nib.load(str(gb.GRAD15_MASK_PATH))
    mni = nib.affines.apply_affine(m.affine, np.argwhere(m.get_fdata() > 0))
    mni[:, 0] = np.abs(mni[:, 0])
    centroid = mni.mean(0)
    _, _, vt = np.linalg.svd(mni - centroid, full_matrices=False)
    pc1 = vt[0]
    return centroid, (-pc1 if pc1[2] < 0 else pc1)


def compute_split_lines(c, axis_col, axis, inm, n):
    """Return (axis_line, [boundary_lines]) as folded-(x,y,z) point arrays:
    a grey line along the split axis and a black perpendicular bar at each
    division boundary (median for n=2; terciles for n=3)."""
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    cx = np.abs(coords[inm, 0]).mean()
    cy, cz = coords[inm, 1].mean(), coords[inm, 2].mean()
    yv, zv = coords[inm, 1], coords[inm, 2]
    edges = ([np.median(axis[inm])] if n == 2
             else list(np.quantile(axis[inm], [1/3, 2/3])))
    if axis_col == 'MNI_z':
        zr = np.linspace(axis[inm].min() - 2, axis[inm].max() + 2, 40)
        axis_line = np.column_stack([np.full(40, cx), np.full(40, cy), zr])
        yw = np.linspace(yv.min() - 1, yv.max() + 1, 24)
        bounds = [np.column_stack([np.full(24, cx), yw, np.full(24, b)])
                  for b in edges]
    elif axis_col == 'MNI_y':
        yr = np.linspace(axis[inm].min() - 2, axis[inm].max() + 2, 40)
        axis_line = np.column_stack([np.full(40, cx), yr, np.full(40, cz)])
        zw = np.linspace(zv.min() - 1, zv.max() + 1, 24)
        bounds = [np.column_stack([np.full(24, cx), np.full(24, b), zw])
                  for b in edges]
    else:  # grad_axis_coord (PC1)
        centroid, pc1 = gradient_pc1()
        tr = np.linspace(axis[inm].min() - 1, axis[inm].max() + 1, 40)
        axis_line = centroid[None] + tr[:, None] * pc1[None]
        perp = np.array([0.0, -pc1[2], pc1[1]]); perp /= np.linalg.norm(perp)
        fold = coords.copy(); fold[:, 0] = np.abs(fold[:, 0])
        hw = np.abs((fold[inm] - centroid) @ perp).max()
        uw = np.linspace(-hw, hw, 24)
        bounds = [(centroid + b * pc1)[None] + uw[:, None] * perp[None]
                  for b in edges]
    return axis_line, bounds


# ── brain overlay (opaque backdrop, haloed cells) ────────────────────
def render_brain(c, lab, order, ang, tag, subtitle, subjects_dir,
                 axis_line, boundary_lines):
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    cols = np.tile(np.array(GREY + (1.0,)), (len(c), 1))
    for g in order:
        cols[lab == g] = gb.angles_to_colours([ang[g]])[0]
    for hemi in ('lh', 'rh'):
        try:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir,
                          alpha=SURF_ALPHA)
        except TypeError:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir)
        data = gb.backdrop_texture(hemi, subjects_dir)
        if np.isfinite(data).any():
            try:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=gb.CIRC_CMAP, alpha=BACKDROP_ALPHA,
                               colorbar=False, smoothing_steps=5)
            except TypeError:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=gb.CIRC_CMAP, alpha=BACKDROP_ALPHA,
                               colorbar=False)
        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(hash((hemi, tag)) & 0xFFFF)
        cj = coords[keep] + rng.uniform(-JITTER_MM, JITTER_MM, coords[keep].shape)
        for cc, col in zip(cj, cols[keep]):
            dark = tuple(np.clip(np.asarray(col[:3]) * CELL_DARKEN, 0, 1))
            brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                           color=dark, scale_factor=CELL_SCALE)
        try:
            brain.show_view('medial')
        except Exception:
            pass
        # version WITHOUT bars
        png = str(OUT / f'{tag}_{hemi}_medial.png')
        brain.save_image(png)
        # add the split axis (grey) + boundary bars (black), version WITH bars.
        # Draw bars at a small fixed |x| so they sit in front of the cells in
        # the medial view (viewer looks from the midline outward).
        sign = -1 if hemi == 'lh' else 1

        def to_h(p):
            q = np.asarray(p, float).copy(); q[:, 0] = sign * BAR_X
            return q
        brain.add_foci(to_h(axis_line), coords_as_verts=False, hemi=hemi,
                       color=AXIS_BAR_COLOR, scale_factor=BAR_SCALE)
        for bl in boundary_lines:
            brain.add_foci(to_h(bl), coords_as_verts=False, hemi=hemi,
                           color=BOUND_BAR_COLOR, scale_factor=BAR_SCALE)
        try:                              # foci can reset the camera; re-apply
            brain.show_view('medial')
        except Exception:
            pass
        png_bars = str(OUT / f'{tag}_{hemi}_medial_bars.png')
        brain.save_image(png_bars)
        try:
            brain.close()
        except Exception:
            pass
        _brain_cbar(png, hemi, subtitle)
        _brain_cbar(png_bars, hemi, subtitle + ' | axis=grey, boundary=black')
        print(f'  wrote {png} + _bars (+pdf)')


def _brain_cbar(png, hemi, subtitle):
    img = plt.imread(png)
    fig = plt.figure(figsize=(img.shape[1]/300, img.shape[0]/300 + 0.7), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 70],
                          hspace=0.02, left=0, right=1, bottom=0, top=1)
    a = fig.add_subplot(gs[0]); a.imshow(img); a.axis('off')
    cax = fig.add_subplot(gs[1]); pos = cax.get_position()
    cax.set_position([0.24, pos.y0, 0.5, pos.height * 0.32])
    sm = ScalarMappable(cmap=gb.CIRC_CMAP, norm=Normalize(-180, 180))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([-180, -90, 0, 90, 180])
    cb.set_label(f'cell group preferred lag (deg) on fMRI-gradient backdrop.  '
                 f'{subtitle}  [{hemi}]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── profile overlay (bootstrap bands + grey outside line) ────────────
def render_profiles(c, R, subj, consist, lab, order, tag, weighting):
    fig, ax = plt.subplots(figsize=(4.5 * CM, 3 * CM), constrained_layout=True)
    out = lab == 'outside'
    po = profile(R, out, subj, consist, weighting)
    lo, hi = boot_band(R, out, subj, consist, weighting)
    ax.fill_between(L, lo, hi, color='0.6', alpha=0.18, linewidth=0)
    ax.plot(L, po, '-', color='0.55', lw=1.0)
    for g in order:
        sel = lab == g
        p = profile(R, sel, subj, consist, weighting)
        lo, hi = boot_band(R, sel, subj, consist, weighting)
        col = gb.angles_to_colours([int(L[np.nanargmax(p)])])[0]
        ax.fill_between(L, lo, hi, color=col, alpha=0.2, linewidth=0)
        ax.plot(L, p, '-', color=col, lw=1.2)
        ax.plot(L[np.nanargmax(p)], np.nanmax(p), 'o', color=col, ms=2.6)
    ax.axhline(0, color='0.6', lw=0.5, ls='--')
    ax.set_xticks([0, 120, 240])
    ax.set_xlabel('lag (deg)', fontsize=FS, labelpad=1)
    ax.set_ylabel('CV r', fontsize=FS, labelpad=1)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=FS, length=2, pad=1, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{tag}_profiles.{ext}', dpi=600)
    plt.close(fig)
    print(f'  wrote {tag}_profiles.pdf/png')


# ── main ─────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    consist = c['argmax_r'].to_numpy(float)
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    subjects_dir = os.path.dirname(fetch_fsaverage())

    inm = c['in_gradient_mask'].to_numpy(bool)
    per_cell = c[['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z',
                  'in_gradient_mask']].copy()
    per_cell['fmri_angle'] = fmri
    summary_rows = []

    for tag, col, n, w, names in SCHEMES:
        axis = c[col].to_numpy(float)
        lab, order, ang, fm = build_scheme(c, R, subj, consist, fmri, axis, n, w)
        rename = {g: nm for g, nm in zip(order, names)}
        sub = ', '.join(f'{rename[g]}={ang[g]:.0f}deg(fMRI{fm[g]:.0f})'
                        for g in order)
        print(f'\n=== {tag} ({w}) ===  ' + sub)
        per_cell[f'{tag}_group'] = [rename.get(g, 'outside') for g in lab]
        per_cell[f'{tag}_angle'] = [ang.get(g, np.nan) for g in lab]
        summary_rows += summarize_scheme(tag, col, axis, lab, order, rename, w,
                                         n, R, subj, consist, fmri, inm)
        axis_line, boundary_lines = compute_split_lines(c, col, axis, inm, n)
        render_brain(c, lab, order, ang, tag,
                     f'{tag} [{w}]: {sub}, outside=grey', subjects_dir,
                     axis_line, boundary_lines)
        render_profiles(c, R, subj, consist, lab, order, tag, w)

    per_cell.to_csv(OUT / 'final_splits_per_cell.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'final_splits_summary.csv', index=False)
    pd.set_option('display.width', 240); pd.set_option('display.max_columns', 20)
    print('\n=== GROUP SUMMARY ===')
    print(summary.drop(columns=['boundary_rule', 'n_cells_total']).to_string(
        index=False))
    print(f'\nWrote {OUT / "final_splits_per_cell.csv"}')
    print(f'Wrote {OUT / "final_splits_summary.csv"}')
    print(f'All final_splits outputs in {OUT}')


if __name__ == '__main__':
    main()
