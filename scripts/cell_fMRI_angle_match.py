#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do human mPFC single units show the same dorsoventral future-lag
gradient that the 7T fMRI shows?

This is the reported cell <-> fMRI correspondence analysis. Cells are
ordered along an anatomical axis derived from the fMRI gradient mask,
split into groups, and each group's pooled spatial-tuning profile is read
out for its preferred future lag. If the cell population carries the same
gradient, the preferred lag should increase from ventral to dorsal.

PIPELINE POSITION
-----------------
  per_lag_encoding.py        -> per-cell 12-lag spatial-tuning profiles
                                (`r_lag{000..330}_noctrl`: leave-one-
                                configuration-out correlation between
                                lag-shifted 9-location rate maps)
  cell_gradient_master_table.py -> per_cell_master.csv: adds the gradient
                                axis coordinate, gradient-mask membership,
                                and the fMRI preferred angle sampled at
                                each cell
  THIS SCRIPT                -> the splits, their pooled profiles, the
                                brain overlays, and the two CSVs

This script is self-contained: it imports no other script in this repo.

USAGE
-----
    python cell_fMRI_angle_match.py [--master-csv per_cell_master.csv]
                                    [--out DIR] [--brains | --no-brains]

Set `PLOT_BRAINS` at the top of this file to turn the fsaverage brain
overlays on or off (--brains / --no-brains override it for one run).

Everything numeric — the splits, the pooled profiles, both CSVs and the
2D profile plots — is reproduced from `per_cell_master.csv` alone (~80 KB)
on numpy + pandas + matplotlib. nibabel, nilearn and mne are imported
lazily and ONLY when the brains are drawn, which additionally needs the
fMRI group volumes, the gradient mask, a working 3D backend and an
fsaverage download.

THE BACKDROP IS NOT CONFIGURABLE HERE — AND THAT IS DELIBERATE
--------------------------------------------------------------
The brain figure shows cells coloured by their group's preferred lag on
top of the fMRI preferred-angle map. Those two things are only comparable
if the map is processed exactly as the per-cell angles were. So the
symmetrisation, the pre-projection smoothing, the harmonic volumes and
the gradient mask are all READ FROM the `config.json` that
cell_gradient_master_table.py wrote next to `per_cell_master.csv`
(`load_master_provenance`), and the script refuses to render if that file
is missing rather than guessing. To see a raw, unsmoothed map, change the
settings in cell_gradient_master_table.py and regenerate the master table
— then the cells and the backdrop move together and the figure still
means something. The only rendering knob left here is
`SURFACE_SMOOTHING_STEPS`, which is display-side mesh smoothing with no
counterpart in the per-cell sampling and cannot change a reported number.

THE ANATOMICAL AXIS
-------------------
Defined in cell_gradient_master_table.py as PC1 of the MNI coordinates of
the fMRI gradient-mask voxels, with x folded to |x| so the axis is
bilaterally symmetric, oriented +z = dorsal. It is essentially
dorsoventral (r = 0.98 with MNI z) with a slight posterior tilt. Note it
comes from the GEOMETRY of the mask, not from the fMRI angle values, so
the anatomical ordering of cells is independent of the fMRI effect it is
compared against.

SPLITS (in-mask cells only; outside-mask cells shown grey, never used to
define the gradient)
  1. z_tercile_consist : MNI z, 3 terciles, consistency-weighted
                         -> ventral 0deg, mid 30deg, dorsal 60deg
                            (fMRI at those cells: 63 / 76 / 80deg)
  2. pc1_ventral_dorsal: PC1 axis, median split, unweighted
                         -> ventral 30deg (n=42) / dorsal 60deg (n=32)
                            *** the split reported in the manuscript ***
  3. y_antpost_subj    : MNI y, median split, subject-weighted
                         -> posterior 60deg / anterior 90deg

Profiles are POOLED FIRST, then the peak is read off — never an average
of per-cell argmaxes, which are unreliable at this SNR. Three weightings
are available: unweighted, consistency-weighted (by each cell's peak r),
and subject-weighted (mean within subject, then across subjects).
Uncertainty is the 16-84th percentile of 1000 bootstrap resamples,
resampling cells (unweighted / consistency) or subjects (subject-
weighted).

TWO CAVEATS THAT BELONG WITH ANY REPORT OF THIS RESULT
------------------------------------------------------
* Effective N is recording SITES, not cells. The 74 in-mask cells come
  from 16 unique microwire bundles; cells on one bundle share identical
  coordinates. The pc1 median (-13.81) falls exactly on a 15-cell bundle,
  which is why that split is 42/32 rather than 37/37 (ties go to the
  `<= median` ventral side). The per-cell circular correlation between
  cell lag and fMRI angle is null (approx -0.15 to -0.21); the agreement
  here is coarse and directional, at group level.
* The axis/n_groups/weighting combinations reported above were picked
  from a much larger sweep for the cleanest gradient-direction match.
  Those exploratory scripts now live in
  `scripts/old/cell_fMRI_gradient_exploration/` (see its README). The
  splits are therefore descriptive, not an inferential test, and should
  be presented as such.

FIGURES
-------
Brain: opaque fMRI-gradient backdrop; cells as spheres with a black halo,
coloured by their group's preferred angle on the same cyclic wheel,
jittered +-3 mm so co-located cells read individually. Saved twice —
plain, and with thin "t-bars" marking the split axis (grey) and the
division boundaries (black).
Profiles: 4.5 x 3 cm, Arial >= 9 pt, bootstrap bands, grey outside-mask
line.

OUTPUTS (in <master>/final_splits/)
  {scheme}_{lh,rh}_medial.{png,pdf}       -- brain overlay (no bars)
  {scheme}_{lh,rh}_medial_bars.{png,pdf}  -- brain overlay + axis/boundary
  {scheme}_profiles.{png,pdf}             -- profile overlay
  final_splits_per_cell.csv               -- per-cell group + angle
  final_splits_summary.csv                -- per-group summary
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

# NOTE: nibabel / nilearn / mne are imported LAZILY inside the figure code
# only. The numeric result (splits, pooled profiles, both CSVs) runs on
# numpy + pandas + matplotlib alone, so a broken 3D backend or a missing
# fsaverage download cannot block reproducing the reported numbers.

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9
mpl.rcParams['pdf.fonttype'] = 42

# ── Paths (defaults; override with --master-csv / --out) ─────────────
MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-22_12-18-21'
)
DEFAULT_MASTER_CSV = MASTER_DIR / 'per_cell_master.csv'
DEFAULT_OUT = MASTER_DIR / 'final_splits'

# ── Render the fsaverage brain overlays? ─────────────────────────────
# True  -> also write {scheme}_{lh,rh}_medial[_bars].{png,pdf}. Needs mne +
#          nilearn, a working 3D backend, and an fsaverage download.
# False -> CSVs and 2D profile plots only (all the numeric results).
PLOT_BRAINS = True

# The column of per_cell_master.csv carrying the fMRI angle at each cell.
# The backdrop is built from the SAME dataset this column came from.
FMRI_ANGLE_COLUMN = 'fmri_angle_quarters_deg'

L = np.arange(0, 360, 30)
CM, FS = 1 / 2.54, 9
GREY = (0.7, 0.7, 0.7)

AMP_EPS = 1e-6

# Display-only: MNE mesh-neighbour smoothing of the projected backdrop.
# This has NO counterpart in the per-cell sampling (which reads the volume
# inside a sphere and never touches a surface), so it is a rendering
# choice, not a data-processing one, and cannot change any reported number.
SURFACE_SMOOTHING_STEPS = 5

# ── Provenance: everything that determines the ANGLE VALUES ──────────
# Symmetrisation, pre-projection smoothing, the harmonic volumes and the
# gradient mask are deliberately NOT settable here. They are read from the
# config.json that cell_gradient_master_table.py wrote next to
# per_cell_master.csv, so the backdrop is always built exactly the way the
# per-cell angles were. Choosing them independently would put the cells and
# the map underneath them on two different conventions and make any
# apparent cell-vs-backdrop mismatch uninterpretable.
def load_master_provenance(master_csv):
    """Return the fMRI-sampling settings used to build `master_csv`.

    Raises if the sibling config.json is missing rather than falling back
    to defaults — a guessed convention is exactly what this prevents.
    """
    import json
    cfg_path = Path(master_csv).parent / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.json next to {master_csv}. The brain overlay must be "
            f"built with the same fMRI sampling settings as the per-cell "
            f"angles, and those settings live in the master run's "
            f"config.json. Re-run cell_gradient_master_table.py, or point "
            f"--master-csv at a run directory that has one.")
    cfg = json.load(open(cfg_path))
    missing = [k for k in ('harmonic_root', 'gradient_mask',
                           'fmri_symmetrise', 'fmri_smooth_fwhm_mm')
               if k not in cfg]
    if missing:
        raise KeyError(
            f"{cfg_path} is missing {missing} — it predates the provenance "
            f"fields. Re-run cell_gradient_master_table.py.")
    dataset = FMRI_ANGLE_COLUMN.replace('fmri_angle_', '').replace('_deg', '')
    prov = {
        'harmonic_root': Path(cfg['harmonic_root']),
        'gradient_mask': Path(cfg['gradient_mask']),
        'dataset':       dataset,
        'symmetrise':    bool(cfg['fmri_symmetrise']),
        'smooth_fwhm':   float(cfg['fmri_smooth_fwhm_mm']),
        'sphere_radius': cfg.get('sphere_radius_mm'),
        'config_path':   str(cfg_path),
    }
    print(f"  [provenance] {cfg_path.parent.name}/config.json -> "
          f"dataset={prov['dataset']}, symmetrise={prov['symmetrise']}, "
          f"smooth_fwhm={prov['smooth_fwhm']} mm, "
          f"per-cell sphere={prov['sphere_radius']} mm")
    return prov

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

# Cyclic wheel: -180 blue, -90 green, 0 yellow, +90 red, +180 blue.
CIRCULAR_ANCHORS_HEX = ['#1E88E5', '#43A047', '#FCE300', '#E53935', '#1E88E5']
CIRC_CMAP = LinearSegmentedColormap.from_list('circular_wheel',
                                              CIRCULAR_ANCHORS_HEX)


def angles_to_colours(angles_360):
    """[0,360) angle -> signed (-180,180] -> cyclic wheel RGBA."""
    a = np.asarray(angles_360, float)
    signed = ((a + 180.0) % 360.0) - 180.0
    return CIRC_CMAP(Normalize(vmin=-180, vmax=180)(signed))


def backdrop_texture(hemi, subjects_dir, prov):
    """+181-shifted fMRI preferred-angle texture for the gradient-masked
    backdrop on this hemisphere (NaN outside mask / no signal).

    Angle is circular, so the angle volume is NOT projected directly (that
    would smear across the +-180 wrap): the cos and sin group volumes are
    projected and the angle recomputed on the surface.

    `prov` comes from `load_master_provenance` — the settings the per-cell
    angles were sampled with. It is not user-settable, so the backdrop and
    the cells always share one angle convention.
    """
    import nibabel as nib
    from nilearn import surface as _surface
    from nilearn.image import new_img_like, smooth_img

    symmetrise, smooth_fwhm = prov['symmetrise'], prov['smooth_fwhm']

    def _symmetrise(img):
        d = img.get_fdata()
        return new_img_like(img, (d + d[::-1, ...]) / 2.0)

    surf = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.pial')
    cos_img = nib.load(str(prov['harmonic_root'] / prov['dataset'] / 'cos_group.nii.gz'))
    sin_img = nib.load(str(prov['harmonic_root'] / prov['dataset'] / 'sin_group.nii.gz'))
    if symmetrise:
        cos_img, sin_img = _symmetrise(cos_img), _symmetrise(sin_img)
    if smooth_fwhm:
        cos_img = smooth_img(cos_img, smooth_fwhm)
        sin_img = smooth_img(sin_img, smooth_fwhm)
    cos_txt = _surface.vol_to_surf(cos_img, surf, interpolation='linear')
    sin_txt = _surface.vol_to_surf(sin_img, surf, interpolation='linear')
    angle = np.degrees(np.arctan2(sin_txt, cos_txt))
    amp = np.hypot(cos_txt, sin_txt)
    mask_txt = _surface.vol_to_surf(
        nib.load(str(prov['gradient_mask'])), surf, interpolation='nearest')
    keep = (amp > AMP_EPS) & (mask_txt >= 0.5) & np.isfinite(angle)
    return np.where(keep, angle + 181.0, np.nan)

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
def gradient_pc1(prov):
    """Gradient-mask centroid + PC1 unit vector (folded x), oriented +z.

    Recomputed here only to draw the split-axis t-bars; the per-cell
    projection itself (`grad_axis_coord`) already comes from
    cell_gradient_master_table.py using the identical definition."""
    import nibabel as nib
    m = nib.load(str(prov['gradient_mask']))
    mni = nib.affines.apply_affine(m.affine, np.argwhere(m.get_fdata() > 0))
    mni[:, 0] = np.abs(mni[:, 0])
    centroid = mni.mean(0)
    _, _, vt = np.linalg.svd(mni - centroid, full_matrices=False)
    pc1 = vt[0]
    return centroid, (-pc1 if pc1[2] < 0 else pc1)


def compute_split_lines(c, axis_col, axis, inm, n, prov):
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
        centroid, pc1 = gradient_pc1(prov)
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
                 axis_line, boundary_lines, out_dir, prov):
    """Backdrop built from `prov` (the master run's own sampling settings),
    so cells and map always share one convention."""
    from mne.viz import Brain
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    cols = np.tile(np.array(GREY + (1.0,)), (len(c), 1))
    for g in order:
        cols[lab == g] = angles_to_colours([ang[g]])[0]
    for hemi in ('lh', 'rh'):
        try:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir,
                          alpha=SURF_ALPHA)
        except TypeError:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir)
        data = backdrop_texture(hemi, subjects_dir, prov)
        if np.isfinite(data).any():
            try:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=CIRC_CMAP, alpha=BACKDROP_ALPHA,
                               colorbar=False,
                               smoothing_steps=(SURFACE_SMOOTHING_STEPS or None))
            except TypeError:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=CIRC_CMAP, alpha=BACKDROP_ALPHA,
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
        png = str(out_dir / f'{tag}_{hemi}_medial.png')
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
        png_bars = str(out_dir / f'{tag}_{hemi}_medial_bars.png')
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
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(-180, 180))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([-180, -90, 0, 90, 180])
    cb.set_label(f'cell group preferred lag (deg) on fMRI-gradient backdrop.  '
                 f'{subtitle}  [{hemi}]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── profile overlay (bootstrap bands + grey outside line) ────────────
def render_profiles(c, R, subj, consist, lab, order, tag, weighting, out_dir):
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
        col = angles_to_colours([int(L[np.nanargmax(p)])])[0]
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
        fig.savefig(out_dir / f'{tag}_profiles.{ext}', dpi=600)
    plt.close(fig)
    print(f'  wrote {tag}_profiles.pdf/png')


# ── main ─────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--master-csv', type=Path, default=DEFAULT_MASTER_CSV,
                   help='per_cell_master.csv from cell_gradient_master_table.py. '
                        'This one file is all the numeric result needs.')
    p.add_argument('--out', type=Path, default=None,
                   help='output directory (default: <master-csv dir>/final_splits)')
    p.add_argument('--brains', dest='plot_brains', action='store_true',
                   default=PLOT_BRAINS,
                   help=f'render the fsaverage brain overlays '
                        f'(PLOT_BRAINS={PLOT_BRAINS} at the top of this file)')
    p.add_argument('--no-brains', dest='plot_brains', action='store_false',
                   help='skip the brain overlays; write CSVs + profile plots '
                        'only. Needs no mne / 3D backend / fsaverage.')
    # NOTE: symmetrisation, pre-projection smoothing, the harmonic volumes
    # and the gradient mask are intentionally NOT exposed here — they are
    # read from the master run's config.json (see load_master_provenance).
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or args.master_csv.parent / 'final_splits'
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(args.master_csv)
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    consist = c['argmax_r'].to_numpy(float)
    if FMRI_ANGLE_COLUMN not in c.columns:
        raise KeyError(f"{args.master_csv} has no column {FMRI_ANGLE_COLUMN!r}")
    fmri = c[FMRI_ANGLE_COLUMN].to_numpy(float)

    subjects_dir, prov = None, None
    if args.plot_brains:
        prov = load_master_provenance(args.master_csv)
        from mne.datasets import fetch_fsaverage
        subjects_dir = os.path.dirname(fetch_fsaverage())
    else:
        print('(brains off: writing CSVs + profile plots only. Set '
              'PLOT_BRAINS = True at the top, or pass --brains.)')

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
        if args.plot_brains:
            axis_line, boundary_lines = compute_split_lines(
                c, col, axis, inm, n, prov)
            render_brain(c, lab, order, ang, tag,
                         f'{tag} [{w}]: {sub}, outside=grey', subjects_dir,
                         axis_line, boundary_lines, out, prov)
        render_profiles(c, R, subj, consist, lab, order, tag, w, out)

    per_cell.to_csv(out / 'final_splits_per_cell.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'final_splits_summary.csv', index=False)
    pd.set_option('display.width', 240); pd.set_option('display.max_columns', 20)
    print('\n=== GROUP SUMMARY ===')
    print(summary.drop(columns=['boundary_rule', 'n_cells_total']).to_string(
        index=False))
    print(f'\nWrote {out / "final_splits_per_cell.csv"}')
    print(f'Wrote {out / "final_splits_summary.csv"}')
    print(f'All outputs in {out}')


if __name__ == '__main__':
    main()
