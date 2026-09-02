#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project the DSR main effect onto the fsaverage surface so that the surface
rendering looks like the FSLeyes volumetric view.

Why the old rendering broke into "islands"
------------------------------------------
``nilearn.surface.vol_to_surf(vol, pial)`` with a single mesh defaults to
``kind='line'``: it samples the volume along the vertex normal over +-3 mm and
returns the *mean*. For a binary cluster mask, thresholding that mean at 0.5
means "more than half of a 6 mm line must be inside the cluster" — a strong
erosion. Where the elongated medial-wall cluster only grazes the ribbon, the
projected label falls apart into disconnected gyral patches, which is exactly
the two-islands look.

What this script does instead (the FreeSurfer ``--projfrac-max`` recipe)
-----------------------------------------------------------------------
* Samples the volume at N equally spaced depths *between the white and the pial
  surface* — i.e. strictly inside the cortical ribbon, no white-matter or CSF
  contamination.
* Aggregates with ``max`` (t-maps) / ``any`` (masks) rather than ``mean``, so a
  cluster that touches the ribbon anywhere across its thickness shows up.
* Optionally closes 1-2 mesh rings, which welds the remaining pinholes that come
  from the mesh being finer than the 2 mm functional grid.
* Optionally drops surface components below ``MIN_PATCH_VERTICES`` — purely to
  remove single-vertex speckle from clusters that are nowhere near the rendered
  view (e.g. the ventral lOFC cluster shining through the translucent surface).
  The number of dropped vertices is printed and logged in the settings JSON.

Render modes
------------
``cluster_t``   t values inside the FWE-significant cluster only. This is "the
                main effect on the surface" and is the direct analogue of the
                black blob in FSLeyes.
``dual``        the FSLeyes two-layer look: t > ``UNC_T`` in a translucent
                red-yellow ramp, with the FWE-significant cluster drawn on top
                in saturated colour plus a black boundary.
``binary``      solid fill of the FWE cluster + black boundary — the shape that
                the harmonic gradient overlay outlines.

Outputs (PNG + PDF + settings JSON):
    <derivatives>/group/Main_Results_fMRI/dsr_main_effect_surface/<date>/

@author: Svenja Kuchenhoff
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mne.datasets import fetch_fsaverage
from mne.viz import Brain
from mne import Label
from nilearn import surface as nsurf


# ── Settings ─────────────────────────────────────────────────────────
PALM_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group'
    '/Main_Results_fMRI/RSA_quarters_DSR_controls_glmbase_all-paths-fixed'
    '_stickrews_split-buttons_smooth5_palm_p0_01')
STEM = ('cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but'
        '-mask_reward-path_beta_std')
TSTAT_PATH = PALM_DIR / f'{STEM}_vox_tstat_c1.nii'
FWEP_PATH  = PALM_DIR / f'{STEM}_clusterm_tstat_fwep_c1.nii'

OUT_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group'
    '/Main_Results_fMRI/dsr_main_effect_surface')

# PALM stores 1-p, so p < 0.05  <=>  value > 0.95.
P_THRESHOLD = 0.05
# Uncorrected display threshold for the second layer of the `dual` mode.
# t = 1.7 is roughly p < 0.05 one-sided at n = 33.
UNC_T = 1.7
# Colour-scale limits for the t overlay (matches the FSLeyes red-yellow ramp).
T_MIN, T_MAX = 2.5, 3.5

# Ribbon sampling: fractions of the way from white (0) to pial (1).
RIBBON_DEPTHS = np.linspace(0.0, 1.0, 11)
# Mesh closing (dilate then erode) applied to binary labels only.
CLOSE_RINGS = 2
# Surface patches smaller than this are not drawn (speckle removal only).
MIN_PATCH_VERTICES = 40
# Extra mesh-neighbour smoothing MNE applies to the continuous t overlay.
SURFACE_SMOOTHING_STEPS = 5

SURFACES = ('pial', 'inflated')
MODES = ('cluster_t', 'dual', 'binary')
HEMIS = ('lh', 'rh')
VIEWS = ('medial', 'lateral', 'ventral')

BRAIN_SIZE = (1000, 900)
SURF_ALPHA = 0.30           # translucency of the grey cortex (glass brain)
OVERLAY_ALPHA = 0.95        # opacity of the significant-cluster overlay
UNC_ALPHA = 0.40            # opacity of the sub-threshold t layer
OUTLINE_COLOR = 'black'
OUTLINE_WIDTH = 2

# FSL "Red-Yellow" look-alike.
RED_YELLOW = LinearSegmentedColormap.from_list(
    'red_yellow', ['#B30000', '#FF0000', '#FF8C00', '#FFE000', '#FFFF66'])
# Solid fill colour for the `binary` mode.
BINARY_FILL = '#D2321E'

SAVE_PDF = True


# ── Surface helpers ──────────────────────────────────────────────────
def fsaverage_dir():
    return os.path.dirname(fetch_fsaverage(verbose=False))


def load_hemi_meshes(hemi, subjects_dir):
    """Return (white_vertices, pial_vertices, faces) for one hemisphere."""
    surf_dir = os.path.join(subjects_dir, 'fsaverage', 'surf')
    white, faces = nsurf.load_surf_mesh(os.path.join(surf_dir, f'{hemi}.white'))
    pial, _ = nsurf.load_surf_mesh(os.path.join(surf_dir, f'{hemi}.pial'))
    return np.asarray(white), np.asarray(pial), np.asarray(faces)


def project_ribbon(img, white, pial, agg='max', depths=RIBBON_DEPTHS):
    """Sample `img` inside the cortical ribbon and aggregate across depths.

    This is the FreeSurfer ``mri_vol2surf --projfrac-max`` recipe: the value at
    a vertex is the strongest value anywhere between its white and pial
    position, rather than the mean along an arbitrary normal segment.
    """
    data = np.asarray(img.dataobj, dtype=float)
    inv = np.linalg.inv(img.affine)
    shape = np.array(data.shape[:3])
    out = np.full(white.shape[0], -np.inf)
    for frac in depths:
        world = white * (1.0 - frac) + pial * frac
        ijk = np.rint(world @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        vals = np.full(white.shape[0], -np.inf)
        vals[inside] = data[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
        out = np.maximum(out, vals)
    out[~np.isfinite(out)] = 0.0
    if agg == 'max':
        return out
    raise ValueError(f'unknown agg {agg!r}')


def _adjacency(faces, n_vertices):
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    both_r = np.concatenate([rows, cols])
    both_c = np.concatenate([cols, rows])
    return coo_matrix((np.ones(both_r.size), (both_r, both_c)),
                      shape=(n_vertices, n_vertices)).tocsr()


def _dilate(mask_bool, adj, rings):
    out = mask_bool.copy()
    for _ in range(rings):
        out = (adj @ out.astype(float)) > 0
    return out


def close_mesh(mask_bool, faces, rings=CLOSE_RINGS):
    """Morphological closing on the cortical mesh: dilate then erode."""
    if rings <= 0:
        return mask_bool
    adj = _adjacency(faces, mask_bool.size)
    grown = _dilate(mask_bool, adj, rings)
    shrunk = ~_dilate(~grown, adj, rings)
    return shrunk | mask_bool          # never lose original vertices


def drop_small_patches(mask_bool, faces, min_vertices=MIN_PATCH_VERTICES):
    """Remove connected surface patches below `min_vertices`. Returns
    (cleaned_mask, n_dropped_vertices, n_dropped_patches)."""
    if min_vertices <= 1 or not mask_bool.any():
        return mask_bool, 0, 0
    adj = _adjacency(faces, mask_bool.size)
    sub = adj[mask_bool][:, mask_bool]
    n_comp, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    keep_lab = np.flatnonzero(sizes >= min_vertices)
    keep_sub = np.isin(labels, keep_lab)
    cleaned = np.zeros_like(mask_bool)
    cleaned[np.flatnonzero(mask_bool)[keep_sub]] = True
    return (cleaned, int(mask_bool.sum() - cleaned.sum()),
            int(n_comp - keep_lab.size))


def describe_patches(mask_bool, faces, tag=''):
    if not mask_bool.any():
        print(f'    {tag}: empty')
        return
    adj = _adjacency(faces, mask_bool.size)
    sub = adj[mask_bool][:, mask_bool]
    n_comp, labels = connected_components(sub, directed=False)
    sizes = np.sort(np.bincount(labels, minlength=n_comp))[::-1]
    print(f'    {tag}: {int(mask_bool.sum())} vertices in {n_comp} patches '
          f'(largest: {sizes[:5].tolist()})')


# ── Rendering ────────────────────────────────────────────────────────
def _force_surface_alpha(brain, alpha):
    """Older MNE Brain versions reject ``alpha=`` in the constructor; walk the
    underlying VTK actors and set opacity directly instead."""
    renderer = getattr(brain, '_renderer', None)
    fig = (getattr(renderer, 'plotter', None)
           or getattr(renderer, 'figure', None)) if renderer else None
    actors = getattr(fig, 'actors', None) if fig is not None else None
    if actors is None and getattr(fig, 'renderer', None) is not None:
        actors = getattr(fig.renderer, 'actors', None)
    if not actors:
        return
    for actor in (actors.values() if hasattr(actors, 'values') else actors):
        try:
            actor.GetProperty().SetOpacity(float(alpha))
        except Exception:
            continue


def make_brain(hemi, surface, subjects_dir):
    """The one place the glass-brain look is defined.

    ``harmonic_maps_brain_overlay`` calls this too, so the gradient overlay and
    the main-effect panel are the same brain, size, background and shading.
    """
    kwargs = dict(subject='fsaverage', hemi=hemi, surf=surface,
                  background='white', size=BRAIN_SIZE,
                  subjects_dir=subjects_dir, cortex='low_contrast')
    alpha = SURF_ALPHA if surface == 'pial' else 1.0
    try:
        return Brain(alpha=alpha, **kwargs)
    except TypeError:
        brain = Brain(**kwargs)
        _force_surface_alpha(brain, alpha)
        return brain


def add_outline(brain, mask_bool, hemi, name):
    verts = np.flatnonzero(mask_bool)
    if not verts.size:
        return
    brain.add_label(Label(vertices=verts, hemi=hemi, name=name),
                    color=OUTLINE_COLOR, alpha=1.0, borders=OUTLINE_WIDTH)


def add_scalar(brain, values, hemi, fmin, fmax, cmap, alpha,
               vertex_alpha=None):
    """Add ONE scalar overlay.

    MNE keeps a single overlay named 'data' per hemisphere, so calling this
    twice replaces rather than stacks the layer. To show two thresholds at
    once, pass a single scalar map plus `vertex_alpha` — a per-vertex opacity
    vector — which is what the `dual` mode does.
    """
    data = np.where(np.isfinite(values), values, np.nan)
    kwargs = dict(hemi=hemi, fmin=fmin, fmid=(fmin + fmax) / 2.0, fmax=fmax,
                  colormap=cmap, alpha=alpha, colorbar=False)
    try:
        brain.add_data(data, smoothing_steps=int(SURFACE_SMOOTHING_STEPS),
                       **kwargs)
    except TypeError:
        brain.add_data(data, **kwargs)
    if vertex_alpha is not None:
        # MNE's public `alpha` is scalar-only, but the layered mesh accepts
        # one opacity per vertex.
        brain._layered_meshes[hemi].update_overlay(name='data',
                                                   opacity=vertex_alpha)
        brain._renderer._update()


def save_with_colorbar(brain, png_path, pdf_path, cbar):
    brain.save_image(str(png_path))
    if cbar is None:
        return
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    img = plt.imread(str(png_path))
    h_in, w_in = img.shape[0] / 300, img.shape[1] / 300
    fig = plt.figure(figsize=(w_in, h_in + 0.55), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 55], hspace=0.02,
                          left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis('off')
    ax_cb = fig.add_subplot(gs[1])
    pos = ax_cb.get_position()
    ax_cb.set_position([0.28, pos.y0, 0.44, pos.height * 0.4])
    sm = ScalarMappable(cmap=cbar['cmap'],
                        norm=Normalize(vmin=cbar['vmin'], vmax=cbar['vmax']))
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_ticks(cbar['ticks'])
    cb.set_ticklabels([f'{t:g}' for t in cbar['ticks']])
    cb.set_label(cbar['label'], fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


def render(mode, surface, hemi, view, textures, faces, subjects_dir, out_dir):
    sig_mask, t_sig, t_unc = textures
    if not sig_mask.any() and mode != 'dual':
        print(f'    [skip] no significant vertices on {hemi}')
        return
    brain = make_brain(hemi, surface, subjects_dir)
    cbar = {'cmap': RED_YELLOW, 'vmin': T_MIN, 'vmax': T_MAX,
            'ticks': [T_MIN, (T_MIN + T_MAX) / 2, T_MAX],
            'label': 't (DSR main effect)'}

    if mode == 'cluster_t':
        add_scalar(brain, t_sig, hemi, T_MIN, T_MAX, RED_YELLOW, OVERLAY_ALPHA)
        add_outline(brain, sig_mask, hemi, 'DSR_sig')
    elif mode == 'dual':
        # One overlay spanning UNC_T..T_MAX, with the FWE-significant vertices
        # opaque and the merely uncorrected ones faded. This reproduces the
        # FSLeyes two-layer look in a single MNE data layer.
        show = np.isfinite(t_unc) | sig_mask
        combined = np.where(show, np.fmax(np.nan_to_num(t_unc, nan=-np.inf),
                                          np.nan_to_num(t_sig, nan=-np.inf)),
                            np.nan)
        vertex_alpha = np.where(sig_mask, OVERLAY_ALPHA,
                                np.where(np.isfinite(t_unc), UNC_ALPHA, 0.0))
        add_scalar(brain, combined, hemi, UNC_T, T_MAX, RED_YELLOW,
                   OVERLAY_ALPHA, vertex_alpha=vertex_alpha)
        add_outline(brain, sig_mask, hemi, 'DSR_sig')
        cbar = {'cmap': RED_YELLOW, 'vmin': UNC_T, 'vmax': T_MAX,
                'ticks': [UNC_T, (UNC_T + T_MAX) / 2, T_MAX],
                'label': f't (faded: t > {UNC_T:g} uncorrected;\n'
                         f'opaque: cluster-mass FWE p < {P_THRESHOLD:g})'}
    elif mode == 'binary':
        brain.add_label(Label(vertices=np.flatnonzero(sig_mask), hemi=hemi,
                              name='DSR_sig_fill'),
                        color=BINARY_FILL, alpha=0.85, borders=False)
        add_outline(brain, sig_mask, hemi, 'DSR_sig')
        cbar = None
    else:
        raise ValueError(mode)

    try:
        brain.show_view(view)
    except Exception as exc:
        print(f'    [warn] show_view({view!r}) failed: {exc}')

    stem = out_dir / f'DSRmain__{mode}__{surface}__{hemi}_{view}'
    try:
        save_with_colorbar(brain, stem.with_suffix('.png'),
                           stem.with_suffix('.pdf'), cbar)
        print(f'    wrote {stem.name}.png')
    except Exception as exc:
        print(f'    [warn] save failed: {exc}')
    finally:
        try:
            brain.close()
        except Exception:
            pass


def main():
    subjects_dir = fsaverage_dir()
    out_dir = OUT_ROOT / datetime.now().strftime('%Y%m%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    t_img = nib.load(str(TSTAT_PATH))
    p_img = nib.load(str(FWEP_PATH))
    print(f'  t map:   {TSTAT_PATH.name}')
    print(f'  fwe map: {FWEP_PATH.name}')

    dropped_log = {}
    for hemi in HEMIS:
        white, pial, faces = load_hemi_meshes(hemi, subjects_dir)
        print(f'\n=== {hemi} ===')
        t_surf = project_ribbon(t_img, white, pial, agg='max')
        p_surf = project_ribbon(p_img, white, pial, agg='max')

        raw_sig = p_surf > (1.0 - P_THRESHOLD)
        describe_patches(raw_sig, faces, 'ribbon-max, raw')
        closed = close_mesh(raw_sig, faces, CLOSE_RINGS)
        describe_patches(closed, faces, f'after {CLOSE_RINGS}-ring closing')
        sig, n_drop_v, n_drop_p = drop_small_patches(closed, faces)
        describe_patches(sig, faces, f'after dropping <{MIN_PATCH_VERTICES}-vertex patches')
        print(f'    dropped {n_drop_v} vertices in {n_drop_p} small patches')
        dropped_log[hemi] = {'dropped_vertices': n_drop_v,
                             'dropped_patches': n_drop_p,
                             'kept_vertices': int(sig.sum())}

        t_sig = np.where(sig, t_surf, np.nan)
        t_unc = np.where(t_surf > UNC_T, t_surf, np.nan)
        textures = (sig, t_sig, t_unc)

        for surface in SURFACES:
            for mode in MODES:
                for view in VIEWS:
                    print(f'  [{surface} | {mode} | {view}]')
                    render(mode, surface, hemi, view, textures, faces,
                           subjects_dir, out_dir)

    settings = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'tstat_map': str(TSTAT_PATH),
        'fwep_map': str(FWEP_PATH),
        'p_threshold_cluster_fwe': P_THRESHOLD,
        'uncorrected_display_t': UNC_T,
        'colour_limits_t': [T_MIN, T_MAX],
        'projection': 'ribbon max across white->pial depths (projfrac-max)',
        'ribbon_depths': RIBBON_DEPTHS.tolist(),
        'mesh_closing_rings': CLOSE_RINGS,
        'min_patch_vertices': MIN_PATCH_VERTICES,
        'surface_smoothing_steps': SURFACE_SMOOTHING_STEPS,
        'surfaces': list(SURFACES), 'modes': list(MODES),
        'hemis': list(HEMIS), 'views': list(VIEWS),
        'surface_alpha': SURF_ALPHA, 'overlay_alpha': OVERLAY_ALPHA,
        'unc_layer_alpha': UNC_ALPHA,
        'speckle_removal': dropped_log,
    }
    with open(out_dir / 'settings.json', 'w') as fh:
        json.dump(settings, fh, indent=2)
    print(f'\nDone → {out_dir}')


if __name__ == '__main__':
    main()
