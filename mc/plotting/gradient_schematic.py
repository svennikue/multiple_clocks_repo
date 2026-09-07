"""Method schematic for the anatomical-gradient (harmonic angle) analysis.

Explains, panel by panel, what ``scripts/harmonic_angle_maps.py`` does:

  a) the concurrent code is cut into four quarters — current, next, +2, +3 —
     which enter one searchlight GLM as four competing regressors
  b) every voxel therefore has a profile of four β's, one per future quarter
  c) that profile is projected onto the first Fourier harmonic, using the
     CENTRE of each angular bin (45°, 135°, 225°, 315°) as its angle
  d) cos and sin are the two components of one vector per voxel: its ANGLE is
     the preferred future step, its LENGTH is how strong the effect is
  e) each subject contributes their own vector; the group mean is tested
     against (0, 0) with Hotelling T² — does this voxel carry harmonic signal?
  f) projected onto the unit circle instead, the same subjects give the mean
     resultant length R̄ and the Rayleigh test — do subjects agree on the
     ANGLE, whatever its size? (this is what ``USE_UNIT_VECTOR_MAPS`` switches on)
  g) doing that at every voxel gives the preferred-angle map

Everything plotted is real: the β profiles, the per-subject vectors and the
angle map are read from the analysis outputs, and the two example voxels are
picked by a stated rule (see ``pick_example_voxels``), not by hand.

Style, colours and the cm-based layout are shared with
``mc.plotting.method_schematic``.

@author: Svenja Kuechenhoff
"""

import os
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyArrowPatch, Arc

from mc.plotting.method_schematic import (
    CM, FONT_TITLE, FONT_LABEL, FONT_TICK, FUTURE_CMAP, lag_colors,
    STATE_COLORS, MATCH_COLOR, OFF_COLOR, RDM_CM, W_CONCURRENT,
    rc_context, _add_ax_cm, _cm_canvas, _new_page, _strip_cm, _lag_bar_cm,
    _save, _title, build_fmri_examples)

QUARTER_NAMES = ['current', 'next', '+2', '+3']
VOXEL_COLORS = {'ventral': '#111111', 'dorsal': '#8a8a8a'}
VOXEL_MARKERS = {'ventral': 'o', 'dorsal': 's'}


# ══════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════

def _resample_mask(mask_path, ref_img):
    img = nib.load(str(mask_path))
    if img.shape[:3] == ref_img.shape[:3] and np.allclose(
            img.affine, ref_img.affine, atol=1e-3):
        return img.get_fdata() > 0.5
    from nilearn.image import resample_img
    return resample_img(img, target_affine=ref_img.affine,
                        target_shape=ref_img.shape[:3],
                        interpolation='nearest').get_fdata() > 0.5


def load_gradient_data(beta_dir, harmonic_dir, mask_path, beta_files):
    """Per-subject β per quarter, the harmonic maps, and the mPFC mask."""
    imgs = [nib.load(str(Path(beta_dir) / f)) for f in beta_files]
    ref = imgs[0]
    betas = np.stack([i.get_fdata() for i in imgs], axis=-1)  # X,Y,Z,subj,step
    H = Path(harmonic_dir)
    out = dict(
        ref=ref, betas=betas, n_subj=betas.shape[3], n_steps=betas.shape[4],
        cos_persubj=nib.load(str(H / 'cos_persubj.nii.gz')).get_fdata(),
        sin_persubj=nib.load(str(H / 'sin_persubj.nii.gz')).get_fdata(),
        angle_deg=nib.load(str(H / 'angle_deg.nii.gz')).get_fdata(),
        amplitude=nib.load(str(H / 'amplitude.nii.gz')).get_fdata(),
        hotelling_F=nib.load(str(H / 'hotelling_F.nii.gz')).get_fdata(),
        hotelling_p=nib.load(str(H / 'hotelling_p.nii.gz')).get_fdata(),
        rayleigh_R=nib.load(str(H / 'rayleigh_R.nii.gz')).get_fdata(),
        rayleigh_p=nib.load(str(H / 'rayleigh_p.nii.gz')).get_fdata(),
        mask=_resample_mask(mask_path, ref),
        harmonic_dir=H,
    )
    # Fourier weights, as in harmonic_angle_maps.py: BIN CENTRES.
    n = out['n_steps']
    out['theta_deg'] = 180.0 / n + 360.0 / n * np.arange(n)
    out['theta'] = np.radians(out['theta_deg'])
    return out


def pick_example_voxels(D, quantile=20):
    """Two illustrative voxels, chosen by a fixed rule (never by hand).

    Among mPFC voxels whose Hotelling T² is significant at p < 0.05, take the
    ventral and dorsal quintiles along MNI z, and within each take the voxel
    with the largest group amplitude.
    """
    sig = (D['hotelling_p'] < 0.05) & D['mask']
    ii, jj, kk = np.where(sig)
    mni = nib.affines.apply_affine(D['ref'].affine, np.stack([ii, jj, kk], 1))
    z = mni[:, 2]
    lo, hi = np.percentile(z, [quantile, 100 - quantile])
    picks = {}
    for tag, sel in (('ventral', z <= lo), ('dorsal', z >= hi)):
        idx = np.where(sel)[0]
        best = idx[np.argmax(D['amplitude'][ii[idx], jj[idx], kk[idx]])]
        v = (int(ii[best]), int(jj[best]), int(kk[best]))
        picks[tag] = dict(
            vox=v, mni=mni[best].astype(int).tolist(),
            beta=D['betas'][v],                       # (n_subj, n_steps)
            cos=D['cos_persubj'][v], sin=D['sin_persubj'][v],
            angle=float(D['angle_deg'][v]), amp=float(D['amplitude'][v]),
            F=float(D['hotelling_F'][v]), p=float(D['hotelling_p'][v]),
            R=float(D['rayleigh_R'][v]), p_rayleigh=float(D['rayleigh_p'][v]),
            n_sig_pool=int(sig.sum()), z_cut=(float(lo), float(hi)))
    return picks


# ══════════════════════════════════════════════════════════════════════════
#  small drawing helpers
# ══════════════════════════════════════════════════════════════════════════

def _angle_ring(ax, r_in=0.86, r_out=1.0, n=180, lw=0):
    """A thin cyclic colour ring: which angle means which future step."""
    from matplotlib.patches import Wedge
    for k in range(n):
        th1, th2 = 360 * k / n, 360 * (k + 1) / n
        ax.add_patch(Wedge((0, 0), r_out, th1, th2, width=r_out - r_in,
                           facecolor=FUTURE_CMAP((k + 0.5) / n),
                           edgecolor='none', zorder=1))


def _polar_axes(ax, lim, ring=True, ticks_deg=(0, 90, 180, 270),
                tick_labels=None, ring_r=1.30, tick_r=1.12, width=0.13):
    """Square cos/sin plane with the cyclic angle ring around it.

    All radii are multiples of ``lim`` so a panel can push the ring outwards
    to make room for annotations inside it.
    """
    ax.set_aspect('equal')
    ax.axis('off')
    span = lim * (ring_r + 0.03)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    if ring:
        from matplotlib.patches import Wedge
        n = 180
        for k in range(n):
            th1, th2 = 360 * k / n, 360 * (k + 1) / n
            ax.add_patch(Wedge((0, 0), lim * ring_r, th1, th2,
                               width=lim * width,
                               facecolor=FUTURE_CMAP((k + 0.5) / n),
                               edgecolor='none', zorder=1))
    ax.plot([-lim * (ring_r - width), lim * (ring_r - width)], [0, 0],
            color='#bbbbbb', lw=0.5, zorder=0)
    ax.plot([0, 0], [-lim * (ring_r - width), lim * (ring_r - width)],
            color='#bbbbbb', lw=0.5, zorder=0)
    for d, lab in zip(ticks_deg, tick_labels or [f"{d}°" for d in ticks_deg]):
        a = np.radians(d)
        ax.text(lim * tick_r * np.cos(a), lim * tick_r * np.sin(a), lab,
                ha='center', va='center', fontsize=FONT_TICK, zorder=5)


def _arrow(ax, x, y, color, lw=1.6, alpha=1.0, zorder=6, scale=8):
    ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle='-|>',
                                 mutation_scale=scale, linewidth=lw,
                                 color=color, alpha=alpha, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def _conf_ellipse(ax, xs, ys, conf=0.95, **kw):
    """95 % confidence ellipse for the MEAN — the region Hotelling T² tests."""
    from scipy.stats import f as f_dist
    n = len(xs)
    S = np.cov(np.vstack([xs, ys]))
    t2_crit = 2 * (n - 1) / (n - 2) * f_dist.ppf(conf, 2, n - 2)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    w, h = 2 * np.sqrt(vals * t2_crit / n)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((xs.mean(), ys.mean()), w, h, angle=ang, **kw))
    return w, h


# ══════════════════════════════════════════════════════════════════════════
#  PANELS
# ══════════════════════════════════════════════════════════════════════════

A_LAG = 0.20
A_STRIP = 0.55
A_BRACKET = 0.34
A_LABEL = 0.44
A_H = 0.42 + A_LAG + A_STRIP + A_BRACKET + A_LABEL + 0.16


def _dark(c, f=0.45):
    import matplotlib.colors as mc_
    r, g, b = mc_.to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def panel_a_quarters(ax, ex, bin_idx=1, w=W_CONCURRENT):
    """The concurrent code cut into four quarters = four GLM regressors."""
    n, block = ex['n_bins'], ex['len_per_bin']
    _cm_canvas(ax, w, A_H)
    y0 = 0.42
    cols = lag_colors(4)

    for k, txt in enumerate(['now', '+90°', '+180°', '+270°']):
        ax.text((k + 0.5) * w / 4, y0 - 0.06, txt, ha='center', va='bottom',
                fontsize=FONT_TICK)
    _lag_bar_cm(ax, n, 0, w, y0, A_LAG)
    _strip_cm(ax, ex['dsr'][bin_idx], 0, w, y0 + A_LAG, A_STRIP, block=block)

    y_b = y0 + A_LAG + A_STRIP + 0.10
    for q in range(4):
        x0, x1 = q * w / 4, (q + 1) * w / 4
        ax.plot([x0 + 0.03, x0 + 0.03, x1 - 0.03, x1 - 0.03],
                [y_b - 0.06, y_b + 0.10, y_b + 0.10, y_b - 0.06],
                color=_dark(cols[q]), lw=1.2, solid_joinstyle='miter')
        ax.text((x0 + x1) / 2, y_b + A_BRACKET, QUARTER_NAMES[q],
                ha='center', va='bottom', fontsize=FONT_TICK,
                color=_dark(cols[q]), fontweight='bold')
    ax.text(w / 2, A_H - 0.02,
            'four competing regressors in one searchlight GLM',
            ha='center', va='bottom', fontsize=FONT_TICK, color='#555555')


def panel_b_profiles(ax, picks, theta_deg):
    """The four β's per voxel — the profile the harmonic is fitted to."""
    n_steps = picks['ventral']['beta'].shape[1]
    x = np.arange(n_steps)
    cols = lag_colors(4)
    for k in range(n_steps):
        ax.add_patch(Rectangle((k - 0.5, -1), 1, 2, facecolor=cols[k],
                               alpha=0.16, edgecolor='none', zorder=0))
    for tag, pk in picks.items():
        m = pk['beta'].mean(axis=0)
        se = pk['beta'].std(axis=0, ddof=1) / np.sqrt(pk['beta'].shape[0])
        ax.errorbar(x, m, yerr=se, color=VOXEL_COLORS[tag], lw=1.4,
                    marker=VOXEL_MARKERS[tag], ms=3.2, capsize=2,
                    elinewidth=0.9, zorder=4,
                    label=f"{tag}  z = {pk['mni'][2]:+d}")
    ax.axhline(0, color='#666666', lw=0.6, zorder=1)
    lim = 1.35 * max(np.abs(pk['beta'].mean(0)).max() for pk in picks.values())
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.5, n_steps - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n{d:.0f}°" for q, d in zip(QUARTER_NAMES, theta_deg)],
                       fontsize=FONT_TICK)
    ax.set_ylabel('β (z-scored)', fontsize=FONT_LABEL, labelpad=1)
    ax.set_xlabel('future quarter', fontsize=FONT_LABEL, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK, length=1.6, pad=1.5)
    ax.legend(fontsize=FONT_TICK, frameon=False, loc='lower left',
              handlelength=1.1, borderpad=0.1, labelspacing=0.15,
              bbox_to_anchor=(0.0, 0.0))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.6)


def panel_c_weights(ax, theta_deg):
    """Each quarter gets an angle; cos and sin of it are its weights."""
    cols = lag_colors(4)
    _polar_axes(ax, 1.0, tick_labels=['0°\nnow', '90°', '180°', '270°'])
    for k, (d, q) in enumerate(zip(theta_deg, QUARTER_NAMES)):
        a = np.radians(d)
        _arrow(ax, np.cos(a), np.sin(a), cols[k], lw=1.5)
        ax.text(0.62 * np.cos(a), 0.62 * np.sin(a), q, ha='center',
                va='center', fontsize=FONT_TICK, color=cols[k],
                fontweight='bold', zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.75))
        ax.text(1.02 * np.cos(a), 1.02 * np.sin(a), f"{d:.0f}°", ha='center',
                va='center', fontsize=FONT_TICK, zorder=8,
                bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.8))
    ax.text(0, -1.45, r'$\cos = \sum_k \cos\theta_k \cdot \beta_k$'
                      '\n'
                      r'$\sin = \sum_k \sin\theta_k \cdot \beta_k$',
            ha='center', va='top', fontsize=FONT_LABEL)


def panel_d_vector(ax, picks, lim=None):
    """cos and sin are one vector: angle = preferred step, length = effect."""
    if lim is None:
        lim = max(pk['amp'] for pk in picks.values())
    _polar_axes(ax, lim, ring_r=1.62, tick_r=1.44,
                tick_labels=['0°\nnow', '90°', '180°', '270°'])
    for r_frac, off, (tag, pk) in zip((0.34, 0.52), (-34, 34), picks.items()):
        cx, cy = pk['cos'].mean(), pk['sin'].mean()
        _arrow(ax, cx, cy, VOXEL_COLORS[tag], lw=1.8)
        ax.plot([cx, cx], [0, cy], color=VOXEL_COLORS[tag], lw=0.6, ls=':')
        ax.plot([0, cx], [cy, cy], color=VOXEL_COLORS[tag], lw=0.6, ls=':')
        a = np.radians(pk['angle'])
        ax.add_patch(Arc((0, 0), lim * r_frac, lim * r_frac, theta1=0,
                         theta2=pk['angle'], color=VOXEL_COLORS[tag], lw=0.7))
        a_lab = np.radians(pk['angle'] + off)
        ax.plot([cx, lim * 1.05 * np.cos(a_lab)],
                [cy, lim * 1.05 * np.sin(a_lab)],
                color=VOXEL_COLORS[tag], lw=0.5, zorder=8)
        ax.text(lim * 1.18 * np.cos(a_lab), lim * 1.18 * np.sin(a_lab),
                f"{tag}\n{pk['angle']:.0f}°", fontsize=FONT_TICK,
                color=VOXEL_COLORS[tag], zorder=9, ha='center', va='center',
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.9))
    ax.text(lim * 0.55, -lim * 0.15, 'cos', ha='center', va='center',
            fontsize=FONT_TICK, color='#999999')
    ax.text(-lim * 0.34, lim * 0.55, 'sin', ha='center', va='center',
            fontsize=FONT_TICK, color='#999999')
    ax.text(0, -lim * 1.72, 'angle = preferred future step\n'
                            'length = effect size (amplitude)',
            ha='center', va='top', fontsize=FONT_LABEL)


def panel_e_subjects(ax, pk, tag='ventral'):
    """Every subject has their own vector; Hotelling T² tests the mean."""
    cs, ss = pk['cos'], pk['sin']
    lim = 1.05 * np.nanpercentile(np.hypot(cs, ss), 97)
    _polar_axes(ax, lim, tick_labels=['0°', '90°', '180°', '270°'])
    for c, s in zip(cs, ss):
        _arrow(ax, c, s, '#9d9d9d', lw=0.6, alpha=0.75, zorder=3, scale=4)
    _conf_ellipse(ax, cs, ss, facecolor=VOXEL_COLORS[tag], alpha=0.22,
                  edgecolor=VOXEL_COLORS[tag], lw=0.8, zorder=5)
    _arrow(ax, cs.mean(), ss.mean(), VOXEL_COLORS[tag], lw=2.0, zorder=7)
    ax.plot(0, 0, marker='o', ms=3.5, mfc='white', mec='#333333', mew=0.8,
            zorder=8)
    ax.text(0, -lim * 1.46,
            f"{len(cs)} subject vectors, group mean\n"
            f"Hotelling F = {pk['F']:.2f}, p = {pk['p']:.4f}",
            ha='center', va='top', fontsize=FONT_LABEL)


def panel_f_rayleigh(ax, pk, tag='ventral'):
    """The same subjects on the unit circle: do they agree on the ANGLE?"""
    cs, ss = pk['cos'], pk['sin']
    th = np.arctan2(ss, cs)
    _polar_axes(ax, 1.0, tick_labels=['0°', '90°', '180°', '270°'])
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor='#cccccc', lw=0.6,
                        zorder=2))
    for t in th:
        ax.plot(np.cos(t), np.sin(t), marker='o', ms=3.0,
                mfc='#9d9d9d', mec='white', mew=0.4, zorder=4)
        ax.plot([0, np.cos(t)], [0, np.sin(t)], color='#dddddd', lw=0.4,
                zorder=3)
    C, S = np.cos(th).mean(), np.sin(th).mean()
    _arrow(ax, C, S, VOXEL_COLORS[tag], lw=2.0, zorder=7)
    ax.add_patch(Circle((0, 0), np.hypot(C, S), fill=False,
                        edgecolor=VOXEL_COLORS[tag], lw=0.7, ls=':', zorder=5))
    verdict = ('angles agree' if pk['p_rayleigh'] < 0.05
               else 'angles scattered')
    ax.text(0, -1.46,
            f"equal weight per subject\n"
            f"Rayleigh $\\bar{{R}}$ = {pk['R']:.2f}, p = {pk['p_rayleigh']:.3f}\n"
            f"→ {verdict}",
            ha='center', va='top', fontsize=FONT_LABEL)


def best_sagittal_slice(D, sig_file='hotelling_sig_p05_mPFC.nii.gz'):
    """The sagittal slice carrying the most suprathreshold mPFC voxels."""
    sig = nib.load(str(D['harmonic_dir'] / sig_file)).get_fdata() > 0.5
    counts = sig.sum(axis=(1, 2))
    xi = int(np.argmax(counts))
    x_mni = float(nib.affines.apply_affine(D['ref'].affine,
                                           np.array([xi, 0, 0]))[0])
    return xi, x_mni, int(counts[xi])


def panel_g_map(ax_brain, ax_wheel, D, template_path, x_mni=None,
                angle_file='angle_deg.nii.gz',
                sig_file='hotelling_sig_p05_mPFC.nii.gz',
                y_range=(-12, 74), z_range=(-26, 54)):
    """The preferred-angle map: one sagittal slice plus the angle wheel."""
    ref = D['ref']
    aff = ref.affine
    tmpl = nib.load(str(template_path))
    if tmpl.shape[:3] != ref.shape[:3]:
        from nilearn.image import resample_img
        tmpl = resample_img(tmpl, target_affine=aff,
                            target_shape=ref.shape[:3], interpolation='linear')
    ang = nib.load(str(D['harmonic_dir'] / angle_file)).get_fdata()
    sig = nib.load(str(D['harmonic_dir'] / sig_file)).get_fdata() > 0.5
    ang = np.where(sig, ang, np.nan)

    if x_mni is None:
        xi, x_mni, _ = best_sagittal_slice(D, sig_file)
    else:
        xi = int(round(np.linalg.inv(aff).dot([x_mni, 0, 0, 1])[0]))

    ny, nz = ref.shape[1], ref.shape[2]
    y0, dy = aff[1, 3], aff[1, 1]
    z0, dz = aff[2, 3], aff[2, 2]
    extent = [y0 - dy / 2, y0 + (ny - 0.5) * dy,
              z0 - dz / 2, z0 + (nz - 0.5) * dz]
    ax_brain.imshow(tmpl.get_fdata()[xi].T, cmap='gray', origin='lower',
                    extent=extent, interpolation='bilinear', aspect='equal')
    # FUTURE_CMAP is cyclic with 0 deg at its low end, so the angle has to be
    # wrapped into [0, 360) before it is looked up — a -180..180 range would
    # put 0 deg in the middle of the map (blue) instead of at yellow.
    ax_brain.imshow(np.mod(ang[xi].T, 360.0), cmap=FUTURE_CMAP,
                    vmin=0, vmax=360, origin='lower', extent=extent,
                    interpolation='nearest', aspect='equal')
    ax_brain.set_xlim(*y_range)
    ax_brain.set_ylim(*z_range)
    ax_brain.set_xticks([y_range[0] + 6, 0, y_range[1] - 6])
    ax_brain.set_yticks([z_range[0] + 6, 0, z_range[1] - 6])
    ax_brain.set_xlabel('MNI y (mm)', fontsize=FONT_TICK, labelpad=1)
    ax_brain.set_ylabel('MNI z (mm)', fontsize=FONT_TICK, labelpad=1)
    ax_brain.tick_params(labelsize=FONT_TICK, length=1.6, pad=1.5,
                         colors='#333333')
    for sp in ax_brain.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color('#333333')
    ax_brain.text(0.03, 0.97, f"x = {x_mni:+.0f} mm\nHotelling p < 0.05, mPFC",
                  transform=ax_brain.transAxes, ha='left', va='top',
                  fontsize=FONT_TICK, color='white', zorder=9)

    from matplotlib.patches import Wedge
    ax_wheel.set_aspect('equal')
    ax_wheel.axis('off')
    ax_wheel.set_xlim(-1.45, 1.45)
    ax_wheel.set_ylim(-1.45, 1.45)
    for k in range(180):
        ax_wheel.add_patch(Wedge((0, 0), 1.0, 360 * k / 180,
                                 360 * (k + 1) / 180,
                                 facecolor=FUTURE_CMAP((k + 0.5) / 180),
                                 edgecolor='none', zorder=1))
    ax_wheel.add_patch(Circle((0, 0), 0.46, facecolor='white',
                              edgecolor='#333333', lw=0.7, zorder=3))
    ax_wheel.text(0, 0, 'preferred\nangle', ha='center', va='center',
                  fontsize=FONT_TICK, zorder=4)
    for d, lab in zip((0, 90, 180, 270),
                      ('0°\nnow', '90°', '180°', '270°')):
        a = np.radians(d)
        ax_wheel.text(1.18 * np.cos(a), 1.18 * np.sin(a), lab, ha='center',
                      va='center', fontsize=FONT_TICK)
    return x_mni





# ══════════════════════════════════════════════════════════════════════════
#  figures — every plotted box has the physical size declared here
# ══════════════════════════════════════════════════════════════════════════

SQ_CM = RDM_CM            # 4 cm — the square panels (b, c, d, e, f, wheel)
BRAIN_W_CM = 5.5
BRAIN_H_CM = 4.0


def save_panels(D, picks, ex, template_path, out_dir, prefix='gradient',
                x_mni=6, show=False):
    """Render every panel at its declared physical size."""
    made = {}
    theta_deg = D['theta_deg']
    with rc_context():
        # a — the quarters split
        fig, page = _new_page(W_CONCURRENT + 0.5, A_H + 1.0)
        _title(fig, page, 'a  concurrent code → four quarters')
        panel_a_quarters(_add_ax_cm(fig, page, 0.25, 0.75, W_CONCURRENT, A_H), ex)
        _save(fig, os.path.join(out_dir, f'{prefix}_a_quarters'), show)
        made['a_quarters'] = f"{W_CONCURRENT} × {A_H:.2f} cm"

        # b — the per-voxel β profile
        fig, page = _new_page(1.75 + SQ_CM + 0.25, 0.85 + SQ_CM + 1.30)
        _title(fig, page, 'b  four β per voxel')
        panel_b_profiles(_add_ax_cm(fig, page, 1.75, 0.85, SQ_CM, SQ_CM),
                         picks, theta_deg)
        _save(fig, os.path.join(out_dir, f'{prefix}_b_beta_profile'), show)
        made['b_beta_profile'] = f"{SQ_CM}×{SQ_CM} cm"

        # c — the Fourier weights
        fig, page = _new_page(SQ_CM + 1.2, 0.85 + SQ_CM + 1.85)
        _title(fig, page, 'c  each quarter has an angle')
        panel_c_weights(_add_ax_cm(fig, page, 0.6, 0.85, SQ_CM, SQ_CM),
                        theta_deg)
        _save(fig, os.path.join(out_dir, f'{prefix}_c_weights'), show)
        made['c_weights'] = f"{SQ_CM}×{SQ_CM} cm"

        # d — the resulting vector
        fig, page = _new_page(SQ_CM + 1.2, 0.85 + SQ_CM + 1.65)
        _title(fig, page, 'd  cos + sin = one vector')
        panel_d_vector(_add_ax_cm(fig, page, 0.6, 0.85, SQ_CM, SQ_CM), picks)
        _save(fig, os.path.join(out_dir, f'{prefix}_d_vector'), show)
        made['d_vector'] = f"{SQ_CM}×{SQ_CM} cm"

        # e — subjects + Hotelling
        fig, page = _new_page(SQ_CM + 1.2, 0.85 + SQ_CM + 1.65)
        _title(fig, page, 'e  subjects → group mean')
        panel_e_subjects(_add_ax_cm(fig, page, 0.6, 0.85, SQ_CM, SQ_CM),
                         picks['ventral'], 'ventral')
        _save(fig, os.path.join(out_dir, f'{prefix}_e_hotelling'), show)
        made['e_hotelling'] = f"{SQ_CM}×{SQ_CM} cm"

        # f — unit vectors + Rayleigh
        fig, page = _new_page(SQ_CM + 1.2, 0.85 + SQ_CM + 1.65)
        _title(fig, page, 'f  angle agreement only')
        panel_f_rayleigh(_add_ax_cm(fig, page, 0.6, 0.85, SQ_CM, SQ_CM),
                         picks['ventral'], 'ventral')
        _save(fig, os.path.join(out_dir, f'{prefix}_f_rayleigh'), show)
        made['f_rayleigh'] = f"{SQ_CM}×{SQ_CM} cm"

        # g — the map
        fig, page = _new_page(BRAIN_W_CM + 4.8, 0.85 + BRAIN_H_CM + 1.5)
        _title(fig, page, 'g  preferred-angle map')
        panel_g_map(_add_ax_cm(fig, page, 1.05, 0.85, BRAIN_W_CM, BRAIN_H_CM),
                    _add_ax_cm(fig, page, BRAIN_W_CM + 1.6, 1.0, 2.8, 2.8),
                    D, template_path, x_mni=x_mni)
        _save(fig, os.path.join(out_dir, f'{prefix}_g_angle_map'), show)
        made['g_angle_map'] = f"brain {BRAIN_W_CM}×{BRAIN_H_CM} cm, wheel 2.8 cm"
    return made


def make_gradient_figure(D, picks, ex, template_path, save_stem,
                         x_mni=6, show=False):
    """One overview page with every panel at its true physical size."""
    theta_deg = D['theta_deg']
    lab = 1.75
    gap = 1.35
    col = lab + SQ_CM
    page_w = max(W_CONCURRENT + 1.0, 3 * col + 2 * (gap - lab) + 1.0)

    y = 1.25
    y_a = y + 0.55;      y += 0.55 + A_H + 1.2
    y_row1 = y + 0.60;   y += 0.60 + SQ_CM + 2.05
    y_row2 = y + 0.60;   y += 0.60 + SQ_CM + 2.45
    y_g = y + 0.60;      y += 0.60 + BRAIN_H_CM + 1.3
    page_h = y

    with rc_context():
        fig, page = _new_page(page_w, page_h)
        fig.text(0.05 / page_w, 1 - 0.45 / page_h,
                 'Preferred future-step angle: from four quarter-regressors '
                 'to one vector per voxel',
                 fontsize=FONT_TITLE, ha='left', va='center')

        _title(fig, page, 'a  the concurrent code, cut into four quarters',
               0.5, y_a - 0.55)
        panel_a_quarters(_add_ax_cm(fig, page, 0.5, y_a, W_CONCURRENT, A_H), ex)

        xs = [lab, lab + col + (gap - lab), lab + 2 * (col + (gap - lab))]
        for i, (t, fn) in enumerate([
                ('b  four β per voxel', lambda a: panel_b_profiles(a, picks, theta_deg)),
                ('c  each quarter has an angle', lambda a: panel_c_weights(a, theta_deg)),
                ('d  cos + sin = one vector', lambda a: panel_d_vector(a, picks))]):
            _title(fig, page, t, xs[i] - lab + 0.1, y_row1 - 0.55)
            fn(_add_ax_cm(fig, page, xs[i], y_row1, SQ_CM, SQ_CM))

        for i, (t, fn) in enumerate([
                ('e  subjects → group mean',
                 lambda a: panel_e_subjects(a, picks['ventral'], 'ventral')),
                ('f  angle agreement only',
                 lambda a: panel_f_rayleigh(a, picks['ventral'], 'ventral'))]):
            _title(fig, page, t, xs[i] - lab + 0.1, y_row2 - 0.55)
            fn(_add_ax_cm(fig, page, xs[i], y_row2, SQ_CM, SQ_CM))

        _title(fig, page, 'g  the same at every voxel → preferred-angle map',
               0.5, y_g - 0.55)
        panel_g_map(_add_ax_cm(fig, page, 1.5, y_g, BRAIN_W_CM, BRAIN_H_CM),
                    _add_ax_cm(fig, page, BRAIN_W_CM + 2.6, y_g + 0.4,
                               2.8, 2.8),
                    D, template_path, x_mni=x_mni)
        _save(fig, save_stem, show)
    return page_w, page_h
