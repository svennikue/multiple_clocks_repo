"""
Per-TR RSA group analysis in an a-priori mask: SVC peak test + LOSO
cross-validated timecourse + figures + whole-brain maps for FSLeyes.

Edit the CONFIGURATION block below and hit Run (F5) in Spyder. Nothing else
needs changing.

WHAT THIS RUNS
--------------
1. SVC peak test (small-volume-corrected max-t sign-flip permutation).
   Corrects over the mask's voxels x TRs JOINTLY. Answers: "is there any
   voxel x time point in this region where the model fits above chance?"
   Localises the effect, but its threshold is high because it pays for the
   whole joint search space.

2. LOSO cross-validated timecourse (optional, RUN_LOSO).
   Answers a DIFFERENT question: "how big is the effect, and when?"
   See the long note in run_loso() for why both are worth reporting.

3. Two figures at journal spec (4.0 x 3.5 cm, Arial 9 pt), one per analysis,
   each shading the time points that survive ITS OWN correction.

4. Whole-brain NIfTIs for FSLeyes: uncorrected group t-map, and 1-p_FWE maps
   (FSL's *_corrp_* convention -- threshold at 0.95 for p < .05).

CONVENTIONS THIS SCRIPT ASSUMES
-------------------------------
- Beta maps are 4D (x, y, z, subject), one directory per TR.
- TR = 1 s, so TR index k covers k to k+1 seconds from cue onset. All x-axes
  are in SECONDS, never TR index.
- Peak locations are recovered with np.unravel_index on the 2-D (vox, TR)
  t-map -- NEVER a hand-written divmod on a flat index. D is
  (subj, vox, TR), so a C-order flat index divides by n_tr, not n_vox;
  getting that backwards returns the right peak VALUE with the wrong voxel
  and TR attached, silently.
"""

# ============================================================================
# CONFIGURATION -- edit this block only
# ============================================================================

# Which set of per-TR results to analyse. See ANALYSES below for what each
# one points at and which models it offers.
ANALYSIS = 'split_rew'          # 'instruction' | 'split_rew'

# Which RSA model regressor. Set LIST_MODELS = True to print the ones
# actually present on disk for the chosen ANALYSIS, then pick from that list.
MODEL = 'rewDSR'
# 'CURR_REW-split_rew_DSR_combo', 'NEXT_REW-split_rew_DSR_combo', 'THREE_NEXT_REW-split_rew_DSR_combo', 'TWO_NEXT_REW-split_rew_DSR_combo', 'curr_rew', 
MODELLIST = ['next_rew', 'three_next_rew', 'two_next_rew']

# Print available models/TRs for ANALYSIS and exit without computing.
# Use this first when exploring a results directory you have not looked at yet.
LIST_MODELS = False

# A-priori mask (any resolution; resampled to the beta grid automatically).
MASK = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks/mask_PFC_LR_smoothed_resampled.nii.gz'

N_PERM = 10000                    # 200 for a fast sanity check, 10000 to report
SEED = 0

# --- LOSO cross-validated timecourse ---
RUN_LOSO = True
K_VALUES = [50, 100, 200]         # top-k voxels selected on n-1 subjects

# --- Whole-brain maps for FSLeyes ---
WHOLE_BRAIN_MAPS = True
WB_N_PERM = 10000                 # whole-brain FWE null; set 0 to skip that map

OUTPUT_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/per_TR_mask_stats'
TAG = None                        # None -> auto from ANALYSIS + MODEL + mask name

# ============================================================================
# End of configuration
# ============================================================================

import os
import json
import numpy as np
import nibabel as nib
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import blended_transform_factory

ROOT = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/per_TR'

# Each analysis differs only in directory naming and how model names appear in
# the beta filenames; everything downstream is shared.
ANALYSES = {
    'instruction': dict(
        dir_pattern='group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped',
        beta_template='cropped_masked_smooth_fwhm5_{model}_beta_std',
        note='instruction-period RSA, single rewDSR regressor',
    ),
    'split_rew': dict(
        dir_pattern='group_RSA_split_rew_DSR_per_TR_glmbase_01-TR{tr}_cropped',
        beta_template='cropped_masked_smooth_fwhm5_{model}_beta_std',
        note='reward-split DSR: separate regressors per reward position',
    ),
}

TRS = list(range(12))

# Reward schedule in SECONDS from cue onset (start, end, label, colour).
REWARD_SCHEDULE = [
    (0.0, 1.5, 'A', '#F15A29'), (1.5, 3.0, 'B', '#F7931E'),
    (3.0, 4.5, 'C', '#C7C6E2'), (4.5, 6.0, 'D', '#6B60AA'),
    (6.0, 7.0, 'A', '#F15A29'), (7.0, 8.0, 'B', '#F7931E'),
    (8.0, 9.0, 'C', '#C7C6E2'), (9.0, 10.0, 'D', '#6B60AA'),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve(path_stem):
    """Return path_stem with whichever of .nii / .nii.gz exists.

    The instruction directories store .nii, the split_rew directories .nii.gz.
    """
    for ext in ('.nii', '.nii.gz'):
        if os.path.exists(path_stem + ext):
            return path_stem + ext
    raise FileNotFoundError(f"neither {path_stem}.nii nor {path_stem}.nii.gz exists")


def list_available(cfg):
    """Print the models and TRs actually present on disk for this analysis."""
    print(f"\n{'='*70}\nAVAILABLE IN: {ANALYSIS}  ({cfg['note']})\n{'='*70}")
    trs_found, models = [], set()
    for tr in TRS:
        d = os.path.join(ROOT, cfg['dir_pattern'].format(tr=tr))
        if not os.path.isdir(d):
            continue
        trs_found.append(tr)
        for f in os.listdir(d):
            if '_beta_std' not in f:
                continue
            stem = f.split('_beta_std')[0]
            prefix = 'cropped_masked_smooth_fwhm5_'
            if stem.startswith(prefix):
                models.add(stem[len(prefix):])
    print(f"TR directories found: {trs_found}")
    print(f"\nModels ({len(models)}) -- set MODEL to one of these:")
    for m in sorted(models):
        print(f"    {m}")
    print(f"{'='*70}\n")


def load_ref(cfg):
    """Reference image + group brain mask (voxels present in all subjects)."""
    d = os.path.join(ROOT, cfg['dir_pattern'].format(tr=TRS[0]))
    ref = nib.load(_resolve(os.path.join(d, 'mask_all_32_subjects')))
    return ref, ref.get_fdata() > 0


def load_mask(mask_path, ref):
    """Load an a-priori mask, resampling to the beta grid if needed."""
    img = nib.load(mask_path)
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
        from nibabel.processing import resample_from_to
        img = resample_from_to(img, (ref.shape[:3], ref.affine), order=0)
        print("[info] mask resampled to beta grid (nearest-neighbour)")
    return img.get_fdata() > 0.5


def extract_betas(cfg, model, brain):
    """Read every TR once. Returns (n_subj, n_brain_vox, n_tr) + brain coords.

    Loading the WHOLE BRAIN (not just the mask) means the same read serves both
    the mask analysis and the whole-brain maps -- reading these 12 4D niftis is
    the slowest part of the script, so it happens exactly once.
    """
    ix, iy, iz = np.where(brain)
    out = None
    for j, tr in enumerate(TRS):
        d = os.path.join(ROOT, cfg['dir_pattern'].format(tr=tr))
        f = _resolve(os.path.join(d, cfg['beta_template'].format(model=model)))
        data = nib.load(f).get_fdata()
        if data.ndim == 3:
            data = data[..., None]
        if out is None:
            out = np.empty((data.shape[-1], len(ix), len(TRS)), dtype=np.float32)
        out[:, :, j] = data[ix, iy, iz, :].T
        print(f"[read] TR{tr}: {data.shape[-1]} subjects")
    return out, (ix, iy, iz)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def tstat(A):
    """One-sample t against zero over axis 0. A is (n_subj, ...)."""
    n = A.shape[0]
    m = A.mean(0)
    s = A.std(0, ddof=1)
    return np.where(s > 0, m / np.where(s > 0, s, 1.0) * np.sqrt(n), 0.0)


def null_max_t(A, n_perm, seed, mem_budget_mb=400):
    """Sign-flip max-t null over all columns of A (n_subj, n_cols).

    Exploits that sign-flipping preserves sum(x^2) per column, so the permuted
    t-map follows from the permuted mean alone -- no full recompute per
    permutation. Permutations are processed in blocks sized to the memory
    budget, because the whole-brain case has ~1.8M columns and a fixed block
    size would either thrash or waste time.
    """
    rng = np.random.RandomState(seed)
    n, n_cols = A.shape
    S2 = (A ** 2).sum(0)
    pblock = max(1, min(1000, int(mem_budget_mb * 1e6 / (4 * max(n_cols, 1)))))
    out = np.empty(n_perm)
    for s0 in range(0, n_perm, pblock):
        b = min(pblock, n_perm - s0)
        F = rng.choice([-1.0, 1.0], size=(b, n)).astype(np.float32)
        M = (F @ A) / n
        var = (S2[None, :] - n * M ** 2) / (n - 1)
        T = np.where(var > 0, M * np.sqrt(n) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
        out[s0:s0 + b] = T.max(1)
    return out


def p_fwe_from_null(t_values, null):
    """Corrected one-sided p for each t: proportion of max-t null at or above.

    One-sided (positive model fit) -- the hypothesis is that the model is
    expressed, not merely non-zero.
    """
    ns = np.sort(null)
    idx = np.searchsorted(ns, t_values, side='left')
    return (len(ns) - idx) / float(len(ns))


def benjamini_hochberg(p, alpha=0.05):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    return q < alpha, np.clip(q, 0, 1)


def run_svc(D, ref, ijk_mask, n_perm, seed):
    """Small-volume-corrected peak test over the mask's voxels x TRs."""
    n_subj, n_vox, n_tr = D.shape
    Tobs = tstat(D)                                   # (n_vox, n_tr)
    nmt = null_max_t(D.reshape(n_subj, -1), n_perm=n_perm, seed=seed)

    # unravel on the 2-D t-map; never a hand-written divmod (see module docstring)
    vox_idx, tr_idx = np.unravel_index(np.argmax(Tobs), Tobs.shape)
    peak_t = float(Tobs[vox_idx, tr_idx])
    tcrit = float(np.percentile(nmt, 95))

    ix, iy, iz = ijk_mask
    mni = nib.affines.apply_affine(ref.affine, [ix[vox_idx], iy[vox_idx], iz[vox_idx]])

    # Which TRs at the peak voxel survive the SAME correction? This is the
    # band drawn in the SVC figure. Do NOT re-correct over TRs only to widen
    # it -- that is a weaker test over 12 rather than n_vox*n_tr comparisons.
    t_at_peak = Tobs[vox_idx, :]
    p_per_tr = p_fwe_from_null(t_at_peak, nmt)

    p_uncorr = 2 * (1 - stats.t.cdf(np.abs(Tobs.reshape(-1)), n_subj - 1))
    _, q_vals = benjamini_hochberg(p_uncorr, alpha=0.05)

    return dict(
        Tobs=Tobs, null=nmt, vox_idx=int(vox_idx), tr_idx=int(tr_idx),
        peak_t=peak_t, peak_p_FWE=float(p_fwe_from_null(np.array([peak_t]), nmt)[0]),
        tcrit=tcrit, peak_mni=[round(float(v)) for v in mni],
        t_at_peak_voxel_per_TR=[float(v) for v in t_at_peak],
        p_FWE_at_peak_voxel_per_TR=[float(v) for v in p_per_tr],
        sig_TRs=[int(k) for k in np.where(p_per_tr < 0.05)[0]],
        n_surviving_voxTR=int((Tobs > tcrit).sum()),
        n_tests=int(n_vox * n_tr),
        peak_q_FDR=float(q_vals[int(vox_idx) * n_tr + int(tr_idx)]),
    )


def run_loso(D, k_values, n_perm, seed0):
    """Leave-one-subject-out cross-validated timecourse.

    WHY THIS EXISTS, AND WHAT IT IS *NOT*
    -------------------------------------
    The SVC test tells you an effect is present somewhere in the mask, and
    where its maximum is. It does NOT give you an honest effect size at that
    maximum: the voxel and TR were chosen as the largest, so plotting that
    voxel's mean beta overstates the effect, and its SEM is too narrow. This
    is the standard selection-bias / "circular analysis" problem in ROI
    analyses (Kriegeskorte et al. 2009; Vul et al. 2009).

    LOSO fixes it by never letting a subject's data influence which voxels
    that subject is read out from:

        for each subject s:
            select the top-k voxels using the OTHER n-1 subjects only
            record subject s's mean beta in those voxels, per TR

    Every one of the n resulting timecourses is therefore a genuinely
    held-out measurement, so their mean and SEM are unbiased estimates and
    the t-test over them is valid.

    IMPORTANT -- THIS IS NOT A MORE STRINGENT TEST
    ----------------------------------------------
    Cross-validation removes selection bias from the EFFECT SIZE; it does not
    change the multiple-comparisons problem, which is a separate issue. Here
    the spatial search is absorbed into the cross-validation, so only the 12
    TRs remain to correct over -- which makes the LOSO threshold much LOWER
    than the SVC one (t ~ 2.6 vs ~ 5.0 for a 4000-voxel mask), and its
    surviving window correspondingly WIDER. A wider LOSO band is not a
    stronger result than a narrow SVC one; the two are answering different
    questions and paying for different search spaces. Never present a LOSO
    band as though it were SVC-corrected.

    DOES LOSO NEED A MASK?
    ----------------------
    Yes -- D is already restricted to the a-priori mask, and top-k selection
    happens strictly within it. LOSO removes bias from voxel selection INSIDE
    a region; it does nothing about the choice of region itself. If the mask
    was drawn after seeing this data, both analyses are equally compromised.
    """
    n, n_vox, n_tr = D.shape
    out, curves = {}, {}
    for k in k_values:
        k = min(k, n_vox)
        held = np.zeros((n, n_tr))
        for s in range(n):
            train = np.delete(np.arange(n), s)
            Ttr = tstat(D[train])                     # (vox, TR), TRAIN ONLY
            top = np.argsort(-Ttr.max(1))[:k]         # spatial selection
            held[s] = D[s, top, :].mean(0)            # held-out readout
        t = tstat(held)
        nm = null_max_t(held, n_perm=n_perm, seed=seed0 + k)
        p = p_fwe_from_null(t, nm)
        out[str(k)] = dict(
            k=int(k),
            mean=[float(v) for v in held.mean(0)],
            sem=[float(v) for v in held.std(0, ddof=1) / np.sqrt(n)],
            t=[float(v) for v in t],
            p_FWE=[float(v) for v in p],
            tcrit=float(np.percentile(nm, 95)),
            sig_TRs=[int(j) for j in np.where(p < 0.05)[0]],
            n_tests=int(n_tr),
            correction='max-t over 12 TRs only (spatial search absorbed by CV)',
        )
        curves[str(k)] = held
        print(f"[loso] k={k:>4}: t_crit={out[str(k)]['tcrit']:.2f}  "
              f"sig TRs={out[str(k)]['sig_TRs']}")
    return out, curves


# ---------------------------------------------------------------------------
# Figure -- journal spec: 4.0 x 3.5 cm, Arial >= 9 pt
# ---------------------------------------------------------------------------

def readable_text_colour(hex_colour):
    r, g, b = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in (r, g, b)]
    Y = .2126 * lin[0] + .7152 * lin[1] + .0722 * lin[2]
    return 'white' if (1.05 / (Y + .05)) > ((Y + .05) / .05) else 'black'


def plot_panel(mean, sem, sig_trs, n_tr, save_stem,
               second=None, fig_w_cm=4.0, fig_h_cm=3.5, font_pt=9):
    """One timecourse panel at journal spec, shading `sig_trs`.

    Journal spec forces these choices; none of them is cosmetic:
      * EXACT canvas size, never bbox_inches='tight' -- a tight box silently
        grows the figure past spec to fit labels, so the panel that reaches
        the manuscript is not the size requested. Margins are hand-set and
        the AXES shrinks instead.
      * y in units of 1e-2, so tick labels stay one character wide. At 4 cm,
        '0.04' costs more width than the data.
      * no legend -- name the traces in the caption; a 9 pt legend eats a
        quarter of a 3.5 cm panel.
      * schedule strip ABOVE the axes, letters only in bins wide enough for a
        9 pt glyph: a 9 pt letter is taller than any in-axes strip that fits.
      * last x-tick dropped -- its label would overhang the canvas edge.
    """
    CM = 1.0 / 2.54
    SCALE = 1e2
    DARK_SAGE, GREY, BAND = '#3E6B4D', '#9A9A9A', '#C9B458'
    FS = font_pt

    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': FS, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 1.8, 'ytick.major.size': 1.8,
        'xtick.major.pad': 1.0, 'ytick.major.pad': 1.0,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.labelpad': 1.0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    mean = np.asarray(mean) * SCALE
    sem = np.asarray(sem) * SCALE
    xs = np.arange(n_tr) + 0.5                      # bin centres, in seconds

    fig, ax = plt.subplots(figsize=(fig_w_cm * CM, fig_h_cm * CM))
    fig.subplots_adjust(left=0.315, right=0.95, bottom=0.315, top=0.80)

    for k in sig_trs:
        ax.axvspan(k, k + 1, color=BAND, alpha=0.30, lw=0, zorder=1)

    ax.axhline(0, color='0.65', lw=0.4, zorder=2)
    ax.fill_between(xs, mean - sem, mean + sem, color=DARK_SAGE, alpha=0.30,
                    linewidth=0, zorder=3)
    lo_vals = [(mean - sem).min()]
    if second is not None:
        second = np.asarray(second) * SCALE
        ax.plot(xs, second, '-', color=GREY, lw=0.8, zorder=4)
        lo_vals.append(second.min())
    ax.plot(xs, mean, '-', color=DARK_SAGE, lw=1.2, zorder=5)

    dmax = float((mean + sem).max())
    dmin = float(min(lo_vals))
    pad = 0.06 * (dmax - dmin) if dmax > dmin else 0.001
    ax.set_ylim(dmin - pad, dmax + pad)
    ax.set_xlim(0, n_tr)
    ax.set_xticks([t for t in range(0, n_tr + 1, 4) if t < n_tr])
    yl, yh = ax.get_ylim()
    ax.set_yticks([v for v in range(-20, 21, 2) if yl < v < yh])

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    strip_lo, strip_hi = 1.04, 1.25                 # ~one 9 pt line tall
    for (a, b, label, col) in REWARD_SCHEDULE:
        if b <= 0 or a >= n_tr:
            continue
        a, b = max(a, 0), min(b, n_tr)
        ax.add_patch(plt.Rectangle((a, strip_lo), b - a, strip_hi - strip_lo,
                                   transform=trans, facecolor=col,
                                   edgecolor='white', lw=0.3, zorder=6,
                                   clip_on=False))
        if (b - a) >= 1.4:
            ax.text((a + b) / 2, (strip_lo + strip_hi) / 2, label,
                    transform=trans, ha='center', va='center', fontsize=FS,
                    color=readable_text_colour(col), zorder=7, clip_on=False)

    ax.set_xlabel('time (s)', fontsize=FS)
    ax.set_ylabel(r'$\beta$ ($\times 10^{-2}$)', fontsize=FS)
    ax.tick_params(labelsize=FS)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f"{save_stem}.{ext}", dpi=600)   # exact canvas

    # self-check: a panel that violates spec should say so, not reach the paper
    r = fig.canvas.get_renderer()
    txt = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
           if t.get_text().strip() and t.get_visible()]
    overlaps = sum(1 for i, (a, ba) in enumerate(txt)
                   for b, bb in txt[i + 1:] if ba.overlaps(bb))
    sizes = sorted({round(t.get_fontsize(), 1) for t, _ in txt})
    clipped = [t.get_text() for t, bb in txt
               if bb.x0 < -0.5 or bb.y0 < -0.5
               or bb.x1 > fig.bbox.width + 0.5 or bb.y1 > fig.bbox.height + 0.5]
    w_cm, h_cm = fig.get_size_inches() * 2.54
    ok = (min(sizes) >= font_pt) and not clipped and not overlaps
    print(f"[spec] {os.path.basename(save_stem)}: {w_cm:.2f} x {h_cm:.2f} cm | "
          f"font pt {sizes} | overlaps {overlaps} | clipped {clipped} | "
          f"{'OK' if ok else 'VIOLATES SPEC -- fix before use'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Whole-brain maps for FSLeyes
# ---------------------------------------------------------------------------

def write_maps(D_wb, brain_ijk, ref, svc, in_mask, n_perm, seed, out_stem):
    """Write uncorrected group t-map and 1-p_FWE maps as 4D NIfTIs (x,y,z,TR).

    1-p maps follow the FSL randomise *_corrp_* convention: threshold at 0.95
    for p_FWE < .05. Two are written because they answer different questions:

      _corrp_mask : corrected within the a-priori mask only (the SVC test).
                    Use this one for the a-priori claim.
      _corrp_wholebrain : corrected across every brain voxel x TR. Far more
                    conservative -- this is the exploratory map.

    A voxel can pass the mask-corrected map and fail the whole-brain one; that
    is not a contradiction, it is the price of the larger search space.
    """
    ix, iy, iz = brain_ijk
    n_subj, n_brain, n_tr = D_wb.shape
    shape4 = ref.shape[:3] + (n_tr,)

    def _to_img(values_per_brainvox_tr):
        vol = np.zeros(shape4, dtype=np.float32)
        vol[ix, iy, iz, :] = values_per_brainvox_tr
        return nib.Nifti1Image(vol, ref.affine, ref.header)

    T_wb = tstat(D_wb)                                  # (n_brain, n_tr)
    nib.save(_to_img(T_wb), f"{out_stem}_t_uncorr.nii.gz")
    print(f"[map] uncorrected group t -> {out_stem}_t_uncorr.nii.gz")

    # mask-corrected: reuse the SVC null, so this map matches the reported p
    corrp_mask = np.zeros((n_brain, n_tr), dtype=np.float32)
    corrp_mask[in_mask, :] = 1.0 - p_fwe_from_null(
        T_wb[in_mask, :].reshape(-1), svc['null']).reshape(-1, n_tr)
    nib.save(_to_img(corrp_mask), f"{out_stem}_corrp_mask.nii.gz")
    print(f"[map] 1-p_FWE within mask -> {out_stem}_corrp_mask.nii.gz "
          f"(threshold 0.95)")

    if n_perm and n_perm > 0:
        print(f"[map] whole-brain max-t null ({n_brain} vox x {n_tr} TR, "
              f"{n_perm} perms)...")
        nm_wb = null_max_t(D_wb.reshape(n_subj, -1), n_perm=n_perm, seed=seed + 1)
        corrp_wb = (1.0 - p_fwe_from_null(T_wb.reshape(-1), nm_wb)
                    ).reshape(n_brain, n_tr).astype(np.float32)
        nib.save(_to_img(corrp_wb), f"{out_stem}_corrp_wholebrain.nii.gz")
        n_pass = int((corrp_wb > 0.95).sum())
        print(f"[map] 1-p_FWE whole brain -> {out_stem}_corrp_wholebrain.nii.gz "
              f"(threshold 0.95; t_crit={np.percentile(nm_wb, 95):.2f}, "
              f"{n_pass} voxel x TR survive)")
        return dict(wholebrain_tcrit=float(np.percentile(nm_wb, 95)),
                    wholebrain_n_surviving=n_pass)
    return {}


# ---------------------------------------------------------------------------

def main():
    if ANALYSIS not in ANALYSES:
        raise SystemExit(f"ANALYSIS must be one of {list(ANALYSES)}")
    cfg = ANALYSES[ANALYSIS]

    if LIST_MODELS:
        list_available(cfg)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for MODEL in MODELLIST:
        tag = TAG or f"{ANALYSIS}_{MODEL}_{os.path.basename(MASK).split('.nii')[0]}"
    
        print(f"\n{'='*70}")
        print(f"Analysis : {ANALYSIS}  ({cfg['note']})")
        print(f"Model    : {MODEL}")
        print(f"Mask     : {os.path.basename(MASK)}")
        print(f"Perms    : {N_PERM}   LOSO: {RUN_LOSO}   maps: {WHOLE_BRAIN_MAPS}")
        print(f"Tag      : {tag}")
        print(f"{'='*70}\n")
    
        ref, brain = load_ref(cfg)
        mask = load_mask(MASK, ref)
    
        try:
            D_wb, brain_ijk = extract_betas(cfg, MODEL, brain)
        except FileNotFoundError as e:
            print(f"\n[error] {e}\n")
            list_available(cfg)
            raise SystemExit("MODEL not found on disk -- pick one from the list above")
    
        ix, iy, iz = brain_ijk
        in_mask = mask[ix, iy, iz]                     # boolean over brain voxels
        D = D_wb[:, in_mask, :]
        ijk_mask = (ix[in_mask], iy[in_mask], iz[in_mask])
        n_subj, n_vox, n_tr = D.shape
        print(f"\n[info] {n_subj} subjects | {n_vox} mask voxels "
              f"({int(brain.sum())} brain) | {n_tr} TRs\n")
        if n_vox == 0:
            raise SystemExit("mask has no voxels inside the group brain mask")
    
        print(f"[running] SVC max-t permutation ({N_PERM} perms)...")
        svc = run_svc(D, ref, ijk_mask, N_PERM, SEED)
    
        loso, loso_curves = ({}, {})
        if RUN_LOSO:
            print(f"[running] LOSO cross-validation, k={K_VALUES}...")
            loso, loso_curves = run_loso(D, K_VALUES, N_PERM, SEED)
    
        # ---- figures ----
        pk = D[:, svc['vox_idx'], :]
        zoom = np.abs(np.diag(ref.affine)[:3])
        centre = np.array([ijk_mask[0][svc['vox_idx']], ijk_mask[1][svc['vox_idx']],
                           ijk_mask[2][svc['vox_idx']]])
        dist = np.sqrt((((np.vstack(ijk_mask).T - centre) * zoom) ** 2).sum(1))
        sphere = dist <= 6.0
    
        plot_panel(pk.mean(0), pk.std(0, ddof=1) / np.sqrt(n_subj),
                   svc['sig_TRs'], n_tr, os.path.join(OUTPUT_DIR, f"{tag}_svc"),
                   second=D[:, sphere, :].mean(1).mean(0))
    
        if RUN_LOSO:
            kk = str(min(K_VALUES))
            h = loso_curves[kk]
            plot_panel(h.mean(0), h.std(0, ddof=1) / np.sqrt(n_subj),
                       loso[kk]['sig_TRs'], n_tr,
                       os.path.join(OUTPUT_DIR, f"{tag}_loso_k{kk}"))
    
        # ---- whole-brain maps ----
        map_info = {}
        if WHOLE_BRAIN_MAPS:
            map_info = write_maps(D_wb, brain_ijk, ref, svc, in_mask,
                                  WB_N_PERM, SEED, os.path.join(OUTPUT_DIR, tag))
    
        # ---- results ----
        results = dict(
            analysis=ANALYSIS, model=MODEL, mask_path=MASK,
            n_subjects=int(n_subj), mask_nvox=int(n_vox), n_tr=int(n_tr),
            n_perm=N_PERM, seed=SEED,
            svc={k: v for k, v in svc.items() if k not in ('Tobs', 'null')},
            loso=loso, whole_brain=map_info,
            peak_voxel_timecourse=dict(
                mean=[float(v) for v in pk.mean(0)],
                sem=[float(v) for v in pk.std(0, ddof=1) / np.sqrt(n_subj)],
                sphere6mm_mean=[float(v) for v in D[:, sphere, :].mean(1).mean(0)]),
            whole_mask_mean=[float(v) for v in D.mean(1).mean(0)],
        )
        out_json = os.path.join(OUTPUT_DIR, f"{tag}_stats.json")
        json.dump(results, open(out_json, 'w'), indent=2)
    
        print(f"\n{'='*70}\nRESULTS: {tag}\n{'='*70}")
        print(f"\n1. SVC PEAK TEST  (corrected over {svc['n_tests']} voxel x TR)")
        print(f"   peak t({n_subj-1})  = {svc['peak_t']:.2f}")
        print(f"   time         = {svc['tr_idx']}-{svc['tr_idx']+1} s")
        print(f"   MNI          = {svc['peak_mni']}")
        print(f"   p_FWE        = {svc['peak_p_FWE']:.4f}   "
              f"{'PASS (p < .05)' if svc['peak_p_FWE'] < 0.05 else 'FAIL'}")
        print(f"   t_crit(.05)  = {svc['tcrit']:.2f}")
        print(f"   surviving    = {svc['n_surviving_voxTR']} / {svc['n_tests']} "
              f"voxel x TR pairs")
        print(f"   window at peak voxel = "
              f"{[f'{k}-{k+1}s' for k in svc['sig_TRs']] or 'none'}")
        print(f"   (FDR reference only: q = {svc['peak_q_FDR']:.3f})")
    
        if RUN_LOSO:
            print(f"\n2. LOSO CROSS-VALIDATED  (corrected over {n_tr} TRs only)")
            print(f"   {'k':>5}  {'t_crit':>7}  {'peak t':>7}  {'window':>14}")
            for k in K_VALUES:
                v = loso[str(min(k, n_vox))]
                w = f"{v['sig_TRs'][0]}-{v['sig_TRs'][-1]+1}s" if v['sig_TRs'] else 'none'
                print(f"   {v['k']:>5}  {v['tcrit']:>7.2f}  {max(v['t']):>7.2f}  {w:>14}")
            print(f"\n   Reminder: the LOSO threshold is LOWER than the SVC one "
                  f"because\n   the spatial search is absorbed by cross-validation, "
                  f"not corrected.\n   A wider LOSO band is NOT a stronger result "
                  f"than a narrow SVC one.")
    
        print(f"\nOUTPUTS")
        print(f"   json    {out_json}")
        print(f"   figures {os.path.join(OUTPUT_DIR, tag)}_svc.*"
              + (f", _loso_k{min(K_VALUES)}.*" if RUN_LOSO else ""))
        if WHOLE_BRAIN_MAPS:
            print(f"   maps    {os.path.join(OUTPUT_DIR, tag)}_t_uncorr.nii.gz, "
                  f"_corrp_mask.nii.gz"
                  + (", _corrp_wholebrain.nii.gz" if WB_N_PERM else ""))
            print(f"\n   FSLeyes: overlay _corrp_* and threshold at 0.95 for p<.05;\n"
                  f"   the 4th dimension is time (12 x 1 s volumes).")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
