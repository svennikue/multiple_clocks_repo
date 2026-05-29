"""ROI atlas lookups + anatomical mask building.

Factored out of `scripts/cell_to_roi_MNI.py` so the same masks can be
reused for plotting (e.g. shading each ROI on a glass-brain by a
heatmap value) without re-running the cell-to-ROI assignment script.

Atlases are loaded lazily on first call to :func:`make_roi_mask` and
cached at module level.  Harvard-Oxford and Juelich come from nilearn;
the Brainnetome atlas is loaded from a user-supplied path (defaults
to the location used in `scripts/cell_to_roi_MNI.py`).
"""

from __future__ import annotations

import os
import re
import numpy as np
import nibabel as nib


# Defaults that match scripts/cell_to_roi_MNI.py.
DEFAULT_BRAINNETOME_NII = (
    "/Users/xpsy1114/Documents/toolboxes/Brainnatome/BN_Atlas_246_1mm.nii.gz"
)
DEFAULT_BRAINNETOME_LUT = (
    "/Users/xpsy1114/Documents/toolboxes/Brainnatome/BN_Atlas_246_LUT.txt"
)

# Anatomical cutoffs (mirroring cell_to_roi_MNI.py).
HC_AP_CUTOFF = -21        # y >= cutoff -> HC_anterior; y < cutoff -> HC_mid
ACC_Y_CUTOFF = 10         # for alt_final_roi: y >= cutoff stays ACC, else medial_CC


_ATLAS_CACHE: dict = {}

# Anatomical masks, keyed by (roi, roi_label_column, brainnetome_nii).
# Sphere-restricted masks, keyed by (id(anat_mask), coords-bytes, radius_mm).
# Both caches grow within a session; call `clear_mask_caches()` to reset.
_ANAT_MASK_CACHE: dict = {}
_RESTRICTED_MASK_CACHE: dict = {}


def clear_mask_caches():
    """Drop both mask caches. Atlases stay loaded."""
    _ANAT_MASK_CACHE.clear()
    _RESTRICTED_MASK_CACHE.clear()


# ── Internal helpers ────────────────────────────────────────────────
def _get_img(atlas_or_img):
    if hasattr(atlas_or_img, "maps"):
        img = atlas_or_img.maps
    else:
        img = atlas_or_img
    return nib.load(img) if isinstance(img, str) else img


def _load_brainnetome_lut(lut_path):
    labels = {0: "background"}
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = None
            for p in parts:
                try:
                    idx = int(p)
                    break
                except ValueError:
                    pass
            if idx is None:
                continue
            text_parts = [p for p in parts
                          if not re.match(r"^-?\d+(\.\d+)?$", p)]
            label = " ".join(text_parts).strip()
            labels[idx] = label if label else f"Brainnetome_{idx}"
    return labels


def ensure_atlases_loaded(brainnetome_nii=None, brainnetome_lut=None):
    """Lazy-load the atlases the first time this is called.

    Subsequent calls re-use the cached atlases; Brainnetome is reloaded
    only if the user supplies a new path.
    """
    if "ho_cort" not in _ATLAS_CACHE:
        from nilearn import datasets
        _ATLAS_CACHE["ho_cort"] = datasets.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm")
        _ATLAS_CACHE["ho_sub"] = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-2mm")
        _ATLAS_CACHE["juelich"] = datasets.fetch_atlas_juelich(
            "maxprob-thr25-2mm")

    nii = brainnetome_nii or DEFAULT_BRAINNETOME_NII
    lut = brainnetome_lut or DEFAULT_BRAINNETOME_LUT
    if _ATLAS_CACHE.get("brainnetome_nii") == nii:
        return
    if os.path.isfile(nii) and os.path.isfile(lut):
        _ATLAS_CACHE["brainnetome_img"] = nib.load(nii)
        _ATLAS_CACHE["brainnetome_data"] = _ATLAS_CACHE["brainnetome_img"].get_fdata()
        _ATLAS_CACHE["brainnetome_labels"] = _load_brainnetome_lut(lut)
    else:
        _ATLAS_CACHE["brainnetome_img"] = None
        _ATLAS_CACHE["brainnetome_data"] = None
        _ATLAS_CACHE["brainnetome_labels"] = None
    _ATLAS_CACHE["brainnetome_nii"] = nii


def _label_indices_containing(labels, patterns):
    return [i for i, lab in enumerate(labels)
            if any(p.lower() in str(lab).lower() for p in patterns)]


def _mask_from_atlas(atlas, patterns):
    from nilearn.image import new_img_like
    img = _get_img(atlas)
    data = img.get_fdata()
    idxs = _label_indices_containing(atlas.labels, patterns)
    mask = np.isin(data, idxs).astype(float) if idxs else np.zeros(data.shape)
    return new_img_like(img, mask)


def _mask_from_brainnetome(patterns):
    from nilearn.image import new_img_like
    img = _ATLAS_CACHE.get("brainnetome_img")
    data = _ATLAS_CACHE.get("brainnetome_data")
    labels = _ATLAS_CACHE.get("brainnetome_labels")
    if img is None or data is None or labels is None:
        return None
    idxs = [idx for idx, lab in labels.items()
            if any(p.lower() in str(lab).lower() for p in patterns)]
    mask = np.isin(data, idxs).astype(float) if idxs else np.zeros(data.shape)
    return new_img_like(img, mask)


def _y_split_mask(base_img, part, y_cutoff):
    """Apply a y-axis cut to `base_img` (anatomical mask).

    `part` is 'anterior' (y >= cutoff) or 'posterior' (y < cutoff).
    """
    from nilearn.image import new_img_like
    img = _get_img(base_img)
    data = img.get_fdata() > 0
    coords = np.indices(data.shape).reshape(3, -1).T
    mni = nib.affines.apply_affine(img.affine, coords)
    y_coords = mni[:, 1].reshape(data.shape)
    if part == "anterior":
        data = data & (y_coords >= y_cutoff)
    elif part == "posterior":
        data = data & (y_coords < y_cutoff)
    return new_img_like(img, data.astype(float))


def _hc_mask(ap="both"):
    ho_sub = _ATLAS_CACHE["ho_sub"]
    img = _get_img(ho_sub)
    hc_idxs = _label_indices_containing(ho_sub.labels, ["hippocampus"])
    from nilearn.image import new_img_like
    base = new_img_like(img, np.isin(img.get_fdata(), hc_idxs).astype(float))
    if ap == "both":
        return base
    return _y_split_mask(base, "anterior" if ap == "anterior" else "posterior",
                         HC_AP_CUTOFF)


def _acc_alt_mask(part="anterior"):
    """ACC split by ACC_Y_CUTOFF for the alt_final_roi labelling.

    `part='anterior'` -> alt 'ACC' (y >= cutoff)
    `part='posterior'` -> alt 'medial_CC' (y < cutoff)
    """
    base = _mask_from_atlas(
        _ATLAS_CACHE["ho_cort"], ["cingulate gyrus, anterior division"])
    return _y_split_mask(base, part, ACC_Y_CUTOFF)


# ── Public API ──────────────────────────────────────────────────────
def make_roi_mask(roi, roi_label_column="alt_final_roi",
                  brainnetome_nii=None, brainnetome_lut=None):
    """Return an MNI152 nilearn-compatible mask for `roi`.

    Recognises labels from both `final_roi` and `alt_final_roi` from
    scripts/cell_to_roi_MNI.py. Returns ``None`` if the ROI is unknown
    or the required atlas is unavailable (e.g. Brainnetome path absent).

    Cached per ``(roi, roi_label_column, brainnetome_nii)`` — repeated
    calls return the same nibabel object instantly.
    """
    nii_key = brainnetome_nii or DEFAULT_BRAINNETOME_NII
    cache_key = (roi, roi_label_column, nii_key)
    if cache_key in _ANAT_MASK_CACHE:
        return _ANAT_MASK_CACHE[cache_key]
    img = _build_roi_mask(roi, roi_label_column,
                          brainnetome_nii, brainnetome_lut)
    _ANAT_MASK_CACHE[cache_key] = img
    return img


def _build_roi_mask(roi, roi_label_column, brainnetome_nii, brainnetome_lut):
    """Actual mask construction; see :func:`make_roi_mask` for the cached
    public entry point."""
    ensure_atlases_loaded(brainnetome_nii, brainnetome_lut)
    ho_cort = _ATLAS_CACHE["ho_cort"]
    juelich = _ATLAS_CACHE["juelich"]

    if roi == "EC":
        return _mask_from_atlas(juelich, ["entorhinal"])
    if roi == "Parahippocampal":
        return _mask_from_atlas(ho_cort, ["parahippocampal"])
    if roi == "HC_anterior":
        return _hc_mask("anterior")
    if roi in ("HC_mid", "HC_posterior"):
        return _hc_mask("posterior")
    if roi == "ACC":
        if roi_label_column == "alt_final_roi":
            return _acc_alt_mask("anterior")
        return _mask_from_atlas(
            ho_cort, ["cingulate gyrus, anterior division"])
    if roi == "medial_CC":
        if roi_label_column == "alt_final_roi":
            return _acc_alt_mask("posterior")
        return _mask_from_atlas(
            ho_cort, ["cingulate gyrus, posterior division"])
    if roi == "ventral_ACC":
        return _mask_from_brainnetome(["a14m"])
    if roi in ("PCC", "posterior_CC"):
        return _mask_from_atlas(
            ho_cort,
            ["cingulate gyrus, posterior division", "precuneous", "precuneus"])
    if roi == "medialOFC":
        from nilearn.image import new_img_like
        parts = [_mask_from_brainnetome(["a11m"]),
                 _mask_from_brainnetome(["a13"]),
                 _mask_from_brainnetome(["a14m"])]
        parts = [m for m in parts if m is not None]
        if not parts:
            return None
        data = np.zeros(parts[0].shape, dtype=float)
        for m in parts:
            data = np.maximum(data, m.get_fdata())
        return new_img_like(parts[0], data)
    if roi == "OFC11":
        return _mask_from_brainnetome(["a11m"])
    if roi == "OFC13":
        return _mask_from_brainnetome(["a13"])
    if roi == "Visual":
        return _mask_from_atlas(ho_cort, [
            "occipital", "cuneal", "lingual",
            "intracalcarine", "supracalcarine", "occipital pole",
        ])
    return None


def restrict_mask_to_cell_spheres(mask_img, coords, radius_mm=8.0):
    """Intersect an anatomical mask with the union of spheres around cells.

    Only voxels that (a) belong to the anatomical mask AND (b) lie within
    `radius_mm` of at least one cell coordinate are kept.  Useful for
    shading only the part of a large ROI (e.g. visual cortex) that is
    actually sampled by the recording.

    Cached per ``(id(mask_img), coords-bytes, radius_mm)`` — within a
    session, the second call with the same anatomical mask and the same
    cell coordinates returns immediately.

    Parameters
    ----------
    mask_img : nibabel image
        Anatomical mask (binary, 0/1).
    coords : (N, 3) ndarray
        Cell positions in MNI mm.
    radius_mm : float

    Returns
    -------
    nibabel image with the restricted (binary) mask. If no cells fall
    inside the anatomical mask, returns the union of cell spheres alone
    so the figure still shows something where the cells actually are.
    """
    from nilearn.image import new_img_like

    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    coords = coords[np.isfinite(coords).all(axis=1)]

    # Cache lookup. id(mask_img) is stable because make_roi_mask itself
    # returns a cached object — same anatomical mask -> same id.
    coords_key = coords.tobytes() if coords.size else b''
    cache_key = (id(mask_img), coords_key, float(radius_mm))
    if cache_key in _RESTRICTED_MASK_CACHE:
        return _RESTRICTED_MASK_CACHE[cache_key]

    affine = mask_img.affine
    shape = mask_img.shape
    anat = mask_img.get_fdata() > 0.5

    if coords.size == 0 or not anat.any():
        result = new_img_like(mask_img, np.zeros(shape, dtype=float))
        _RESTRICTED_MASK_CACHE[cache_key] = result
        return result

    # 1) Voxels inside the anatomical mask, intersected with cell spheres.
    anat_voxels = np.argwhere(anat)
    anat_mni = nib.affines.apply_affine(affine, anat_voxels)
    keep_anat = _within_radius(anat_mni, coords, radius_mm)
    out = np.zeros(shape, dtype=float)
    out[tuple(anat_voxels[keep_anat].T)] = 1.0
    if out.sum() > 0:
        result = new_img_like(mask_img, out)
        _RESTRICTED_MASK_CACHE[cache_key] = result
        return result

    # 2) Fallback: cells fell outside the anatomical mask. Show the cell
    # spheres themselves over the whole image so the user still sees a
    # blob where the recording actually is.
    i = np.arange(shape[0])
    j = np.arange(shape[1])
    k = np.arange(shape[2])
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    all_mni = nib.affines.apply_affine(affine, vox)
    keep_all = _within_radius(all_mni, coords, radius_mm)
    out = keep_all.reshape(shape).astype(float)
    result = new_img_like(mask_img, out)
    _RESTRICTED_MASK_CACHE[cache_key] = result
    return result


def _within_radius(points_mni, centres_mni, radius_mm):
    """Boolean mask: True for points within `radius_mm` of any centre."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centres_mni)
        dists, _ = tree.query(points_mni, k=1)
        return dists <= radius_mm
    except ImportError:
        keep = np.zeros(points_mni.shape[0], dtype=bool)
        r2 = float(radius_mm) ** 2
        for cx, cy, cz in centres_mni:
            d2 = ((points_mni[:, 0] - cx) ** 2
                  + (points_mni[:, 1] - cy) ** 2
                  + (points_mni[:, 2] - cz) ** 2)
            keep |= d2 <= r2
        return keep
