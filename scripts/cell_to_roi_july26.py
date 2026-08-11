#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cell_to_roi_july26.py — rebuild the human single-unit ROI labelling from
scratch, one step at a time.

STEP 1 (this file): coordinate loading + provenance.

Re-running after new source files arrive
----------------------------------------
This script is fully data-driven. To incorporate corrected Baylor
`{CODE}-electrodes_v2026.csv` files (older/broken copies sit in
`<v2026>/incorrect/`), just move the corrected files into the top-level
`<v2026>/` folder and re-run:

    conda activate env_multiple_clocks
    python scripts/cell_to_roi_july26.py

Nothing else to edit. Subject files inside subfolders (like `incorrect/`)
are ignored — the loader only reads top-level `-electrodes_v2026.csv`
(so browser-duplicated names like `YEU-electrodes_v2026[37].csv` are
skipped too). Once the new file passes the reliability gate (mean
152-vs-305transform <= 8 mm across bundles), its cells switch from
`baylor_bigtable_pre2026_macro_position` /
`baylor_v2026_bundle_305to152_unreliable_file` to the fully-verified
`baylor_v2026_bundle_micro`.

Coordinate source priority per site:

  Baylor
    - v2026 CSV micro-bundle MNI152, matched by stripping the trailing
      2 channel digits of `electrode label` (mRT2bHaEa04 ->
      mRT2bHaEa01). Two row types describe the same bundle:
        `microwires`  — the micro bundle itself (label `mLT2bHb01`)
        `sEEG-micro`  — contact 01 of the macro probe the bundle sits
                        in (label `LT2bHb01`, ~3 mm shallower)
      `microwires` is preferred; `sEEG-micro` is used only for bundles
      that have no `microwires` row (both are keyed on the m-prefixed
      bundle name so either can match the big table).
      Reliability gate: file's MNI152 must agree with file's MNI305 via
      the Fischl 305->152 transform to <= 8 mm mean.
    - If subject file is unreliable: use file MNI305 -> 152 transform.
    - If the subject's file ships NO micro rows at all (YER), rebuild
      each bundle from its macro probe: the micro tip sits a fixed
      3.15 mm beyond the deepest macro contact along the insertion
      axis. That constant is Baylor's own — it is identical in all 119
      bundles that do have `microwires` rows — so the rebuild
      reproduces their supplied positions to 0.25 mm median / 1.07 mm
      max. Tagged `..._micro_reconstructed_from_macro` and left
      `coord_verified = False` because it is inferred, not supplied.
    - If subject has no v2026 file at all: keep big-table coord
      transformed 305->152, but flag as `pre2026_macro_position`
      (systematically ~4 mm off from true micro position — Baylor
      originally reported the last macro contact where the micro bundle
      is now correctly reported).

  UCLA
    - Coord-based match to the v2026 xlsx (Sheet1). For every big-table
      cell, find the nearest electrode in that subject's xlsx; if within
      COORD_MATCH_TOL_MM (0.5 mm) accept and attach the v2026 electrode
      name + NMM region label. Big-table coord itself is kept (it was
      derived from the same source; the match just verifies + names).

  Utah
    - Coord independently reconstructed from the subject's
      `s{NN}/electrodes/Electrodes.mat` via:
        1. Big-table `chan{N}` -> find (r, c) in `ChannelMap1`
           (or `ChannelMap2`) where value == N.
        2. Enumerate every `LabelMap[r, c]` starting with 'm'
           (microwires) in canonical order (column asc, row desc =
           deepest microwire first per probe). This ordered list
           corresponds 1:1 to `MicroElec`, whose entries are 1-indexed
           rows into `ElecXYZMNIProj`.
        3. Coord = `ElecXYZMNIProj[MicroElec[i] - 1]` for the
           microwire that sits at (r, c).
        4. Cross-check: `LabelMap[r, c]` is the micro-label (e.g.
           mLOFC3, mLVCG3), stored as `source_region_hint`.
      Where the independent coord disagrees with the big-table coord
      (>0.5 mm), we trust the independent coord and record the shift.

Output columns per cell:
  MNI_x_final, MNI_y_final, MNI_z_final   – coord we will use downstream
  coord_source                             – provenance tag (see below)
  coord_verified                           – True if matched to source
  source_electrode                         – v2026 electrode / Baylor
                                             bundle / Utah row index
  source_region_hint                       – NMM/region label from source
  source_match_dist_mm                     – 0 for exact, NaN if no match

Nothing here touches probabilistic atlases yet — that's step 2.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")   # non-interactive; plt.show() would block on headless
import matplotlib.pyplot as plt
import nibabel as nib

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


# =============================================================================
# PATHS
# =============================================================================

path_to_cell_table = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives/neurons_MNI_latest.csv"
)

path_to_v2026 = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026"
)

path_to_subject_folders = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
)

output_dir = os.path.dirname(path_to_cell_table)
# All intermediate step outputs live in a subfolder so they don't clutter
# the main derivatives/ folder. Only the final authoritative table is
# saved directly to derivatives/ (see FINAL_TABLE_OUT below).
roi_assignment_dir = os.path.join(output_dir, "ROI_assignment")
os.makedirs(roi_assignment_dir, exist_ok=True)

# The FINAL canonical table (used by all downstream analyses)
FINAL_TABLE_OUT = os.path.join(output_dir, "neurons_with_ROI_labels.csv")

# Snapshot of the previous run of THIS script, kept so every re-run can
# report exactly what the new source files changed (step 8 at the very
# bottom). Point this at the archived copy, not at the live table —
# the live one is overwritten by this script.
REFERENCE_TABLE = os.path.join(
    output_dir, "old_electrode_tables", "aug-09-2026",
    "neurons_with_ROI_labels.csv")

step1_out = os.path.join(roi_assignment_dir, "cells_step1_coords.csv")
step1_utah_recon_out = os.path.join(roi_assignment_dir,
                                     "cells_step1_utah_reconstruction_log.csv")

COORD_MATCH_TOL_MM = 0.5
BAYLOR_RELIABILITY_TOL_MM = 8.0


# =============================================================================
# MNI305 -> MNI152 (Fischl affine)
# =============================================================================

MNI305_TO_MNI152 = np.array([
    [ 0.9975, -0.0073,  0.0176, -0.0429],
    [ 0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093,  0.9971,  1.1840],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
])


def mni305_to_mni152(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[None, :]
    coords_h = np.c_[coords, np.ones(len(coords))]
    return (coords_h @ MNI305_TO_MNI152.T)[:, :3]


# =============================================================================
# BAYLOR v2026 LOADER
# =============================================================================

def _bundle_key(label):
    """mRT2bHaEa04 -> mrt2bhaea (strip trailing 2 digits, lowercase)."""
    if pd.isna(label):
        return None
    return re.sub(r"\d{2}$", "", str(label)).lower()


# Row types in the v2026 CSVs that describe a microwire bundle. Each
# bundle usually appears twice: once as `microwires` (the bundle itself,
# label `mLT2bHb01`) and once as `sEEG-micro` (contact 01 of the macro
# probe carrying it, label `LT2bHb01`, ~3 mm shallower). Preference
# order matters — index 0 wins.
MICRO_TYPES = ["microwires", "sEEG-micro"]


def _micro_bundle_rows(d):
    """Return one row per micro bundle, keyed on the m-prefixed bundle
    name in a new `bundle_key` column. `microwires` rows win over
    `sEEG-micro` rows for the same bundle; `sEEG-micro` only fills in
    bundles that have no `microwires` row at all."""
    micro = d[d["Type"].isin(MICRO_TYPES)].copy()
    if micro.empty:
        return micro
    micro["bundle_key"] = micro["Label"].apply(_bundle_key)
    # sEEG-micro labels lack the leading 'm' — add it so both row types
    # land on the same key as the big table's `electrode label`.
    is_seeg = micro["Type"].eq("sEEG-micro")
    micro.loc[is_seeg, "bundle_key"] = "m" + micro.loc[is_seeg, "bundle_key"]
    micro["_pref"] = micro["Type"].map({t: i for i, t in enumerate(MICRO_TYPES)})
    return (micro.sort_values("_pref")
                 .drop_duplicates("bundle_key", keep="first"))


# Distance the microwire bundle protrudes beyond the deepest macro
# contact, along the probe insertion axis. This is not a per-subject
# measurement: in all 119 bundles across all 19 v2026 files that carry
# `microwires` rows, the offset is exactly 3.15 mm, i.e. Baylor applies a
# nominal Behnke-Fried protrusion rather than localising the wires
# individually. Reconstructing a bundle from its macro probe with this
# constant reproduces Baylor's own `microwires` row to a median of
# 0.25 mm (max 1.07 mm) over those 113 checkable bundles.
MICRO_PROTRUSION_MM = 3.15


def reconstruct_micro_from_macro(d, probe, cols):
    """Rebuild a micro-bundle position from its macro probe alone, for
    subject files that ship no `microwires` / `sEEG-micro` rows at all
    (YER as of 2026-08). Contact 01 is the deepest contact and
    01 -> 02 points back out along the shaft, so the bundle sits at
    `contact01 - MICRO_PROTRUSION_MM * unit(contact02 - contact01)`.

    NB `Label` (zero-padded contact number) is the sort key — the
    `ElectrodeID` column is mixed int/str and sorts lexicographically."""
    pr = d[(d["ProbeName"].astype(str) == probe)
           & (d["Type"].isin(["sEEG", "sEEG-micro"]))].sort_values("Label")
    if len(pr) < 2:
        return None
    p1 = pr.iloc[0][cols].to_numpy(float)
    p2 = pr.iloc[1][cols].to_numpy(float)
    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return None
    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return None
    return p1 - MICRO_PROTRUSION_MM * (v / np.linalg.norm(v))


def _file_152_is_selfconsistent(d):
    """True if the file's MNI152 column agrees with its own MNI305
    column under the Fischl transform (same gate as the micro-bundle
    reliability check, applied to every contact in the file)."""
    chk = d[["MNI305_x", "MNI305_y", "MNI305_z",
             "MNI152_x", "MNI152_y", "MNI152_z"]].dropna()
    if chk.empty:
        return False
    a305 = chk[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
    a152 = chk[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
    return bool(np.mean(np.linalg.norm(a152 - mni305_to_mni152(a305), axis=1))
                <= BAYLOR_RELIABILITY_TOL_MM)


def load_baylor_v2026(folder):
    """Return `(reliable, fallback_305, reconstructed, reliability_df)`.

    * reliable: {(subj_code, bundle_key): (x, y, z)} in MNI152 from
      subjects whose file passed the reliability check.
    * fallback_305: same shape, but coords are the file's MNI305 for
      subjects that failed the check (caller applies 305->152).
    * reconstructed: same shape, in MNI152, for subjects whose file has
      no micro rows at all — rebuilt from the macro probe geometry (see
      `reconstruct_micro_from_macro`). Flagged separately downstream
      because it is inferred, not supplied.
    """
    reliable = {}
    fallback = {}
    reconstructed = {}
    rel_rows = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith("-electrodes_v2026.csv"):
            continue
        subj_code = fn.split("-")[0]
        d = pd.read_csv(os.path.join(folder, fn))
        micro = _micro_bundle_rows(d)
        if micro.empty:
            # No micro rows shipped at all. The macro probes are still
            # there, and Baylor's own micro positions are a fixed
            # 3.15 mm extension of them, so rebuild every probe. Use the
            # file's MNI152 when it is self-consistent, else rebuild in
            # MNI305 and transform.
            use_152 = _file_152_is_selfconsistent(d)
            cols = (["MNI152_x", "MNI152_y", "MNI152_z"] if use_152
                    else ["MNI305_x", "MNI305_y", "MNI305_z"])
            n_rec = 0
            for probe in d["ProbeName"].dropna().astype(str).unique():
                xyz = reconstruct_micro_from_macro(d, probe, cols)
                if xyz is None:
                    continue
                if not use_152:
                    xyz = mni305_to_mni152(xyz)[0]
                reconstructed[(subj_code, "m" + probe.lower())] = tuple(xyz)
                n_rec += 1
            rel_rows.append({"subject_code": subj_code, "n_bundles": 0,
                             "n_microwires_rows": 0, "n_seegmicro_rows": 0,
                             "n_reconstructed_probes": n_rec,
                             "mean_mm": np.nan, "max_mm": np.nan,
                             "reliable": False})
            continue

        chk = micro[["MNI305_x", "MNI305_y", "MNI305_z",
                     "MNI152_x", "MNI152_y", "MNI152_z"]].dropna()
        if chk.empty:
            mean_d = max_d = np.nan
            is_rel = False
        else:
            a305 = chk[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
            a152 = chk[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
            recomp = mni305_to_mni152(a305)
            d152 = np.linalg.norm(a152 - recomp, axis=1)
            mean_d = float(np.mean(d152))
            max_d = float(np.max(d152))
            is_rel = mean_d <= BAYLOR_RELIABILITY_TOL_MM

        rel_rows.append({
            "subject_code": subj_code,
            "n_bundles": len(micro),
            "n_microwires_rows": int(micro["Type"].eq("microwires").sum()),
            "n_seegmicro_rows": int(micro["Type"].eq("sEEG-micro").sum()),
            "n_reconstructed_probes": 0,
            "mean_mm": mean_d, "max_mm": max_d, "reliable": is_rel})

        for _, r in micro.iterrows():
            key = (subj_code, r["bundle_key"])
            if is_rel:
                try:
                    x, y, z = float(r["MNI152_x"]), float(r["MNI152_y"]), float(r["MNI152_z"])
                except (TypeError, ValueError):
                    continue
                if any(np.isnan(v) for v in (x, y, z)):
                    continue
                reliable[key] = (x, y, z)
            else:
                try:
                    x, y, z = float(r["MNI305_x"]), float(r["MNI305_y"]), float(r["MNI305_z"])
                except (TypeError, ValueError):
                    continue
                if any(np.isnan(v) for v in (x, y, z)):
                    continue
                fallback[key] = (x, y, z)

    return reliable, fallback, reconstructed, pd.DataFrame(rel_rows)


# =============================================================================
# UCLA v2026 xlsx LOADER
# =============================================================================

# All 6 UCLA subjects now have a v2026 xlsx (UC3-0559 moved in by the user).
UCLA_SUBJECT_TO_FILE = {
    "UC3-0559": "sub-559",
    "UC3-0573": "sub-573",
    "UC2-0576": "sub-576",
    "UC3-0577": "sub-577",
    "UC2-0578": "sub-578",
    "UC3-0582": "sub-582",
}


def load_ucla_v2026(folder):
    """Return `{subject_label: DataFrame}` where each df has columns
    `electrode`, `region_hint`, `MNI_x`, `MNI_y`, `MNI_z`. Used to
    coord-match big-table cells (no label-based rules)."""
    tables = {}
    for subj, prefix in UCLA_SUBJECT_TO_FILE.items():
        fpath = os.path.join(folder, f"{prefix}_localizations.xlsx")
        if not os.path.exists(fpath):
            continue
        d = pd.read_excel(fpath, sheet_name="Sheet1")
        d = d[["electrode", "MNI_x", "MNI_y", "MNI_z", "NMM"]].copy()
        d = d.dropna(subset=["MNI_x", "MNI_y", "MNI_z"]).reset_index(drop=True)
        d = d.rename(columns={"NMM": "region_hint"})
        tables[subj] = d
    return tables


# =============================================================================
# UTAH .mat LOADER
# =============================================================================


def _load_mat(fpath):
    """Load .mat as a plain dict, supporting both v7 (scipy) and v7.3 (h5)."""
    try:
        return sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if not HAVE_H5:
            return None
        try:
            out = {}
            with h5py.File(fpath, "r") as f:
                for k in f.keys():
                    arr = np.array(f[k])
                    # h5py returns column-major; transpose 2D arrays.
                    if arr.ndim == 2:
                        arr = arr.T
                    out[k] = arr
            return out
        except Exception as e:
            print(f"  [mat load] h5 failed for {fpath}: {e}")
            return None
    except Exception as e:
        print(f"  [mat load] scipy failed for {fpath}: {e}")
        return None


def build_micro_map(mat):
    """Build {chan_value: (mni_xyz, micro_label)} for microwires.

    Two attempts:
      (A) MicroElec + ElecXYZMNIProj (gray-matter-projected).
      (B) MicroElecRaw + ElecXYZMNIRaw (raw electrode positions, used
          when MicroElec is empty — happens for some subjects, e.g. s23).

    In each attempt, LabelMap entries starting with 'm' mark
    microwires; each such (r, c) carries an amplifier chan value in
    ChannelMap1 or ChannelMap2. Sort micros by chan-value ascending —
    that ordering matches the sorted MicroElec[i] entries (validated on
    s02, s06). Return a map keyed by chan_value so we can look up any
    big-table `chan{N}` directly."""
    lm = mat.get("LabelMap")
    cm1 = mat.get("ChannelMap1")
    cm2 = mat.get("ChannelMap2")
    if lm is None:
        return {}

    entries = []  # (chan, r, c, label)
    for r in range(lm.shape[0]):
        for c in range(lm.shape[1]):
            lab = lm[r, c]
            if not (isinstance(lab, str) and lab.startswith("m")):
                continue
            chan = None
            for cm in (cm1, cm2):
                if cm is None:
                    continue
                v = cm[r, c]
                if v is not None and not np.isnan(v) and v > 0:
                    chan = int(v)
                    break
            if chan is not None:
                entries.append((chan, r, c, lab))
    entries.sort(key=lambda x: x[0])
    if not entries:
        return {}

    for micro_key, coord_key in (("MicroElec", "ElecXYZMNIProj"),
                                 ("MicroElecRaw", "ElecXYZMNIRaw")):
        micro = np.atleast_1d(mat.get(micro_key, np.array([]))).astype(int).ravel()
        coords = mat.get(coord_key)
        if not len(micro) or coords is None or len(coords) == 0:
            continue
        if len(entries) != len(micro):
            continue  # try the next attempt
        coords = np.asarray(coords, dtype=float)
        out = {}
        for i, (chan, r, c, lab) in enumerate(entries):
            row = int(micro[i]) - 1
            if 0 <= row < len(coords):
                out[chan] = (coords[row], lab, coord_key)
        if out:
            return out
    return {}


def channel_position(mat, chan):
    """Find (r, c) where `ChannelMap1[r, c] == chan`, else fall back to
    ChannelMap2. Returns None if not found."""
    for key in ("ChannelMap1", "ChannelMap2"):
        cm = mat.get(key)
        if cm is None:
            continue
        pos = np.argwhere(cm == chan)
        if len(pos):
            return int(pos[0][0]), int(pos[0][1])
    return None


def discover_utah_mats():
    """For each Utah subject, find which `s{NN}/electrodes/*.mat` file
    corresponds to it. We don't trust that s{NN} == subject_index
    (e.g. subject 24 lives in s23), so we coord-match: pool every
    folder's `ElecXYZMNIProj` coords and pick the folder that covers
    >= 50% of the subject's big-table cells at <= 0.5 mm. Returns
    {subject_label: (folder_name, mat_dict)} — the whole mat dict is
    kept so the independent-reconstruction step below can use it."""
    folder_mats = {}
    for f in sorted(os.listdir(path_to_subject_folders)):
        if not (f.startswith("s") and f[1:].isdigit()):
            continue
        edir = os.path.join(path_to_subject_folders, f, "electrodes")
        if not os.path.isdir(edir):
            continue
        # Prefer Electrodes.mat; fall back to ChannelMap.mat.
        chosen = None
        for name in ("Electrodes.mat", "ChannelMap.mat"):
            p = os.path.join(edir, name)
            if os.path.exists(p):
                chosen = p
                break
        if not chosen:
            continue
        mat = _load_mat(chosen)
        if mat is None or "ElecXYZMNIProj" not in mat:
            continue
        folder_mats[f] = (chosen, mat)

    bt = pd.read_csv(path_to_cell_table)
    bt = bt[bt["Recording Site"].astype(str).str.lower() == "utah"]
    mapping = {}
    for subj_label, grp in bt.groupby("Subject Label"):
        pts = grp[["MNI_x", "MNI_y", "MNI_z"]].dropna().to_numpy(float)
        if not len(pts):
            continue
        best = None
        for folder, (fpath, mat) in folder_mats.items():
            # Big-table coords may match Proj OR Raw depending on subject;
            # pool both for the folder-picking step (reconstruction below
            # always uses Proj).
            pool = []
            for k in ("ElecXYZMNIProj", "ElecXYZMNIRaw"):
                if k in mat:
                    a = np.asarray(mat[k], dtype=float)
                    if a.ndim == 2 and a.shape[1] == 3:
                        pool.append(a)
            if not pool:
                continue
            coords = np.vstack(pool)
            n_hit = 0
            for p in pts:
                d = np.linalg.norm(coords - p, axis=1)
                if np.any(~np.isnan(d)) and np.nanmin(d) <= COORD_MATCH_TOL_MM:
                    n_hit += 1
            if best is None or n_hit > best[1]:
                best = (folder, n_hit)
        if best and best[1] >= len(pts) * 0.5:
            mapping[subj_label.strip("'\" ")] = (
                best[0], folder_mats[best[0]][1])
    return mapping


# =============================================================================
# MAIN
# =============================================================================

print("Loading big table...")
df = pd.read_csv(path_to_cell_table)
coord_cols_bt = ["MNI_x", "MNI_y", "MNI_z"]
df[coord_cols_bt] = df[coord_cols_bt].astype(float)

df["subject_label_clean"] = df["Subject Label"].astype(str).str.strip("'\" ")
df["subject_code"] = df["subject_label_clean"].str.split("-").str[-1]

site = df["Recording Site"].astype(str).str.lower()
baylor_mask = site.eq("baylor")
ucla_mask = site.eq("ucla")
utah_mask = site.eq("utah")
print(f"  Baylor {baylor_mask.sum()} | UCLA {ucla_mask.sum()} | Utah {utah_mask.sum()}")


# =============================================================================
# BAYLOR
# =============================================================================

print("\n--- Baylor v2026 CSVs ---")
bay_reliable, bay_fallback, bay_reconstructed, bay_rel_df = load_baylor_v2026(path_to_v2026)
print(f"  {len(bay_rel_df)} subject files. Reliable: "
      f"{int(bay_rel_df['reliable'].sum())} / {len(bay_rel_df)}.")
print(bay_rel_df.round(2).to_string(index=False))

df["baylor_bundle_key"] = np.where(
    baylor_mask, df["electrode label"].apply(_bundle_key), None)


# =============================================================================
# UCLA
# =============================================================================

print("\n--- UCLA v2026 xlsx ---")
ucla_tables = load_ucla_v2026(path_to_v2026)
for subj, t in ucla_tables.items():
    print(f"  {subj}: {len(t)} electrodes")


# =============================================================================
# ASSIGN FINAL COORDS + PROVENANCE
# =============================================================================

df["MNI_x_final"] = np.nan
df["MNI_y_final"] = np.nan
df["MNI_z_final"] = np.nan
df["coord_source"] = "unset"
df["coord_verified"] = False
df["source_electrode"] = ""
df["source_region_hint"] = ""
df["source_match_dist_mm"] = np.nan


# --- Baylor: v2026 bundle overwrite ------------------------------------
for idx in df.index[baylor_mask]:
    code = df.at[idx, "subject_code"]
    key = (code, df.at[idx, "baylor_bundle_key"])
    if key in bay_reliable:
        x, y, z = bay_reliable[key]
        df.at[idx, "MNI_x_final"] = x
        df.at[idx, "MNI_y_final"] = y
        df.at[idx, "MNI_z_final"] = z
        df.at[idx, "coord_source"] = "baylor_v2026_bundle_micro"
        df.at[idx, "coord_verified"] = True
        df.at[idx, "source_electrode"] = key[1]
        df.at[idx, "source_match_dist_mm"] = 0.0
    elif key in bay_fallback:
        x, y, z = mni305_to_mni152(np.array(bay_fallback[key]))[0]
        df.at[idx, "MNI_x_final"] = x
        df.at[idx, "MNI_y_final"] = y
        df.at[idx, "MNI_z_final"] = z
        df.at[idx, "coord_source"] = "baylor_v2026_bundle_305to152_unreliable_file"
        df.at[idx, "coord_verified"] = False
        df.at[idx, "source_electrode"] = key[1]
    elif key in bay_reconstructed:
        # File shipped no micro rows; position inferred from the macro
        # probe with Baylor's own 3.15 mm protrusion (validated to
        # 0.25 mm median on the 113 bundles where a ground-truth
        # `microwires` row exists). Better than the pre-2026 big-table
        # macro position, but marked unverified because it is inferred.
        x, y, z = bay_reconstructed[key]
        df.at[idx, "MNI_x_final"] = x
        df.at[idx, "MNI_y_final"] = y
        df.at[idx, "MNI_z_final"] = z
        df.at[idx, "coord_source"] = "baylor_v2026_micro_reconstructed_from_macro"
        df.at[idx, "coord_verified"] = False
        df.at[idx, "source_electrode"] = key[1]
    else:
        # No v2026 file for this subject: fall back to big table
        # (Baylor big table is MNI305 -> transform to MNI152) and flag
        # as ~4 mm off from true micro position (macro-contact era).
        bt = df.loc[idx, coord_cols_bt].to_numpy(float)
        if not np.any(np.isnan(bt)):
            x, y, z = mni305_to_mni152(bt)[0]
            df.at[idx, "MNI_x_final"] = x
            df.at[idx, "MNI_y_final"] = y
            df.at[idx, "MNI_z_final"] = z
        df.at[idx, "coord_source"] = "baylor_bigtable_pre2026_macro_position"
        df.at[idx, "coord_verified"] = False


# --- UCLA: coord-match big-table -> v2026 xlsx --------------------------
for idx in df.index[ucla_mask]:
    subj = df.at[idx, "subject_label_clean"]
    bt = df.loc[idx, coord_cols_bt].to_numpy(float)
    df.at[idx, "MNI_x_final"] = bt[0]
    df.at[idx, "MNI_y_final"] = bt[1]
    df.at[idx, "MNI_z_final"] = bt[2]

    if np.any(np.isnan(bt)) or subj not in ucla_tables:
        df.at[idx, "coord_source"] = "ucla_bigtable_unverified"
        continue

    t = ucla_tables[subj]
    coords = t[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(float)
    d = np.linalg.norm(coords - bt, axis=1)
    i = int(np.argmin(d))
    dist = float(d[i])
    df.at[idx, "source_match_dist_mm"] = dist
    if dist <= COORD_MATCH_TOL_MM:
        df.at[idx, "coord_source"] = "ucla_v2026_coord_match"
        df.at[idx, "coord_verified"] = True
        df.at[idx, "source_electrode"] = str(t.iloc[i]["electrode"])
        df.at[idx, "source_region_hint"] = str(t.iloc[i]["region_hint"])
    else:
        df.at[idx, "coord_source"] = "ucla_bigtable_no_v2026_match"


# --- Utah: independently reconstruct MNI from ChannelMap+MicroElec+Proj -
print("\n--- Utah folder auto-discovery ---")
utah_mapping = discover_utah_mats()
for subj, (folder, _mat) in utah_mapping.items():
    print(f"  {subj} -> {folder}")

utah_micro_maps = {subj: build_micro_map(mat)
                   for subj, (_f, mat) in utah_mapping.items()}
utah_reconstruction_log = []

# Also precompute a pooled coord array per subject for the coord-match
# fallback (some subjects have no valid MicroElec/MicroElecRaw path).
utah_pooled = {}
for subj, (folder, mat) in utah_mapping.items():
    pool, tags = [], []
    for k in ("ElecXYZMNIProj", "ElecXYZMNIRaw"):
        if k in mat:
            a = np.asarray(mat[k], dtype=float)
            if a.ndim == 2 and a.shape[1] == 3:
                pool.append(a)
                tags.extend([(k, i) for i in range(len(a))])
    if pool:
        utah_pooled[subj] = (np.vstack(pool), tags)

for idx in df.index[utah_mask]:
    bt = df.loc[idx, coord_cols_bt].to_numpy(float)
    subj = df.at[idx, "subject_label_clean"]
    lab = str(df.at[idx, "electrode label"]).strip()

    if subj not in utah_mapping:
        df.loc[idx, ["MNI_x_final", "MNI_y_final", "MNI_z_final"]] = bt
        df.at[idx, "coord_source"] = "utah_bigtable_no_mat"
        continue

    folder, mat = utah_mapping[subj]
    mmap = utah_micro_maps.get(subj, {})

    # Try algorithmic reconstruction first.
    coord = None
    micro_label = None
    coord_source_field = None
    m = re.match(r"^chan(\d+)$", lab, re.IGNORECASE)
    if m:
        chan = int(m.group(1))
        if chan in mmap:
            coord, micro_label, coord_source_field = mmap[chan]
    else:
        # Label might already be a micro label like 'mLHIP4' (some cells
        # in the big table were entered this way). Find (r,c) in
        # LabelMap directly and read its coord via the chan value there.
        lm = mat.get("LabelMap")
        if lm is not None:
            pos = np.argwhere(lm == lab)
            if len(pos):
                r, c = pos[0]
                cm_chan = None
                for cm in (mat.get("ChannelMap1"), mat.get("ChannelMap2")):
                    if cm is None:
                        continue
                    v = cm[r, c]
                    if v is not None and not np.isnan(v) and v > 0:
                        cm_chan = int(v)
                        break
                if cm_chan in mmap:
                    coord, micro_label, coord_source_field = mmap[cm_chan]

    if coord is not None and not np.any(np.isnan(coord)):
        dist = (float(np.linalg.norm(coord - bt))
                if not np.any(np.isnan(bt)) else np.nan)
        utah_reconstruction_log.append({
            "subject": subj, "electrode_label": lab, "micro_label": micro_label,
            "bt_x": bt[0], "bt_y": bt[1], "bt_z": bt[2],
            "recon_x": coord[0], "recon_y": coord[1], "recon_z": coord[2],
            "shift_mm": dist,
        })
        # Conservative policy: only trust the reconstruction when it
        # agrees with the big-table coord within 3 mm. Larger shifts
        # are ambiguous — could be either a big-table error (like the
        # documented UT1-202217 mLVCG cases: chan107/110/112 with wrong
        # coords copy-pasted) or a per-subject quirk my algorithm
        # doesn't handle (e.g. s23 uses MicroElecRaw with a numbering
        # scheme I can't verify against ground truth). Flag those for
        # user review rather than silently overriding.
        UTAH_RECON_TOL_MM = 3.0
        if not np.isnan(dist) and dist <= UTAH_RECON_TOL_MM:
            df.at[idx, "MNI_x_final"] = coord[0]
            df.at[idx, "MNI_y_final"] = coord[1]
            df.at[idx, "MNI_z_final"] = coord[2]
            df.at[idx, "coord_source"] = f"utah_reconstructed_via_{coord_source_field}"
            df.at[idx, "coord_verified"] = True
            df.at[idx, "source_electrode"] = f"{folder}:{micro_label}"
            df.at[idx, "source_region_hint"] = micro_label
            df.at[idx, "source_match_dist_mm"] = dist
            continue
        else:
            df.loc[idx, ["MNI_x_final", "MNI_y_final", "MNI_z_final"]] = bt
            df.at[idx, "coord_source"] = "utah_bigtable_recon_disagrees_gt3mm"
            df.at[idx, "source_electrode"] = f"{folder}:{micro_label} (recon={coord.tolist()})"
            df.at[idx, "source_region_hint"] = micro_label
            df.at[idx, "source_match_dist_mm"] = dist
            continue

    # Fallback: no reconstruction available. Verify big-table coord
    # against the pooled .mat coord set (>= 0.5 mm to a nearest row).
    df.loc[idx, ["MNI_x_final", "MNI_y_final", "MNI_z_final"]] = bt
    if subj in utah_pooled and not np.any(np.isnan(bt)):
        coords, tags = utah_pooled[subj]
        d = np.linalg.norm(coords - bt, axis=1)
        if np.any(~np.isnan(d)):
            i = int(np.nanargmin(d))
            dist = float(d[i])
            df.at[idx, "source_match_dist_mm"] = dist
            if dist <= COORD_MATCH_TOL_MM:
                df.at[idx, "coord_source"] = "utah_bigtable_verified_by_mat_lookup"
                df.at[idx, "coord_verified"] = True
                df.at[idx, "source_electrode"] = f"{folder}:{tags[i][0]}[{tags[i][1]}]"
                continue
    df.at[idx, "coord_source"] = "utah_bigtable_unreconstructed"

if utah_reconstruction_log:
    ul = pd.DataFrame(utah_reconstruction_log)
    ul.to_csv(step1_utah_recon_out, index=False)
    n_agree = int((ul["shift_mm"] <= 3.0).sum())
    n_disagree = int((ul["shift_mm"] > 3.0).sum())
    print(f"\n[Utah] Algorithmically reconstructed {len(ul)} cells: "
          f"{n_agree} agree with big-table (<=3 mm), "
          f"{n_disagree} disagree (>3 mm, flagged for user review).")
    print(f"Full log saved to: {step1_utah_recon_out}")


# =============================================================================
# REPORTS
# =============================================================================

print("\n=== Coord source counts (per site) ===")
print(pd.crosstab(df["Recording Site"], df["coord_source"]).to_string())

print("\n=== Coord verification (per site) ===")
print(pd.crosstab(df["Recording Site"], df["coord_verified"]).to_string())

print("\n=== Per-subject verification summary ===")
per_subj = (df.groupby(["Recording Site", "subject_label_clean"])
              .agg(n_cells=("MNI_x_final", "size"),
                   n_verified=("coord_verified", "sum"),
                   max_match_dist_mm=("source_match_dist_mm", "max"))
              .round(3)
              .reset_index()
              .sort_values(["Recording Site", "n_verified"]))
print(per_subj.to_string(index=False))

# Cells that still need attention
unverified = df[~df["coord_verified"]]
print(f"\n=== {len(unverified)} unverified cells ===")
if len(unverified):
    print(unverified.groupby(["Recording Site", "subject_label_clean", "coord_source"])
                    .size().reset_index(name="n").to_string(index=False))


# =============================================================================
# SAVE
# =============================================================================

keep_cols = [
    "subject", "cell idx", "electrode label", "region label", "roi",
    "Subject Label", "subject_label_clean", "subject_code", "Recording Site",
    "MNI_x", "MNI_y", "MNI_z",              # original big-table (untouched)
    "MNI_x_final", "MNI_y_final", "MNI_z_final",
    "coord_source", "coord_verified",
    "source_electrode", "source_region_hint", "source_match_dist_mm",
    "baylor_bundle_key",
    "correct ROI", "Confirmed region label", "Notes",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df[keep_cols].to_csv(step1_out, index=False)
print(f"\nSaved: {step1_out}")


# =============================================================================
# STEP 2 — PROBABILISTIC-ATLAS ROI ASSIGNMENT
# =============================================================================
#
# For every cell (using its MNI_*_final coord), query 3 probabilistic
# atlases + 1 max-prob atlas and assign an ROI in priority order at a
# 25 % probability threshold:
#
#   1. EC              — Juelich (`GM Hippocampus entorhinal cortex`)
#   2. mPFC            — Brainnetome A32sg/A32p/A24cd/A24rv + A10m + A9m
#   3. mOFC / vOFC    — Brainnetome A14m + A13 + A11m
#   4. Hippocampus     — HO subcortical, split anterior vs mid at y = -21
#                        mm (Poppenk & Moscovitch 2013 — anterior HC has
#                        preferential mPFC connectivity via uncinate)
#   5. Parahippocampal — HO cortical
#   6. PCC             — Brainnetome A23 + HO `Cingulate Gyrus, posterior`
#   7. Precuneus       — Brainnetome A31 / dmPOS + HO `Precuneous Cortex`
#   8. Visual          — Juelich V1/V2/V3 + HO occipital
#
# Cells that no rule assigns become "leftover" and are plotted on a
# glass brain for the user to eyeball. Brainnetome is used as its
# max-prob atlas (per-voxel most-likely region); Juelich and HO come as
# 4D probabilistic images so a real 25 % threshold applies.

from nilearn import datasets as nldatasets

HC_ANT_MID_Y = -21.0  # split y (mm). Poppenk & Moscovitch 2013 (anterior HC
                       # is preferentially connected to mPFC/mOFC via
                       # the uncinate fasciculus). Same value we used
                       # previously in cell_to_roi_MNI.py so results
                       # remain comparable.

step2_out = os.path.join(roi_assignment_dir, "cells_step2_atlas_labels.csv")
step2_leftover_plot = os.path.join(roi_assignment_dir, "cells_step2_leftovers_glassbrain.png")
step2_hc_y_plot = os.path.join(roi_assignment_dir, "cells_step2_hc_y_distribution.png")


# =============================================================================
# LOAD ATLASES (all at 25 % probability threshold via max-prob variants)
# =============================================================================

print("\n\n=========================================================")
print(" STEP 2 — atlas ROI assignment at 25% probability threshold")
print("=========================================================\n")
print("Loading atlases (already cached by nilearn / local)...")

ho_cort_atlas = nldatasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
ho_sub_atlas = nldatasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
juelich_atlas = nldatasets.fetch_atlas_juelich("maxprob-thr25-2mm")

path_to_brainnetome = "/Users/xpsy1114/Documents/toolboxes/Brainnatome"
brainnetome_nii = os.path.join(path_to_brainnetome, "BN_Atlas_246_1mm.nii.gz")
brainnetome_lut = os.path.join(path_to_brainnetome, "BN_Atlas_246_LUT.txt")


def load_brainnetome_labels(lut_path):
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
                    idx = int(p); break
                except ValueError:
                    pass
            if idx is None:
                continue
            text_parts = [p for p in parts
                          if not re.match(r"^-?\d+(\.\d+)?$", p)]
            labels[idx] = " ".join(text_parts).strip() or f"BN_{idx}"
    return labels


class MaxProbAtlas:
    """Single-label-per-voxel lookup. For nilearn Harvard-Oxford/Juelich
    the `maxprob-thr25-2mm` variant assigns each voxel the label of the
    region with highest probability *only if that probability is >= 25 %*
    — voxels below threshold are 0 (background). So a non-background
    return already means the cell passed the 25 % criterion."""
    def __init__(self, atlas_or_path, labels=None):
        maps = getattr(atlas_or_path, "maps", atlas_or_path)
        self.img = nib.load(maps) if isinstance(maps, str) else maps
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        if labels is None:
            self.labels = list(atlas_or_path.labels)
        else:
            self.labels = labels

    def label_at(self, xyz):
        v = np.round(nib.affines.apply_affine(self.inv_affine, xyz)).astype(int)
        if np.any(v < 0) or np.any(v >= self.data.shape):
            return None
        idx = int(self.data[tuple(v)])
        if idx == 0:
            return None
        if isinstance(self.labels, dict):
            return self.labels.get(idx)
        return self.labels[idx] if idx < len(self.labels) else None


ho_cort = MaxProbAtlas(ho_cort_atlas)
ho_sub = MaxProbAtlas(ho_sub_atlas)
juelich = MaxProbAtlas(juelich_atlas)
brainnetome = MaxProbAtlas(brainnetome_nii, labels=load_brainnetome_labels(brainnetome_lut))
print(f"  HO cortical: {len(ho_cort.labels)} labels")
print(f"  HO subcortical: {len(ho_sub.labels)} labels")
print(f"  Juelich: {len(juelich.labels)} labels")
print(f"  Brainnetome: {len(brainnetome.labels)} labels")


# =============================================================================
# ROI ASSIGNMENT RULES (priority order)
# =============================================================================

def _contains_any(label, patterns):
    if not label:
        return False
    low = label.lower()
    return any(p.lower() in low for p in patterns)


def assign_atlas_roi(row):
    """Return (atlas_roi, atlas_source_label, atlas_reason). Priority
    order — first rule that matches wins. atlas_roi == 'leftover' if
    nothing matches at the 25 % threshold."""
    y = row["MNI_y_final"]
    jl = row["_juelich"]        # single label or None
    hc = row["_ho_cort"]
    hs = row["_ho_sub"]
    bn = row["_bn_label"]

    # 1. EC — Juelich histological entorhinal cortex, provided y >= -18.
    #    The Juelich EC probability map has a dense core at y ~= -6 to
    #    -12 mm (Amunts et al. 2005, Insausti et al. 1998) and tails off
    #    posteriorly at y ~= -18 to -20 mm where it transitions into the
    #    subiculum. Cells with y < -18 whose Juelich label at 25% is
    #    still "entorhinal" are at the far posterior EC border, where
    #    the atlas overlaps heavily with the subicular complex — those
    #    fall through to rule 4 (Hippocampus), which captures them as
    #    HC via the Juelich subiculum/CA neighborhood extension. Same
    #    -18 mm cutoff is used in the LEC/REC rescue path.
    if _contains_any(jl, ["entorhinal"]) and float(y) >= -18:
        return ("EC", jl, "juelich_entorhinal_y_ge_-18")

    # 2. mPFC — matched by EITHER
    #    (a) Brainnetome A32sg / A32p / A24cd / A24rv / A10m / A9m
    #        (fine cytoarchitectonic parcellation of medial PFC), OR
    #    (b) Harvard-Oxford cortical `Cingulate Gyrus, anterior
    #        division` (coarser gyral label covering the whole ACC).
    #    A32sg (subgenual) stays with mPFC EXCEPT for cells with
    #    z <= -10 which are on the orbital surface (assigned mOFC).
    #    Only cells with y >= 10 stay mPFC (anterior cingulate proper);
    #    more posterior hits become `medial_CC` (mid-cingulate, following
    #    the old cell_to_roi_MNI.py `acc_y_cutoff = 10` convention).
    _mpfc_bn = _contains_any(bn, ["a32sg", "a32p", "a24cd", "a24rv",
                                    "a10m", "a9m"])
    _mpfc_hc = (hc is not None
                and "cingulate gyrus, anterior" in hc.lower())
    if _mpfc_bn or _mpfc_hc:
        z = row["MNI_z_final"]
        if _contains_any(bn, ["a32sg"]) and float(z) <= -10:
            return ("mOFC", bn, "a32sg_ventral_z_le_-10")
        atlas_lab = bn if _mpfc_bn else hc
        src = "brainnetome_mpfc" if _mpfc_bn else "ho_cort_anterior_cingulate"
        if float(y) >= 10:
            return ("mPFC", atlas_lab, src)
        return ("medial_CC", atlas_lab, src + "_but_y_lt_10")

    # 3. mOFC / vOFC — Brainnetome A14m / A13 / A11m
    if _contains_any(bn, ["a14m", "a13", "a11m"]):
        return ("mOFC", bn, "brainnetome_mofc")

    # 4. Hippocampus — assigned when EITHER
    #    (a) HO subcortical returns "Hippocampus" at 25 %, OR
    #    (b) Juelich histological atlas returns one of the non-EC
    #        hippocampal subfields at 25 %: `subiculum` or `cornu
    #        ammonis`, OR
    #    (c) Juelich subiculum / cornu ammonis is within a ±3 mm
    #        cube neighborhood of the coord (extension of rule b to
    #        cover voxels that sit in the immediately adjacent white
    #        matter or transitional cortex — Juelich MTL parcels are
    #        thin and 25 %-thresholded, so a small neighborhood probe
    #        captures the perihippocampal WM that anatomically belongs
    #        to the hippocampal formation but falls outside the strict
    #        maxprob mask).
    #    Rule (b)+(c) matter because HO subcortical uses a coarser
    #    hippocampal mask, and the strict 25 % threshold on Juelich
    #    otherwise leaves subicular voxels to be captured by the
    #    coarser parahippocampal-gyrus rule (rule 5) despite being
    #    anatomically hippocampal.
    _hc_hs   = _contains_any(hs, ["hippocampus"])
    _hc_jl0  = _contains_any(jl, ["subiculum", "cornu ammonis"])
    _hc_jlN  = None            # (label, dist) if found in ±3 mm neighborhood
    if not (_hc_hs or _hc_jl0):
        xyz_here = np.array([row["MNI_x_final"], row["MNI_y_final"],
                              row["MNI_z_final"]], dtype=float)
        if not np.any(np.isnan(xyz_here)):
            for r_probe in range(1, 4):
                found = False
                for dx in (-r_probe, 0, r_probe):
                    for dy in (-r_probe, 0, r_probe):
                        for dz in (-r_probe, 0, r_probe):
                            if dx == dy == dz == 0:
                                continue
                            lab_probe = juelich.label_at(
                                xyz_here + np.array([dx, dy, dz], float))
                            if lab_probe and (
                                "subiculum" in lab_probe.lower()
                                or "cornu ammonis" in lab_probe.lower()):
                                _hc_jlN = (lab_probe, r_probe); found = True; break
                        if found: break
                    if found: break
                if found: break
    if _hc_hs or _hc_jl0 or _hc_jlN:
        if _hc_hs:
            atlas_label = hs; suffix = ""
        elif _hc_jl0:
            atlas_label = jl; suffix = ""
        else:
            atlas_label = _hc_jlN[0]; suffix = f"_neighbor@{_hc_jlN[1]}mm"
        roi = "HC_anterior" if float(y) >= HC_ANT_MID_Y else "HC_mid"
        return (roi, atlas_label,
                f"hippocampal_subfield_split_y{HC_ANT_MID_Y}{suffix}")

    # 5. PHC (parahippocampal cortex) — HO cortical
    if _contains_any(hc, ["parahippocampal"]):
        return ("PHC", hc, "ho_cort_parahippocampal")

    # 6. PCC — Brainnetome A23 (PCC proper) + A31/dmPOS (Precuneus).
    #    Both are DMN posterior-medial hubs (Andrews-Hanna 2010,
    #    Utevsky 2014); collapsed under the CLAUDE.md name `PCC`.
    if _contains_any(bn, ["a23", "a31", "dmpos"]):
        return ("PCC", bn, "brainnetome_pcc_precuneus")
    if hc and "cingulate" in hc.lower() and "posterior" in hc.lower():
        return ("PCC", hc, "ho_cort_pcc")
    if _contains_any(hc, ["precuneous"]):
        return ("PCC", hc, "ho_cort_precuneous")

    # 8. Visual — Juelich V1/V2/V3 or HO cortical occipital
    if _contains_any(jl, ["v1", "v2", "v3", "visual", "calcarine"]):
        return ("Visual", jl, "juelich_visual")
    if _contains_any(hc, ["occipital", "cuneal", "lingual",
                          "intracalcarine", "supracalcarine"]):
        return ("Visual", hc, "ho_cort_occipital")

    # 9. Amygdala — HO subcortical
    if _contains_any(hs, ["amygdala"]):
        return ("Amygdala", hs, "ho_sub_amygdala")

    # 10. Thalamus — HO subcortical
    if _contains_any(hs, ["thalamus"]):
        return ("Thalamus", hs, "ho_sub_thalamus")

    # 11. Insula — HO cortical `Insular Cortex`
    if _contains_any(hc, ["insular"]):
        return ("Insula", hc, "ho_cort_insula")

    # 12. Atlas-neighborhood extension for cells whose exact voxel
    #    sits in white matter or falls outside any 25 %-thresholded
    #    atlas mask, but a target-ROI atlas region is immediately
    #    adjacent (±3 mm cube grid). Purely atlas-based — probes the
    #    same atlases at slightly-offset voxels, no text lookup.
    #    Priority ORDER matches the priority order of the exact-coord
    #    rules 1-11 above (EC first, then mPFC/mOFC, then PHC, PCC,
    #    Visual, Amygdala, Thalamus, Insula). mPFC is checked here
    #    via the HO cortical anterior-cingulate probe used for the
    #    perigenual cingulum bundle.
    xyz_here = np.array([row["MNI_x_final"], row["MNI_y_final"],
                          row["MNI_z_final"]], dtype=float)
    if np.any(np.isnan(xyz_here)):
        return ("leftover", "", "")

    NEIGH_PROBES = [
        # (target_roi, atlas_obj, patterns, extra_y_constraint)
        ("EC",       juelich, ["entorhinal"],
                                     ("y_ge", -18)),   # EC only if y >= -18
        ("mPFC",     ho_cort, ["cingulate gyrus, anterior"],
                                     ("y_ge",  10)),   # mPFC if y >= 10
        ("medial_CC",ho_cort, ["cingulate gyrus, anterior"],
                                     ("y_lt",  10)),   # else medial_CC
        ("mOFC",     brainnetome, ["a11m", "a13", "a14m"], None),
        ("mOFC",     ho_cort, ["frontal orbital", "subcallosal"], None),
        ("PHC",      ho_cort, ["parahippocampal"], None),
        ("PCC",      brainnetome, ["a23", "a31", "dmpos"], None),
        ("PCC",      ho_cort, ["cingulate gyrus, posterior", "precuneous"], None),
        ("Visual",   ho_cort, ["occipital", "cuneal", "lingual",
                                "intracalcarine", "supracalcarine"], None),
        ("Amygdala", ho_sub,  ["amygdala"], None),
        ("Thalamus", ho_sub,  ["thalamus"], None),
        ("Insula",   ho_cort, ["insular"], None),
    ]

    for r_probe in range(1, 4):    # 1, 2, 3 mm
        # Enumerate all offsets at radius r_probe (cube grid).
        offsets = [(dx, dy, dz)
                   for dx in (-r_probe, 0, r_probe)
                   for dy in (-r_probe, 0, r_probe)
                   for dz in (-r_probe, 0, r_probe)
                   if not (dx == dy == dz == 0)]
        for target_roi, atlas_obj, patterns, ycon in NEIGH_PROBES:
            # skip if y-constraint fails at the cell's own coord
            if ycon is not None:
                op, thr = ycon
                if op == "y_ge" and float(y) < thr: continue
                if op == "y_lt" and float(y) >= thr: continue
            for dx, dy, dz in offsets:
                lab_probe = atlas_obj.label_at(
                    xyz_here + np.array([dx, dy, dz], float))
                if lab_probe and any(p.lower() in lab_probe.lower()
                                       for p in patterns):
                    return (target_roi, lab_probe,
                            f"atlas_neighbor@{r_probe}mm")

    return ("leftover", "", "")


# =============================================================================
# QUERY EVERY CELL
# =============================================================================

print("\nQuerying atlas labels at each cell's MNI_*_final...")

records = []
for idx, row in df.iterrows():
    xyz = np.array([row["MNI_x_final"], row["MNI_y_final"],
                    row["MNI_z_final"]], dtype=float)
    if np.any(np.isnan(xyz)):
        records.append({"_juelich": None, "_ho_cort": None,
                        "_ho_sub": None, "_bn_label": None})
        continue
    records.append({
        "_juelich": juelich.label_at(xyz),
        "_ho_cort": ho_cort.label_at(xyz),
        "_ho_sub":  ho_sub.label_at(xyz),
        "_bn_label": brainnetome.label_at(xyz),
    })

atlas_df = pd.DataFrame(records, index=df.index)
df = pd.concat([df, atlas_df], axis=1)

assigns = df.apply(assign_atlas_roi, axis=1, result_type="expand")
assigns.columns = ["atlas_roi", "atlas_source_label", "atlas_reason"]
df = pd.concat([df, assigns], axis=1)

print("\nATLAS ROI COUNTS:")
print(df["atlas_roi"].value_counts(dropna=False).to_string())

print("\nPer-ROI cell/subject counts:")
per_roi = (df.groupby("atlas_roi")
             .agg(n_cells=("subject", "size"),
                  n_subjects=("subject", "nunique"))
             .sort_values("n_cells", ascending=False))
print(per_roi.to_string())


# =============================================================================
# HC ANT/MID SPLIT DIAGNOSTIC PLOT
# =============================================================================

print(f"\nPlotting HC y-distribution (split at y = {HC_ANT_MID_Y} mm)...")
hc_cells = df[df["atlas_roi"].isin(["HC_anterior", "HC_mid"])].copy()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(hc_cells["MNI_y_final"], bins=40, color="#5b9b8d",
        edgecolor="black", linewidth=0.4)
ax.axvline(HC_ANT_MID_Y, color="#5C1027", linestyle="--", linewidth=1.5,
           label=f"split @ y = {HC_ANT_MID_Y} mm\n(Poppenk & Moscovitch 2013)")
ax.set_xlabel("MNI y (mm)  — posterior ← → anterior")
ax.set_ylabel("Number of hippocampal cells")
ax.set_title(f"HO-subcortical hippocampal cells (n = {len(hc_cells)}) — "
             "anterior/mid split")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(step2_hc_y_plot, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"Saved: {step2_hc_y_plot}")


# =============================================================================
# LEFTOVER CELLS GLASS-BRAIN
# =============================================================================

leftovers = df[df["atlas_roi"] == "leftover"].copy()
print(f"\n{len(leftovers)} leftover cells (no atlas ROI at 25% threshold).")

if len(leftovers):
    from nilearn.plotting import plot_glass_brain
    fig = plt.figure(figsize=(12, 4))
    gb = plot_glass_brain(None, display_mode="lyrz", figure=fig,
                          title=f"Leftover cells (n = {len(leftovers)}) "
                                "— no ROI at 25% threshold")
    coords = leftovers[["MNI_x_final", "MNI_y_final",
                        "MNI_z_final"]].to_numpy(float)
    gb.add_markers(coords, marker_color="#0e3d3a", marker_size=18)
    fig.savefig(step2_leftover_plot, dpi=200, bbox_inches="tight")
    plt.close("all")
    plt.close("all")
    print(f"Saved: {step2_leftover_plot}")


# =============================================================================
# SAVE STEP 2 OUTPUT
# =============================================================================

drop_scratch = [c for c in df.columns if c.startswith("_")]
out_df = df.drop(columns=drop_scratch)
out_df.to_csv(step2_out, index=False)
print(f"\nSaved: {step2_out}")


# =============================================================================
# STEP 3 — LEFTOVER DIAGNOSIS
# =============================================================================
# For each cell that stayed 'leftover' at the 25 % threshold, look up
# which region it WOULD be in at the permissive thr0 variant of each
# atlas (winning region regardless of probability). Group by primary
# hint (first non-empty atlas label in a priority-ordered list), then
# plot each cluster on a glass brain so the user can decide per cluster:
# add to an existing ROI, create a new ROI, or leave leftover.
#
# `source_region_hint` from step 1 (the .mat LabelMap micro-label or
# UCLA xlsx NMM region) is displayed alongside for extra context —
# especially useful for testing Svenja's hypothesis that some Utah cells
# are ~3 mm off (last-macro-contact confusion à la Baylor pre-2026): if
# many white-matter leftovers carry a `mLOFC*`/`mLHIP*`-style intent
# label, that's evidence for a systematic shift.

print("\n\n=========================================================")
print(" STEP 3 — leftover cluster diagnosis")
print("=========================================================\n")

step3_summary_out = os.path.join(roi_assignment_dir, "cells_step3_leftover_clusters.csv")
step3_plot_dir = os.path.join(roi_assignment_dir, "cells_step3_leftover_plots")
os.makedirs(step3_plot_dir, exist_ok=True)

# Load permissive (no-threshold) atlas variants — same file cache, just
# different threshold param, so no downloads.
print("Loading thr0 atlas variants for permissive leftover labeling...")
ho_cort_0 = MaxProbAtlas(
    nldatasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm"))
ho_sub_0 = MaxProbAtlas(
    nldatasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-2mm"))
juelich_0 = MaxProbAtlas(
    nldatasets.fetch_atlas_juelich("maxprob-thr0-2mm"))

leftovers = df[df["atlas_roi"] == "leftover"].copy()
print(f"{len(leftovers)} leftover cells to analyse.\n")

# Query permissive atlases at each leftover coord
def _label_or_empty(atlas, xyz):
    if np.any(np.isnan(xyz)):
        return ""
    return atlas.label_at(xyz) or ""


rows = []
for idx, r in leftovers.iterrows():
    xyz = np.array([r["MNI_x_final"], r["MNI_y_final"],
                    r["MNI_z_final"]], dtype=float)
    rows.append({
        "ho_cort_thr0": _label_or_empty(ho_cort_0, xyz),
        "ho_sub_thr0":  _label_or_empty(ho_sub_0, xyz),
        "juelich_thr0": _label_or_empty(juelich_0, xyz),
        "bn_thr0":      _label_or_empty(brainnetome, xyz),
    })
leftover_atlas = pd.DataFrame(rows, index=leftovers.index)
leftovers = pd.concat([leftovers, leftover_atlas], axis=1)


# Primary hint priority: HO subcortical > Juelich (histological) >
# Brainnetome (fine cortical) > HO cortical (coarse). Subcortical wins
# because white matter / hippocampus / amygdala labels are the most
# useful for spotting "just outside gray-matter" cells.
def _primary_hint(row):
    for field in ("ho_sub_thr0", "juelich_thr0", "bn_thr0", "ho_cort_thr0"):
        v = row[field]
        if v and v.lower() != "background":
            return v
    return "no atlas label"


leftovers["primary_hint"] = leftovers.apply(_primary_hint, axis=1)


def _sample_source_hints(series):
    hints = [h for h in series.dropna().unique() if h]
    return "; ".join(hints[:5]) + (" ..." if len(hints) > 5 else "")


cluster_summary = (
    leftovers.groupby("primary_hint")
    .agg(n_cells=("subject", "size"),
         n_subjects=("subject", "nunique"),
         sample_source_region_hints=("source_region_hint",
                                     _sample_source_hints),
         sample_recording_sites=("Recording Site",
                                 lambda s: "; ".join(sorted(s.unique()))))
    .sort_values("n_cells", ascending=False)
    .reset_index())

cluster_summary.to_csv(step3_summary_out, index=False)
print("Leftover clusters by primary atlas hint "
      "(HO_sub > Juelich > Brainnetome > HO_cort priority):\n")
print(cluster_summary.to_string(index=False))
print(f"\nSaved: {step3_summary_out}")


# Per-cluster glass-brain plot (only clusters with n_cells >= 3)
from nilearn.plotting import plot_glass_brain

MIN_CLUSTER_PLOT = 3
plotted = 0
for _, row in cluster_summary.iterrows():
    hint = row["primary_hint"]
    if row["n_cells"] < MIN_CLUSTER_PLOT:
        continue
    grp = leftovers[leftovers["primary_hint"] == hint]
    coords = grp[["MNI_x_final", "MNI_y_final",
                  "MNI_z_final"]].to_numpy(float)
    fig = plt.figure(figsize=(12, 4))
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", hint)[:60]
    title = (f"{hint}  |  n={row['n_cells']} cells, "
             f"{row['n_subjects']} subjects\n"
             f"electrode hints: {row['sample_source_region_hints']}")
    gb = plot_glass_brain(None, display_mode="lyrz", figure=fig, title=title)
    gb.add_markers(coords, marker_color="#0e3d3a", marker_size=22)
    fpath = os.path.join(step3_plot_dir,
                         f"leftover_{plotted:02d}_{safe_name}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close("all")
    plotted += 1

print(f"\nSaved {plotted} per-cluster glass-brain plots to: "
      f"{step3_plot_dir}")


# Save annotated leftovers table
step3_leftovers_out = os.path.join(roi_assignment_dir,
                                    "cells_step3_leftovers_annotated.csv")
leftover_keep = [
    "subject", "cell idx", "electrode label", "region label",
    "Subject Label", "Recording Site",
    "MNI_x_final", "MNI_y_final", "MNI_z_final",
    "coord_source", "source_region_hint",
    "ho_cort_thr0", "ho_sub_thr0", "juelich_thr0", "bn_thr0",
    "primary_hint",
]
leftover_keep = [c for c in leftover_keep if c in leftovers.columns]
leftovers[leftover_keep].to_csv(step3_leftovers_out, index=False)
print(f"Saved: {step3_leftovers_out}")


# =============================================================================
# STEP 4 — RESCUE LEFTOVERS VIA ELECTRODE INTENT
# =============================================================================
# Many leftovers sit ~1-5 mm outside a gray-matter ROI whose label is
# right there in `source_region_hint` (from the .mat LabelMap for Utah,
# or v2026 xlsx for UCLA — e.g. mLOFC3, mRAHIP4, mLACC1, mRPHG). Use
# that intent to search a small neighborhood for the atlas label the
# electrode was targeting; if found within ±5 mm, promote the cell out
# of 'leftover'. This is the operational test of Svenja's "coord is 3 mm
# too far" hypothesis: only rescue when the intended gray-matter region
# is genuinely nearby.

RESCUE_MAX_DIST_MM = 8.0

# Map (intent-label substring) -> (target_roi, atlas, [patterns])
# atlas keys: 'ho_sub', 'ho_cort', 'juelich', 'brainnetome'
INTENT_MAP = [
    # Note: text patterns matching UCLA v2026 xlsx region_hint (e.g.
    # "phg parahippocampal", "posterior cingulate", "insula") were
    # removed on 2026-07-29. They were catching cells whose electrode
    # was targeting a DIFFERENT region (e.g. sub60 GA1-REC* electrodes
    # labelled REC by the researcher, but annotated "Right PHG
    # parahippocampal gyrus" by UCLA's coord-based v2026 xlsx). The
    # rescue now only uses site-standard electrode labels (`mlphg`,
    # `lins`, `lec`, ...) which reflect actual electrode intent
    # rather than post-hoc coord-based annotation.

    # ==== Utah .mat LabelMap micro-labels (m-prefixed, specific)
    ("mlofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    ("mrofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    # HO fallback for mOFC — the coord `RmOFC (8.6, 22.9, -14.9)` sits in
    # WM at exact position; Brainnetome doesn't cover it within 8 mm but
    # HO cortical `Frontal Orbital Cortex` / `Subcallosal Cortex` do.
    ("mlofc",  "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("mrofc",  "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("mlamcc", "mPFC", "brainnetome", ["a24", "a32"]),
    ("mramcc", "mPFC", "brainnetome", ["a24", "a32"]),
    ("mlacc",  "mPFC", "brainnetome", ["a24", "a32"]),
    ("mracc",  "mPFC", "brainnetome", ["a24", "a32"]),
    ("mlmcg",  "mPFC", "brainnetome", ["a24", "a32", "a23"]),
    ("mrmcg",  "mPFC", "brainnetome", ["a24", "a32", "a23"]),
    # HC / PHG intents: try EC first (Juelich entorhinal at ≤5 mm) —
    # EC boundary is diffuse and the electrode label is often wrong
    # about EC vs HC head; prefer EC on the brink.
    ("mlahip", "EC",   "juelich", ["entorhinal"]),
    ("mrahip", "EC",   "juelich", ["entorhinal"]),
    ("mlahc",  "EC",   "juelich", ["entorhinal"]),
    ("mrahc",  "EC",   "juelich", ["entorhinal"]),
    ("mlhip",  "EC",   "juelich", ["entorhinal"]),
    ("mrhip",  "EC",   "juelich", ["entorhinal"]),
    ("mlphg",  "EC",   "juelich", ["entorhinal"]),
    ("mrphg",  "EC",   "juelich", ["entorhinal"]),
    ("mlahip", "HC",   "ho_sub", ["hippocampus"]),
    ("mrahip", "HC",   "ho_sub", ["hippocampus"]),
    ("mlahc",  "HC",   "ho_sub", ["hippocampus"]),
    ("mrahc",  "HC",   "ho_sub", ["hippocampus"]),
    ("mlhip",  "HC",   "ho_sub", ["hippocampus"]),
    ("mrhip",  "HC",   "ho_sub", ["hippocampus"]),
    ("mlphg",  "PHC", "ho_cort", ["parahippocampal"]),
    ("mrphg",  "PHC", "ho_cort", ["parahippocampal"]),

    # ==== Big-table `region label` (all sites) — coarser, applied as fallback
    ("l_precuneus", "PCC", "brainnetome", ["a31", "dmpos", "a23"]),
    ("r_precuneus", "PCC", "brainnetome", ["a31", "dmpos", "a23"]),
    ("precuneus",   "PCC", "brainnetome", ["a31", "dmpos", "a23"]),
    ("thalamus",    "Thalamus",  "ho_sub", ["thalamus"]),
    ("lamyg",       "Amygdala",  "ho_sub", ["amygdala"]),
    ("ramyg",       "Amygdala",  "ho_sub", ["amygdala"]),
    ("lvofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    ("rvofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    # `RmOFC` / `LmOFC` (Baylor `right/left medial OFC` labels) —
    # letter order differs from Utah's `mLOFC`, so add explicit patterns.
    ("lmofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    ("rmofc",  "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    ("lmofc",  "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("rmofc",  "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("lofc",   "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    ("rofc",   "mOFC", "brainnetome", ["a11m", "a13", "a14m"]),
    # HO cortical fallback (same rationale as mlofc/mrofc above).
    ("lofc",   "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("rofc",   "mOFC", "ho_cort", ["frontal orbital", "subcallosal"]),
    ("ldacc",  "mPFC", "brainnetome", ["a24", "a32"]),
    ("rdacc",  "mPFC", "brainnetome", ["a24", "a32"]),
    ("lpgacc", "mPFC", "brainnetome", ["a24", "a32", "a10m", "a9m"]),
    ("rpgacc", "mPFC", "brainnetome", ["a24", "a32", "a10m", "a9m"]),
    ("pgacc",  "mPFC", "brainnetome", ["a24", "a32", "a10m", "a9m"]),
    ("lacc",   "mPFC", "brainnetome", ["a24", "a32"]),
    ("racc",   "mPFC", "brainnetome", ["a24", "a32"]),
    ("lins",   "Insula", "ho_cort", ["insular"]),
    ("rins",   "Insula", "ho_cort", ["insular"]),
    ("mlins",  "Insula", "ho_cort", ["insular"]),
    ("mrins",  "Insula", "ho_cort", ["insular"]),
    ("lpcc",   "PCC",  "brainnetome", ["a23", "a31", "dmpos"]),
    ("rpcc",   "PCC",  "brainnetome", ["a23", "a31", "dmpos"]),
    ("lvcc",   "medial_CC", "brainnetome", ["a24", "a32"]),  # ventral CC
    ("rvcc",   "medial_CC", "brainnetome", ["a24", "a32"]),
    ("lec",    "EC",   "juelich", ["entorhinal"]),
    ("rec",    "EC",   "juelich", ["entorhinal"]),
    # LEC/REC labels can also land in HC when the electrode overshoots
    # posteriorly (e.g. y = -25 is HC body territory, not EC). Try EC
    # first (rule above); if entorhinal isn't within 8 mm, this fallback
    # rescues to HC (split by y downstream).
    ("lec",    "HC",   "ho_sub", ["hippocampus"]),
    ("rec",    "HC",   "ho_sub", ["hippocampus"]),
    ("phg",    "PHC", "ho_cort", ["parahippocampal"]),
    ("lhc",    "HC",   "ho_sub", ["hippocampus"]),
    ("rhc",    "HC",   "ho_sub", ["hippocampus"]),
]


def _neighborhood_search(atlas, xyz, patterns, max_dist=RESCUE_MAX_DIST_MM):
    """Search a small integer-mm cube grid around xyz for an atlas
    label containing any of the given patterns. Return
    (matched_label, distance_mm) or (None, None)."""
    xyz = np.asarray(xyz, dtype=float)
    for r in range(0, int(max_dist) + 1):
        if r == 0:
            offsets = [(0, 0, 0)]
        else:
            offsets = [(dx, dy, dz)
                       for dx in (-r, 0, r)
                       for dy in (-r, 0, r)
                       for dz in (-r, 0, r)
                       if not (dx == 0 and dy == 0 and dz == 0)]
        for off in offsets:
            probe = xyz + np.array(off, dtype=float)
            lab = atlas.label_at(probe)
            if lab and any(p.lower() in lab.lower() for p in patterns):
                return lab, float(np.linalg.norm(off))
    return None, None


print("\n\n=========================================================")
print(" STEP 4 — [DISABLED] no text-based intent rescue")
print("=========================================================\n")
# Removed on 2026-07-29 (was: neighborhood-search rescue keyed on the
# text of the electrode's `source_region_hint` / big-table `region
# label`, e.g. matching `lec`/`rec`/`mlofc`/`phg parahippocampal`).
# The rescue was pulling cells whose researcher-annotated intent
# conflicted with the atlas verdict at the cell's actual MNI
# coordinate. All ROI labels are now derived exclusively from the
# atlas-at-coord priority rules in `assign_atlas_roi` (rules 1-10),
# with the Juelich hippocampal-subfield extension in rule 4.
df["rescue_dist_mm"] = np.nan
df["rescue_source_atlas_label"] = ""
n_rescued = 0
rescue_rows = []


# Amygdala-vs-EC sanity: EC boundary is diffuse and amygdala electrodes
# sometimes bracket the entorhinal cortex. For every cell currently
# labeled Amygdala, check if Juelich entorhinal is within 3 mm; if yes,
# reassign to EC (which is our priority ROI).
n_amyg_to_ec = 0
for idx in df.index[df["atlas_roi"] == "Amygdala"]:
    xyz = np.array([df.at[idx, "MNI_x_final"],
                    df.at[idx, "MNI_y_final"],
                    df.at[idx, "MNI_z_final"]], dtype=float)
    if np.any(np.isnan(xyz)):
        continue
    lab, dist = _neighborhood_search(juelich, xyz, ["entorhinal"],
                                       max_dist=3.0)
    if lab:
        df.at[idx, "atlas_roi"] = "EC"
        df.at[idx, "atlas_source_label"] = lab
        df.at[idx, "atlas_reason"] = f"amyg_to_ec_neighborhood@{dist:.1f}mm"
        df.at[idx, "rescue_dist_mm"] = dist
        n_amyg_to_ec += 1
if n_amyg_to_ec:
    print(f"Reassigned {n_amyg_to_ec} Amygdala cells to EC "
          f"(entorhinal within 3 mm).")


# Note: cells previously handled as one-off "functional QC" per-coord
# overrides are now covered systemically by the Juelich hippocampal-
# subfield rule inside assign_atlas_roi (rule 4). No per-cell overrides
# remain — every assignment is derived from the atlases + priority
# order + intent rescue, so the pipeline is fully anatomical and
# reproducible from coordinate alone.
if rescue_rows:
    rescue_df = pd.DataFrame(rescue_rows)
    print("\nRescue breakdown (target ROI x intent):")
    print(rescue_df.groupby(["target_roi", "intent"])
                    .size().reset_index(name="n").to_string(index=False))
    print("\nDistance distribution (mm) at rescue:")
    print(rescue_df["rescue_dist_mm"].describe().round(2).to_string())

print("\nFINAL ROI counts:")
print(df["atlas_roi"].value_counts(dropna=False).to_string())


# =============================================================================
# STEP 5 — FINAL VISUALISATIONS
# =============================================================================
# (a) Master glass brain: every cell as a marker, colored by final ROI.
# (b) Per-ROI: atlas mask contour on glass brain + markers for the
#     cells assigned to that ROI. This is the "ROIs plotted under
#     electrode locations" figure you asked for.

from nilearn.plotting import plot_glass_brain
from matplotlib.lines import Line2D

step5_master_out = os.path.join(roi_assignment_dir, "cells_step5_all_cells_by_roi.png")
step5_per_roi_dir = os.path.join(roi_assignment_dir, "cells_step5_per_roi_atlas_overlay")
os.makedirs(step5_per_roi_dir, exist_ok=True)

# Colors — use CLAUDE.md palette where defined, else fall back
try:
    sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
    from mc.plotting.cell_results import SHOWGIRL2_DISCRETE
except Exception:
    SHOWGIRL2_DISCRETE = ["#7B0000", "#F15A29", "#8B5A2B", "#F7E7B0",
                          "#B5D3B0", "#5B9B8D", "#0e3d3a"]

# Canonical ROI colour palette — matches CLAUDE.md `roi_colour_dict`.
# ROIs not listed there use neutral filler colours.
ROI_COLORS = {
    "EC":          SHOWGIRL2_DISCRETE[0],
    "mPFC":        SHOWGIRL2_DISCRETE[1],
    "HC_anterior": SHOWGIRL2_DISCRETE[2],
    "PCC":         SHOWGIRL2_DISCRETE[3],
    "mOFC":        SHOWGIRL2_DISCRETE[4],
    "HC_mid":      "#a30d6c",   # magenta (CLAUDE.md override)
    "PHC":         "#23677E",   # teal    (CLAUDE.md override)
    # Non-target ROIs
    "Visual":      "#bdbdbd",
    "Thalamus":    "#7d3c98",
    "Amygdala":    "#e67e22",
    "medial_CC":   "#4a4a4a",
    "Insula":      "#1abc9c",
    "leftover":    "#888888",
}

# ---------- (a) master plot ----------
fig = plt.figure(figsize=(14, 5))
gb = plot_glass_brain(None, display_mode="lyrz", figure=fig,
                       title="All cells color-coded by final ROI "
                             f"(n = {len(df)})")
for roi in df["atlas_roi"].dropna().unique():
    sub = df[df["atlas_roi"] == roi]
    coords = sub[["MNI_x_final", "MNI_y_final", "MNI_z_final"]].to_numpy(float)
    if not len(coords):
        continue
    color = ROI_COLORS.get(roi, "#000000")
    gb.add_markers(coords, marker_color=color, marker_size=20)
# Legend as separate patch below
legend_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=ROI_COLORS.get(roi, "#000"),
                          markersize=8,
                          label=f"{roi} (n={int((df['atlas_roi']==roi).sum())})")
                   for roi in sorted(df["atlas_roi"].dropna().unique())]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)
fig.savefig(step5_master_out, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"\nSaved master plot: {step5_master_out}")


# ---------- (b) per-ROI: atlas mask + cells ----------
# For each ROI we care about, build a binary mask image from the
# corresponding atlas at 25 % threshold, then plot as contour +
# markers.
def _mask_img(atlas_obj, label_patterns):
    """Return a nibabel image whose voxels are 1 where the atlas label
    contains any of the substrings (case-insensitive), else 0."""
    data = atlas_obj.data
    labels = atlas_obj.labels
    match_idx = []
    iterator = (labels.items() if isinstance(labels, dict)
                else list(enumerate(labels)))
    for idx, lab in iterator:
        if isinstance(lab, str) and any(p.lower() in lab.lower()
                                          for p in label_patterns):
            match_idx.append(idx)
    mask = np.isin(data, match_idx).astype(np.uint8)
    return nib.Nifti1Image(mask, atlas_obj.img.affine)


ROI_ATLAS_SPECS = {
    "EC":              (juelich,     ["entorhinal"]),
    "mPFC":            (brainnetome, ["a32", "a24", "a10m", "a9m"]),
    "mOFC":           (brainnetome, ["a11m", "a13", "a14m"]),
    "HC_anterior":     (ho_sub,      ["hippocampus"]),  # split by y
    "HC_mid":          (ho_sub,      ["hippocampus"]),
    "PHC": (ho_cort,     ["parahippocampal"]),
    "PCC":   (brainnetome, ["a23", "a31", "dmpos"]),
    "Visual":          (ho_cort,     ["occipital", "cuneal", "lingual",
                                       "intracalcarine", "supracalcarine"]),
    "Thalamus":        (ho_sub,      ["thalamus"]),
    "Amygdala":        (ho_sub,      ["amygdala"]),
    "medial_CC":       (brainnetome, ["a32", "a24"]),
    "Insula":          (ho_cort,     ["insular"]),
}

for roi, (atlas_obj, patterns) in ROI_ATLAS_SPECS.items():
    sub = df[df["atlas_roi"] == roi]
    if not len(sub):
        continue
    coords = sub[["MNI_x_final", "MNI_y_final", "MNI_z_final"]].to_numpy(float)
    mask_img = _mask_img(atlas_obj, patterns)
    color = ROI_COLORS.get(roi, "#000000")
    fig = plt.figure(figsize=(12, 4))
    gb = plot_glass_brain(None, display_mode="lyrz", figure=fig,
                           title=f"{roi}: atlas mask (blue) + {len(coords)} "
                                 f"cells (colored, incl. rescued)")
    gb.add_contours(mask_img, levels=[0.5], colors="#4a90d9",
                     linewidths=1.0)
    # Split rescued vs originally-assigned so rescued shows in a
    # slightly different marker
    rescued_mask = sub["rescue_dist_mm"].notna()
    if rescued_mask.any():
        gb.add_markers(coords[rescued_mask.to_numpy()],
                        marker_color=color, marker_size=34)
    orig_mask = ~rescued_mask
    if orig_mask.any():
        gb.add_markers(coords[orig_mask.to_numpy()],
                        marker_color=color, marker_size=20)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", roi)
    fpath = os.path.join(step5_per_roi_dir, f"{safe}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close("all")

print(f"Saved per-ROI atlas-overlay plots to: {step5_per_roi_dir}")

# Save the final table (includes rescue columns)
step5_final_out = os.path.join(roi_assignment_dir, "cells_step5_final_labels.csv")
df.to_csv(step5_final_out, index=False)
print(f"Saved final table: {step5_final_out}")


# =============================================================================
# STEP 6 — LEFTOVER DEEP DIVE (ROI contours + labels around each cell)
# =============================================================================
# For the cells that STILL couldn't be assigned, show them on a brain
# with ALL target-ROI atlas masks overlaid as color-coded contours.
# This tells you at a glance whether a leftover is 1 mm outside mOFC,
# 20 mm from anything, or genuinely in a region we're not tracking.
#
# Two plot types:
#   (a) master leftover brain: all leftovers as markers, every ROI mask
#       as a color-coded contour, with a legend.
#   (b) per-cluster (grouped by permissive `primary_hint` from step 3):
#       zoom into that cluster, overlay only the ROI contours that
#       overlap the cluster's bounding box (± 20 mm) so it's readable.

from nilearn.plotting import plot_glass_brain, plot_stat_map
from matplotlib.lines import Line2D

step6_dir = os.path.join(roi_assignment_dir, "cells_step6_leftover_deepdive")
os.makedirs(step6_dir, exist_ok=True)

# Recompute leftovers after steps 4+5 (rescue may have moved some)
leftovers_final = df[df["atlas_roi"] == "leftover"].copy()
print(f"\n\n=========================================================")
print(f" STEP 6 — deep dive on {len(leftovers_final)} remaining leftovers")
print(f"=========================================================\n")

# Rebuild permissive primary_hint for these too (in case we're re-running
# without step 3 having been called first).
if "primary_hint" not in leftovers_final.columns:
    ho_cort_0 = MaxProbAtlas(
        nldatasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm"))
    ho_sub_0 = MaxProbAtlas(
        nldatasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-2mm"))
    juelich_0 = MaxProbAtlas(
        nldatasets.fetch_atlas_juelich("maxprob-thr0-2mm"))
    hints = []
    for _, r in leftovers_final.iterrows():
        xyz = np.array([r["MNI_x_final"], r["MNI_y_final"],
                        r["MNI_z_final"]], dtype=float)
        for atlas_obj in (ho_sub_0, juelich_0, brainnetome, ho_cort_0):
            lab = atlas_obj.label_at(xyz) if not np.any(np.isnan(xyz)) else None
            if lab and lab.lower() != "background":
                hints.append(lab); break
        else:
            hints.append("no atlas label")
    leftovers_final["primary_hint"] = hints


# Precompute one mask image per ROI spec, and pick a distinct color
# for each. Use nilearn contour + a Line2D legend.
roi_masks = {}
for roi, (atlas_obj, patterns) in ROI_ATLAS_SPECS.items():
    roi_masks[roi] = _mask_img(atlas_obj, patterns)

# ---------- (a) master leftover plot ----------
fig = plt.figure(figsize=(15, 5.5))
gb = plot_glass_brain(None, display_mode="lyrz", figure=fig,
                       title=f"Leftover cells (n = {len(leftovers_final)}) "
                             "with all target-ROI atlas contours (25 %)")
for roi, mask_img in roi_masks.items():
    color = ROI_COLORS.get(roi, "#000000")
    try:
        gb.add_contours(mask_img, levels=[0.5], colors=color, linewidths=0.7)
    except Exception:
        pass
coords = leftovers_final[["MNI_x_final", "MNI_y_final",
                          "MNI_z_final"]].to_numpy(float)
gb.add_markers(coords, marker_color="#0e3d3a", marker_size=22)

roi_handles = [Line2D([0], [0], color=ROI_COLORS.get(r, "#000"),
                       linewidth=2.5, label=r)
                for r in ROI_ATLAS_SPECS.keys()]
roi_handles.append(Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#0e3d3a", markersize=9,
                           label="leftover cell"))
fig.legend(handles=roi_handles, loc="lower center", ncol=7,
           bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
master_out = os.path.join(step6_dir, "leftover_MASTER_all_ROI_contours.png")
fig.savefig(master_out, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"Saved master leftover plot: {master_out}")


# ---------- (b) per-cluster axial-slice deep dive ----------
# Use nilearn plot_stat_map (background + colored ROI overlay + marker)
# at the mean coord of each cluster with n >= 3.
cluster_counts = leftovers_final["primary_hint"].value_counts()
for hint, n in cluster_counts.items():
    if n < 3:
        continue
    grp = leftovers_final[leftovers_final["primary_hint"] == hint]
    coords = grp[["MNI_x_final", "MNI_y_final",
                  "MNI_z_final"]].to_numpy(float)
    center = coords.mean(axis=0)
    # find the ROI whose mask is closest to the cluster centroid, for
    # a hint about where they *could* land — we contour ALL ROIs but
    # cut display to slices around the cluster center for readability.
    fig = plt.figure(figsize=(15, 4.5))
    hint_hints = "; ".join(sorted(set(grp["source_region_hint"].dropna()))[:5])
    title = (f"Leftover cluster: {hint}  |  n={n} cells\n"
             f"electrode-intent hints: {hint_hints}")
    gb = plot_glass_brain(None, display_mode="lyrz", figure=fig, title=title)
    for roi, mask_img in roi_masks.items():
        color = ROI_COLORS.get(roi, "#000000")
        try:
            gb.add_contours(mask_img, levels=[0.5], colors=color,
                             linewidths=0.9)
        except Exception:
            pass
    gb.add_markers(coords, marker_color="#0e3d3a", marker_size=28)
    fig.legend(handles=roi_handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", hint)[:60]
    fpath = os.path.join(step6_dir, f"leftover_cluster__{safe}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close("all")

print(f"Saved per-cluster leftover plots to: {step6_dir}")

# Save the final leftover table with primary_hint attached for later
step6_leftover_out = os.path.join(roi_assignment_dir,
                                   "cells_step6_final_leftovers.csv")
leftovers_final[
    [c for c in ["subject", "cell idx", "electrode label",
                 "region label", "Subject Label", "Recording Site",
                 "MNI_x_final", "MNI_y_final", "MNI_z_final",
                 "source_region_hint", "primary_hint"]
     if c in leftovers_final.columns]
].to_csv(step6_leftover_out, index=False)
print(f"Saved: {step6_leftover_out}")


# =============================================================================
# STEP 7 — FINAL SUMMARY PLOTS + OLD-vs-NEW COMPARISON
# =============================================================================
# (a) Master plot restricted to ROIs with n_subjects >= 3 (analysis-ready).
# (b) Per-ROI old-vs-new plot: overlay the atlas contour and show
#     - green markers: cells that agree between old and new labeling
#     - red markers  : new-label cells whose OLD label differed (plotted
#                       at their NEW coord)
#     - blue markers : same cells plotted at their OLD coord
#     - grey arrows  : linking old -> new position
#     - text list    : old labels of disagreeing cells (in title/legend)
#
# "Old" here = the atlas assignment from the previous pipeline output
# at `neurons_with_final_roi_labels.csv` (produced by cell_to_roi_MNI.py),
# using its `alt_final_roi` column and its `MNI_x/y/z`.

from nilearn.plotting import plot_glass_brain
from matplotlib.lines import Line2D

MIN_SUBJECTS_FOR_ANALYSIS = 3

step7_master_analysis_out = os.path.join(
    roi_assignment_dir, "cells_step7_master_analysis_ROIs_only.png")
step7_master_all_out = os.path.join(
    roi_assignment_dir, "cells_step7_master_all_cells.png")
step7_old_vs_new_dir = os.path.join(
    roi_assignment_dir, "cells_step7_old_vs_new_comparison")
os.makedirs(step7_old_vs_new_dir, exist_ok=True)
step7_diff_out = os.path.join(
    roi_assignment_dir, "cells_step7_old_vs_new_differences.csv")


# ---------- Master plots ----------
def _master_plot(df_sub, title, save_path):
    fig = plt.figure(figsize=(14, 5))
    gb = plot_glass_brain(None, display_mode="lyrz", figure=fig, title=title)
    for roi in df_sub["atlas_roi"].dropna().unique():
        s = df_sub[df_sub["atlas_roi"] == roi]
        coords = s[["MNI_x_final", "MNI_y_final",
                    "MNI_z_final"]].to_numpy(float)
        if len(coords):
            gb.add_markers(coords, marker_color=ROI_COLORS.get(roi, "#000"),
                           marker_size=40)
    legend = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=ROI_COLORS.get(r, "#000"),
                      markersize=8,
                      label=f"{r} (n={int((df_sub['atlas_roi']==r).sum())})")
               for r in sorted(df_sub["atlas_roi"].dropna().unique())]
    fig.legend(handles=legend, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close("all")

_master_plot(df, f"All cells by final ROI (n = {len(df)})",
             step7_master_all_out)
print(f"Saved: {step7_master_all_out}")

subj_counts = df.groupby("atlas_roi")["subject"].nunique()
analysis_rois = subj_counts[subj_counts >= MIN_SUBJECTS_FOR_ANALYSIS].index.tolist()
df_analysis = df[df["atlas_roi"].isin(analysis_rois)]

# `alt_final_roi` — the column name downstream analysis scripts look for
# (`mc.analyse.roi_relabel.relabel_per_cell(..., roi_col_in_table=
# "alt_final_roi")`). Set to the atlas ROI name when that ROI has
# >= MIN_SUBJECTS_FOR_ANALYSIS distinct subjects; NaN otherwise, so
# under-covered ROIs (Visual/Thalamus/Amygdala/Insula/medial_CC/PHC/
# leftover) are automatically excluded from analyses filtering on it.
# Defined here (not at the bottom) because the RSA-ready plot below
# already needs it.
_analysis_rois = set(analysis_rois)
_analysis_rois.discard("leftover")   # never treat leftover as a real ROI
df["alt_final_roi"] = df["atlas_roi"].where(
    df["atlas_roi"].isin(_analysis_rois), other=np.nan)
_master_plot(df_analysis,
             f"Analysis-ready cells (ROI with >= {MIN_SUBJECTS_FOR_ANALYSIS} "
             f"subjects; n = {len(df_analysis)})",
             step7_master_analysis_out)
step7_master_analysis_pdf = step7_master_analysis_out.replace(".png", ".pdf")
_master_plot(df_analysis,
             f"Analysis-ready cells (ROI with >= {MIN_SUBJECTS_FOR_ANALYSIS} "
             f"subjects; n = {len(df_analysis)})",
             step7_master_analysis_pdf)
print(f"Saved: {step7_master_analysis_out}")
print(f"Saved: {step7_master_analysis_pdf}")
print(f"Analysis ROIs (n_subj >= {MIN_SUBJECTS_FOR_ANALYSIS}): {analysis_rois}")


# ---------- Overview bar chart: n_cells and n_sessions per ROI ----------
# import pdb; pdb.set_trace() 
per_roi_summary = (df.groupby("atlas_roi")
                     .agg(n_cells=("subject", "size"),
                          n_subjects=("subject", "nunique"))
                     .sort_values("n_cells", ascending=False))

roi_order = per_roi_summary.index.tolist()
bar_colors = [ROI_COLORS.get(r, "#888888") for r in roi_order]

fig_ov, axes_ov = plt.subplots(1, 2, figsize=(14, 5))
fig_ov.suptitle("ROI overview — cells and sessions per region",
                fontsize=12, fontweight="bold", y=1.01)

x = np.arange(len(roi_order))

# n_cells
axes_ov[0].bar(x, per_roi_summary["n_cells"], color=bar_colors, edgecolor="white", linewidth=0.6)
axes_ov[0].set_xticks(x)
axes_ov[0].set_xticklabels(roi_order, rotation=45, ha="right", fontsize=9)
axes_ov[0].set_ylabel("Number of cells", fontsize=10)
axes_ov[0].set_title("Cells per ROI", fontsize=11)
for xi, v in enumerate(per_roi_summary["n_cells"]):
    axes_ov[0].text(xi, v + 2, str(v), ha="center", va="bottom", fontsize=8)
axes_ov[0].axhline(0, color="black", linewidth=0.5)
axes_ov[0].spines[["top", "right"]].set_visible(False)

# n_subjects (sessions)
axes_ov[1].bar(x, per_roi_summary["n_subjects"], color=bar_colors, edgecolor="white", linewidth=0.6)
axes_ov[1].set_xticks(x)
axes_ov[1].set_xticklabels(roi_order, rotation=45, ha="right", fontsize=9)
axes_ov[1].set_ylabel("Number of sessions (subjects)", fontsize=10)
axes_ov[1].set_title("Sessions per ROI", fontsize=11)
axes_ov[1].axhline(MIN_SUBJECTS_FOR_ANALYSIS, color="#e74c3c",
                    linestyle="--", linewidth=1.2,
                    label=f"analysis threshold (n={MIN_SUBJECTS_FOR_ANALYSIS})")
for xi, v in enumerate(per_roi_summary["n_subjects"]):
    axes_ov[1].text(xi, v + 0.4, str(v), ha="center", va="bottom", fontsize=12)
axes_ov[1].legend(fontsize=8, frameon=False)
axes_ov[1].spines[["top", "right"]].set_visible(False)

fig_ov.tight_layout()
step7_overview_out = os.path.join(roi_assignment_dir, "cells_step7_overview_per_roi.png")
fig_ov.savefig(step7_overview_out, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"Saved: {step7_overview_out}")


# ---------- RSA-ready brain plot (same-task subjects + RSA ROIs only) ----------
# Shows exactly the cells that RSA_DSR_ROIs_simple.py will use:
#   - subject in all_sessions_dsrRSA_grouping_summary.json (completed same task)
#   - alt_final_roi ∈ {EC, mPFC, mOFC, HC_anterior, HC_mid, PCC}
import json as _json
_grouping_json = os.path.join(output_dir, "all_sessions_dsrRSA_grouping_summary.json")
if os.path.exists(_grouping_json):
    with open(_grouping_json) as _f:
        _rsa_subjects = set(int(s) for s in _json.load(_f).keys())
else:
    _rsa_subjects = set(df["subject"].unique())   # fallback: no task filter
    print(f"[warning] {_grouping_json} not found — RSA plot uses all subjects")

RSA_ROIS = ["EC", "mPFC", "mOFC", "HC_anterior", "HC_mid", "PCC"]
df_rsa = df[df["alt_final_roi"].isin(RSA_ROIS) & df["subject"].isin(_rsa_subjects)].copy()

# Count per ROI for legend labels
rsa_roi_counts = df_rsa.groupby("alt_final_roi")["subject"].agg(
    n_cells="size", n_subjects="nunique")

fig_rsa = plt.figure(figsize=(14, 5))
gb_rsa = plot_glass_brain(
    None, display_mode="lyrz", figure=fig_rsa,
    title=f"RSA-ready cells — same-task subjects only "
          f"(n = {len(df_rsa)} cells / {df_rsa['subject'].nunique()} sessions)")
for roi in RSA_ROIS:
    s = df_rsa[df_rsa["alt_final_roi"] == roi]
    if not len(s):
        continue
    coords = s[["MNI_x_final", "MNI_y_final", "MNI_z_final"]].to_numpy(float)
    gb_rsa.add_markers(coords, marker_color=ROI_COLORS.get(roi, "#000"),
                       marker_size=20)

legend_rsa = []
for roi in RSA_ROIS:
    if roi not in rsa_roi_counts.index:
        continue
    nc = int(rsa_roi_counts.loc[roi, "n_cells"])
    ns = int(rsa_roi_counts.loc[roi, "n_subjects"])
    legend_rsa.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=ROI_COLORS.get(roi, "#000"),
                              markersize=9,
                              label=f"{roi}  n={nc} cells / {ns} sess"))
fig_rsa.legend(handles=legend_rsa, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

step7_rsa_out_png = os.path.join(roi_assignment_dir,
                                  "cells_step7_RSA_ready_cells.png")
step7_rsa_out_pdf = os.path.join(roi_assignment_dir,
                                  "cells_step7_RSA_ready_cells.pdf")
fig_rsa.savefig(step7_rsa_out_png, dpi=200, bbox_inches="tight")
fig_rsa.savefig(step7_rsa_out_pdf, bbox_inches="tight")
plt.close("all")
print(f"Saved: {step7_rsa_out_png}")
print(f"Saved: {step7_rsa_out_pdf}")


# ---------- Old-vs-new comparison ----------
old_path = os.path.join(output_dir, "neurons_with_final_roi_labels.csv")
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    # Canonical rename map: naming-only or intentional-collapse changes
    # don't count as real disagreement.
    OLD_TO_NEW_CANON = {
        "medialOFC": "mOFC",              # naming
        "ACC":       "mPFC",              # naming (old script's ACC covered
                                          # dorsomedial A32/A24; A32sg re-
                                          # classifications to mOFC still
                                          # show as ACC->mOFC below).
        "PCC":       "PCC",         # unchanged (kept as name)
        "Precuneus": "PCC",         # collapsed into PCC
        "Parahippocampal": "PHC",   # renamed
    }
    def _canon(x):
        s = str(x)
        return OLD_TO_NEW_CANON.get(s, s)

    def _is_unassigned(x):
        s = str(x).strip().lower()
        return s in ("", "nan", "leftover", "none")

    key_cols = ["subject", "cell idx", "electrode label"]
    old_key = (old[key_cols].astype(str).agg("|".join, axis=1))
    new_key = (df[key_cols].astype(str).agg("|".join, axis=1))

    old_lookup = {}
    for i, k in enumerate(old_key):
        old_lookup[k] = (
            str(old.iloc[i].get("alt_final_roi", "")),
            float(old.iloc[i].get("MNI_x", np.nan)),
            float(old.iloc[i].get("MNI_y", np.nan)),
            float(old.iloc[i].get("MNI_z", np.nan)),
        )

    diffs = []
    for i, k in enumerate(new_key):
        if k not in old_lookup:
            continue
        old_roi, oX, oY, oZ = old_lookup[k]
        new_roi = df.iloc[i]["atlas_roi"]
        nX = df.iloc[i]["MNI_x_final"]; nY = df.iloc[i]["MNI_y_final"]; nZ = df.iloc[i]["MNI_z_final"]
        # Skip newly-rescued cells (old was unassigned) — they're not
        # 'disagreements', just fills. Also skip if new is unassigned.
        if _is_unassigned(old_roi) or _is_unassigned(new_roi):
            continue
        if _canon(old_roi) != str(new_roi):
            diffs.append({
                "subject": df.iloc[i]["subject"],
                "cell idx": df.iloc[i]["cell idx"],
                "electrode label": df.iloc[i]["electrode label"],
                "old_roi": old_roi, "new_roi": new_roi,
                "old_x": oX, "old_y": oY, "old_z": oZ,
                "new_x": nX, "new_y": nY, "new_z": nZ,
                "coord_shift_mm": float(np.linalg.norm(
                    np.array([nX-oX, nY-oY, nZ-oZ]))),
            })
    diffs_df = pd.DataFrame(diffs)
    diffs_df.to_csv(step7_diff_out, index=False)
    print(f"\n{len(diffs_df)} cells changed ROI label between old and new.")
    print(f"Saved detailed diff table: {step7_diff_out}")

    if len(diffs_df):
        top = (diffs_df.groupby(["old_roi", "new_roi"])
                        .size().reset_index(name="n")
                        .sort_values("n", ascending=False))
        print("\nTop transitions (old -> new):")
        print(top.head(20).to_string(index=False))

    # Per-new-ROI plot with disagreements highlighted
    for new_roi in sorted(df["atlas_roi"].dropna().unique()):
        cells_new = df[df["atlas_roi"] == new_roi]
        keys = (cells_new[key_cols].astype(str).agg("|".join, axis=1))
        agree, disagree_new_coords, disagree_old_coords, disagree_labels = (
            [], [], [], [])
        for _, r in cells_new.iterrows():
            k = "|".join(str(r[c]) for c in key_cols)
            if k not in old_lookup:
                continue
            old_roi, oX, oY, oZ = old_lookup[k]
            nc = [r["MNI_x_final"], r["MNI_y_final"], r["MNI_z_final"]]
            if _is_unassigned(old_roi):
                # newly-rescued — count as agree (was previously
                # leftover / unassigned; new label is our best call).
                agree.append(nc)
            elif _canon(old_roi) == str(new_roi):
                agree.append(nc)
            else:
                disagree_new_coords.append(nc)
                if not np.isnan(oX):
                    disagree_old_coords.append([oX, oY, oZ])
                disagree_labels.append(str(old_roi))
        # Build the plot
        atlas_obj_pat = ROI_ATLAS_SPECS.get(new_roi)
        fig = plt.figure(figsize=(13, 4.5))
        n_ag = len(agree); n_dis = len(disagree_new_coords)
        title = (f"{new_roi} — old vs new comparison  |  "
                 f"agree n={n_ag}, disagree n={n_dis}")
        gb = plot_glass_brain(None, display_mode="lyrz", figure=fig,
                               title=title)
        if atlas_obj_pat is not None:
            try:
                mask_img = _mask_img(atlas_obj_pat[0], atlas_obj_pat[1])
                gb.add_contours(mask_img, levels=[0.5], colors="#4a90d9",
                                 linewidths=1.0)
            except Exception:
                pass
        if agree:
            gb.add_markers(np.array(agree), marker_color="#5b9b8d",
                           marker_size=16)
        if disagree_new_coords:
            gb.add_markers(np.array(disagree_new_coords),
                           marker_color="#F15A29", marker_size=28)
        if disagree_old_coords:
            gb.add_markers(np.array(disagree_old_coords),
                           marker_color="#6B60AA", marker_size=14)
        legend = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#5b9b8d", markersize=8,
                          label=f"agree (n={n_ag})"),
                   Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#F15A29", markersize=8,
                          label=f"disagree, NEW coord (n={n_dis})"),
                   Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#6B60AA", markersize=8,
                          label=f"same cells, OLD coord (n={len(disagree_old_coords)})"),
                   Line2D([0], [0], color="#4a90d9", linewidth=2,
                          label="atlas mask (25%)")]
        fig.legend(handles=legend, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
        # Annotate old labels of disagreers below the figure
        from collections import Counter
        if disagree_labels:
            c = Counter(disagree_labels).most_common()
            labs_txt = ", ".join(f"{lab}({n})" for lab, n in c)
            fig.text(0.5, -0.12, f"old labels of disagreers: {labs_txt}",
                     ha="center", fontsize=7.5, style="italic")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", new_roi)
        fpath = os.path.join(step7_old_vs_new_dir, f"{safe}.png")
        fig.savefig(fpath, dpi=180, bbox_inches="tight")
        plt.close("all")
    print(f"Saved per-ROI old-vs-new plots: {step7_old_vs_new_dir}")
else:
    print(f"[old table not found at {old_path}, skipping old-vs-new step]")


# =============================================================================
# FINAL OUTPUT — canonical table for downstream analyses
# =============================================================================
# Save the authoritative per-cell table directly to derivatives/. All
# intermediate step outputs (audit trail, plots, diagnostics) stay in
# derivatives/ROI_assignment/ so downstream code doesn't have to sift
# through them.

# `alt_final_roi` was already assigned in step 7 (the RSA-ready plot
# needs it); see the comment there for the >= MIN_SUBJECTS_FOR_ANALYSIS
# rule.

FINAL_KEEP_COLS = [
    # identifiers
    "subject", "cell idx", "electrode label", "Subject Label",
    "Recording Site",
    # final coord (source of truth) + provenance
    "MNI_x_final", "MNI_y_final", "MNI_z_final",
    "coord_source", "coord_verified",
    "source_electrode", "source_region_hint",
    # final ROI + reason
    "atlas_roi",             # every cell's raw atlas label (13 values)
    "alt_final_roi",         # analysis-ready ROI (>= 3 subjects) else NaN
    "atlas_source_label", "atlas_reason",
    "rescue_dist_mm",
    # original big-table coords (for reference)
    "MNI_x", "MNI_y", "MNI_z",
    "region label",
]
FINAL_KEEP_COLS = [c for c in FINAL_KEEP_COLS if c in df.columns]
df[FINAL_KEEP_COLS].to_csv(FINAL_TABLE_OUT, index=False)


# =============================================================================
# OVERVIEW REPORT (printed at the end)
# =============================================================================

print("\n\n=========================================================")
print(" OVERVIEW — input table vs final output table")
print("=========================================================\n")

orig = pd.read_csv(path_to_cell_table)
print(f"Input table:  {path_to_cell_table}")
print(f"              {len(orig)} rows, {orig['subject'].nunique()} subjects, "
      f"{len(orig['Recording Site'].unique())} recording sites")
print(f"              site breakdown: "
      f"{dict(orig['Recording Site'].value_counts())}")

print(f"\nOutput table: {FINAL_TABLE_OUT}")
print(f"              {len(df)} rows (matches input: "
      f"{'yes' if len(df) == len(orig) else 'NO'})")
print(f"              {df['atlas_roi'].nunique()} distinct atlas_roi values")

print(f"\nCoord provenance breakdown:")
print(df["coord_source"].value_counts().to_string())

print(f"\nFinal ROI counts (sorted):")
per_roi = (df.groupby("atlas_roi")
             .agg(n_cells=("subject", "size"),
                  n_subjects=("subject", "nunique"))
             .sort_values("n_cells", ascending=False))
print(per_roi.to_string())

# analysis vs excluded ROIs (n_subj threshold)
threshold = MIN_SUBJECTS_FOR_ANALYSIS
excluded = per_roi[per_roi["n_subjects"] < threshold]
included = per_roi[per_roi["n_subjects"] >= threshold]
print(f"\nAnalysis-ready ROIs (n_subjects >= {threshold}):  "
      f"{included.index.tolist()}")
print(f"  covering {int(included['n_cells'].sum())} cells")
print(f"Excluded ROIs (n_subjects < {threshold}):  "
      f"{excluded.index.tolist()}")
print(f"  covering {int(excluded['n_cells'].sum())} cells")

# Old-vs-new one-line summary if old table is present
old_path_ref = os.path.join(output_dir, "neurons_with_final_roi_labels.csv")
if os.path.exists(old_path_ref):
    old_ref = pd.read_csv(old_path_ref)
    if "alt_final_roi" in old_ref.columns:
        matched = min(len(old_ref), len(df))
        n_diff = 0
        old_map = {}
        for _, r in old_ref.iterrows():
            key = f"{r['subject']}|{r['cell idx']}|{r['electrode label']}"
            old_map[key] = str(r["alt_final_roi"])
        CANON = {"medialOFC": "mOFC", "ACC": "mPFC",
                 "PCC": "PCC", "Precuneus": "PCC",
                 "Parahippocampal": "PHC"}
        for _, r in df.iterrows():
            key = f"{r['subject']}|{r['cell idx']}|{r['electrode label']}"
            if key not in old_map:
                continue
            o = old_map[key]
            if o.strip().lower() in ("", "nan", "leftover", "none"):
                continue
            if CANON.get(o, o) != str(r["atlas_roi"]):
                n_diff += 1
        print(f"\nVs old atlas assignment "
              f"(neurons_with_final_roi_labels.csv, `alt_final_roi`):")
        print(f"  {n_diff} cells changed ROI label after canonical "
              f"renames (medialOFC->mOFC, ACC->mPFC, "
              f"PCC/Precuneus->PCC).")
        print(f"  Detailed diff: {step7_diff_out}")

print(f"\nAll audit / plots / intermediate tables: {roi_assignment_dir}")
print(f"Canonical table for analyses:            {FINAL_TABLE_OUT}")


# =============================================================================
# STEP 8 — WHAT CHANGED vs THE PREVIOUS RUN OF THIS SCRIPT
# =============================================================================
# Diff against `REFERENCE_TABLE` (an archived copy of this script's own
# previous output) on the SAME columns downstream analyses consume:
# `alt_final_roi` (what `mc.analyse.roi_relabel.relabel_per_cell` reads)
# and `MNI_*_final`. Two questions are answered:
#   (1) how many cells landed in a different final ROI, and which?
#   (2) did the coordinates just move a few mm (expected: better micro
#       localisation) or jump somewhere else entirely (a red flag)?

print("\n\n=========================================================")
print(" STEP 8 — change report vs previous run")
print("=========================================================\n")

step8_diff_out = os.path.join(roi_assignment_dir,
                              "cells_step8_change_vs_previous_run.csv")

if not os.path.exists(REFERENCE_TABLE):
    print(f"[reference table not found at {REFERENCE_TABLE} — skipping]")
else:
    print(f"Reference: {REFERENCE_TABLE}")
    ref = pd.read_csv(REFERENCE_TABLE)

    CELL_KEY = ["subject", "cell idx", "electrode label"]

    def _cellkey(frame):
        return frame[CELL_KEY].astype(str).agg("|".join, axis=1)

    ref = ref.assign(_key=_cellkey(ref)).set_index("_key")
    new = df.assign(_key=_cellkey(df)).set_index("_key")

    shared = new.index.intersection(ref.index)
    print(f"  {len(ref)} reference cells, {len(new)} new cells, "
          f"{len(shared)} matched by subject|cell idx|electrode label.")
    if len(shared) < len(new):
        print(f"  [{len(new) - len(shared)} new cells absent from reference]")

    r = ref.loc[shared]
    n = new.loc[shared]

    # ---- (1) coordinate shifts -------------------------------------
    shift = np.linalg.norm(
        n[["MNI_x_final", "MNI_y_final", "MNI_z_final"]].to_numpy(float)
        - r[["MNI_x_final", "MNI_y_final", "MNI_z_final"]].to_numpy(float),
        axis=1)

    comp = pd.DataFrame({
        "subject": n["subject"].to_numpy(),
        "Subject Label": n["Subject Label"].to_numpy(),
        "Recording Site": n["Recording Site"].to_numpy(),
        "electrode label": n["electrode label"].to_numpy(),
        "cell idx": n["cell idx"].to_numpy(),
        "old_coord_source": r["coord_source"].to_numpy(),
        "new_coord_source": n["coord_source"].to_numpy(),
        "old_x": r["MNI_x_final"].to_numpy(), "new_x": n["MNI_x_final"].to_numpy(),
        "old_y": r["MNI_y_final"].to_numpy(), "new_y": n["MNI_y_final"].to_numpy(),
        "old_z": r["MNI_z_final"].to_numpy(), "new_z": n["MNI_z_final"].to_numpy(),
        "coord_shift_mm": shift,
        "old_atlas_roi": r["atlas_roi"].astype(str).to_numpy(),
        "new_atlas_roi": n["atlas_roi"].astype(str).to_numpy(),
        "old_alt_final_roi": r["alt_final_roi"].astype(str).to_numpy(),
        "new_alt_final_roi": n["alt_final_roi"].astype(str).to_numpy(),
    }, index=shared)

    moved = comp[comp["coord_shift_mm"] > 0.01]
    print(f"\n--- Coordinates ---")
    print(f"  {len(moved)} / {len(comp)} cells moved at all.")
    if len(moved):
        print(f"  shift (mm): median {moved['coord_shift_mm'].median():.2f}, "
              f"mean {moved['coord_shift_mm'].mean():.2f}, "
              f"max {moved['coord_shift_mm'].max():.2f}")
        print("\n  Per subject (only subjects whose coords moved):")
        print(moved.groupby(["Subject Label", "old_coord_source",
                             "new_coord_source"])["coord_shift_mm"]
                   .agg(n="size", median="median", mean="mean", max="max")
                   .round(2).to_string())
        # Sanity flag: a corrected micro position should be a few mm, not
        # a different structure.
        BIG_SHIFT_MM = 10.0
        big = moved[moved["coord_shift_mm"] > BIG_SHIFT_MM]
        print(f"\n  Cells shifted > {BIG_SHIFT_MM} mm: {len(big)}"
              + ("  <-- INSPECT THESE" if len(big) else "  (none — all shifts"
                 " are small local corrections, as expected)"))
        if len(big):
            print(big[["Subject Label", "electrode label", "cell idx",
                       "old_x", "old_y", "old_z", "new_x", "new_y", "new_z",
                       "coord_shift_mm", "old_alt_final_roi",
                       "new_alt_final_roi"]]
                  .round(2).to_string(index=False))

    # ---- (2) ROI label changes -------------------------------------
    print(f"\n--- Final ROI (`alt_final_roi`, the column used by "
          f"mc.analyse.roi_relabel.relabel_per_cell) ---")
    counts = pd.DataFrame({
        "old": comp["old_alt_final_roi"].value_counts(),
        "new": comp["new_alt_final_roi"].value_counts(),
    }).fillna(0).astype(int)
    counts["delta"] = counts["new"] - counts["old"]
    print(counts.sort_values("new", ascending=False).to_string())

    changed = comp[comp["old_alt_final_roi"] != comp["new_alt_final_roi"]]
    print(f"\n  {len(changed)} / {len(comp)} cells changed `alt_final_roi`.")
    if len(changed):
        print("\n  Transitions (old -> new):")
        print(changed.groupby(["old_alt_final_roi", "new_alt_final_roi"])
                     .agg(n_cells=("cell idx", "size"),
                          n_subjects=("subject", "nunique"),
                          median_shift_mm=("coord_shift_mm", "median"))
                     .round(2)
                     .sort_values("n_cells", ascending=False)
                     .to_string())
        print("\n  Per subject:")
        print(changed.groupby("Subject Label")
                     .agg(n_changed=("cell idx", "size"),
                          median_shift_mm=("coord_shift_mm", "median"))
                     .round(2).to_string())

    comp.to_csv(step8_diff_out, index=False)
    print(f"\nSaved per-cell change table: {step8_diff_out}")
