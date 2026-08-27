#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomy for LFP MACRO contacts, across all three recording sites.

Produces one row per LFP channel with an MNI152 coordinate, grey/white, a
native-atlas region label and an ROI assigned by the SAME rule ladder the cell
pipeline uses (`mc.analyse.anatomy_atlas.assign_atlas_roi`). Then pairs contacts
into bipolar derivations.

This replaces contact selection by string matching. The old code used
`'H' in channel and 'T' in channel` (matches any label containing both letters)
and `'1' in label` (matches HIP10, HIP11, HIP14), plus a hard-coded
contact01-contact04 bipolar pair spanning ~15 mm.

Site specifics, all verified against the data rather than assumed
-----------------------------------------------------------------
Baylor
    `channels.npy` entries look like `RF1bIb01-001`. Strip the trailing
    `-\\d+` and the remainder matches the v2026 CSV `Label` exactly. The
    trailing number is a Blackrock electrode ID, NOT the array position
    (s05 runs to 256 for 156 channels), so data is always indexed by
    position in `channels.npy`. Macro rows are `Type == 'sEEG'`; they
    carry MNI152 directly.

Utah
    `Electrodes.mat` -> `ElecMapRaw`, an (n, 3) object array of
    `[label, amplifier_channel, other_channel]`.

    IMPORTANT: coordinates come from the **direct row index** into
    `ElecXYZMNIRaw` / `ElecAtlasRaw` / `ElecTypeRaw`, not from indexing by
    the channel number. Verified on s02: row 41 is `bRAHIP2` and
    `ElecAtlasRaw[41]` is "Right Hippocampus", whereas indexing by column 1
    gives "Right fusiform gyrus" -- a different electrode entirely, and it
    would not raise. Column 1 is the amplifier channel; it matches
    `utah_elec_labels_{XX}.csv` for all 114 rows where it is finite (the
    other 30 are unlocalised and are dropped with a logged reason).

    Macros are labels NOT starting with 'm'. Naming: `m*` = microwire,
    `b*` = the Behnke-Fried macro on the same shaft as those micros,
    bare = plain sEEG depth contact.

    We use `ElecXYZMNIRaw`, whereas cell_to_roi_july26.py uses
    `ElecXYZMNIProj` for micros. Gray-matter projection is right for a
    spiking micro tip and wrong for a 2 mm macro ring that legitimately
    straddles the GM/WM boundary. Both are stored so the choice is
    reversible.

UCLA
    `sub-NNN_localizations.xlsx` Sheet1, rows with `isMicro == False`.
    Carries `grayWhite`, `MNI_x/y/z`, `NMM`, `AnatMacro_1`, and
    `ASHS_ABC` hippocampal subfields -- the last is independent ground
    truth for validating the atlas ROI assignment. `electrode` is
    `LMH-1`; the .ncs file is `LMH1.ncs`, i.e. drop the hyphen.

@author: Svenja Kuchenhoff
"""

import os
import re

import numpy as np
import pandas as pd

import mc.analyse.anatomy_sources as anat_src
import mc.analyse.anatomy_atlas as anat_atlas


HPC_ROIS = ("HC_anterior", "HC_mid")


# =============================================================================
# NATIVE-SPACE ROI  (primary criterion for macro contacts)
# =============================================================================
#
# A contact's ROI is taken from the SUBJECT'S OWN segmentation, not from the
# group-space atlas. Each site supplies one: Baylor `Area_fs_vox` (FreeSurfer
# aseg), Utah `ElecAtlasRaw` NMM, UCLA `NMM` (+ `ASHS_ABC` subfields).
#
# Why native rather than the shared atlas ladder: measured on s05/s02/s03, the
# atlas rules have 100% recall but only 22% precision against native
# segmentation -- they label 49 contacts hippocampal where the subjects' own
# MRIs say 11, the excess being amygdala, lingual gyrus, VentralDC and
# parahippocampal cortex. That is expected: the ladder was tuned for microwire
# TIPS (where +-2-3 mm neighbourhood rescue is desirable) and is applied here to
# MNI152 coordinates, where MTL registration error is large. Chen et al.
# identified hippocampal contacts "via visual inspection of postoperative
# T1-weighted anatomical MRI scans" -- i.e. in native space.
#
# `atlas_roi` is still computed and stored as a cross-check. Because the atlas
# ladder is a strict superset, disagreement in the HC direction is impossible;
# `roi_concordant` therefore flags only the reverse case.

_NATIVE_PATTERNS = [
    ("PHC",      ("parahippocamp",)),          # must precede the HC test
    ("HC",       ("hippocamp",)),
    ("Amygdala", ("amygdala",)),
    ("EC",       ("entorhinal", "ent ")),
    ("Insula",   ("insula",)),
    ("Thalamus", ("thalamus",)),
    ("WM",       ("cerebral white matter", "white matter")),
]


def native_roi_label(native_region, mni_y=None,
                     hc_ant_mid_y=anat_atlas.HC_ANT_MID_Y):
    """Normalise a site's own region string to a project ROI.

    Hippocampus is split anterior/mid at the same y used for cells
    (HC_ANT_MID_Y = -21.0, Poppenk & Moscovitch 2013), so native and atlas
    ROIs are directly comparable.
    """
    if native_region is None or (isinstance(native_region, float)
                                 and np.isnan(native_region)):
        return None
    s = str(native_region).strip().lower()
    if not s or s in ("nan", "unknown", "none"):
        return None
    for roi, pats in _NATIVE_PATTERNS:
        if any(p in s for p in pats):
            if roi != "HC":
                return roi
            if mni_y is None or not np.isfinite(mni_y):
                return "HC"
            return "HC_anterior" if float(mni_y) >= hc_ant_mid_y else "HC_mid"
    return None

# Contact labels split into a probe/shaft name and a contact number.
_CONTACT_RE = re.compile(r'^(?P<probe>.*?)[-_]?(?P<num>\d+)$')
_NS_SUFFIX_RE = re.compile(r'-\d+$')


def split_contact(label):
    """'LT2HbE02' -> ('LT2HbE', 2); 'LMH-1' -> ('LMH', 1). (None, nan) if
    the label carries no trailing contact number."""
    m = _CONTACT_RE.match(str(label).strip())
    if not m:
        return None, np.nan
    return m.group('probe'), int(m.group('num'))


def _hemisphere(label, fallback=None):
    lab = str(label).lstrip('mb').upper()
    if lab.startswith('L'):
        return 'L'
    if lab.startswith('R'):
        return 'R'
    return fallback


# =============================================================================
# PER-SITE MACRO LOADERS
# =============================================================================

def load_baylor_macros(v2026_folder):
    """{subject_code: DataFrame of sEEG (macro) contacts}.

    Applies the same MNI152 self-consistency gate as the micro loader: if a
    file's MNI152 column disagrees with its own MNI305 under the Fischl
    transform, use MNI305 transformed instead and flag it.
    """
    out = {}
    for fn in sorted(os.listdir(v2026_folder)):
        if not fn.endswith("-electrodes_v2026.csv"):
            continue
        code = fn.split("-")[0]
        d = pd.read_csv(os.path.join(v2026_folder, fn))
        macro = d[d["Type"].astype(str) == "sEEG"].copy()
        if macro.empty:
            continue

        use_152 = anat_src._file_152_is_selfconsistent(d)
        if use_152:
            xyz = macro[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
            src = "baylor_v2026_macro_mni152"
        else:
            a305 = macro[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
            xyz = np.full_like(a305, np.nan)
            ok = ~np.any(np.isnan(a305), axis=1)
            if ok.any():
                xyz[ok] = anat_src.mni305_to_mni152(a305[ok])
            src = "baylor_v2026_macro_305to152"

        # Native region uses the 3 mm-neighbourhood parcellation
        # (ROI_DK2005_3mm / Matter_3mm), NOT the single-voxel Area_fs_vox.
        # A macro contact is a ~2 mm ring recording from a volume, so the
        # label should be sampled at that scale. Measured on YEJ: the voxel
        # column calls 1 contact hippocampal, the 3 mm column calls 4 -- e.g.
        # RT2bHaEa02 is "Right-Hippocampus" at 3 mm and
        # "Right-Cerebral-White-Matter" at the voxel. The voxel columns are
        # kept alongside so the choice is reversible.
        res = pd.DataFrame({
            "anat_label": macro["Label"].astype(str).str.strip(),
            "mni_x": xyz[:, 0], "mni_y": xyz[:, 1], "mni_z": xyz[:, 2],
            "matter": macro.get("Matter_3mm"),
            "native_region": macro.get("ROI_DK2005_3mm"),
            "native_region_vox": macro.get("Area_fs_vox"),
            "matter_vox": macro.get("Matter_fs_vox"),
            "probe_src": macro.get("ProbeName"),
            "hemisphere_src": macro.get("Hemisphere"),
            "coord_source": src,
        }).reset_index(drop=True)
        out[code] = res
    return out


def load_utah_macros(mat):
    """Macro contacts from an Electrodes.mat. See the module docstring for
    why coordinates use the direct row index and not the channel number."""
    em = np.asarray(mat.get("ElecMapRaw"), dtype=object)
    if em is None or em.ndim != 2 or em.shape[1] < 2:
        return pd.DataFrame()

    xyz_raw = np.asarray(mat.get("ElecXYZMNIRaw"), dtype=float)
    xyz_proj = mat.get("ElecXYZMNIProjRaw", mat.get("ElecXYZMNIProj"))
    xyz_proj = (np.asarray(xyz_proj, dtype=float)
                if xyz_proj is not None else None)
    matter = np.atleast_1d(mat.get("ElecTypeRaw"))
    atlas = mat.get("ElecAtlasRaw")
    atlas = np.asarray(atlas, dtype=object) if atlas is not None else None
    atlas_names = [str(a) for a in np.atleast_1d(mat.get("AtlasNames", []))]
    nmm_col = atlas_names.index("NMM") if "NMM" in atlas_names else 0

    rows = []
    for i in range(len(em)):
        label = str(em[i][0]).strip()
        if label.startswith('m'):
            continue                              # microwire, not an LFP macro
        chan = em[i][1]
        try:
            chan = int(chan) if np.isfinite(float(chan)) else None
        except (TypeError, ValueError):
            chan = None

        c = xyz_raw[i] if i < len(xyz_raw) else np.full(3, np.nan)
        cp = (xyz_proj[i] if xyz_proj is not None and i < len(xyz_proj)
              else np.full(3, np.nan))
        rows.append({
            "anat_label": label,
            "utah_chan": chan,
            "mni_x": c[0], "mni_y": c[1], "mni_z": c[2],
            "mni_proj_x": cp[0], "mni_proj_y": cp[1], "mni_proj_z": cp[2],
            "matter": (str(matter[i]) if i < len(matter) else None),
            "native_region": (str(atlas[i][nmm_col])
                              if atlas is not None and i < len(atlas) else None),
            "coord_source": "utah_elecmapraw_mni_raw",
        })
    return pd.DataFrame(rows)


def load_ucla_macros(v2026_folder):
    """{subject_label: DataFrame of macro contacts} from the v2026 xlsx."""
    out = {}
    for subj, prefix in anat_src.UCLA_SUBJECT_TO_FILE.items():
        fpath = os.path.join(v2026_folder, f"{prefix}_localizations.xlsx")
        if not os.path.exists(fpath):
            continue
        d = pd.read_excel(fpath, sheet_name="Sheet1")
        if "isMicro" in d.columns:
            d = d[d["isMicro"] == False]                          # noqa: E712
        d = d.dropna(subset=["MNI_x", "MNI_y", "MNI_z"])
        res = pd.DataFrame({
            "anat_label": d["electrode"].astype(str).str.strip(),
            "ncs_stem": d["electrode"].astype(str).str.replace("-", "", regex=False),
            "mni_x": d["MNI_x"].to_numpy(float),
            "mni_y": d["MNI_y"].to_numpy(float),
            "mni_z": d["MNI_z"].to_numpy(float),
            "matter": d.get("grayWhite"),
            "native_region": d.get("NMM"),
            "ashs_subfield": d.get("ASHS_ABC"),
            "coord_source": "ucla_v2026_xlsx",
        }).reset_index(drop=True)
        out[subj] = res
    return out


# =============================================================================
# JOIN ANATOMY TO ACTUAL LFP CHANNELS
# =============================================================================

def build_macro_table(session, recording_site, subject_label, channels,
                      baylor_macros=None, utah_mat=None, ucla_macros=None):
    """One row per LFP channel in `channels` (the `channels.npy` order).

    Never drops a channel: unmatched ones come back with `resolved=False`
    and an `unresolved_reason`, so the QC report can account for every
    channel in the recording rather than silently losing some.

    `ns_pos` is the 0-based column index into the LFP array. For Baylor the
    trailing `-NNN` in a channel name is a Blackrock electrode ID, not the
    array position, so it must never be used to index data.
    """
    channels = [str(c).strip() for c in channels]
    site = str(recording_site).lower()
    subject_key = re.sub(r'^BY\d*[-_]?', '', str(subject_label).strip().strip("'"))

    base = pd.DataFrame({"ns_pos": np.arange(len(channels)),
                         "ns_label": channels})

    if site == 'baylor':
        table = (baylor_macros or {}).get(subject_key)
        if table is None:
            base["unresolved_reason"] = f"no v2026 macro table for '{subject_key}'"
            anat = pd.DataFrame()
        else:
            base["anat_label"] = base["ns_label"].str.replace(
                _NS_SUFFIX_RE, "", regex=True)
            anat = table
        key = "anat_label"

    elif site == 'utah':
        if utah_mat is None:
            base["unresolved_reason"] = "no Electrodes.mat found"
            anat = pd.DataFrame()
        else:
            anat = load_utah_macros(utah_mat)
            # channels.npy is 'chan1'..'chanN' plus trailing analog channels
            # (EyeX, EyeY, Pupil, BP). Match on the integer, never on order.
            base["utah_chan"] = base["ns_label"].str.extract(
                r'^chan(\d+)$', expand=False).astype(float)
        key = "utah_chan"

    elif site == 'ucla':
        table = (ucla_macros or {}).get(str(subject_label).strip().strip("'"))
        if table is None:
            base["unresolved_reason"] = f"no v2026 xlsx for '{subject_label}'"
            anat = pd.DataFrame()
        else:
            base["ncs_stem"] = base["ns_label"].str.replace(
                r'\.ncs$|_\d{4}$', "", regex=True)
            anat = table
        key = "ncs_stem"

    else:
        base["unresolved_reason"] = f"unknown recording site '{recording_site}'"
        anat = pd.DataFrame()
        key = None

    if anat is not None and len(anat) and key in base.columns:
        # pandas merges NaN keys to each other, so a channel with no join key
        # (Utah's trailing EyeX/Pupil/BP analog channels) would fan out against
        # every unlocalised anatomy row. s02 went 132 -> 248 rows before this
        # guard. Drop null keys on both sides, then reattach the unmatched
        # channels so no channel is ever silently lost.
        anat_j = anat[anat[key].notna()].drop_duplicates(subset=[key])
        joinable = base[base[key].notna()]
        merged = joinable.merge(anat_j, on=key, how="left", suffixes=("", "_anat"))
        leftover = base[base[key].isna()].copy()
        if len(leftover):
            leftover["unresolved_reason"] = leftover.get(
                "unresolved_reason", pd.Series([None] * len(leftover))
            ).fillna("channel has no anatomy join key (e.g. analog channel)")
            merged = pd.concat([merged, leftover], ignore_index=True)
        merged = merged.sort_values("ns_pos").reset_index(drop=True)
    else:
        merged = base.copy()
        for c in ["anat_label", "mni_x", "mni_y", "mni_z", "matter",
                  "native_region", "coord_source"]:
            merged.setdefault(c, np.nan)

    merged["session"] = int(session)
    merged["subject_label"] = str(subject_label).strip().strip("'")
    merged["recording_site"] = site

    probe_num = merged["anat_label"].apply(
        lambda l: split_contact(l) if isinstance(l, str) else (None, np.nan))
    merged["probe"] = [p for p, _ in probe_num]
    merged["contact_no"] = [n for _, n in probe_num]
    if "probe_src" in merged.columns:
        merged["probe"] = merged["probe_src"].fillna(merged["probe"])
    merged["hemisphere"] = [
        _hemisphere(l, h) if isinstance(l, str) else None
        for l, h in zip(merged["anat_label"],
                        merged.get("hemisphere_src", pd.Series([None] * len(merged))))]

    has_xyz = merged[["mni_x", "mni_y", "mni_z"]].notna().all(axis=1)
    merged["resolved"] = has_xyz
    if "unresolved_reason" not in merged.columns:
        merged["unresolved_reason"] = None
    merged.loc[~has_xyz & merged["unresolved_reason"].isna(),
               "unresolved_reason"] = "no coordinate for this channel"
    merged.loc[has_xyz, "unresolved_reason"] = None
    return merged


# =============================================================================
# ATLAS LABELLING  (identical rules to the cell pipeline)
# =============================================================================

def label_contacts_with_atlas(df, atlases=None):
    """Add `_juelich/_ho_cort/_ho_sub/_bn_label`, then `atlas_roi`,
    `atlas_source_label`, `atlas_reason` via anatomy_atlas.assign_atlas_roi.

    Mirrors the per-cell query block of cell_to_roi_july26.py exactly, so a
    contact and a cell at the same coordinate get the same ROI.
    """
    # The atlas is a CROSS-CHECK, not the ROI criterion: `native_roi` (from each
    # site's own segmentation) drives `is_hpc` and the bipolar pairing. So if the
    # atlases are unavailable -- e.g. on a cluster node with no nilearn_data and
    # no network -- degrade gracefully rather than failing. This removes a 314 MB
    # dependency from the cluster run; contact selection is unaffected.
    # `atlases=False` means "explicitly none" (caller already knows they are
    # unavailable); `atlases=None` means "not supplied, try to load". Without
    # that distinction a caller passing None would retry the failed load once
    # per session and print the same error 60 times.
    if atlases is None:
        try:
            atlases = anat_atlas.get_atlases()
        except Exception as e:
            print(f"  [atlas unavailable: {type(e).__name__}: {e}]")
            print("  [continuing with native-space ROI only; atlas_roi will be NaN]")
            atlases = False
    if not atlases:
        df = df.reset_index(drop=True).copy()
        for c in ["_juelich", "_ho_cort", "_ho_sub", "_bn_label",
                  "atlas_roi", "atlas_source_label", "atlas_reason"]:
            df[c] = np.nan
        df["atlas_available"] = False
        if "native_region" in df.columns:
            df["native_roi"] = [native_roi_label(r, y) for r, y
                                in zip(df["native_region"], df["mni_y"])]
        else:
            df["native_roi"] = None
        df["is_hpc"] = df["native_roi"].isin(HPC_ROIS)
        df["roi_concordant"] = np.nan          # cannot be assessed without atlas
        return df
    ho_cort, ho_sub, juelich, brainnetome = atlases

    if df.empty:
        for c in ["_juelich", "_ho_cort", "_ho_sub", "_bn_label",
                  "atlas_roi", "atlas_source_label", "atlas_reason"]:
            df[c] = pd.Series(dtype=object)
        return df

    xyz_all = df[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    records = []
    for xyz in xyz_all:
        if np.any(np.isnan(xyz)):
            records.append({"_juelich": None, "_ho_cort": None,
                            "_ho_sub": None, "_bn_label": None})
            continue
        records.append({
            "_juelich": juelich.label_at(xyz),
            "_ho_cort": ho_cort.label_at(xyz),
            "_ho_sub": ho_sub.label_at(xyz),
            "_bn_label": brainnetome.label_at(xyz),
        })
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(records)], axis=1)

    # assign_atlas_roi reads MNI_y_final and the four _atlas columns.
    df["MNI_x_final"] = df["mni_x"]
    df["MNI_y_final"] = df["mni_y"]
    df["MNI_z_final"] = df["mni_z"]

    assigns = df.apply(anat_atlas.assign_atlas_roi, axis=1, result_type="expand")
    assigns.columns = ["atlas_roi", "atlas_source_label", "atlas_reason"]
    df = pd.concat([df, assigns], axis=1)
    df = df.drop(columns=["MNI_x_final", "MNI_y_final", "MNI_z_final"])

    # Native-space ROI is the PRIMARY criterion for macro contacts.
    if "native_region" in df.columns:
        df["native_roi"] = [
            native_roi_label(r, y)
            for r, y in zip(df["native_region"], df["mni_y"])
        ]
    else:
        df["native_roi"] = None
    df["atlas_available"] = True
    df["is_hpc"] = df["native_roi"].isin(HPC_ROIS)
    # The atlas ladder is a strict superset of native HC, so a native-HC
    # contact that the atlas does NOT call HC would be an anomaly worth seeing.
    df["roi_concordant"] = ~(df["is_hpc"] & ~df["atlas_roi"].isin(HPC_ROIS))
    return df


# =============================================================================
# BIPOLAR PAIRING
# =============================================================================

_WM_TOKENS = ("white",)
# Grey structures a reference contact must NOT sit in: referencing a
# hippocampal contact against another MTL grey contact subtracts out the very
# signal we are trying to measure.
_MTL_GREY = ("HC_anterior", "HC_mid", "HC", "PHC", "EC", "Amygdala")


def _pick_anchor(hpc_rows):
    """The most medial contact of the probe, i.e. the one closest to the
    midline. Defined geometrically as min |MNI x| rather than by contact
    number, so it does not depend on each site's numbering convention
    (Baylor 01 = deepest; Utah/UCLA differ)."""
    return hpc_rows.loc[hpc_rows["mni_x"].abs().idxmin()]


def _pick_reference(probe_rows, anchor, scheme, max_gap_mm):
    """Choose the reference contact for `anchor` on the same probe."""
    others = probe_rows[probe_rows["anat_label"] != anchor["anat_label"]].copy()
    if others.empty:
        return None, "no other contact on probe"

    a_xyz = np.array([anchor["mni_x"], anchor["mni_y"], anchor["mni_z"]], float)
    xyz = others[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    others["_dist"] = np.linalg.norm(xyz - a_xyz, axis=1)
    others = others[np.isfinite(others["_dist"]) & (others["_dist"] <= max_gap_mm)]
    if others.empty:
        return None, f"no contact within {max_gap_mm} mm"

    if scheme == "neighbour":
        # Chen et al.: the immediate neighbour (second-most medial). Prefer
        # the next contact outward; fall back to the one inward.
        if np.isfinite(anchor.get("contact_no", np.nan)):
            n = int(anchor["contact_no"])
            for cand in (n + 1, n - 1):
                hit = others[others["contact_no"] == cand]
                if len(hit):
                    return hit.iloc[0], "immediate neighbour"
        return others.nsmallest(1, "_dist").iloc[0], "nearest contact"

    if scheme == "white_matter":
        wm = others[
            others["matter"].astype(str).str.lower().str.contains("|".join(_WM_TOKENS))
            & ~others["native_roi"].isin(_MTL_GREY)
        ]
        if len(wm):
            return wm.nsmallest(1, "_dist").iloc[0], "nearest white matter"
        return None, "no white-matter contact on probe"

    raise ValueError(f"unknown scheme '{scheme}'")


def build_bipolar_pairs(contacts, target_rois=HPC_ROIS,
                        scheme="neighbour", max_gap_mm=12.0):
    """**One** bipolar derivation per probe -- not every adjacent pair.

    Follows Chen et al.: "bipolar referencing was performed using the most
    medial hippocampal contact and its immediate neighbour (i.e. the
    second-most medial contact) on each hippocampal probe". That yields ~2
    derivations per participant (they report 34 contacts across 17 patients),
    and every derivation is independent -- no contact appears in two pairs.

    Generating all adjacent pairs instead would inflate the count several-fold
    and make the derivations non-independent (contact 3 would appear in both
    pair 2-3 and pair 3-4), which the GLM cannot account for.

    scheme:
      'neighbour'    -- Chen: anchor + immediate neighbour (default; always
                        available).
      'white_matter' -- anchor + nearest same-probe white-matter contact that
                        is not itself in MTL grey. Cleaner subtraction where a
                        suitable contact exists, but not every probe has one.

    Returns one row per (probe, target ROI family) with the anchor's native
    ROI as `pair_roi_native`. The pair coordinate is the midpoint, which is
    where the derivation is actually sensitive.
    """
    rows, skipped = [], []
    for probe, grp in contacts.groupby("probe", dropna=True):
        grp = grp[grp["resolved"]].dropna(subset=["contact_no"])
        if grp.empty:
            continue
        hpc = grp[grp["native_roi"].isin(target_rois)]
        if hpc.empty:
            continue

        anchor = _pick_anchor(hpc)
        ref, why = _pick_reference(grp, anchor, scheme, max_gap_mm)
        if ref is None:
            skipped.append({"probe": probe, "anchor": anchor["anat_label"],
                            "reason": why})
            continue

        d = float(np.linalg.norm(
            np.array([anchor["mni_x"], anchor["mni_y"], anchor["mni_z"]], float)
            - np.array([ref["mni_x"], ref["mni_y"], ref["mni_z"]], float)))
        rows.append({
            "pair_id": f"{anchor['anat_label']}-{ref['anat_label']}",
            "probe": probe,
            "anat_label_a": anchor["anat_label"], "anat_label_b": ref["anat_label"],
            "ns_pos_a": anchor.get("ns_pos"), "ns_pos_b": ref.get("ns_pos"),
            "ns_label_a": anchor.get("ns_label"), "ns_label_b": ref.get("ns_label"),
            "contact_no_a": anchor.get("contact_no"),
            "contact_no_b": ref.get("contact_no"),
            "hemisphere": anchor.get("hemisphere"),
            "matter_a": anchor.get("matter"), "matter_b": ref.get("matter"),
            "native_roi_a": anchor.get("native_roi"),
            "native_roi_b": ref.get("native_roi"),
            "atlas_roi_a": anchor.get("atlas_roi"), "atlas_roi_b": ref.get("atlas_roi"),
            "native_region_a": anchor.get("native_region"),
            "native_region_b": ref.get("native_region"),
            "pair_roi_native": anchor.get("native_roi"),
            "n_hpc_on_probe": int(len(hpc)),
            "mni_x": (anchor["mni_x"] + ref["mni_x"]) / 2.0,
            "mni_y": (anchor["mni_y"] + ref["mni_y"]) / 2.0,
            "mni_z": (anchor["mni_z"] + ref["mni_z"]) / 2.0,
            "inter_contact_mm": d,
            "ref_rule": scheme,
            "ref_reason": why,
        })
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out


# =============================================================================
# UTAH ELECTRODE FILE RESOLUTION
# =============================================================================

def resolve_utah_mat(session, subject_key, data_root, manifest=None,
                     verbose=False):
    """Find the Electrodes.mat that genuinely belongs to `session`.

    Deliberately NOT `anat_src.discover_utah_mats()`, which coord-matches each
    subject's cells against every folder's electrode pool and picks the best.
    That has no uniqueness constraint, and Utah subjects have only 3-16 cells
    each, so the match is weak: measured on this dataset, **s47's mat was
    assigned to six different patients** (UT202302, UT202311, UT202418,
    UT202421, UT202422b, UT202503), and s23's to two. Sessions would silently
    receive another patient's electrode positions.

    Resolution order, strongest first:
      1. the session's own `s{NN}/electrodes/*.mat`  -- definitive
      2. another session of the SAME patient          -- same implant
      3. nothing: return None so the caller excludes the session with a
         reason, rather than borrowing a stranger's anatomy

    `anat_src.discover_utah_mats` is left untouched: the cell pipeline calls it
    and changing it would alter published cell ROIs. That pipeline appears to
    have the same weakness -- worth reviewing separately.
    """
    for name in ("Electrodes.mat", "ChannelMap.mat"):
        p = os.path.join(data_root, f"s{int(session):02d}", "electrodes", name)
        if os.path.isfile(p):
            return anat_src._load_mat(p), f"own ({name})"

    # same patient, different session
    if manifest is not None and subject_key:
        same = manifest[(manifest.subject_key == subject_key)
                        & (manifest.session != int(session))]
        for other in sorted(same.session.astype(int)):
            for name in ("Electrodes.mat", "ChannelMap.mat"):
                p = os.path.join(data_root, f"s{other:02d}", "electrodes", name)
                if os.path.isfile(p):
                    if verbose:
                        print(f"    s{int(session):02d}: using s{other:02d}'s "
                              f"{name} (same patient {subject_key})")
                    return anat_src._load_mat(p), f"same patient s{other:02d}"
    return None, "no Electrodes.mat for this session or patient"
