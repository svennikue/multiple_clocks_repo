#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 2 of the SWR pipeline: build the macro-contact anatomy table.

One row per LFP channel per session, with an MNI152 coordinate, grey/white,
a native-atlas region label, and an ROI assigned by the SAME rule ladder the
cell pipeline uses. Then adjacent-bipolar pairs for the contacts we care about.

Group-level on purpose: the nilearn fetches plus four MaxProbAtlas
instantiations cost ~30 s, and paying that once per session across 60 sessions
would be pointless.

Outputs, per session:
    derivatives/s{XX}/LFP/macro_contacts_{XX}.csv
    derivatives/s{XX}/LFP/bipolar_pairs_{XX}.csv
and group-level:
    derivatives/group/swr/macro_contacts_all.csv
    derivatives/group/swr/contact_qc.csv
    derivatives/group/swr/settings.json

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_build_contacts.py
    python scripts/swr_build_contacts.py --sessions="[2,5,26]" --verbose=True

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.anatomy_sources as anat_src
import mc.analyse.anatomy_atlas as anat_atlas
import mc.analyse.contact_anatomy as ca
import mc.analyse.swr_report as swr_report

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_contacts"
V2026_DIRNAME = "ABCD_pts_elecFilesForSvenja_v2026"


def _settings_dict(sessions, v2026_folder):
    return {
        "analysis_name": ANALYSIS_NAME,
        "sessions": list(sessions),
        "v2026_folder": v2026_folder,
        "hpc_rois": list(ca.HPC_ROIS),
        "bipolar_scheme": "neighbour (Chen: most medial HC contact + immediate neighbour, ONE pair per probe)",
        "utah_coords": "ElecXYZMNIRaw (direct row index, NOT via channel no.)",
        "atlas_rules": "mc.analyse.anatomy_atlas.assign_atlas_roi (shared with cells)",
        "hc_ant_mid_y": anat_atlas.HC_ANT_MID_Y,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _load_channels(session, data_root):
    """Channel names in LFP-array order.

    Prefers the cached channels.npy, but that only exists for the handful of
    sessions the old preprocessing ever ran on. Falls back to reading the
    channel list straight out of the raw file header, so contact anatomy can
    be built for every session with raw data -- not just the preprocessed ten.
    """
    lfp_dir = os.path.join(swr_io.session_deriv_dir(session, data_root), "LFP")
    p = os.path.join(lfp_dir, "channels.npy")
    if os.path.isfile(p):
        return [str(c) for c in np.load(p, allow_pickle=True)], "channels.npy"

    cfg_s = swr_io.session_config(session, data_root=data_root)
    files, kind, _ = swr_io.discover_raw_files(session, cfg_s, data_root=data_root)
    if not files:
        return [], "none"

    if kind == 'neuralynx':
        stems = [os.path.splitext(os.path.basename(f))[0] for f in files[0]]
        return stems, "ncs filenames"

    try:
        import neo
        # Use the format DETECTED from disk, not the config's. s50/s51 are
        # marked `LFP_file_format: ncs` but are Blackrock .ns3, so int('ncs')
        # raised and both sessions silently produced no contacts. Take the
        # extension of the file discover_raw_files actually found.
        nsx = cfg_s.get('LFP_file_format', 3)
        try:
            nsx = int(nsx)
        except (TypeError, ValueError):
            nsx = None
        ext = os.path.splitext(files[0])[1].lstrip('.')          # e.g. 'ns3'
        if ext.startswith('ns') and ext[2:].isdigit():
            nsx = int(ext[2:])
        if nsx is None:
            nsx = 3
        reader = neo.io.BlackrockIO(filename=files[0], nsx_to_load=nsx)
        names = [str(e) for e in reader.header['signal_channels']]
        return [n.split(",")[0].strip("('") for n in names], "raw header"
    except Exception as e:
        print(f"    s{session:02d}: could not read channel names from raw: "
              f"{type(e).__name__}: {e}")
        return [], "none"


def build_contacts(sessions=None, save_all=True, verbose=False,
                   use_atlas=True):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_build_contacts")
    data_root = swr_io.get_data_root()
    v2026 = os.path.join(data_root, V2026_DIRNAME)

    manifest_p = os.path.join(swr_io.derivatives_dir(data_root),
                              "group", "swr", "session_manifest.csv")
    if not os.path.isfile(manifest_p):
        raise FileNotFoundError(
            f"{manifest_p} not found -- run scripts/swr_audit_sessions.py first")
    manifest = pd.read_csv(manifest_p)

    if sessions is not None:
        manifest = manifest[manifest.session.isin([int(s) for s in sessions])]

    print(f"Loading atlases and source tables ({len(manifest)} sessions)...")
    # The atlas is REQUIRED: hippocampal contacts are selected from the MNI
    # coordinate alone (mc.analyse.contact_anatomy.select_hpc_contacts), so
    # without it no contact can be chosen. Site-supplied region strings are not
    # consulted. A missing atlas therefore yields zero pairs, loudly.
    atlases = False          # False = explicitly none (see label_contacts_with_atlas)
    if use_atlas:
        try:
            atlases = anat_atlas.get_atlases()
        except Exception as e:
            print(f"  [atlas unavailable] {e}")
            print("  [continuing with native-space ROI only -- contact selection "
                  "is unaffected; atlas_roi will be NaN]")
    else:
        print("  [--use_atlas=False: native-space ROI only]")
    baylor_macros = ca.load_baylor_macros(v2026)
    ucla_macros = ca.load_ucla_macros(v2026)
    print(f"  baylor macro tables: {len(baylor_macros)} subjects")
    print(f"  ucla macro tables:   {len(ucla_macros)} subjects")

    # Utah electrode files are resolved PER SESSION (own folder, then same
    # patient), not by coord-matching cells against every folder -- see
    # ca.resolve_utah_mat for why that was unsafe here.

    all_rows, qc_rows = [], []
    for _, m in manifest.iterrows():
        session = int(m.session)
        site = str(m.recording_site).lower()
        label = str(m.subject_label).strip().strip("'") if pd.notna(m.subject_label) else None

        channels, chan_src = _load_channels(session, data_root)
        if not channels:
            qc_rows.append({"session": session, "recording_site": site,
                            "n_channels": 0, "n_resolved": 0, "n_hpc": 0,
                            "n_pairs": 0, "n_hpc_pairs": 0,
                            "note": "no channel list available"})
            if verbose:
                print(f"  s{session:02d}: no channel list")
            continue

        utah_mat, utah_src = None, ""
        if site == 'utah':
            utah_mat, utah_src = ca.resolve_utah_mat(
                session, m.subject_key, data_root, manifest=manifest,
                verbose=verbose)
            if utah_mat is None and verbose:
                print(f"  s{session:02d}: {utah_src}")

        contacts = ca.build_macro_table(
            session, site, label, channels,
            baylor_macros=baylor_macros, utah_mat=utah_mat,
            ucla_macros=ucla_macros)
        contacts = ca.label_contacts_with_atlas(contacts, atlases=atlases)

        pairs = ca.build_bipolar_pairs(contacts)
        if len(pairs):
            pairs = ca.label_contacts_with_atlas(pairs, atlases=atlases)
            pairs = pairs.rename(columns={"atlas_roi": "pair_roi"})
            pairs.insert(0, "session", session)
            pairs.insert(1, "subject_label", label)

        n_res = int(contacts["resolved"].sum())
        n_hpc = int(contacts["is_hpc"].sum())
        n_hpc_pairs = int(pairs["pair_roi_atlas"].isin(ca.HPC_ROIS).sum()) if len(pairs) else 0
        qc_rows.append({
            "session": session, "recording_site": site, "subject_label": label,
            "channel_source": chan_src, "n_channels": len(contacts),
            "n_resolved": n_res,
            "frac_resolved": round(n_res / max(len(contacts), 1), 3),
            "n_hpc": n_hpc, "n_pairs": len(pairs), "n_hpc_pairs": n_hpc_pairs,
            "note": "",
        })
        if verbose:
            print(f"  s{session:02d} {site:7s} ch={len(contacts):4d} "
                  f"resolved={n_res:4d} hpc={n_hpc:3d} pairs={len(pairs):3d} "
                  f"hpc_pairs={n_hpc_pairs:3d}")

        all_rows.append(contacts)
        if save_all:
            out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root), "LFP")
            os.makedirs(out_dir, exist_ok=True)
            contacts.to_csv(os.path.join(out_dir, f"macro_contacts_{session:02d}.csv"),
                            index=False)
            pair_path = os.path.join(out_dir, f"bipolar_pairs_{session:02d}.csv")
            if len(pairs):
                pairs.to_csv(pair_path, index=False)
            elif os.path.isfile(pair_path):
                # A session that yields no pairs now must not keep a file from an
                # earlier run: swr_check_inputs counts files, so a stale one reads
                # as "ready for stage 2" and stage 2 would run on outdated contacts.
                os.remove(pair_path)
                print(f"    s{session:02d}: removed stale {os.path.basename(pair_path)} "
                      f"(this build produced no pairs)")

    qc = pd.DataFrame(qc_rows)

    # inclusion report: every session accounted for, with a reason
    rep = swr_report.InclusionReport(
        "contacts", ANALYSIS_NAME,
        "Sessions yielding at least one hippocampal bipolar derivation.")
    for _, r in qc.iterrows():
        u = f"s{int(r.session):02d}"
        if int(r.get("n_hpc_pairs", 0)) > 0:
            rep.include(u, "", site=r.recording_site,
                        channels=int(r.n_channels), resolved=int(r.n_resolved),
                        hpc=int(r.n_hpc), pairs=int(r.n_hpc_pairs))
        elif int(r.get("n_channels", 0)) == 0:
            rep.exclude(u, "no channel list could be read "
                           "(no channels.npy and raw header unreadable)",
                        site=r.recording_site)
        elif int(r.get("n_resolved", 0)) == 0:
            rep.exclude(u, "no channel matched an electrode table "
                           "(missing/mismatched anatomy source)",
                        site=r.recording_site, channels=int(r.n_channels))
        elif int(r.get("n_hpc", 0)) == 0:
            rep.exclude(u, "no contact in hippocampus by native segmentation",
                        site=r.recording_site, resolved=int(r.n_resolved))
        else:
            rep.exclude(u, "hippocampal contacts found but no valid bipolar "
                           "partner on the same probe",
                        site=r.recording_site, hpc=int(r.n_hpc))
    print("\n" + "=" * 74)
    print(" CONTACT ANATOMY SUMMARY")
    print("=" * 74)
    print(qc.groupby("recording_site")[
        ["n_channels", "n_resolved", "n_hpc", "n_pairs", "n_hpc_pairs"]
    ].sum().to_string())
    print(f"\nsessions with >=1 hippocampal bipolar pair: "
          f"{int((qc.n_hpc_pairs > 0).sum())}/{len(qc)}")

    if all_rows:
        allc = pd.concat(all_rows, ignore_index=True)
        print("\n--- ROI counts across all resolved contacts ---")
        print(allc.loc[allc.resolved, "atlas_roi"].value_counts().head(12).to_string())

        if save_all:
            gdir = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr")
            os.makedirs(gdir, exist_ok=True)
            allc.to_csv(os.path.join(gdir, "macro_contacts_all.csv"), index=False)
            qc.to_csv(os.path.join(gdir, "contact_qc.csv"), index=False)
            rep.write(gdir)
            swr_io.write_settings(gdir, _settings_dict(
                list(manifest.session.astype(int)), v2026))
            print(f"\nSaved -> {gdir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(build_contacts)
    else:
        build_contacts()
