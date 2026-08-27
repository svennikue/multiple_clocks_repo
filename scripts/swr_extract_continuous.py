#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 3 of the SWR pipeline: continuous preprocessing, per session.

raw -> 1000 Hz (resample_poly) -> bipolar -> notch 60/120/180, written out on
the behavioural clock so that `sample = round(t_session * fs)`.

Separate from detection on purpose: this stage is I/O-bound on multi-GB raw
files, while detection is CPU-bound and gets re-run whenever a detector
parameter changes. Fusing them would force a re-read of the raw data on every
parameter tweak (~4 h across all sessions instead of ~15 min).

Outputs (per session, keyed by ANALYSIS_NAME rather than a timestamp so a
60-task array job produces one joinable set):
    derivatives/s{XX}/LFP-clean/{name}/continuous.npy    float32 (n_pairs, n_samples)
    derivatives/s{XX}/LFP-clean/{name}/pairs.csv
    derivatives/s{XX}/LFP-clean/{name}/meta.json
    derivatives/s{XX}/LFP-clean/{name}/settings.json
    derivatives/s{XX}/LFP-clean/{name}/qc_psd.png

Usage:
    python scripts/swr_extract_continuous.py --session=38
    python scripts/swr_extract_continuous.py --session=38 --analysis_name=swr_v1

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_preproc as pp

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"

# A behavioural event may fall at most this far outside its recording before
# the block is rejected. Three sessions (s09, s10, s33) overrun by 0.9-2.0 s,
# always by exactly one trailing repeat, because the amplifier was stopped just
# before the final trial completed.
CLOCK_TOLERANCE_S = 5.0


def _settings_dict(session, analysis_name):
    return {
        "analysis_name": analysis_name,
        "session": int(session),
        "target_fs": pp.TARGET_FS,
        "notch_hz": list(pp.LINE_FREQS),
        "notch_q": pp.NOTCH_Q,
        "resample": "scipy.signal.resample_poly (NOT scipy.signal.resample)",
        "montage": "bipolar, one pair per probe, anchor + immediate neighbour",
        "clock": "behavioural = cumulative file duration",
        "clock_tolerance_s": CLOCK_TOLERANCE_S,
        "chunk_s": pp.CHUNK_S,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _qc_psd(sig, fs, pair_ids, out_png):
    """Welch PSD of every derivation, with the notch frequencies marked.

    The check that matters: no residual peak at 60/120/180, and no edge
    artifact from resampling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    f, P = welch(sig, fs=fs, nperseg=int(4 * fs), axis=-1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i in range(P.shape[0]):
        ax.semilogy(f, P[i], lw=0.7, alpha=0.75)
    for lf in pp.LINE_FREQS:
        ax.axvline(lf, color="0.5", ls=":", lw=0.8)
    ax.axvspan(80, 120, color="#F15A29", alpha=0.12, label="ripple band 80–120 Hz")
    ax.set_xlim(0, 250)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu$V$^2$/Hz)")
    ax.set_title(f"Bipolar derivations after notch (n={P.shape[0]})", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # residual line noise: power at the notch vs the local neighbourhood
    res = {}
    for lf in pp.LINE_FREQS:
        i = int(np.argmin(np.abs(f - lf)))
        side = np.r_[P[:, max(0, i - 12):i - 4], P[:, i + 5:i + 13]]
        res[f"{lf:.0f}Hz_ratio"] = float(np.median(P[:, i]) / max(np.median(side), 1e-30))
    return res


def extract_session(session, analysis_name=ANALYSIS_NAME, save_all=True,
                    verbose=True):
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-clean", analysis_name), "swr_extract_continuous")
    session = int(session)
    data_root = swr_io.get_data_root()

    pairs_p = os.path.join(swr_io.session_deriv_dir(session, data_root),
                           "LFP", f"bipolar_pairs_{session:02d}.csv")
    if not os.path.isfile(pairs_p):
        raise FileNotFoundError(
            f"{pairs_p} not found -- run scripts/swr_build_contacts.py first")
    pairs = pd.read_csv(pairs_p)
    if not len(pairs):
        print(f"s{session:02d}: no bipolar pairs, nothing to do")
        return None

    # ---- hard gate: does every behavioural event land inside its recording?
    blocks, bt, warn = pp.session_block_table(session, data_root=data_root)
    print(f"\ns{session:02d}: {len(blocks)} block(s), {len(pairs)} pairs")
    print(blocks[["block", "duration_s", "fs_raw", "offset_s",
                  "head_margin_s", "tail_margin_s"]].to_string(index=False))
    if len(blocks) != len(bt):
        # s32 is the known case: one behavioural block, two recordings with a
        # 444.8 s hole between them (file t_starts 12042.7 and 14465.0 on a
        # shared amplifier clock). The cumulative-duration clock cannot express
        # a gap inside a block, so the mapping is not recoverable here. Skip
        # with a reason rather than crashing a batch run.
        reason = (f"{len(blocks)} recordings vs {len(bt)} behavioural blocks "
                  f"-- recording gap inside a behavioural block")
        print(f"  SKIPPED s{session:02d}: {reason}")
        if save_all:
            out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                                   "LFP-clean", analysis_name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "SKIPPED.json"), "w") as f:
                json.dump({"session": session, "reason": reason,
                           "n_recordings": len(blocks),
                           "n_behavioural_blocks": len(bt),
                           "block_durations_s": list(blocks.duration_s)}, f, indent=2)
        return None
    bad = blocks[(blocks.head_margin_s < -CLOCK_TOLERANCE_S)
                 | (blocks.tail_margin_s < -CLOCK_TOLERANCE_S)]
    if len(bad):
        raise RuntimeError(
            f"s{session:02d}: behavioural events fall >{CLOCK_TOLERANCE_S}s "
            f"outside their recording -- block alignment failed:\n{bad}")
    n_over = int((blocks.tail_margin_s < 0).sum())
    if n_over:
        print(f"  note: {n_over} block(s) overrun by <{CLOCK_TOLERANCE_S}s "
              f"(trailing repeat past recording end; will be dropped downstream)")

    sig, meta = pp.preprocess_session(session, pairs, data_root=data_root,
                                      verbose=verbose)
    meta["behaviour_blocks"] = bt.to_dict("records")
    meta["clock_overrun_blocks"] = n_over
    meta["discovery_warnings"] = warn

    print(f"  -> {sig.shape[0]} pairs x {sig.shape[1]} samples "
          f"({sig.shape[1]/pp.TARGET_FS:.0f}s) "
          f"= {sig.nbytes/1e6:.0f} MB float32")

    if save_all:
        out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                               "LFP-clean", analysis_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "continuous.npy"), sig)
        pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)

        res = _qc_psd(sig, pp.TARGET_FS, list(pairs.pair_id),
                      os.path.join(out_dir, "qc_psd.png"))
        meta["line_noise_residual"] = res
        print("  residual line noise (1.0 = fully removed): "
              + ", ".join(f"{k}={v:.2f}" for k, v in res.items()))

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(session, analysis_name))
        print(f"  saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(extract_session)
    else:
        extract_session(38)
