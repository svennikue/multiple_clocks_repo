#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The bundle: ripples, intervals and channel QC for every session, from one place.

The cluster holds the raw recordings; the analysis of what the ripples MEAN does
not need them. `export_bundle` writes a few MB that reproduces every result in
this project on a laptop, and `RippleStore` reads either that bundle or the
per-session detection output behind one interface -- so the same analysis code
runs on the cluster and locally and cannot diverge.

    import mc.analyse.swr_bundle as swb_
    store = swb_.RippleStore(bundle="<bundle dir>")
    ev, intervals, qc = store.get(18)

@author: Svenja Kuchenhoff
"""

import os
import glob

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io

ANALYSIS_NAME = "swr_v1"


class RippleStore:
    """Where the detected ripples come from, behind one interface.

    Two sources:
      `sessions`  the per-session detection output on this machine. What the
                  cluster has.
      `bundle`    a bundle downloaded from the cluster, carrying the same three
                  tables for every session in a few MB. What the laptop has.

    The bundle exists so these statistics can be redone without moving the LFP.
    Everything that reads ripples goes through here, so both paths run the
    IDENTICAL analysis code -- the source cannot change a result.
    """

    def __init__(self, analysis_name=ANALYSIS_NAME, data_root=None, bundle=None):
        self.analysis_name = analysis_name
        self.R = data_root or swr_io.get_data_root()
        self.bundle_path = None
        if bundle is None:
            self.source = "sessions"
            paths = sorted(glob.glob(os.path.join(
                swr_io.derivatives_dir(self.R), "s*", "LFP-ripples",
                analysis_name, "ripple_events.csv")))
            self._sessions = [int(p.split(os.sep)[-4][1:]) for p in paths]
            self._dirs = {s: os.path.dirname(p)
                          for s, p in zip(self._sessions, paths)}
        else:
            self.source = "bundle"
            self._load_bundle(bundle)

    def _load_bundle(self, bundle):
        """`bundle` is the bundle directory, or the .pkl inside it."""
        if os.path.isdir(bundle):
            d = bundle
            tabs = {k: pd.read_csv(os.path.join(d, f"{k}.csv"))
                    for k in ("ripples", "intervals", "channel_qc")}
        else:
            import pickle
            with open(bundle, "rb") as f:
                b = pickle.load(f)
            d = os.path.dirname(bundle)
            tabs = {k: b[k] for k in ("ripples", "intervals", "channel_qc")}
        self.bundle_path = d
        self._rip = {s: g for s, g in tabs["ripples"].groupby("session")}
        self._iv = {s: g for s, g in tabs["intervals"].groupby("session")}
        self._qc = {s: g.set_index("pair_id")
                    for s, g in tabs["channel_qc"].groupby("session")}
        self._sessions = sorted(self._rip)
        # the bundle carries its own subject key, so a stale local manifest
        # cannot silently re-cluster the robust standard errors
        r = tabs["ripples"]
        self._subj = (r[["session", "subject_key", "recording_site"]]
                      .drop_duplicates("session").set_index("session")
                      .to_dict("index"))

    def sessions(self):
        return list(self._sessions)

    def get(self, sess):
        """(accepted events, artifact-free intervals, channel QC) for a session.

        Events are already filtered to those that passed detection, in both
        sources -- the bundle stores only accepted ripples.
        """
        sess = int(sess)
        if self.source == "bundle":
            return (self._rip[sess], self._iv[sess], self._qc[sess])
        d = self._dirs[sess]
        ev = pd.read_csv(os.path.join(d, "ripple_events.csv"))
        ev = ev[ev.passed.fillna(False)]
        iv = pd.read_csv(os.path.join(d, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(d, "channel_qc.csv")).set_index("pair_id")
        return ev, iv, qc

    def subject_map(self):
        if self.source == "bundle":
            return self._subj
        m = pd.read_csv(os.path.join(swr_io.derivatives_dir(self.R), "group",
                                     "swr", "session_manifest.csv"))
        return m.set_index("session")[["subject_key",
                                       "recording_site"]].to_dict("index")

    def describe(self):
        n = sum(len(self._rip[s]) for s in self._sessions) \
            if self.source == "bundle" else None
        where = self.bundle_path or f"{swr_io.derivatives_dir(self.R)}/s*/LFP-ripples"
        print(f"  source: {self.source}  ({len(self._sessions)} sessions"
              + (f", {n} accepted ripples" if n is not None else "") + ")")
        print(f"          {where}")
