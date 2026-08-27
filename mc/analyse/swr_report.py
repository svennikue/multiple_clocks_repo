#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inclusion / exclusion reporting.

Every analysis stage records which sessions and derivations it kept, which it
dropped, and **why** -- written as both JSON (machine-readable) and Markdown
(for the methods section). Without this the final n in a paper is an
archaeological exercise: each stage silently drops a few units and by the end
nobody can reconstruct where they went.

Design points:

* **Nothing is dropped silently.** A stage calls `.include()` or `.exclude()`
  for every unit it considers, so counts always reconcile:
  `n_considered = n_included + n_excluded`.
* **Reasons are grouped**, so the report says "6 sessions excluded: no raw
  files (4), recording gap inside a behavioural block (1), ..." rather than
  listing 60 lines.
* **Reports chain.** `load()` reads an upstream stage's report so a later
  stage can state the cumulative attrition from raw data to final analysis.

Usage:

    rep = InclusionReport("detection", analysis_name="swr_v1")
    rep.include(38, "3 derivations, 608 ripples", n_derivations=3)
    rep.exclude(32, "recording gap inside a behavioural block")
    rep.write(out_dir)

@author: Svenja Kuchenhoff
"""

import os
import json
from datetime import datetime
from collections import Counter, OrderedDict

import pandas as pd


class InclusionReport:
    """Accumulates include/exclude decisions for one analysis stage."""

    def __init__(self, stage, analysis_name=None, description=None):
        self.stage = stage
        self.analysis_name = analysis_name
        self.description = description
        self.created = datetime.now().isoformat(timespec="seconds")
        self.units = []          # one record per unit considered
        self.notes = []

    # ---- recording ------------------------------------------------------
    def include(self, unit, reason="", **extra):
        self.units.append({"unit": unit, "included": True,
                           "reason": reason, **extra})

    def exclude(self, unit, reason, **extra):
        if not reason:
            raise ValueError("an exclusion must always carry a reason")
        self.units.append({"unit": unit, "included": False,
                           "reason": reason, **extra})

    def note(self, text):
        """Free-text caveat that belongs in the report but is not a decision."""
        self.notes.append(text)

    # ---- summarising ----------------------------------------------------
    @property
    def included(self):
        return [u for u in self.units if u["included"]]

    @property
    def excluded(self):
        return [u for u in self.units if not u["included"]]

    def reason_counts(self):
        return OrderedDict(Counter(u["reason"] for u in self.excluded).most_common())

    def summary(self):
        return {
            "stage": self.stage,
            "analysis_name": self.analysis_name,
            "description": self.description,
            "created": self.created,
            "n_considered": len(self.units),
            "n_included": len(self.included),
            "n_excluded": len(self.excluded),
            "included_units": [u["unit"] for u in self.included],
            "excluded_units": {u["unit"]: u["reason"] for u in self.excluded},
            "exclusion_reasons": self.reason_counts(),
            "notes": self.notes,
        }

    # ---- writing --------------------------------------------------------
    def write(self, out_dir, basename=None, verbose=True):
        """Write `<stage>_inclusion.json` and `.md`. Returns (json, md) paths."""
        os.makedirs(out_dir, exist_ok=True)
        base = basename or f"{self.stage}_inclusion"
        jp = os.path.join(out_dir, base + ".json")
        mp = os.path.join(out_dir, base + ".md")

        with open(jp, "w") as f:
            json.dump({"summary": self.summary(), "units": self.units},
                      f, indent=2, default=str)

        with open(mp, "w") as f:
            f.write(self._markdown())

        if verbose:
            print(f"\n  inclusion report: {len(self.included)}/{len(self.units)} "
                  f"included -> {mp}")
            for r, n in self.reason_counts().items():
                print(f"     excluded ({n}): {r}")
        return jp, mp

    def _markdown(self):
        s = self.summary()
        L = [f"# Inclusion report — {self.stage}", ""]
        if self.analysis_name:
            L.append(f"**Analysis:** `{self.analysis_name}`  ")
        if self.description:
            L.append(f"{self.description}  ")
        L += [f"**Generated:** {self.created}", "",
              f"**Considered {s['n_considered']} · included "
              f"{s['n_included']} · excluded {s['n_excluded']}**", ""]

        if self.excluded:
            L += ["## Why units were excluded", "",
                  "| reason | n | units |", "|---|---|---|"]
            by = {}
            for u in self.excluded:
                by.setdefault(u["reason"], []).append(str(u["unit"]))
            for r, us in sorted(by.items(), key=lambda kv: -len(kv[1])):
                shown = ", ".join(us[:12]) + (f" … (+{len(us)-12})" if len(us) > 12 else "")
                L.append(f"| {r} | {len(us)} | {shown} |")
            L.append("")

        if self.included:
            L += ["## Included", "",
                  "| unit | detail |", "|---|---|"]
            extra_keys = [k for k in (self.included[0].keys())
                          if k not in ("unit", "included", "reason")]
            for u in self.included:
                det = u.get("reason", "")
                bits = [f"{k}={u[k]}" for k in extra_keys if u.get(k) not in (None, "")]
                if bits:
                    det = (det + " · " if det else "") + ", ".join(bits)
                L.append(f"| {u['unit']} | {det} |")
            L.append("")

        if self.notes:
            L += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(L)

    # ---- chaining -------------------------------------------------------
    @staticmethod
    def load(path):
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def chain(paths, out_dir, basename="pipeline_attrition"):
        """Cumulative attrition across stages -> one table.

        Answers, in one place, "we started with N sessions and analysed M, and
        here is every unit lost along the way."
        """
        rows, prev = [], None
        for p in paths:
            if not os.path.isfile(p):
                continue
            d = InclusionReport.load(p)["summary"]
            inc = set(map(str, d["included_units"]))
            lost = sorted(prev - inc) if prev is not None else []
            rows.append({
                "stage": d["stage"],
                "considered": d["n_considered"],
                "included": d["n_included"],
                "excluded": d["n_excluded"],
                "lost_vs_previous": len(lost),
                "lost_units": ", ".join(lost) if lost else "",
                "reasons": "; ".join(f"{r} ({n})"
                                     for r, n in d["exclusion_reasons"].items()),
            })
            prev = inc
        t = pd.DataFrame(rows)
        os.makedirs(out_dir, exist_ok=True)
        t.to_csv(os.path.join(out_dir, basename + ".csv"), index=False)
        with open(os.path.join(out_dir, basename + ".md"), "w") as f:
            f.write("# Pipeline attrition\n\n")
            f.write("Units entering and leaving each stage.\n\n")
            f.write(t.to_markdown(index=False) if hasattr(t, "to_markdown")
                    else t.to_string(index=False))
            f.write("\n")
        return t
