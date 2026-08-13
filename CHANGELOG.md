# CHANGELOG

## 2026-08-09 (later) — YER micro positions reconstructed from macro probes

**Script:** `scripts/cell_to_roi_july26.py`

YER's v2026 file ships no `microwires` / `sEEG-micro` rows (202 rows, all
`Type == sEEG`, no `m`-prefixed labels, `NSxSource`/`NSxIndex` empty), so
its 26 cells were the last Baylor cells still on the pre-2026 big-table
macro position.

**The 3.15 mm constant.** Across **all 119 bundles in all 19 files that do
carry `microwires` rows**, the `microwires` position sits exactly
**3.15 mm** from the `sEEG-micro` position (min 3.15, max 3.15), always
*beyond* the probe tip along the insertion axis. Baylor is applying a
nominal Behnke-Fried protrusion, not localising wires individually.

**Validation.** Rebuilding each bundle from its macro probe as
`contact01 − 3.15 mm × unit(contact02 − contact01)` reproduces Baylor's
own supplied `microwires` coordinate to **median 0.25 mm, max 1.07 mm**
over the 113 checkable bundles (YEN excluded; its MNI152 is the corrupt
column). 112/113 within 1 mm. So this is Baylor's own construction, not
an approximation.
*(Sort key must be `Label` — `ElectrodeID` is a mixed int/str column and
sorts lexicographically, which silently picks the wrong contact.)*

YER's own MNI152 column is sound (median 0.00 mm from Fischl(MNI305),
sensible whole-head extent), so the rebuild uses it directly. The
supplied `YER_electrodes.pptx` independently confirms the probe
inventory: 6 probes labelled "microwire" — RT2cHbEb, RT2bHaEa, RF2aCa,
LT2cHbEb, LT2bHaEa, LF2Ca — matching the CSV `ProbeName`s.

Implemented as `reconstruct_micro_from_macro()`, applied only to files
with no micro rows at all, tagged
`baylor_v2026_micro_reconstructed_from_macro` with
`coord_verified = False` (inferred, not supplied).

**Result:** 26 YER cells move by median 3.15 mm (max 10.19 mm). **2 cells
change ROI** (`mLT2bHaEa03`, cells 1 & 2: HC_mid → HC_anterior; that
bundle moves 10.19 mm because the big-table macro coord was on a
different contact). No Baylor cell now uses the big table.

Cumulative vs the aug-09-2026 reference: 112/984 cells moved,
**3/984 changed `alt_final_roi`** (HC_anterior 275→276, HC_mid 232→231;
all other ROIs unchanged).

### How much does the MNI152 provenance actually matter?
Measured on the 78 cell-carrying bundles in the 18 subjects where both a
supplied MNI152 and MNI305 exist: supplied-152 vs Fischl(305→152) differ
by **median 1.79 mm, 95th pct 5.0 mm, max 9.0 mm**, and the ROI verdict
flips for **5/78 bundles = 50/608 cells (8.2 %)**. So coordinates derived
via the plain Fischl affine — YEN (broken 152, 35 cells), YER (26), and
the five affine-only files YEL/YEP/YEQ/YEU/YFT (146) = **207/669 Baylor
cells** — carry roughly a 2 mm positional and ~8 % ROI-label uncertainty
relative to a proper MNI152 normalisation.

## 2026-08-09 — Baylor v2026 electrode tables for YEL / YEN / YFT added to the cell→ROI pipeline

**Script:** `scripts/cell_to_roi_july26.py`
**Output:** `data/ephys_humans/derivatives/neurons_with_ROI_labels.csv`
**Reference (previous run):** `data/ephys_humans/derivatives/old_electrode_tables/aug-09-2026/neurons_with_ROI_labels.csv`
**Per-cell diff:** `data/ephys_humans/derivatives/ROI_assignment/cells_step8_change_vs_previous_run.csv`

### Source data
New `-electrodes_v2026.csv` files placed at the top level of
`data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026/`: **YEL, YEN, YFT**
(byte-identical re-sends of YEU and YFI arrived as `...[37].csv` /
`...[94].csv` and are ignored by the loader's `-electrodes_v2026.csv`
suffix filter — no content change). **YER** still contains no
`microwires` / `sEEG-micro` rows at all, so its 26 cells remain on
`baylor_bigtable_pre2026_macro_position`.

### Code changes
- `load_baylor_v2026` now reads micro-bundle rows of **either** `Type ==
  "microwires"` **or** `Type == "sEEG-micro"` (new helper
  `_micro_bundle_rows`). `microwires` (the bundle itself, label
  `mLT2bHb01`) wins; `sEEG-micro` (contact 01 of the carrying macro
  probe, label `LT2bHb01`, ~3 mm shallower) only fills bundles with no
  `microwires` row. Both are keyed on the m-prefixed bundle name.
  *In the current data this fallback never fires* — every bundle that
  has an `sEEG-micro` row also has a `microwires` row, so all coordinates
  still come from `microwires`.
- Fixed a pre-existing ordering bug: `alt_final_roi` was consumed by the
  step-7 RSA-ready plot but only assigned at the very bottom of the
  script (`KeyError`). It is now assigned in step 7 where
  `analysis_rois` is computed.
- New **step 8**: change report vs `REFERENCE_TABLE` (an archived copy of
  this script's previous output), printing coordinate shifts per subject,
  cells shifted > 10 mm, `alt_final_roi` counts old vs new, and the
  transition matrix.

### Result
984 cells, unchanged row count. 86 cells moved coordinate; all of them in
YEL (17), YEN (35), YFT (34).

| provenance | before | after |
|---|---|---|
| `baylor_v2026_bundle_micro` | 557 | 608 |
| `baylor_v2026_bundle_305to152_unreliable_file` | 0 | 35 |
| `baylor_bigtable_pre2026_macro_position` | 112 | 26 |

YEN's `MNI152_*` columns are internally inconsistent with its `MNI305_*`
columns (mean 55.6 mm, max 71.2 mm disagreement under the Fischl
305→152 transform), so the existing reliability gate correctly rejects
them and uses `MNI305 → 152` instead (`..._305to152_unreliable_file`).

**Coordinate sanity:** 85 / 86 moved cells shifted **3.13–3.15 mm** —
exactly the known macro-last-contact → micro-bundle offset, i.e. a small
local correction, not a relocation. The single exception is
`BY2-YEN`, electrode `mLT2bHb07`, cell idx 2, which moved 60.6 mm:
its big-table coordinate was `(32.3, -19.8, -17.9)` — the **right**-
hemisphere `mRT2bHb` coordinate — despite an `mL...` (left) electrode
label. The label-driven v2026 lookup places it at `(-28.1, -24.2, -14.9)`
together with its seven `mLT2bHb` siblings. This is a big-table
data-entry error being corrected, not a localisation change.

**ROI changes (`alt_final_roi`, the column read by
`mc.analyse.roi_relabel.relabel_per_cell`):** exactly **1 of 984** cells
changed label — the `mLT2bHb07` cell above, `HC_anterior → HC_mid`.

| ROI | before | after | Δ |
|---|---|---|---|
| HC_anterior | 275 | 274 | −1 |
| HC_mid | 232 | 233 | +1 |
| mOFC | 163 | 163 | 0 |
| mPFC | 155 | 155 | 0 |
| PCC | 61 | 61 | 0 |
| EC | 38 | 38 | 0 |
| (NaN / excluded) | 60 | 60 | 0 |

Downstream RSA results are therefore essentially unaffected; the value of
the update is 86 cells now sitting on verified micro-bundle coordinates
rather than inferred macro-contact positions.

---

## 2026-08-13 — Cell ↔ fMRI future-lag gradient: extensive exploration (mostly null)

**Question:** do human mPFC single units recapitulate the fMRI DSR "preferred
future angle" gradient — i.e. if we average cells, does anatomical position
predict their preferred spatial-tuning lag the way the fMRI angle map does?

**Scripts (all read `per_cell_ALL_ROIs.csv`, mPFC, 155 cells / 32 subjects):**
`cell_gradient_master_table.py` (master per-cell + group tables),
`gradient_brain_cells_by_lag.py` (MNE medial surfaces, cells coloured by lag),
`cell_gradient_principal_curve.py` (bent-axis sliding window + shift test),
`cell_gradient_split_table.py`, `cell_gradient_split_permgated.py`,
`cell_gradient_full_factorial.py` (320-row robustness grid).
Outputs under `data/ephys_humans/derivatives/group/cell_gradient_master/2026-08-13_09-45-28/`.

**Methods settled on:** pool the 12-lag profiles within a cell group, then read
the preferred lag off the pooled profile (argmax); per-cell argmax and the
continuous first-harmonic angle are both too noisy (harmonic vector length
≈ 0.07; per-cell harmonic angle ~ uniform). fMRI angle sampled at each cell
via symmetrised + 3 mm cos/sin smoothing + 6 mm sphere (quarters map).
Anatomical axis = PC1 of the gradient-mask voxels (folded x) = essentially
dorsoventral (loads [-0.04, -0.41, 0.91], r = 0.98 with MNI z).

**NULL / dead ends (do not re-run):**
- **Continuous gradient does not exist.** Spearman(future-score, arc-length on a
  bent principal curve) = +0.058, circular-shift p = 0.25, subject-bootstrap
  95% CI [-0.10, +0.19]. In the full factorial, 31/32 correlation configs are
  n.s. (r ≈ 0.0–0.17); ctrl-mode correlations ≈ 0 or negative.
- **Continuous first-harmonic angle** per cell is unusable (near-uniform;
  group bootstrap CIs span the whole circle).
- **Controlled tuning (`_ctrl` columns) shows nothing** reproducible — binned
  pooled lags are chaotic across weighting/gating (confirms prior expectation).
- **Subject-first vs cell averaging** does not stabilise; it only *diverges from*
  cell-weighting where the pooled profile is flat (peak r ≲ 0.05), i.e.
  divergence is a noise flag, not a fixable choice.
- **Perm-gating** (only cells significant at a lag contribute to that lag) does
  not sharpen results; it thins data (median 2–7 cells/lag) and only confirms
  the already-robust groups.
- A real **240° "backward" signal** sits mid-axis (pc1-Q2 / z-middle,
  peak r 0.085–0.10) — genuine, and it blocks any monotone ventral→dorsal ramp.

**The one robust positive:** under **noctrl**, the **future-end bin of the
gradient axis** pools to **60°**, invariant to cell-vs-subject weighting AND to
perm-gating: `all/pc1/half:end`, `all/pc1/quartile:Q3`, and `all/z/quartile:Q4`
all give 60/60/60/60, matching the local fMRI angle (~63°, err 3–9°; pooled
peak r ≈ 0.08–0.16). Interpretation is **local, not a gradient**: "cells in the
deep-future (dorsal/high-z) end of the DSR gradient prefer 60°, matching fMRI
there," NOT "a cell gradient mirrors the fMRI gradient."

**One nominally-significant correlation (EXPLORATORY — treat with caution):**
`noctrl / in_mask / z / perm-gated (≥1 sig lag) / cell-weighted`:
Spearman(future-score, MNI z) = **0.238, shift-p = 0.032, n = 54**. This is
1 hit in 32 tests and does **not survive subject-weighting** (r = 0.028,
p = 0.45), so it is not corrected-significant. Splitting those 54 cells at
median z: low-z (n=27) argmax 240°, future-score +0.00; high-z (n=27) argmax
**60°**, peak r 0.159, future-score +0.087 — the same "high-z → 60°" story.
