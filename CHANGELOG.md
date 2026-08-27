# CHANGELOG

## 2026-08-27 — SVC max-t + LOSO over all 27 maps of `instr_test_full`, TR5 only

**Scripts:** `scripts/svc_loso_batch.py` (new), `scripts/svc_loso_test.py` (patched
to resolve `.nii`/`.nii.gz`).
**Output:** `data/derivatives/group/per_TR_svc_instr_test_full_TR5_2026-08-27/`
(`settings.json`, `summary_table.csv`, per-mask/per-model `_svc_summary.json`,
`_loso_results.json`, `_loso_k{K}.npy`, `run.log`).

**Test.** Identical to the reported instruction-phase test — one-sample t over
32 subjects, sign-flip max-t permutation (10 000 perms, seed 0) corrected over
all voxels in an a-priori mask, plus the LOSO cross-validated readout at
k = 50/100/200. `tstat`, `null_max_t` and the LOSO selection are imported from
`svc_loso_test.py`, so empirical and permutation values come from the same code.

**Masks (3).**
- `mPFC` = `masks/mask_PFC_LR_smoothed_resampled.nii.gz` (BA32/mBA9/mBA10), 4182 vox
- `MTL` = `masks/Garvert_MTL_2mm.nii.gz` (HC/EC), 2695 vox
- `visual` = `masks/visual_occipital_HO25_2mm.nii.gz` (NEW: union of the 8
  occipital Harvard-Oxford cortical labels at thr25 — LOC sup/inf,
  Intracalcarine, Cuneal, Lingual, Occipital Fusiform, Supracalcarine,
  Occipital Pole), 12 497 vox in-brain of 27 220 in the atlas. The 7T FOV
  truncates the occipital pole (3 % covered) and occipital fusiform (1 %), so
  this mask is effectively dorsal/medial occipital + LOC-superior.

**BLOCKER — only TR5 could be analysed.** Of the 12
`group_RSA_instr_test_full_glmbase_01-TR{n}_cropped` folders, only TR5 finished
downloading. All 26 beta maps in each of the other 11 folders fail `gzip -t`,
truncated at exact 256 KB boundaries (6–10 MB of an expected ~28 MB) — an
interrupted transfer, 275 corrupt files. So this run is `--trs 5`: max-t is
corrected over voxels ONLY, not over voxels x TRs, and there is no timecourse.
Re-run across all 12 TRs once the transfer completes.

**Two maps are entirely zero** and were dropped from the tables (25 of 27
remain): `REWDSR-rewDSR_noInstr` and `SIMPLE-rewDSR_noInstr` — i.e. the whole
`rewDSR_noInstr` combo model in `condition_files/rsa_instruction_full.json`
wrote all-zero output. `splitDSR_noInstr` is fine, so it is specific to that
combo. Upstream bug, not a bug in this test.

**Multiple comparisons.** By request, no correction across the family:
each p_FWE is corrected within its own mask only. 25 maps x 3 masks = 75 tests.

**Result — the `_instr` (visual instruction similarity) models dominate
everywhere.** All p_FWE below are within-mask, one-sided positive; LOSO p at
k=100.

| mask | top map | peak t | MNI | p_FWE | LOSO t | LOSO p |
|------|---------|--------|-----|-------|--------|--------|
| mPFC   | curr_rew_instr | 4.79 | -4/38/8 | **.0037** | 3.26 | .0010 |
| mPFC   | CURR_REW_INSTR-splitDSR_vs_instr | 4.70 | -8/42/10 | **.0051** | 3.02 | .0026 |
| mPFC   | rewDSR_instr (= REWDSR_INSTR-rewDSR_vs_instr) | 4.46 | 4/20/34 | **.0107** | 2.26 | .0151 |
| mPFC   | two_next_rew_instr | 4.28 | 4/48/36 | **.0208** | 3.56 | .0007 |
| MTL    | rewDSR_instr | 8.30 | -32/-2/-34 | **<.0001** | 7.18 | <.0001 |
| MTL    | two_next_rew_instr | 6.26 | -30/-24/-22 | **.0002** | 4.87 | <.0001 |
| MTL    | curr_rew_instr | 6.11 | -32/-22/-24 | **.0004** | 4.60 | <.0001 |
| MTL    | three_next_rew_instr | 5.75 | -24/-8/-24 | **.0006** | 4.04 | .0001 |
| MTL    | next_rew_instr | 5.32 | -22/-10/-28 | **.0023** | 4.53 | <.0001 |
| visual | curr_rew_instr | 6.06 | -10/-90/36 | **.0003** | 3.70 | <.0001 |
| visual | rewDSR_instr | 6.16 | 24/-84/26 | **.0006** | 3.82 | <.0001 |
| visual | two_next_rew_instr | 5.45 | -44/-62/2 | **.0030** | 3.81 | .0002 |

**Not significant anywhere:** every execution-similarity model on its own —
`rewDSR` (mPFC p = .74, MTL .86, visual .49), `simple` (.96 / .99 / .99),
`curr_rew`, `next_rew`, `two_next_rew`, `three_next_rew`, and all four
`*-splitDSR_noInstr` regressors. In the `*_vs_instr` combos the execution
regressor is likewise null while its `_INSTR` partner carries the effect —
i.e. at TR5 the instruction-similarity regressor explains the variance and
leaves nothing for the execution regressor.

**Caveat on the effect sizes.** MTL t = 8.3 and visual t = 6.2 for
`rewDSR_instr` are far above anything the execution models produce, and the
`_instr` models are uniform within (task_i, task_j) 2x2 sub-blocks by
construction. A block-structured regressor of that kind can be picked up by any
residual block structure in the data RDM (e.g. run/session or scanner-drift
structure aligned with task identity), so these should be read as "the
instruction-block model fits" and not yet as evidence about representation.
Worth a control before interpreting.

**Also recorded, descriptive only:** `peak_t_neg` / `p_FWE_neg` — the negative
peak against the same (symmetric) sign-flip null, a second one-sided test not
corrected for testing both directions. 11 of 75 cells have p_neg < .05, the
strongest being `NEXT_REW-splitDSR_noInstr` in MTL (t = -5.02, 32/0/-20,
p = .0035). At an uncorrected .05 with 75 tests, ~3.75 are expected by chance.

**Method note:** the group brain mask here is the intersection of
`mask_all_32_subjects` over all included TRs, not TR0's mask alone as in
`svc_loso_test.load_ref` (the per-TR group masks differ by ~21 voxels; a voxel
entering the max-t search must be valid at every TR the search runs over).

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

## 2026-08-26 — SWR pipeline rewrite to Chen/Staresina standard: session audit (Milestone 0)

**Scripts:** `mc/analyse/swr_io.py` (new), `scripts/swr_audit_sessions.py` (new).
Outputs under `data/ephys_humans/derivatives/group/swr/`
(`session_manifest.csv`, `session_blocks.csv`, `session_manifest.json`, `settings.json`).

Motivation: comparison of the existing ripple pipeline against Chen, Staresina et al.
2025 (J Neurosci 45:e1502252025) found seven blocking defects. The most consequential
is that `identify_HPC_ripples.py:122` estimates the detection threshold *within each
cropped snippet*, which partially normalises ripple rate to be constant per snippet and
so suppresses exactly the between-window rate difference the planning hypothesis is
about. Also absent entirely: artifact/IED rejection, a line-noise notch, spectral
validation of candidate events. Full rationale in the plan file.

**Two loaders are currently broken and cannot have run.** `all_trial_times_{XX}.csv`
has **14 columns**, but `scripts/identify_HPC_ripples.py:57` and
`scripts/preprocess_LFP.py:128` assign 13 names, so `df.columns = column_names` raises
`ValueError: Length mismatch` on every session. The correct 14-name contract already
existed at `scripts/behaviour_summary.py:47` and `mc/analyse/helpers_human_cells.py:379`
— the 14th column is `correct`. `swr_io.BEH_COLS` now mirrors it.

**`correct` is per-repeat accuracy, NOT a "plan known" state.** Verified on s05: grids 2
and 9 contain errors *after* the first correct solve. The planning boundary is therefore
derived as a cumulative max within grid (`swr_io.load_behaviour` adds `plan_known`),
which is cleaner than the `found_first_D` heuristic at
`mc/analyse/ripple_helpers.py:69-72`, but the raw column must not be used as a
per-repeat planning state.

**The behavioural clock is continuous across recording blocks; the LFP files are not.**
**25 of 60 sessions are multi-block.** Block *k+1* always continues the block-*k* clock,
separated by a real recording gap: measured range **+7.3 s (s33) to +2910.1 s (s21, a
48-minute break)**; s18 and s27 have three blocks (+229.6 s, +148.8 s). This falsifies
the assumption at `scripts/preprocess_LFP.py:213`, which maps behaviour into block 2 by
subtracting the *file duration* — valid only if recording never stopped. Block offsets
must be estimated per session against three independent references and hard-validated
(every behavioural event inside `[0, duration_k]` with >=5 s margin, else the block
emits no ripples). This is now the highest-risk item in the build; a 20 s error
misassigns every block-2 ripple systematically, which would read as a null rather than
as noise.

**Subject clustering needs a normalised key.** 63 sessions map to 43 distinct
`Subject Label` values but fewer real subjects: s29 is `'UT1-202314'` and s30 is
`'UT202314'` — the same patient in two of the four Utah label formats. 16 labels span
more than one session (`BY2-YEK` = s07/08/09; `BY2-YEX` = s43/44/49). Clustering
robust SEs on the raw label is anticonservative. `swr_io.normalise_subject_key` collapses
them and the audit prints the full map for manual sign-off.

**Audit result (run locally, 60 sessions):** 24 `ok`, 4 `needs_review`, 32
`no_raw_files`. Sites: baylor 36, utah 18, ucla 6. The 32 are simply not on the laptop
— **the audit must be re-run on ceph**, where those sessions' config defects will
surface. The 4 genuine structural defects to resolve by hand: **s03** (UCLA, 2 `.ncs`
recording blocks not represented in the YAML at all), **s18** and **s28** (duplicate
block names in `blocks`), **s32** (yaml=2 blocks, behaviour=1, files=2).

**Config YAML is demoted to a hint.** It disagrees with behaviour and with disk for a
large minority of sessions (`segment: null` for 13; no `blocks` key for s57/58/59;
duplicate block names; block counts that do not match). `session_manifest.csv` is now
the authority for everything downstream.

**NULL / dead ends (do not re-run):**
- Do not use `scipy.signal.resample` on continuous traces — it is FFT-based and wraps
  the end of the recording into the beginning. Silent, and visible only at the two ends.
  Use `resample_poly`. This is also the source of the per-snippet edge ringing in the
  old pipeline.
- Do not downsample to 500 Hz (`preprocess_LFP.py:30-32`). It makes Chen's >250 Hz RMS
  artifact criterion and the 120-200 Hz spectral rejection impossible. Keep 1000 Hz.
- Do not store full stepwise TFR power. The existing `ripple_power_dict_s05` is **5 GB**
  and is not needed by anything.

## 2026-08-26 (later) — Block structure resolved from the raw data, not the config

**Script:** `scripts/swr_diagnose_blocks.py` (new). Reads every neo segment's duration
and channel count out of the raw files and matches them against behavioural block spans.
Written because excluding sessions on a config warning discards data that the recordings
themselves can disambiguate.

**s18 and s28 are fine — the "duplicate block names" warning was a false alarm.** The
duplicates are *filename labels*, not duplicate recordings. Matching by duration is
unambiguous:

| session | behavioural block span | matched recording | slack |
|---|---|---|---|
| s18 | 363.3 s / 4006.1 s / 3432.4 s | EMU-058 seg1 / EMU-059 seg1 / EMU-060 seg0 | +197.0 / +33.9 / +226.4 s |
| s28 | 1819.7 s / 654.6 s / 927.4 s | EMU-045 seg1 / EMU-047 seg1 / EMU-048 seg1 | +41.6 / +58.7 / +36.1 s |

Ordering by **EMU number** (the acquisition counter) is the reliable chronological key;
`blk-NN` labels are not. Note each Baylor file carries a ~2.3 s stub segment plus the
real recording; picking the longest segment per file independently reproduces the YAML
`segment` field for both sessions, so that field is trustworthy where it is populated.

**s32 has a real 444.8 s hole in the middle of a single behavioural block.** Behaviour is
one continuous block (101.8 -> 3883.0 s, 304 repeats). The recording is two files with
**non-zero `t_start` on a common amplifier clock**: 12042.7 s + 1977.5 s, then 14465.0 s
+ 1516.1 s. Recorded wall span 3938.4 s brackets the behavioural span 3781.2 s, but
~445 s of task has no LFP. Usable if the repeats falling in the hole are dropped.

**`t_start` is the amplifier wall clock and gives exact inter-block offsets where it is
populated** (s32: yes; s18/s28: zero). This is the "header wall-clock" offset estimator,
and it is the principled replacement for the file-duration subtraction at
`scripts/preprocess_LFP.py:213`. It is also what the `t_start > 10000` hack at
`preprocess_LFP.py:76-83` was groping at.

**s03 (UCLA) is a SINGLE-block session and is usable.** 164 of its 308 `.ncs` files are
**header-only 16384-byte stubs** with zero data records ("TimeClosed File was not closed
properly") — an aborted first recording. All real data is in the `_0001.ncs` files, which
include both macros (`LMH*`, `RMH*`, `LA*`, `LAC*`, `ROF*`, `RPH*`) and micros (`GA*-`).
Counting the stubs as a second block is what made the YAML look inconsistent.
`swr_io.discover_raw_files` now skips any `.ncs` of exactly the header size.

**UCLA sampling rate in the YAML is wrong.** `config_human_ABCD_iEEG.yaml` declares
`sampling_rate: 1000` for all six UCLA sessions. The s03 `.ncs` headers report
**macros at 2000 Hz** (micros at 32000 Hz). Read the rate from the file header, never
from the config.

**neo cannot read these `.ncs` files.** `NeuralynxRawIO.parse_header()` raises
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` at
`neuralynxrawio.py:505` because `global_t_start`/`global_t_stop` are None. A direct
header/record reader (16 KB header, 1044-byte records of
`uint64 ts, uint32 ch, uint32 fs, uint32 n_valid, int16[512]`) works and additionally
exposes the Unix-epoch record timestamps needed for block alignment. UCLA support must
not depend on neo.

**Audit after these fixes: 26 `ok`, 2 `needs_review` (s03 informational, s32 the 445 s
hole), 32 `no_raw_files` (not on the laptop; re-run on ceph).** s18 and s28 moved to `ok`.

**NULL / dead ends (do not re-run):**
- Do not order Baylor blocks by the `blk-NN` label or by the YAML `blocks` list. Both
  contain duplicates. Order by EMU number.
- Do not use `neo.io.NeuralynxIO` / `NeuralynxRawIO` on the UCLA data in this env; it
  raises before returning anything. Parse the `.ncs` header directly.
- Do not trust `sampling_rate` in the YAML for UCLA (says 1000, actually 2000).

## 2026-08-26 (later still) — Anatomy loaders extracted to shared modules (Milestone 1)

**New:** `mc/analyse/anatomy_sources.py` (coordinate sources, all three sites),
`mc/analyse/anatomy_atlas.py` (MaxProbAtlas + the ROI rule ladder).
**Changed:** `scripts/cell_to_roi_july26.py` now imports them; 2514 -> 1880 lines.
**Unchanged:** every output of the cell pipeline.

Rationale: the LFP/ripple pipeline needs to pick hippocampal and mPFC *macro*
contacts by anatomy rather than by string matching (the old
`'H' in channel and 'T' in channel`, and `'1' in label` which also matches
HIP10/11/14). Rather than duplicating the rules, both pipelines now call one
implementation — so "the same anatomical criteria defined HC for cells and for LFP
contacts" is literally true, and a future change to `assign_atlas_roi` propagates to
both instead of silently diverging.

The move was verbatim: `sed`-extracted line ranges 221-617 (loaders) and 982-1029 +
1040-1245 (atlas layer), not retyped. Two structural edits only:
- `discover_utah_mats()` took two module globals of the script; it now takes them as
  keyword arguments defaulted to the same literal paths, so the bare call still works.
- The four atlas objects were built by `nldatasets.fetch_*` at import time; they are
  now built by `get_atlases()`, which populates the same module-level names.
  `assign_atlas_roi` still reads the module globals `juelich` and `HC_ANT_MID_Y`, so
  its body is byte-identical.

**Regression gate — PASSED.** Procedure: (1) confirm the *unmodified* script is
deterministic by re-running it and checking it reproduces its own output
(md5 `e0e758a303831cfc614a2490dcaf6aac`) — it does, so a post-refactor diff cannot be
blamed on pre-existing nondeterminism; (2) refactor; (3) re-run.

`derivatives/neurons_with_ROI_labels.csv` is **byte-identical** after the refactor, as
are all nine `ROI_assignment/cells_step*.csv` intermediates. Baseline copies kept at
`/tmp/swr_refactor_baseline/` for this session.

Note the gate must be against a *fresh run of the current script*, not against
`old_electrode_tables/aug-09-2026/` — that archive legitimately differs
(md5 `9ebfceda...`, 205394 bytes vs 204383) because new electrode files arrived since.

**NULL / dead ends (do not re-run):**
- A static AST check (names used by the script that are defined in the new modules but
  not imported) caught `_bundle_key`, which a syntax check cannot: `py_compile` passes
  on an undefined global. Use the AST check, not `py_compile`, when moving definitions
  out of a flat script.
- The new modules are deliberately NOT added to `mc/analyse/__init__.py`, which imports
  eagerly — adding nilearn/neo-heavy modules there would make every `import mc` in the
  repo pay for them. Import explicitly.

## 2026-08-26 (Milestone 2) — Macro-contact anatomy: hippocampal LFP contacts selected by anatomy, not string matching

**New:** `mc/analyse/contact_anatomy.py`, `scripts/swr_build_contacts.py`.
**Outputs:** `derivatives/s{XX}/LFP/macro_contacts_{XX}.csv` + `bipolar_pairs_{XX}.csv`
per session; `derivatives/group/swr/macro_contacts_all.csv`, `contact_qc.csv`,
`settings.json`.

Replaces contact selection by string matching — the old `'H' in channel and 'T' in
channel` (matches any label containing both letters) and `'1' in label` (also matches
HIP10/11/14) — and the hard-coded contact01−contact04 bipolar pair spanning ~15 mm.

**Result: 160 hippocampal contacts / 164 hippocampal bipolar pairs across 27 sessions**
(baylor 94, utah 60, ucla 6). The remaining 33 sessions have no raw data on the laptop;
re-run on ceph.

### Utah `ElecMapRaw` indexing — a trap that would not have raised
`ElecMapRaw` is an (n, 3) object array `[label, amplifier_channel, other_channel]`.
Coordinates must be read by the **direct row index** into `ElecXYZMNIRaw` /
`ElecAtlasRaw` / `ElecTypeRaw` — **not** by indexing with the channel number.
Verified on s02: row 41 is `bRAHIP2`, and `ElecAtlasRaw[41]` = "Right Hippocampus",
whereas indexing by column 1 gives "Right fusiform gyrus" — a different electrode
entirely, silently. Column 1 is the amplifier channel and matches
`utah_elec_labels_{XX}.csv` for all 114 rows where it is finite (the other 30 are
unlocalised and are dropped with a logged reason).

Also: `m*` = microwire, `b*` = the Behnke-Fried macro on the same shaft, bare = plain
sEEG depth. Macros use `ElecXYZMNIRaw`, not `ElecXYZMNIProj` — gray-matter projection
is right for a spiking micro tip and wrong for a 2 mm macro ring that legitimately
straddles the GM/WM boundary. Both are stored.

### ROI definition: native-space primary, shared atlas ladder as cross-check
Measured on s05/s02/s03, `anatomy_atlas.assign_atlas_roi` has **100% recall but 22%
precision** against the subjects' own segmentations: it calls 49 contacts hippocampal
where the native labels say 11, the excess being amygdala (`RAMG1-4`), lingual gyrus
(`RPHIP1-2`), VentralDC (`RT1cCM04`) and parahippocampal cortex. Expected — that ladder
was tuned for microwire *tips*, where ±2–3 mm neighbourhood rescue is desirable, and is
applied here to MNI152, where MTL registration error is large.

So a macro contact's ROI comes from the **subject's own segmentation**, matching Chen
et al., who identified hippocampal contacts "via visual inspection of postoperative
T1-weighted anatomical MRI scans". The shared ladder is still computed and stored as a
cross-check; because it is a strict superset, only the reverse disagreement is
meaningful. The cell pipeline is unaffected — `assign_atlas_roi` is untouched.

**Use the 3 mm-neighbourhood parcellation, not the single-voxel one.** For Baylor,
`ROI_DK2005_3mm` / `Matter_3mm`, not `Area_fs_vox` / `Matter_fs_vox`. A macro contact is
a ~2 mm ring recording from a volume, so the label must be sampled at that scale. On
YEJ the voxel column calls 1 contact hippocampal and the 3 mm column calls 4 — e.g.
`RT2bHaEa02` is `Right-Hippocampus` at 3 mm and `Right-Cerebral-White-Matter` at the
voxel. Voxel columns retained as `native_region_vox` / `matter_vox`.

**Pair rule:** a bipolar pair is hippocampal if **at least one** of its two contacts is
natively hippocampal — the standard montage (Chen take the most medial hippocampal
contact and its immediate neighbour, frequently white matter). Requiring both would
have given s05 zero pairs.

### Gate — PASSED (5/5)
1. **Probe names plausible.** baylor `RT2bHaEa02`, `LT2HbE02` (Ha/Hb); utah `LAHC1`,
   `bRAHC1`, `LPHC1`; ucla `LMH-1`, `LPH-1`, `RMH-1`.
2. **Atlas concordance 158/160.** The 2 exceptions are `LT2aA01` in s31 and s35 (same
   subject YFF): native says Left-Hippocampus, atlas says EC — an amygdala-probe
   contact 01 sitting on the HC/EC border. Borderline, not a defect.
3. **UCLA independent ground truth: 6/6.** Every UCLA HPC contact carries an ASHS
   hippocampal subfield (DG, CA1, SUB, CA2), and **0 contacts with an ASHS subfield
   were missed**. ASHS is a completely independent segmentation — this is the strongest
   validation available.
4. **Anterior/mid split clean.** HC_anterior y ∈ [−20.8, −3.8]; HC_mid y ∈ [−34.4,
   −21.4]. No overlap at HC_ANT_MID_Y = −21.0.
5. **Bipolar spans sane.** Median 4.68 mm (range 3.08–7.75), i.e. genuine adjacent sEEG
   spacing, versus ~15 mm for the old 01−04 pairing.

**Open point for Milestone 3/4:** adjacent pairs share contacts — pair (2,3) and pair
(3,4) both contain contact 3 — so the 164 pairs are not independent (median 5 per
session, max 11; Chen used ~2 per participant, one per probe). Decide before detection
whether to keep all pairs, or one anchor pair per probe.

**NULL / dead ends (do not re-run):**
- Do not index Utah coordinates via `ElecMapRaw` column 1. It returns another
  electrode's position and raises nothing.
- Do not use `Area_fs_vox` / `Matter_fs_vox` for macro contacts — single-voxel labels
  understate hippocampal coverage ~4x.
- Do not merge anatomy to channels on a key that can be NaN: pandas matches NaN to NaN,
  and s02 fanned 132 channels out to 248 rows before the null-key guard.
- Do not use the atlas ladder alone to select macro contacts (22% precision).

## 2026-08-26 (CORRECTION) — Block alignment is deterministic; the earlier "highest risk" entry was wrong

**This corrects the 2026-08-26 entry above**, which claimed the behavioural clock
includes recording gaps the LFP files do not, and that
`scripts/preprocess_LFP.py:213` (mapping block *k* by subtracting cumulative file
duration) was therefore "badly wrong". **That claim was mistaken. The old approach is
correct.** Verified rather than assumed, via `/tmp/blockfit.py`.

**The behavioural clock is cumulative file duration.** For block *k*,
`offset_k = sum(durations of files 0..k-1)`, and behavioural times map into file *k* as
`t - offset_k`. Tested on all **14 multi-block sessions with local raw data**: every
behavioural event lands inside its file, with head/tail margins of seconds.

s18 worked out exactly: file durations 560.3 / 4040.0 / 3658.8 s give offsets 0 /
560.3 / 4600.3; behavioural block 2 (580.8–4586.9 s) maps to in-file 20.5–4026.6 s
against a 4040.0 s file. The apparent 180.1 s "behavioural gap" I reported earlier is
just the 159.6 s tail of file 1 plus the 20.5 s head of file 2 — **not** a recording gap.

**Wall-clock timestamps are irrelevant, and would have been actively misleading.** The
NSx 2.2/2.3 basic header does carry a `TimeOrigin` SYSTEMTIME at byte offset 294 (neo
reports `rec_datetime = None`, but it parses directly). For s18 those wall clocks are
1943.4 s and 12459.2 s apart, versus file durations of 560.3 and 4040.0 s — i.e. real
elapsed gaps of 20–200 minutes between recordings. Had I anchored blocks to wall clock,
every block-2 ripple would have been misplaced by tens of minutes. The behaviour was
timestamped against the *concatenated* recording, not against wall time.

Two NSx variants exist: `NEURALCD` (2.2/2.3, has TimeOrigin) and `BRSMPGRP` (s32, s33 —
different layout, no TimeOrigin at that offset; neo reports a non-zero `t_start`
instead). Neither is needed for alignment.

**Three sessions overrun by 1–2 s, each by exactly one repeat.** s09 block 2 (−2.0 s),
s10 block 2 (−1.8 s), s33 block 1 (−0.9 s) — 1 repeat of 275, 274 and 296 respectively.
The amplifier was stopped a second or two before the last trial completed. Handle by
dropping the overrunning repeat and logging it; this is not a misalignment (a real one
would be off by tens or hundreds of seconds, not by one trial).

**Consequence for the plan:** block offsets need no three-way estimation and no
wall-clock anchoring. The rule is `cumsum(file durations)`, with a hard validation that
every behavioural event lands inside its file (margin ≥ 0, allowing a single trailing
repeat to be dropped). This removes what I had flagged as the highest-risk item.

**NULL / dead ends (do not re-run):**
- Do not align blocks by NSx `TimeOrigin` / wall clock. It is present and parseable but
  describes real elapsed time between recordings, which the behavioural clock excludes.
  Using it would misplace block 2+ by tens of minutes.
- Do not treat the behavioural inter-block "gap" as a recording gap. It is the tail of
  one file plus the head of the next.

## 2026-08-26 (Milestone 2, CORRECTION) — One bipolar pair per probe, not every adjacent pair

**Corrects the pair counts in the Milestone 2 entry above (164 pairs).** The final
number is **76 pairs across 27 sessions / 18 subjects**.

`build_bipolar_pairs` was generating *every* adjacent pair containing a hippocampal
contact. That over-generated (median 5 per session, max 11) and made the derivations
non-independent: contact 3 appeared in both pair (2,3) and pair (3,4), which the GLM
cannot account for.

Corrected to follow Chen et al. exactly — *"bipolar referencing was performed using the
most medial hippocampal contact and its immediate neighbour (i.e. the second-most
medial contact) **on each hippocampal probe**"* — i.e. **one derivation per probe**, so
no contact is ever reused. Chen report 34 contacts across 17 patients (~2 each); we get
76 across 27 sessions (~2.8 each).

**Anchor = most medial hippocampal contact on the probe, defined geometrically as
min |MNI x|**, not by contact number. Numbering conventions differ between sites
(Baylor contact 01 is deepest; Utah/UCLA differ), so a geometric definition of "medial"
is convention-free.

**Reference rule = immediate neighbour** (`scheme='neighbour'`, the default). Measured
alternatives on the full contact table:

| scheme | pairs | sessions | median span | ref in MTL grey | probes dropped |
|---|---|---|---|---|---|
| `neighbour` (Chen) | 76 | 27 | 5.28 mm | 49/76 | 0 |
| `white_matter` | 42 | 23 | 10.15 mm | 0/42 | 34 (no WM contact on probe) |

The white-matter montage gives a cleaner subtraction (no ripple signal in the
reference) but doubles the inter-contact distance, which enlarges the lead field,
degrades spatial specificity, and would *increase* volume-conduction confounds for the
H2 HC–mPFC analysis — the opposite of what that analysis needs. It also drops 34 probes
and 4 sessions. `scheme='white_matter'` remains available for a sensitivity analysis.

The 49/76 references that are themselves MTL grey are expected and are what Chen's own
montage produces (the second-most medial contact on a hippocampal probe is usually
still hippocampus). Ripples are spatially local at the millimetre scale, so adjacent
contacts 5 mm apart do not see identical events and the bipolar preserves them. If
common-mode cancellation turns out to matter, it will show up at the Milestone 4
checkpoint as reduced ripple amplitude or rate.

**NULL / dead ends (do not re-run):**
- Do not generate all adjacent pairs. It inflates n several-fold with non-independent
  derivations sharing contacts.
- Do not pick the anchor by contact number; numbering conventions differ per site.
