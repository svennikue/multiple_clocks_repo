# Co-author comment triage — Concurrent_future_in_human_mPFC

Source: `data/derivatives/group/manuscript_comments_2026-08-29.md` (113 threads, 97 open,
16 resolved). Indices in [brackets] are that file's thread numbers.

Split by **what the work actually is**:

| bucket | what it takes | n (open) |
|---|---|---|
| **A** | rephrase / cite / relabel — no new numbers | 72 |
| **B** | explain deeper + verify against the code what I actually did | 19 |
| **C** | new or re-run analysis | 9 distinct (mostly spawned by B) |
| — | praise, no action | 6 |

Some threads appear in two buckets (e.g. [107] is a B that spawns a C).

---

## Four things I checked in the code before triaging

These change what bucket some comments belong in.

### 1. The DSR model IS slot-specific. [107], [85], [109]
`RSA_DSR_ROIs_simple.build_mode_path_dsr` takes the modal trajectory, and for each bin
**rolls the flattened location vector left so the current bin sits at the front**;
similarity is Hamming on *position-matched* entries. Two timepoints are similar only when
the same location sits at the **same lag**. So Kris's and Will's fear that you used a flat
"set of future locations" / SR-like model is unfounded — this is a writing failure, not a
modelling one. Same construction in `create_fMRI_model_RDMs_on_clean_beh.py` (`EVs['DSR']`).

### 2. The rodent n in the manuscript is wrong. [44]
`analysis_rodents_complete_clean.py` L84–86:
> "NOTE ON n: the 8 recdays come from **5 animals** (ah03, ah04 x3, me08, me10, me11 x2),
> so the group test treats **recording days, not animals**, as the unit of inference."

Main text says "eight mice"; Fig 2a panel title says "7 mice" (7 mouse indices listed);
Fig 2b legend says "One dot represents one animal"; and the df's disagree — main text
t(8) = 8.81 vs supplement t(7) = 6.33 for the simplified model you actually report.
Mathias spotted the 8-dots-vs-7-mice symptom. The underlying problem is larger: the
inferential unit is the recording day and that is nowhere stated.

### 3. Left EC / left mOFC may be a data-picked hemisphere. [101]
`fMRI_mask_vs_cluster_extract.py` splits each mask into bilateral/left/right and picks a
`matched` hemisphere by **which one overlaps the significant PALM cluster more**. Its own
docstring warns:
> "NB `matched` is chosen using the data, so it is a descriptive convenience for plotting
> — quote `bilateral` (or a hemisphere fixed a priori)."

You report left EC t(32) = 2.16 and left mOFC t(32) = 2.51. If "left" came from `matched`,
Will's question is a real circularity. Check which variant those two numbers came from in
`mask_vs_cluster_summary.csv`.

### 4. Fig 3b's angles are offset 45 deg from Fig 3a's quarters. [98], [83]
`harmonic_angle_maps.py` uses bin **centres**: theta_k = 45, 135, 225, 315 deg — not
0/90/180/270. That is exactly why Kris reads the loop as "rotated ~90 deg relative to (a)".
Defensible convention, but it is not in the methods, and Kris's guessed construction in [83]
(x = beta_now - beta_t+2, y = beta_next - beta_t+3) assumes the *other* convention. The
schematic he's asking for must not be drawn from his description.

---

## A. Rephrase only (72 open)

No new numbers needed. Grouped by the edit.

**Abstract and framing** — [2] [3] [68] [69]
Habiba is lost in the electrodes sentence and reads the two codes as one population; Alon
wants the hippocampal-sweep sentence cut from the abstract; [68] is redundant with the
opening.

**Narrative / structural decisions** (no new results, but a real rewrite) — [9] [23] [29] [33] [99-main] [103]
Mathias: lead with fMRI to avoid hedged cell results [9]; don't frame cells vs fMRI as
opposition [23]; you hedge only the one result [29]; frame as SNR-vs-coverage trade-off [33].
Mohamady: move the plan-not-memory section earlier [99]. Will + Kris: drop the
"ordering variable is not a property of what is represented" contrast [103].

**Terminology and internal consistency** — [12] [19] [35] [36] [41] [47] [49] [51] [52] [53] [58] [61] [64] [73] [84] [87] [89] [91] [105] [108] [111]
Biggest ones: layout/configuration/task must be one word everywhere [51]; iEEG vs sEEG [47] [61];
"phase" -> "goal progress" per Mohamady [73], which touches figures and methods; define
degrees where first used, not in Fig 3 [84] [53] [87]; mention the sound/feedback at A
[49] [111]; Will is right that fMRI similarity is still across conditions, just with
voxel vectors [108]; "principal axis" is PC1, say so [105].

**One-line additions** — [0] [20] [34] [40] [42] [43] [59] [60]
Includes your own TODO [0] (say you checked El-Gaby's original model, the within-task RDM
and the across-run RDM) and Mathias's request for one extra sentence unpacking the coffee
analogy for naive readers [40].

**Citations** — [11] [13] [22] [54] [55] [62] [65] [66] [70] [71] [78]
[22] is probably unresolvable (Tim thinks Mathias means Badre's abstraction gradient) — decide
and drop. [71] is a discussion paragraph to write, not just a cite.

**Figure cosmetics** — [16] [24] [25] [26] [30] [38] [39] [45] [46] [50] [56] [57] [75] [82] [88]
plus the typography half of [44].
[30] and [18-sub]: the missing star at 0 deg for mid HC is a plotting omission — the text
already reports p = 0.0131.

**Figure rebuilds** (no new numbers, but half a day each) — [17] [18]
[17] Alon's threshold-and-recolour scheme to make the FWE cluster in 2f visibly the same
object as the outline in 3b. [18] is a list: effect-size colourbar in 3a, no overlapping
clusters in 3b, split hemispheres rather than mirrored average in 3d, non-grey "outside
gradient" colour, label the dorsal/ventral split relative to the black line.

**Admin** — [1] demographics from Habiba (X males / X females / mean age in methods).

---

## B. Explain deeper + verify what I did (19)

| # | idx | who | what to check | script |
|---|---|---|---|---|
| 1 | [4] | Mathias | Whole-mask mean or picked cluster? Docstring already answers: per-subject mean over every mask voxel, explicitly *not* a peak readout. Needs one methods sentence. | `fMRI_mask_vs_cluster_extract.py` |
| 2 | [31] | Mathias | Double dipping. Establish and state that the mPFC mask was a-priori (rodent PrL -> BA32 homology) and lOFC was confirmatory after the searchlight. | `fMRI_mask_vs_cluster_extract.py` |
| 3 | [32] | Mathias | "numerically larger" = 1226 vs 960 voxels *and* peak t 4.94 vs 3.85. Decide what you want to convey. | PALM outputs |
| 4 | [37] | Mathias | How much is subject-specific? Models are built per subject from their own modal path per layout; nothing else fitted. Say it. | `create_fMRI_model_RDMs_on_clean_beh.py` |
| 5 | [15] | Alon | Define "overlapping the gradient": mask = voxels where each of the 4 step effects passes t = 1.5; cells assigned by nearest-voxel lookup (87/158). | `cell_gradient_master_table.py`, `harmonic_maps_brain_overlay.py` |
| 6 | [44] | Mathias | See finding 2 — n, dots, df, unit of inference. | `analysis_rodents_complete_clean.py` |
| 7 | [72] | Mohamady | Which test is which: per-region binomial vs chi2 omnibus vs EC-vs-rest Fisher. | `encoding_state_sustained_cv.py` |
| 8 | [74] | Mohamady | Why 330 deg is "previous" not "far future". You have the numbers (39% overlap at +-30 deg, chance by +-90 deg) — move the argument into results. | `per_lag_encoding.py` |
| 9 | [79] [102] | Kris, Will | **The mid-HC story — the biggest coherence problem in the paper.** Is the mid-HC DSR beta carried by the *now* slot of the DSR model? Check the quarter-split DSR fit per ROI. | `RSA_DSR_ROIs_simple.py`, `per_lag_encoding.py` |
| 10 | [83] | Kris | Verify the convention (finding 4) before drawing his schematic. | `harmonic_angle_maps.py` |
| 11 | [85] | Kris | Which regressors are in which model. The three pipelines genuinely differ. Write one table. | all three RSA scripts |
| 12 | [93] | Will | Ambiguous anchor ("And they knew this before right?" on "not significantly"). Find the sentence in Drive. | — |
| 13 | [97] | Will | Is the gradient really ventral-dorsal or oblique inside the black region? You already have PC1 ~ [-0.04, -0.41, 0.91], r = 0.98 with MNI z. Reporting that answers him. | `docs/gradient_split_stats_and_smoothness.md`, `cell_fMRI_angle_match.py` |
| 14 | [98] | Will, Kris | Two black outlines + the 45 deg rotation. See finding 4; say what each outline is. | `harmonic_maps_brain_overlay.py` |
| 15 | [101] | Will | See finding 3 — was "left" a-priori or `matched`? | `fMRI_mask_vs_cluster_extract.py` |
| 16 | [107] | Will, Kris, Alon | See finding 1 — slot vs flat. Highest-stakes comment in the set: two co-authors think the model may be wrong. | `RSA_DSR_ROIs_simple.py` |
| 17 | [109] | Will | Residualisation (phase only, data level) vs competing regressors (location/state/buttons, RDM level) are different operations. Same confusion as [85] and [107] — fix once, clearly. | `analysis_rodents_complete_clean.py` |
| 18 | [110] | Will | "in either analysis" = the two rodent RSA variants (within-config and across-run). Same as TODO [0]. | `analysis_rodents_complete_clean.py` |
| 19 | [112] | Will | Was the unfolding/sweep effect modelled or hypothetical? Identify the sentence, state which models were fitted. | — |

**[85], [107], [109] are one problem wearing three hats**: nobody can tell what was
residualised, what was a competing regressor, and what the DSR model geometry actually is.
Fixing that once, plus the supplementary figure below, closes all three.

---

## C. New or re-run analysis (9)

1. **Gradient smoothness index** — [76] Mohamady.
   ~20 small ROIs along the gradient axis, variance of angle change between adjacent ROIs;
   draw the axis in 3b. Check `docs/gradient_split_stats_and_smoothness.md` first — you may
   already have most of it. `harmonic_angle_maps.py` + axis from `cell_fMRI_angle_match.py`.

2. **Hippocampus in the instruction phase** — [99-sub] Mohamady. *Best payoff per effort.*
   Does HC track the *presented* sequence regardless of executed order? You already have the
   `_instr` model family (`instruction_relabel_dict`) in `fMRI_run_RSA_instruction.py`, so
   this is a re-run in MTL masks, not new machinery. It would turn Fig 3g's dissociation into
   a second, independent one.

3. **Forward vs backward instruction trials separately** — [92] Will.
   Split the instruction-phase RSA by execution direction.
   `fMRI_run_RSA_instruction.py` + `scripts/old/per_TR_loso_pre_refactor/svc_loso_*.py`.

4. **Supplementary figure: model RDMs + full regression coefficients** — [107] [85] [109].
   Requested independently by Will, Kris and Alon. Alon explicitly wants it in the
   supplement, not the main text. Mostly plotting from fits you already have.

5. **mid-HC follow-up** — [79] [102], conditional on B-9.
   If the mid-HC DSR fit is carried by the now-slot, run a future-only DSR (drop the 0 deg
   slot) per ROI. If it isn't, you owe a paragraph explaining a genuine mid-HC future code.
   `RSA_DSR_ROIs_simple.py`.

6. **ROI-mean test across all masks, FDR-corrected** — [31].
   Defuses double-dipping by making the selection visible rather than arguing about it.
   `fMRI_mask_vs_cluster_extract.py`.

7. **Bilateral (or a-priori-fixed) EC/mOFC numbers** — [101], follows from finding 3.

8. **Uncontrolled position-in-sequence fit in rodents** — [109-b] Will.
   He wants the ABCD null shown without any pre-controlling. One extra column in the rodent
   GLM figure. `analysis_rodents_complete_clean.py`.

9. **(cheap, optional) % shortest path between consecutive rewards** — [91] Kris.
   The behavioural metric he expected instead of loop time. `behaviour_summary.py`.

**Not an analysis but a decision:** [99-main] Mohamady wants the plan-not-memory result to
open the results. No new numbers, big restructure.

---

## Suggested order

1. Findings 2 and 3 (rodent n; hemisphere choice) — these are correctness issues a reviewer
   would catch, and both are quick.
2. B-9 / C-5 (mid HC) and B-16 / C-4 (slot vs flat + the coefficients figure) — the two
   things that currently stop co-authors believing the result.
3. C-2 (HC in the instruction phase) — the one new analysis that adds a result rather than
   defending an existing one.
4. Everything in A.
