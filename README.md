# multiple_clocks_repo

Code for the project **"The algorithmic representations of sequential behaviour in the human brain."** Subjects perform the ABCD taks on a 3×3 grid: they plan and
execute trajectories that visit four rewarded locations in sequence. We test
whether medial-PFC represents *all upcoming actions simultaneously*
(duplicated-state representation, **DSR**), as recently shown in rodent
mPFC.

The repository couples a Python library (`mc/`) with a series of analysis
pipelines (`scripts/`) covering three datasets:

| Dataset | n | Stored under |
|---|---|---|
| Rodent single units (El-Gaby et al. 2023, re-analysed) | 8 recdays | `data/ephys_recordings_200423/` |
| Human single units (iEEG, Utah arrays + macro contacts) | 63 sessions | `data/ephys_humans/` |
| Human 7 T fMRI | 35 subjects | `data/derivatives/` |

---

## The four core analyses

These are the canonical pipelines we report. Each script writes a
time-stamped folder under the dataset's `derivatives/` tree, plus a JSON
that records every parameter used.

### 1. Rodent DSR RSA — `scripts/analysis_rodents_complete_clean.py`

Re-runs the El-Gaby task on the rodent mPFC neurons and asks the same DSR
question we ask in humans. Two variants per recday:
- `mode_path / all_trials / full_z` — z-scored full DSR RDM.
- `mode_path / across-halves` — duplicate-config sessions split into halves.

Per recday a four-model RSA (DSR vs. phase vs. location vs. state) is fit;
group betas are pooled across recdays with BH-FDR across models. Outputs:
publication figures 1–3 + a stats JSON. 

### 2. Human single-cell RSA — `scripts/RSA_DSR_ROIs_simple.py`

Per-ROI RSA on the human iEEG cells using the human analogue of the DSR
model. For each ROI we build a population matrix (configs × neurons × 12
sub-conditions), then a data RDM (`mc.analyse.my_RSA.compute_crosscorr_within`,
across-task-halves variant + z-scored variant), then fit a combo OLS that
contains DSR alongside its main controls (`bttn_curr`, `bttn_next`,
`location`, `midnight`, `state`).

- **Permutation null:** per-trial circular shifts of each neuron along its
  binned time axis (1,000 perms; see `RSA_DSR_ROIs_simple.py:382-406`).
  Each shift breaks alignment between activity and task condition while
  preserving within-trial autocorrelation; the data RDM is rebuilt and the
  same combo OLS refit. One-sided p_perm with the `(k+1)/(N+1)` correction.
- **FDR family** (`FDR_*` constants, lines 95-115): one primary combo
  (`MRI_combo-nofdb_midn-state`) × `dsr_old` × all ROIs ⇒ 7 tests. The
  second `midn` combo is treated as a robustness check (its `dsr_old` beta
  is correlated with the primary because the combos differ only by `state`).
- Outputs: `results_summary_combos.csv`, `confirmatory_fdr.csv`, per-ROI
  heatmaps, electrode glass-brain.

### 3. Human single-cell encoding — `scripts/encoding_analysis_simple.py`

ElasticNet encoding model per neuron per task-structure model (DSR, button
identities, location, midnight, state, phase, …).

- **Y:** trial-averaged binned firing-rate trace. For each correct-trial
  configuration we take the across-repeat mean at every one of the 360
  within-trial time bins (`build_design_and_neurons`, line 552), then
  concatenate the configurations.
- **CV:** leave-one-configuration-out. ElasticNet
  `alpha=0.001`, `l1_ratio=0.5`, `positive=false`. Per-neuron
  `mean_r` = Pearson r of held-out predicted vs. observed trace, averaged
  over folds.
- **Per-neuron permutation null** (`analyse_one_neuron`, lines 722-765):
  the ElasticNet is fit once. For 500 permutations the held-out test
  trace of each fold is independently circularly shifted and r is
  recomputed against the unchanged predictions. One-sided
  `p_perm = (k+1)/(N_valid+1)`.
- **ROI-level test:** one-sample one-sided t-test of per-neuron `mean_r`
  against zero (`mc.plotting.cell_results._one_sided_t_greater`,
  line 498). `df = n_cells - 1`. BH-FDR over the same 7-ROI family as
  the RSA pipeline (DSR only; controls are exploratory).
- **Follow-ups:** `encoding_followup_simple.py` (lag-coefficient gradient,
  brain plots, mask overlap), `encoding_publication_panels.py`
  (publication panel A–C with the equal-shade RSA × encoding heatmap pair
  and the ACC histogram).

### 4. Human fMRI RSA — `scripts/fMRI_run_RSA_without_rsatoolbox_clean.py`

Whole-brain RSA on the 7 T fMRI dataset. Builds data RDMs from the
`glm_all-rews-split_buttons` GLM and model RDMs from
`scripts/create_fMRI_model_RDMs_on_clean_beh.py` (cleaned behaviour).
Group inference uses PALM permutation tests; the DSR contrast is then
thresholded to define the DSR-effect mask (`scripts/cell_mask_overlap.py`
builds the binary `p_FWE < 0.05` mask from the cluster-mass FWE map and
writes it to `data/masks/DSR_main_effect_mask.nii.gz`).

---

## Glue / publication scripts (work in progress)

- `encoding_publication_panels.py` — combined RSA × encoding panel for the
  paper; equal-shade colorscale across the two heatmaps, ACC DSR
  histogram, per-ROI significant-cell bars. Reads the time-stamped
  outputs of (2) and (3).
- `publication_figures_human_cells.py` — standalone cell-level figures.
- `cell_mask_overlap.py` — overlap between recorded cells and the
  DSR-main-effect / gradient masks (cell counts inside vs. outside,
  ROI-coloured `mne.viz.Brain` with orange cluster shading on the
  surface). Reports counts for `all` / `dsr_rsa` / `not_dsr_rsa` cell
  subsets.
- `roi_brain_visualization.py` — palette sandbox for brain coverage plots.
- `cell_to_roi_MNI.py` — assigns MNI coordinates and final ROI labels to
  every recorded cell (`neurons_with_final_roi_labels.csv`).

---

## The `mc` package

| Subpackage | Purpose |
|---|---|
| `mc.simulation` | Build task configurations, simulate clock / midnight / DSR / location populations, HRF convolution, pathlength distributions. |
| `mc.analyse` | RDM construction (`my_RSA.py`), regression / model evaluation, rodent ephys analysis (`analyse_ephys_clean.py`), MRI preprocessing. |
| `mc.fmri_analysis` | First-level FEAT EV builders, model-RDM construction. |
| `mc.plotting` | Figure layout (`figure_layout.py`), cell-level results (`cell_results.py`), brain plotting helpers. |
| `mc.latest_experiment` | Behavioural-task assets used to generate stimuli + condition files. |
---

## Where to start when picking the project up cold

1. Run `analysis_rodents_complete_clean.py` to reproduce the rodent
   replication — that's the cleanest pipeline.
2. Then `RSA_DSR_ROIs_simple.py` (human cells) and
   `encoding_analysis_simple.py` (encoding), in that order.
3. Publication panels are assembled by `encoding_publication_panels.py`.
4. Whole-brain fMRI RSA (after fMRI preprocessing, and data preparation): `fMRI_run_RSA_without_rsatoolbox_clean.py`.

