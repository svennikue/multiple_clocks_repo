# ROI assignment — methods

Companion document to `scripts/cell_to_roi_july26.py`. Describes how every
recorded single unit was assigned to one of thirteen anatomical regions of
interest (ROIs), from raw MNI152 coordinate to the final `alt_final_roi`
label used downstream. Every assignment rule is anatomical; nothing in
this pipeline depends on the cell's firing statistics.

Output table: `data/ephys_humans/derivatives/neurons_with_ROI_labels.csv`
(one row per cell; audit trail in `derivatives/ROI_assignment/`).

---

## 1. Coordinate provenance

All ROI queries use MNI152 coordinates. Coordinates were obtained per
recording site:

**Baylor** (n = 669 cells, 20 subjects). Bundle-level MNI152 coordinates
for the Behnke-Fried microwires were read from the manufacturer-provided
v2026 electrode files (`YEJ-electrodes_v2026.csv` and 19 siblings). Each
per-channel `electrode label` (e.g., `mRT2bHaEa04`) was mapped to its
bundle row (`mRT2bHaEa01`) by stripping the trailing two digits. As a
reliability gate, each file's MNI152 column was cross-checked against
its own MNI305 column transformed by the Fischl MNI305→MNI152 affine
(Fischl et al. 1999); files whose two coordinate columns disagreed by
> 8 mm mean were rejected. For four subjects with rejected or missing
files the big table's MNI305 coordinates were transformed to MNI152 and
used as an approximate fallback (~4 mm systematically off from the true
micro position — flagged as `baylor_bigtable_pre2026_macro_position`).

**UCLA** (n = 140 cells, 6 subjects). Coordinates were read from the
per-subject `sub-NNN_localizations.xlsx` Sheet 1 (`MNI_x/y/z`).
Big-table coordinates matched a v2026 xlsx row at 0.00 mm distance in
every case, verifying provenance and attaching the electrode name and
NMM region label.

**Utah** (n = 175 cells, 17 subjects). MNI152 coordinates were
reconstructed independently from each subject's
`s{NN}/electrodes/Electrodes.mat` file: for each cell with an
`electrode label` of the form `chan{N}`, we located `N` in the
`ChannelMap1` / `ChannelMap2` matrix, matched the corresponding
`LabelMap` entry (a micro-label such as `mLOFC3`), and read the MNI152
coordinate from `ElecXYZMNIProj[MicroElec[i] - 1]` where `i` is the
micro's rank in the sorted-by-channel ordering. Reconstructed
coordinates agreed with the big-table coordinates to within 3 mm for
97 cells; the remaining 78 cells (where reconstruction disagreed by
> 3 mm) kept the big-table coordinate and are flagged as
`utah_bigtable_recon_disagrees_gt3mm` in the output.

---

## 2. Atlases and thresholds

Four probabilistic atlases were queried at each cell's MNI152 coordinate.
All were used at the **25 % maximum-probability threshold** (nilearn's
`maxprob-thr25-2mm` variant, or Brainnetome MPM at its native
threshold): a cell is assigned to a region only when that region has at
least 25 % population probability at the voxel containing the cell.

- **Juelich histological atlas** (Amunts et al. 2005; Eickhoff et al.
  2005): cytoarchitectonic labels for the medial temporal lobe. Its
  hippocampal-formation hierarchy explicitly separates *cornu ammonis*,
  *subiculum* and *entorhinal cortex* as three sibling labels under
  a common `GM Hippocampus …` prefix. In this pipeline we treat the
  first two as **hippocampus proper** and the third as our EC ROI.
  Also used for visual cortex (V1/V2/V3).
- **Harvard-Oxford cortical atlas** (Desikan et al. 2006): 48 cortical
  parcels, used for parahippocampal gyrus, insular cortex, posterior
  cingulate, precuneus, occipital cortex, and orbital frontal cortex.
- **Harvard-Oxford subcortical atlas**: hippocampus, amygdala,
  thalamus.
- **Brainnetome atlas** (Fan et al. 2016): fine-grained cortical
  parcellation, used for orbitofrontal (A11m, A13, A14m) and medial
  prefrontal (A32sg, A32p, A24rv, A24cd, A10m, A9m) subdivisions, and
  for posterior cingulate / precuneus (A23, A31, dmPOS).

---

## 3. ROI definitions and priority order

Cells were assigned the first ROI in the following priority list whose
atlas rule matched their coordinate. This priority order encodes our
prior that fine-grained cytoarchitectonic labels (Juelich, Brainnetome)
should take precedence over coarser gyral parcels (Harvard-Oxford)
when both apply.

**1. Entorhinal cortex (EC)** — Juelich `GM Hippocampus entorhinal
cortex` at ≥ 25 %.

**2. Medial prefrontal cortex (mPFC)** — Brainnetome A32sg, A32p,
A24cd, A24rv, A10m, or A9m, provided **y ≥ 10 mm** (anterior
cingulate cortex proper). More posterior hits (y < 10) were relabelled
`medial_CC` (mid-cingulate) so that mPFC represents genuinely anterior
cingulate cortex only, following Vogt et al.'s (2005) partitioning of
the cingulate along its rostrocaudal axis at the level of the anterior
commissure (~ y = 10 mm). One additional anatomical rule: A32sg cells
with **z ≤ -10 mm** lie on the orbital surface below the corpus
callosum, so they are assigned to mOFC rather than mPFC (subgenual →
orbital transition).

**3. Medial orbitofrontal cortex (mOFC)** — Brainnetome A11m, A13, or
A14m. This is the ventromedial prefrontal / medial orbitofrontal
region.

**4. Hippocampus (HC_anterior, HC_mid)** — a cell is assigned HC when
EITHER (a) HO subcortical returns `Hippocampus` at ≥ 25 %, OR (b) the
Juelich atlas assigns the voxel to `GM Hippocampus subiculum` or
`GM Hippocampus cornu ammonis`. Both are explicitly hippocampal
subfields in the Juelich cytoarchitectonic hierarchy; the second
criterion is required because the Harvard-Oxford subcortical
hippocampal mask uses a coarser boundary that excludes a substantial
subicular rim, which the higher-resolution Juelich atlas retains as
hippocampus. Without criterion (b), subicular voxels would incorrectly
fall through to the coarser parahippocampal-gyrus label (rule 5) or be
absorbed by an entorhinal-rescue neighbourhood search.

The hippocampal long axis was then split into anterior and mid at
**y = -21 mm**, following Poppenk & Moscovitch (2013), who showed that
anterior HC is preferentially connected to medial prefrontal cortex
via the uncinate fasciculus and is functionally dissociable from
posterior/mid HC. Our y-histogram of HC-labelled cells shows a clear
bimodal distribution with a natural gap around y ≈ −21 to −22 (see
`ROI_assignment/cells_step2_hc_y_distribution.png`).

**5. Parahippocampal cortex (PHC)** — Harvard-Oxford cortical
`Parahippocampal Gyrus` (anterior or posterior division), assigned
only when the hippocampal-subfield rule 4 has not already claimed the
voxel. This ordering ensures that the subicular complex — which sits
at the medial edge of the parahippocampal gyrus and could otherwise be
absorbed by the HO cortical parahippocampal mask — is preserved for
the hippocampal ROI.

**6. Posterior cingulate / precuneus (PCC)** — Brainnetome A23
(cingulate BA23), A31 (medial precuneus BA31), or dmPOS (dorsomedial
parieto-occipital), OR Harvard-Oxford `Cingulate Gyrus, posterior
division` or `Precuneous Cortex`. **PCC and Precuneus were
intentionally collapsed** into a single ROI (`PCC`) because both
regions are core nodes of the default-mode network's posterior medial
hub (Andrews-Hanna et al. 2010; Utevsky et al. 2014) and neither is a
target region for the present study; the collapse maximises statistical
power for the pooled cluster without over-interpreting a functional
subdivision that our task cannot arbitrate.

**7. Visual cortex (Visual)** — Juelich V1, V2, V3, or any label
matching `visual` / `calcarine`; falling back to Harvard-Oxford
`occipital` / `cuneal` / `lingual` / `intracalcarine` /
`supracalcarine`.

**8. Amygdala** — Harvard-Oxford subcortical `Amygdala`. An additional
anatomical sanity check re-tests every Amygdala cell for Juelich
entorhinal within 3 mm; matches are reassigned to EC on the priority
principle (EC precedes Amygdala in the ROI hierarchy).

**9. Thalamus** — Harvard-Oxford subcortical `Thalamus`.

**10. Insula** — Harvard-Oxford cortical `Insular Cortex`.

Cells not matching any rule at 25 % probability were labelled
`leftover`.

---

## 4. Electrode-intent rescue

Cells labelled `leftover` after the atlas rules were re-examined using
the electrode's *intended* target as a prior. The intent label was
assembled from (a) the source region name recorded in the .mat
`LabelMap` or v2026 xlsx `NMM` column, and (b) the `region label`
column of the big table (e.g. `LEC`, `mLOFC3`, `LINS`). For each
recognised intent pattern (e.g., `mlofc*` → mOFC target), we performed
a small-neighbourhood search (up to ±8 mm on an integer-mm cube grid)
in the target atlas for the intent's anatomical region. If a match was
found, the cell was reassigned; if not, it stayed `leftover`. This
step rescued cells that sit 1–5 mm outside the atlas mask edge — for
example, cells whose reconstructed coordinate falls in white matter
adjacent to their intended gray-matter target. Median rescue distance
was 1.7 mm.

Anatomical constraints applied to the rescue:

- EC rescue is rejected for cells with y < −18 mm: at that
  antero-posterior level the medial temporal cortex is hippocampal
  body / tail, not entorhinal (entorhinal cortex ends anteriorly
  around y ≈ −18 to −20 in MNI152; Insausti et al. 1998).
- mPFC rescue is rejected for cells with y < 10 mm (relabelled
  medial_CC), consistent with the ACC/MCC boundary in rule 2.

---

## 5. Analysis-ready ROIs

The final canonical column, `alt_final_roi`, contains the ROI name
whenever that ROI has ≥ 3 distinct contributing subjects; otherwise
it is NaN. This filters out under-covered ROIs (Amygdala, Thalamus,
Visual, Insula, medial_CC, leftover) so that downstream scripts
filtering on `alt_final_roi.notna()` automatically work on the 916
analysis-ready cells.

**Final ROI counts** (n_cells / n_subjects):

| ROI | n_cells | n_subjects |
|---|---|---|
| HC_anterior | 263 | 46 |
| HC_mid | 176 | 31 |
| mOFC | 163 | 29 |
| mPFC | 155 | 32 |
| PCC | 66 | 10 |
| PHC | 51 | 7 |
| EC | 42 | 10 |

916 cells across seven analysis-ready ROIs.

---

## 6. Colour convention

Colour mappings follow the project-wide `roi_colour_dict` in
`CLAUDE.md`:

| ROI | Colour source |
|---|---|
| EC | era_brewer Showgirl2 index 0 (dark red) |
| mPFC | era_brewer Showgirl2 index 1 (orange) |
| HC_anterior | era_brewer Showgirl2 index 2 (tan) |
| PCC | era_brewer Showgirl2 index 3 (pale yellow) |
| mOFC | era_brewer Showgirl2 index 4 (pale green) |
| HC_mid | `#a30d6c` (magenta) |
| PHC | `#23677E` (teal) |

---

## References

- Amunts, K., Kedo, O., Kindler, M., Pieperhoff, P., Mohlberg, H.,
  Shah, N. J., Habel, U., Schneider, F., & Zilles, K. (2005).
  Cytoarchitectonic mapping of the human amygdala, hippocampal region
  and entorhinal cortex: intersubject variability and probability
  maps. *Anatomy and Embryology*, 210(5–6), 343–352.
- Andrews-Hanna, J. R., Reidler, J. S., Sepulcre, J., Poulin, R., &
  Buckner, R. L. (2010). Functional-anatomic fractionation of the
  brain's default network. *Neuron*, 65(4), 550–562.
- Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson,
  B. C., et al. (2006). An automated labeling system for subdividing
  the human cerebral cortex on MRI scans into gyral based regions of
  interest. *NeuroImage*, 31(3), 968–980.
- Eickhoff, S. B., Stephan, K. E., Mohlberg, H., Grefkes, C., Fink,
  G. R., Amunts, K., & Zilles, K. (2005). A new SPM toolbox for
  combining probabilistic cytoarchitectonic maps and functional
  imaging data. *NeuroImage*, 25(4), 1325–1335.
- Fan, L., Li, H., Zhuo, J., Zhang, Y., Wang, J., et al. (2016). The
  Human Brainnetome Atlas: A new brain atlas based on connectional
  architecture. *Cerebral Cortex*, 26(8), 3508–3526.
- Fischl, B., Sereno, M. I., Tootell, R. B. H., & Dale, A. M. (1999).
  High-resolution intersubject averaging and a coordinate system for
  the cortical surface. *Human Brain Mapping*, 8(4), 272–284.
- Insausti, R., Juottonen, K., Soininen, H., Insausti, A. M.,
  Partanen, K., Vainio, P., Laakso, M. P., & Pitkänen, A. (1998).
  MR volumetric analysis of the human entorhinal, perirhinal, and
  temporopolar cortices. *American Journal of Neuroradiology*, 19(4),
  659–671.
- Poppenk, J., & Moscovitch, M. (2013). Long-axis specialization of
  the human hippocampus. *Trends in Cognitive Sciences*, 17(5),
  230–240.
- Utevsky, A. V., Smith, D. V., & Huettel, S. A. (2014). Precuneus is
  a functional core of the default-mode network. *Journal of
  Neuroscience*, 34(3), 932–940.
- Vogt, B. A., Vogt, L., & Laureys, S. (2005). Cytology and
  functionally correlated circuits of human posterior cingulate
  areas. *NeuroImage*, 29(2), 452–466. (Anterior/posterior cingulate
  boundary at y ≈ 10 mm.)
