# Single-unit quality control — methods text and assessment

Figure: `derivatives/group/cell_qc/cell_qc_criteria.pdf`
Metrics: `derivatives/group/cell_qc/cell_qc_metrics.csv`
Source of record: `derivatives/qc_all_sessions.mat` (run 2026-04-16)
Filtered dataset consumed downstream: `derivatives/abcd_passed.mat`

## Methods text (paper)

> **Unit inclusion.** Spikes were extracted and sorted with WaveClus (Chaure et
> al., 2018) and manually curated. Units then passed an automated quality-control
> stage before entering any analysis. Within each session we defined the task
> window as the interval from the first grid onset to the last trial end, and
> required each unit to fire at least 300 spikes within it. Isolation quality was
> quantified as the proportion of inter-spike intervals shorter than 1.5 ms, and
> units with more than 1 % refractory-period violations were excluded. Finally,
> because a single neuron can be picked up by more than one microwire of the same
> Behnke–Fried bundle, we identified duplicates from the zero-lag correlation of
> spike counts in 100 ms bins across the task window: where two or more accepted
> units correlated at r ≥ 0.5, we retained only one, preferring the unit with more
> spikes, then fewer refractory violations, then more stable firing across task
> configurations. Of 1,042 sorted units, 984 (94.4 %) passed: 36 were excluded for
> insufficient spikes and 22 as within-bundle duplicates. All 22 duplicate pairs
> lay on the same electrode bundle and were assigned to the same anatomical
> region, consistent with a single neuron detected on adjacent wires rather than
> with the loss of distinct units.

## Methods text (thesis — longer)

> Single units entered the analyses only after an automated quality-control stage
> applied identically to all 63 recording sessions. Three criteria were used.
>
> First, a **spike-count floor**. For each session the task window was defined as
> the interval between the first grid onset and the last trial end; units firing
> fewer than 300 spikes in that window were excluded. This threshold is
> deliberately permissive — 300 spikes over a typical session corresponds to
> roughly 0.1 Hz — and is intended to remove units too sparse to estimate a firing
> rate, not to select for high-rate cells. The retained units had a median of
> 4,118 spikes (10th percentile 747) and a median firing rate of 1.6 Hz.
>
> Second, an **isolation criterion**. Spike trains from a well-isolated single
> neuron should contain almost no inter-spike intervals shorter than the
> refractory period. We computed the proportion of ISIs below 1.5 ms and excluded
> units exceeding 1 %. The retained population had a median violation rate of
> 0.00 %, indicating that isolation was not a limiting factor in this dataset.
>
> Third, a **duplicate criterion**. Behnke–Fried electrodes carry eight
> microwires within one bundle, and a single neuron may be recorded on more than
> one of them, which would otherwise enter the analyses as two units and
> artificially inflate the sample. We binned each unit's spikes at 100 ms across
> the task window and computed the zero-lag correlation between every pair of
> accepted units. Pairs correlating at r ≥ 0.5 were treated as one neuron, and a
> single representative retained, chosen by spike count, then refractory
> violations, then firing-rate stability across configurations and finally overall
> rate.
>
> Of 1,042 sorted units, 984 (94.4 %) passed. Thirty-six were excluded for
> insufficient spikes (median 233, range 110–296) and 22 as duplicates. Twenty-one
> of the 22 duplicate pairs lay on the same microwire bundle and all 22 shared an
> anatomical region label, which is the pattern expected if the criterion is
> identifying one neuron recorded twice rather than discarding genuinely distinct
> co-active cells.

## Figure caption

> **Figure X. Single-unit quality control.** (a) Distribution of spike counts
> within the task window, for units included (green) and excluded (rust); dashed
> line, the 300-spike inclusion threshold. (b) Refractory-period violations, the
> proportion of inter-spike intervals below 1.5 ms; dashed line, the 1 %
> threshold. Note the logarithmic ordinate: the great majority of units show no
> violations. (c) Maximum zero-lag correlation of each unit with any other unit in
> the same session, from spike counts in 100 ms bins; units above r = 0.5 (dashed)
> were treated as duplicates and one representative of each correlated group
> retained. (d) Units excluded by each criterion. Of 1,042 sorted units, 984
> (94.4 %) entered the analyses.

## Assessment — is excluding these 58 cells the right call?

**Yes, on all three criteria, with one caveat and one correction needed.**

**Spike floor (300) — sound, and permissive.** The 36 excluded units had 110–296
spikes. Over a session that is well under 0.1 Hz, too sparse to estimate a rate
map or contribute to a population geometry. Retained units sit far above it
(median 4,118), so the threshold is not shaping the sample — it is trimming a
tail. If anything it is *lenient*: the RSA uses 96 conditions (8 configurations ×
12 bins), so a 300-spike unit contributes ~3 spikes per condition. A sensitivity
analysis at a higher floor would be cheap reassurance, but the criterion as
stated is defensible.

**Refractory violations (1 % at 1.5 ms) — sound and standard.** This is the
conventional isolation measure (cf. Hill, Mehta & Kleinfeld 2011). The median
retained unit has 0.00 % violations, so this criterion is effectively not
binding — it is a guard, not a filter. A 1.5 ms refractory window is at the short
end of what is typically used (2–3 ms is common); a shorter window makes the test
*more permissive*, so this cannot be inflating exclusions.

**Duplicate removal (r ≥ 0.5, 100 ms bins) — sound, and I checked the obvious
failure mode.** A 100 ms bin is coarse for duplicate detection: two genuinely
distinct neurons co-modulated by a repeating task structure can correlate
strongly without being the same cell, and that risk is real in this paradigm.
The diagnostic is where the pairs sit. **All 22 excluded pairs were on the same
electrode bundle (21/22 by bundle key, the remaining one adjacent contacts of a
single UCLA probe) and all 22 shared a region label.** Not one pair spans
regions. That is the signature of one neuron on two wires, not of co-active
distinct cells, so the criterion is doing what it claims. Reporting that fact is
worth a sentence in the methods, because it is what makes the criterion credible.

**Caveat.** The stricter test for duplication is a cross-correlogram at ±1–2 ms
(a sharp zero-lag peak with a shared refractory notch) rather than a 100 ms-bin
correlation. The 100 ms result is consistent with duplication but does not
establish it. Given that every pair is within-bundle and within-region, the
conclusion is unlikely to change — but if a reviewer presses, the millisecond
cross-correlogram is the answer.

## ⚠ Two things to fix

1. **The manuscript states the wrong threshold.** It currently reads *"excluded
   units as duplicates that correlated higher than r = 70 with a second unit"*.
   The run that produced `abcd_passed.mat` used **r ≥ 0.50**, not 0.70. At 0.70
   only 9 units would have been removed rather than 22. Correct the text to 0.5
   (and note it is a zero-lag correlation of 100 ms binned counts, which the
   current sentence also omits).

2. **The code that produced `abcd_passed.mat` is not in the repository.** The
   settings recorded in `qc_all_sessions.mat` include `MinOverallFR_Hz = 0.1` and
   `SessionLowFR_Hz = 0.1`, and neither parameter appears in
   `scripts/qc_single_session.m`, `scripts/qc_single_session_v1.m`,
   `scripts/cell_wise_QC.m` or `scripts/call_cell_wise_QC.m`, nor anywhere in git
   history. The version that ran on 2026-04-16 was never saved. The per-cell
   metrics and decisions *are* fully preserved in `qc_all_sessions.mat`, so
   nothing is lost — but the pipeline is not reproducible from the repository as
   it stands and the script should be reconstructed and committed.

   Related: an earlier run (`qc_master_summary.txt`, Aug 2025, 60 sessions, 924
   cells) passed only **347/924 = 37.6 %** at nominally the same thresholds, e.g.
   UT1-202217 1/18 there versus 17/18 in April. One of the two runs is wrong.
   The April run is the one feeding `abcd_passed.mat` and its per-cell numbers are
   internally consistent, so the Aug file is most likely the faulty one — but it
   should be deleted or clearly marked superseded so it cannot be picked up by
   mistake.

## Verified: `abcd_passed.mat` matches the September dataset

`abcd_passed.mat` is `abcd_data_08-Sep-2025.mat` filtered to the 984 passing
units and nothing else. Checked cell by cell across all 63 sessions:

- subject IDs identical in order and content
- 1,042 units in, 984 out, 58 removed
- every retained unit's spike train (count, first and last spike time) is present
  in the original — **0 mismatches**
- every retained unit's `regionLabel` is identical to the original — **0 changes**

So the neural data is untouched, and the labels did not change between the two
files either.
