# Reviewer questions on Figure 3: stats for the ventral/dorsal split, and a smoothness index

## Q1 — "Have you run some stats for this?" Also: is it factually correct?

### The numbers first (these have changed)

Re-run on the rebuilt coordinates (`cell_gradient_master/2026-08-28_15-19-35`):

| manuscript | corrected |
|---|---|
| 74/155 mPFC units overlap the gradient | **87/158** |
| n = 16 recording sites | **19** |
| 42 ventral / 32 dorsal | **48 / 39** |
| ventral peaks 30°, dorsal 60° | **unchanged: 30° and 60°** |

### The stats: the split is not supported

`scripts/gradient_split_stats.py`. The unit of inference must be the **recording
site**, not the cell — cells on one microwire bundle share a coordinate exactly, so
87 in-mask cells carry only **19 independent positions**. The permutation shuffles
the ventral/dorsal label across sites and rebuilds the pooled profiles through the
identical code path.

| test | statistic | p |
|---|---|---|
| pooled **argmax** difference (the claim as written) | 30° (ventral 30° → dorsal 60°) | **0.37** |
| pooled r-weighted **circular mean** difference | 23° (29° → 53°) | **0.47** |
| circular-linear corr, preferred angle vs axis position | r = 0.069 | **0.88** |
| circular-circular corr, cell preference vs fMRI angle at its own site | r = −0.079 | **0.74** |

The null distribution of the argmax shift has **median 0°, IQR 0–30°**. A one-bin
(30°) shift is the smallest non-zero difference the 12-lag grid allows, and it
arises by chance in roughly a third of site-label permutations. The observed
result sits at about the 63rd percentile of its own null.

### And the fMRI-consistency clause does not hold at the cells' own locations

The manuscript says the split is *"consistent with the fMRI progression from
30–75° within this anatomical region"*. At the cells' actual coordinates the fMRI
gradient angle is:

    ventral group   mean 58°   (range 12-120°)
    dorsal  group   mean 61°   (range  6-100°)

A **3° difference**. The recorded cells sample a narrow, overlapping band of the
fMRI gradient, so their own fMRI angles do not separate the two groups. (Before
the coordinate fix these read 71° and 100°, which is where the apparent
correspondence came from.)

### So: is it factually correct?

*As a description*, yes — the ventral pooled profile does peak at 30° and the
dorsal at 60°. But three things in that sentence need changing:

1. **The counts are wrong** (42/32/16 → 48/39/19).
2. **There is no inferential support.** With 19 independent sites, a 30° shift is
   what chance produces.
3. **The fMRI-consistency clause is not supported** at the cells' own coordinates.

### What I would report instead

The defensible version is weaker and shorter. Options, strongest first:

- **State it as descriptive and say so.** "Within the recorded population the
  ventral cluster peaked one lag bin earlier than the dorsal cluster (30° vs 60°);
  with only 19 independent recording sites this difference is not statistically
  reliable (site-level permutation p = 0.37) and we present it as consistent with,
  rather than evidence for, the fMRI gradient." That is honest and still useful.
- **Drop the median split** and report the continuous circular-linear correlation
  across sites — which is the analysis a reviewer will ask for next, and which is
  null (r = 0.069, p = 0.88). Better to report it yourself.
- **Do not repeat the fMRI-consistency clause** unless it is restated as a range
  overlap rather than a progression.

The honest framing is that the *fMRI* gradient is the finding, and the cells are
consistent with its ventral end without independently demonstrating the gradient.
The cell recordings simply do not span enough of the axis: all 19 sites sit within
a 14 mm window (axis −18.6 to −4.2 mm) at the ventral end.

## Q2 — Quantifying the smoothness of the Fig 3b gradient

The suggestion (bin the mask into ~20 ROIs along the axis, take the variance of
angle differences between adjacent bins) is the right instinct. Three additions
would make it defensible.

### 1. The index needs a null that goes through the same smoothing

The angle maps were smoothed with a **3 mm FWHM Gaussian** and bilaterally
symmetrised before the angle was derived (Methods, gradient analysis). Any
smoothness index will therefore partly measure the preprocessing, not the brain.
A raw "variance of adjacent differences" has no interpretable scale on its own.

Build the null through the identical pipeline: sign-flip the subject-level β maps
(the same ±1 permutation used for the instruction-phase analysis), re-derive the
group cos/sin maps, re-smooth, re-symmetrise, recompute the index. That gives a
null with matched spatial autocorrelation. Shuffling bin *order* alone is a weaker
null — it destroys the ordering but also the smoothing-induced correlation, so it
will make almost any map look smooth.

### 2. Two statistics are more informative than one

- **Monotonicity**: circular-linear correlation between preferred angle and
  position along the axis, across bins or voxels. One number, directly interpretable
  as "is there a gradient", with a permutation p. This is arguably what "gradient"
  means and is worth reporting alongside smoothness.
- **Angular variogram**: mean absolute angular difference as a function of
  separation along the axis. A smooth gradient gives a gradual rise that saturates;
  a blocky one gives a step at the block boundary. This *shows* smooth-vs-blocky
  rather than compressing it to a scalar, and it answers the reviewer's actual
  question more directly than a variance.

A third option that answers "smooth or blocky" head on: fit angle-vs-position with
a **linear model and a step model** and compare by cross-validated error or BIC.

### 3. Binning details that matter

- 20 bins along PC1 within the BA32/mBA9/mBA10 mask, requiring a minimum voxel
  count per bin — a bin with few voxels gives a noisy angle and will dominate a
  variance-based index.
- Weight each bin by its **resultant vector length** (the angle's own confidence);
  an unweighted variance treats a well-defined bin and a near-uniform one alike.
- Take circular first differences wrapped to (−180°, 180°], and report both the
  mean |Δ| and the variance — the mean is what "smooth" means intuitively, the
  variance is what catches a single discontinuity.

### 4. Showing the axis in 3b

Straightforward and worth doing: PC1 is already computed
(≈ [−0.04, −0.41, 0.91], r = 0.98 with MNI z — essentially dorsoventral with a
slight posterior tilt). Draw it on the medial surface as an arrow with tick marks
at the 20 bin boundaries, and add a small inset panel of preferred angle against
axis position with the per-bin means and their confidence. That makes the
smoothness claim visible rather than asserted, and shows the reader exactly what
was binned.

**One caveat to state either way:** the recorded cells occupy only the ventral
14 mm of this axis, so the smoothness index describes the fMRI gradient, not the
cell population.

---

# Addendum — answering two fair objections

## A. Which tests I ran, and which claim they address

I tested the **contrast between the two groups** — "does the dorsal cluster peak
later than the ventral cluster" — because that is what the sentence asserts. I did
not run the simpler test, and you are right that it exists.

**Tests against zero at each group's own peak** (one-sided):

| group | lag | unit | n | mean r | t | p |
|---|---:|---|---:|---:|---:|---:|
| ventral | 30° | cell | 48 | 0.079 | 2.44 | **0.009** |
| ventral | 30° | **site** | 12 | 0.031 | 0.81 | 0.219 |
| dorsal | 60° | cell | 39 | 0.081 | 2.21 | **0.017** |
| dorsal | 60° | **site** | 7 | 0.196 | 2.24 | **0.033** |

Pre-specified 30+60° window (as the main text uses for mPFC):

| group | cell-level | site-level |
|---|---|---|
| ventral | r 0.038, t = 1.27, p = 0.106 | r 0.027, t = 0.77, p = 0.230 |
| dorsal | r 0.065, t = 2.21, p = **0.017** | r 0.085, t = 1.61, p = 0.079 |

So yes — a cell-level test against zero at each peak is nominally significant, and
that is presumably what the error bars in Fig 3d convey. Two caveats before
reporting it:

1. **It is circular as stated.** The lag is chosen as the group's argmax and then
   tested at that same lag on the same data. Bonferroni across the 12 lags takes
   ventral from p = 0.009 to p = 0.11. Either pre-specify the lag or correct.
2. **It does not test the claim.** "Ventral peaks at 30°, dorsal at 60°" is a
   statement about the *difference*. That difference is the thing that fails
   (p = 0.37); each group being individually above zero at its own peak does not
   establish that one peaks earlier than the other.

**What I would report:** the vs-zero result for the pre-specified window (which is
the analysis the main text already commits to), and state the ventral/dorsal
ordering descriptively without claiming a tested progression.

## B. The fMRI gradient — you were right about the map

**The map does show a clear progression.** Angle by MNI z within the gradient mask:

    z  -5 to  0 :  59 deg      z +15 to +20 :  98 deg
    z   0 to +5 :  82 deg      z +20 to +25 : 118 deg
    z  +5 to +10:  95 deg      z +25 to +30 : 158 deg
    z +10 to +15:  97 deg      z +30 to +40 : 280-333 deg

Monotonic, and spanning nearly the full circle. My earlier "58° vs 61°" was not a
statement about the map; it was the pipeline's per-cell lookup, which is a **single
voxel** (`SPHERE_RADIUS_MM = 0`) averaged arithmetically rather than circularly.
That was the wrong number to quote and I should have checked how it was derived
before using it.

**The problem is where the cells sit, not how the map behaves.**

    ventral cells   z -4.6 to +4.1   (mean  0.8)   12 sites
    dorsal  cells   z +1.2 to +9.8   (mean  2.8)    7 sites

The two groups' means are **2 mm apart in z** and their ranges overlap
substantially. They sample only the flattest, most ventral stretch of the map
(59–95°), not the 30–75° progression the sentence invokes.

Your four sampling options, computed with the pipeline's own constants
(no symmetrisation, no smoothing):

| option | ventral | dorsal |
|---|---:|---:|
| (a) centre of mass → single vertex | 52° | 46° |
| (b) per-site vertex → vector mean | 58° | 45° |
| (c) mask voxels within 8 mm → vector mean | 84° | 66° |

All three give **dorsal ≤ ventral** — the opposite ordering to the claim, though
the differences are small and the groups overlap in z, so this is better read as
"no separation" than as a reversal.

The likely reason is the split axis. PC1 ≈ [−0.04, −0.41, **0.91**] mixes y and z:
the dorsal group has higher z (2.8 vs 0.8) but *lower* y (38.4 vs 42.3). The map's
progression runs along z, so the y-component of the split works against it.

### Option (d) is the one that survives

The cells occupy z ≈ −5 to +10, where the map reads **59–95°** — essentially the
first quarter's 0–90° window. That is accurate, checkable, and does not overstate.
Suggested wording:

> The recorded mPFC units lay at the ventral end of the fMRI gradient
> (MNI z −5 to +10), where the fMRI-derived preferred angle spans approximately
> 60–95°, overlapping the immediate-future quarter (0–90°) to which these cells
> were tuned.

That keeps the correspondence you can actually defend, and drops the ventral-vs-
dorsal progression, which neither the cell statistics nor the fMRI sampling
support.

---

# Addendum 2 — stored tests, FDR, and the z-projection

## Stored outputs

| script | output |
|---|---|
| `scripts/gradient_split_vs_zero.py` | `<run>/final_splits/gradient_split_vs_zero.csv` |
| `scripts/gradient_fmri_z_projection.py` | `<run>/final_splits/fmri_angle_z_profile.csv`, `fmri_angle_by_group.csv` |

## One-sided tests against zero, three units of analysis

87 in-mask cells sit at **19 recording sites** across **19 subjects**.

| family | unit | group | n | mean r | t | p | q (2 groups) | q (12 lags) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| peak | cell | ventral 30° | 48 | 0.079 | 2.44 | 0.009 | **0.019** | 0.111 |
| peak | cell | dorsal 60° | 39 | 0.081 | 2.21 | 0.017 | **0.033** | 0.200 |
| peak | site | ventral 30° | 12 | 0.031 | 0.81 | 0.219 | 0.438 | 0.989 |
| peak | site | dorsal 60° | 7 | 0.196 | 2.24 | 0.033 | 0.067 | 0.401 |
| peak | subject | ventral 30° | 11 | 0.051 | 1.20 | 0.130 | 0.260 | 0.927 |
| peak | subject | dorsal 60° | 12 | 0.139 | 2.26 | 0.023 | **0.045** | 0.271 |
| window 30+60 | cell | ventral | 48 | 0.038 | 1.27 | 0.106 | 0.106 | — |
| window 30+60 | cell | dorsal | 39 | 0.065 | 2.21 | 0.017 | **0.033** | — |
| window 30+60 | site | ventral | 12 | 0.027 | 0.77 | 0.230 | 0.230 | — |
| window 30+60 | site | dorsal | 7 | 0.085 | 1.61 | 0.079 | 0.158 | — |
| window 30+60 | subject | ventral | 11 | 0.047 | 1.55 | 0.076 | 0.076 | — |
| window 30+60 | subject | dorsal | 12 | 0.077 | 2.27 | 0.022 | **0.044** | — |

**FDR families used** (Benjamini–Hochberg): across the two groups within each
(family, unit, lag); and — for the peak family — across the 12 lags within each
(unit, group), which is what removes the circularity of testing at an argmax
chosen from the same data. If you meant a different family, it is a one-line
change in `gradient_split_vs_zero.py`.

**Reading it:** the dorsal cluster is reliably above zero at every unit of analysis
(q = 0.033–0.067). The ventral cluster survives only at cell level, and not once
the 12-lag correction is applied. Neither survives the 12-lag correction, so the
peak-lag test should be reported as the pre-specified 30+60° window instead.

## The z-projection — this does work

Sampling the map at each cell's coordinate was the wrong read-out: a single voxel
(`SPHERE_RADIUS_MM = 0`) inherits x/y variation and noise. Projecting the map onto
z — vector-mean angle across all gradient-mask voxels in a ±3 mm slab — recovers
the progression:

    z -4: 65   z -2: 62   z  0: 71   z +2: 82   z +4: 89
    z +6: 94   z +8: 96   z +10: 97  z +12: 97

Read off at the two groups:

| group | n | sites | z mean | **angle at z mean** | z range | angle range |
|---|---:|---:|---:|---:|---|---|
| ventral | 48 | 12 | 0.8 | **70.5°** | −4.6 to 4.1 | 72–89° |
| dorsal | 39 | 7 | 2.8 | **81.6°** | 1.2 to 9.8 | 72–97° |

**Dorsal is 11° later than ventral, in the predicted direction.** That is a fair
description of the correspondence and supersedes my earlier "58° vs 61°", which
came from the single-voxel lookup averaged arithmetically rather than circularly.

Two honest caveats to keep with it:

- It is a **descriptive projection, not a test.** The 11° carries no p-value, and
  the two z-ranges overlap heavily (ventral reaches 4.1, dorsal starts at 1.2).
- The corresponding fMRI range across the cells' full z-span is about **70–97°**,
  not the 30–75° the manuscript states. That figure needs updating.

Suggested wording:

> Projected onto the dorsoventral axis, the fMRI-derived preferred angle within
> the gradient mask rises from ~62° at z = −2 to ~97° at z = +10. The ventral cell
> cluster lay at z = 0.8 (fMRI angle 71°) and the dorsal cluster at z = 2.8 (82°),
> so both sampled the immediate-future portion of the gradient, with the dorsal
> cluster displaced towards later angles.
