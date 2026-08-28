# Coordinate provenance audit — `scripts/cell_to_roi_july26.py`

Audit date: 2026-08-27. Table audited: `derivatives/neurons_with_ROI_labels.csv`, 984 cells.

Question asked: *where does every single coordinate come from — no inventions, no
cross-guessing?* This document answers it exhaustively. Every cell in the table
falls into exactly one of the rows below.

## Summary

| # cells | provenance | comes from | trustworthy? |
|--------:|---|---|---|
| 608 | `baylor_v2026_bundle_micro` | `{CODE}-electrodes_v2026.csv`, `microwires` row, MNI152 **as shipped by Baylor** | **yes** — site's own number, used verbatim |
| 140 | `ucla_v2026_coord_match` | big table, **confirmed ≤ 0.5 mm** against `sub-{NNN}_localizations.xlsx` | **yes** — independently corroborated |
| 97 | `utah_reconstructed_via_ElecXYZMNI{Proj,Raw}` | `s{NN}/electrodes/Electrodes.mat`, reconstructed via ChannelMap → LabelMap → MicroElec → ElecXYZMNI | **yes** — site's own file, and agrees with big table ≤ 3 mm |
| 35 | `baylor_v2026_bundle_305to152_unreliable_file` | Baylor file's **MNI305**, re-transformed with the Fischl affine because that file's own MNI152 failed the ≤ 8 mm consistency gate | **mostly** — site's number, one transform applied |
| 26 | `baylor_v2026_micro_reconstructed_from_macro` | Baylor macro probe + 3.15 mm along the insertion axis (subject YER ships no micro rows) | **inferred** — but the 3.15 mm constant is Baylor's own, identical across all 119 bundles that do have micro rows; reproduces them to 0.25 mm median |
| **78** | **`utah_bigtable_recon_disagrees_gt3mm`** | **the hand-entered big table**, after *discarding* the reconstruction from the site's `.mat` | **NO — see below** |

748 / 984 (76 %) come straight from a site file. 61 more are derived from a site
file by a documented, validated transform. **78 (7.9 %) do not.**

## The 78 problem cells

`cell_to_roi_july26.py` reconstructs a Utah coordinate from the patient's own
`Electrodes.mat`, then compares it to the big table. Policy in code:

```python
UTAH_RECON_TOL_MM = 3.0
if dist <= UTAH_RECON_TOL_MM:  # use the reconstruction
else:                          # keep the BIG TABLE, discard the reconstruction
```

So when the site's own electrode file and the hand-entered table disagree, **the
hand-entered table wins**. That is backwards: the big table was filled in by hand,
the `.mat` is the authoritative source.

Disagreements are not marginal — median **37.9 mm**, max **62.3 mm**:

| subject | n | median | max |
|---|---:|---:|---:|
| UT1_sj202308 | 16 | 55.1 mm | 62.3 mm |
| UT202409 | 11 | 38.9 mm | 40.0 mm |
| UT1-202314 | 9 | 18.8 mm | 19.2 mm |
| UT1-202503 | 8 | 52.0 mm | 53.9 mm |
| UT202314 | 6 | 18.8 mm | 19.2 mm |
| UT1-202217 | 5 | 10.4 mm | 15.0 mm |
| UT1-202418 | 5 | 37.5 mm | 37.5 mm |
| UT1-202422b | 4 | 52.5 mm | 53.9 mm |
| UT1_sj202309 | 4 | 55.1 mm | 62.3 mm |
| UT202413 | 4 | 3.5 mm | 3.5 mm |
| UT1-202302 | 3 | 37.5 mm | 37.5 mm |
| UT1-202311 | 3 | 37.6 mm | 37.6 mm |

73 of 78 disagree by > 10 mm; 30 by > 40 mm. At that magnitude one of the two
numbers is simply wrong — this is not localisation error.

These 78 currently contribute: **mOFC 48, mPFC 18, HC_anterior 9, EC 3.**

## The placeholder coordinate

Four Utah subjects have **no `Electrodes.mat` anywhere** — s52 (UT202421),
s53 (UT202422b), s54 (UT202418), s55 (UT202503). Their big-table entries are:

```
UT1-202418    6 cells -> 1 distinct coord
UT1-202421    6 cells -> 5 distinct coords
UT1-202422b   4 cells -> 1 distinct coord   ] all three identical:
UT1-202503    8 cells -> 1 distinct coord   ]  (4.55, 29.50, -20.63)
```

**17 cells sit on that one repeated point**, and the atlas calls it **mOFC**. It is
a placeholder, not anatomy. It is also why `discover_utah_mats()` "matched" those
subjects to s47/s48 at 100 % — a single point trivially matches something in any
128-electrode file.

## Two mechanisms that let this through

1. **`discover_utah_mats()` has no uniqueness constraint.** It coord-matches
   big-table cells (3–16 per subject) against every folder's electrode pool and
   takes any folder covering ≥ 50 % at ≤ 0.5 mm. s47's file won for six patients.
   Measured against the folder numbering, that distrust was unwarranted: **8 of 12
   Utah subjects match their own `s{NN}` folder at 100 %.** The folder mapping is
   reliable; the coord-matching is what is not.
2. **The reference it matches against is the hand-filled big table**, whose
   coordinates are exactly what is in question. The check is circular.

## Recommendations (not yet applied — these change published cell ROIs)

1. **Resolve Utah `.mat` files by folder** (`s{NN}` → own folder, then same patient),
   not by coord-matching. Already done for the SWR pipeline in
   `mc.analyse.contact_anatomy.resolve_utah_mat`.
2. **Invert the `UTAH_RECON_TOL_MM` policy**: prefer the reconstruction from the
   patient's own `.mat`; treat a > 3 mm disagreement as evidence the *big table*
   is wrong, not the file.
3. **Drop or quarantine the 17 placeholder cells** — they carry no anatomical
   information and currently all land in mOFC.
4. **Re-derive the 4 missing files** (s52–s55) from Utah if possible; otherwise
   those 24 cells have no usable coordinate.

Item 3 matters most for interpretation: mOFC is one of the paper's headline ROIs,
and 48 of the 78 disputed cells plus all 17 placeholder cells currently sit there.

## What is NOT affected

- Baylor (669 cells) — every coordinate traces to a v2026 site file.
- UCLA (140 cells) — every coordinate independently corroborated to ≤ 0.5 mm.
- The 97 Utah cells reconstructed from their own `.mat` and agreeing with the big
  table.

That is 906 / 984 cells (92 %) with clean provenance.

---

# Addendum — 2026-08-27, after re-download: is the Utah reconstruction trustworthy?

## Is the reconstruction method sound?

Tested independently of the atlas rules. The microwire **label** (`mLHIP3`) comes
from `LabelMap`; the **coordinate** comes from `ElecXYZMNIProj/Raw` via `MicroElec`.
Different arrays — so if the ordering assumption in `build_micro_map` were wrong,
label and coordinate would decouple.

Two checks per microwire:
- **hemisphere** — does `sign(MNI_x)` match the `mL`/`mR` in the label? (chance 50 %)
- **region** — does the atlas ROI at the coordinate match the label stem
  (`HIP`→hippocampal, `OFC`→orbitofrontal, `ACC`→cingulate, …)? (chance ~15 %)

| subject | folder | bilateral | hemisphere | region |
|---|---|---|---:|---:|
| UT1-202212 | s01/electrodes | yes | 24/24 | 24/24 |
| UT1-202214 | s02/electrodes | yes | 24/24 | 17/24 |
| UT1-202216 | s04/electrodes | no | 24/24 | 16/16 |
| UT1-202217 | s06/electrodes | no | 21/24 | 24/24 |
| UT1-202302 | s47/electrodes | yes | 24/24 | **8/24** |
| UT1-202306 | s17/electrodes | yes | 24/24 | 24/24 |
| UT1-202314 | s29/electrodes | yes | 24/24 | 16/24 |
| UT1_sj202308 | s24/electrodes | yes | **24/24** | **24/24** |
| UT1_sj202309 | s23/electrodes | yes | **8/24** | **0/24** |
| UT202409 | s39/electrodes | yes | 24/24 | **8/24** |
| UT202413 | s41/electrodes | yes | 24/24 | 24/24 |
| **TOTAL** | | | **245/264 (92.8 %)** | **185/256 (72.3 %)** |

**Verdict: the method is sound.** 92.8 % hemisphere and 72.3 % region agreement
against chance rates of 50 % and ~15 % could not arise from a wrong ordering. The
`ElecXYZMNIProj` path scores 96.2 % on hemisphere, the `MicroElecRaw` fallback
88.9 %. But it is **not uniformly safe** — three subjects fail and must be handled
individually, not trusted wholesale.

## Resolving by folder fixes the s23/s24 confusion

`UT1_sj202308` (session 24) was assigned **s23's** file by coord-matching:

| | hemisphere correct |
|---|---:|
| via s23's mat (coord-matched, current) | 1/16 |
| via s24's own folder | **16/16** |

That settles the contradiction in the earlier audit: the published laterality was
right, the *file assignment* was wrong. Resolving by folder fixes it outright.

## s23 (UT1_sj202309) is genuinely broken — exclude, do not guess

Its `MicroElec` is empty, so it uses the `MicroElecRaw` fallback, and the pairing
is misaligned:

```
chan97-104  mROFC1-8   x ≈ -16   (labelled Right, coord is Left)
chan105-112 mRACC1-8   x ≈  +5   (correct)
chan113-120 mLHIP1-8   x ≈  +8   (labelled Left, coord is Right —
                                   and |x|=8 is not hippocampus in EITHER
                                   hemisphere; hippocampus sits at |x| ≈ 25-35)
```

Region agreement 0/24. This subject's coordinates cannot be reconstructed by the
current method and should be **excluded**, not resolved by preferring either source.

`s47` and `s39` pass hemisphere 24/24 but score 8/24 on region — worth a look
before trusting them, though the failure mode is milder than s23's.

## MATLAB v7.3 files are silently unreadable — 4 subjects lost

`_load_mat` returns `LabelMap` as a bare `None` for HDF5-format `.mat` files,
because the string cells are stored as HDF5 object references and are never
dereferenced. Affected:

| folder | subject | format | microwires found |
|---|---|---|---:|
| s48/electrodes | UT1-202311 | **v7.3** | 0 |
| s52/Registered-selected | UT202421 | **v7.3** | 0 |
| s54/Registered | UT1-202418 | **v7.3** | 0 |
| s55/Registered-selected | UT1-202503 | **v7.3** | 0 |
| s24/electrodes (control) | UT1_sj202308 | v7 | 24 |

These four fall back to the big table **every time** — and three of them are
exactly the subjects sitting on the placeholder coordinate. Fixing the HDF5
dereference is a pure read bug with no judgment involved, and would recover them.

## Files: current status after re-download

Three of the four previously-missing files are now present, but under
`Registered/` or `Registered-selected/` rather than `electrodes/`, which is the
only directory the loader searches:

| session | subject | location | status |
|---|---|---|---|
| s52 | UT202421 | `Registered-selected/` | present, **not searched** |
| s54 | UT1-202418 | `Registered/` | present, **not searched** |
| s55 | UT1-202503 | `Registered-selected/` | present, **not searched** |
| s53 | UT202422b | — | **genuinely absent** |
| s30 | UT202314 | — | absent, but same patient as s29 ✓ |
| s42 | UT202413 | — | absent, but same patient as s41 ✓ |

## UCLA: clean

`load_ucla_v2026` reads `Sheet1` (the second sheet, the one carrying `MNI_x/y/z`
and `isMicro`) — correct. It does **not** filter on `isMicro`, so a microwire cell
could in principle match a macro contact. Measured: it never does.

| subject | cells | → micro row | → macro row | unmatched |
|---|---:|---:|---:|---:|
| UC2-0576 | 16 | 16 | 0 | 0 |
| UC2-0578 | 6 | 6 | 0 | 0 |
| UC3-0559 | 4 | 4 | 0 | 0 |
| UC3-0573 | 9 | 9 | 0 | 0 |
| UC3-0577 | 42 | 42 | 0 | 0 |
| UC3-0582 | 63 | 63 | 0 | 0 |
| **TOTAL** | **140** | **140** | **0** | **0** |

All 140 UCLA cells are corroborated against the correct microwire row at ≤ 0.5 mm.
Adding the `isMicro` filter is a free safeguard that changes no current output.

## Priority of fixes

1. **Search `Registered/` and `Registered-selected/`** as well as `electrodes/` — recovers s52, s54, s55.
2. **Dereference HDF5 string cells** in `_load_mat` — recovers s48, s52, s54, s55.
3. **Resolve by folder, not coord-match** — fixes UT1_sj202308 (1/16 → 16/16).
4. **Exclude s23 (UT1_sj202309)** — reconstruction demonstrably misaligned.
5. **Add the `isMicro` filter** to `load_ucla_v2026` — safeguard, no current effect.
6. **Investigate s47 / s39** region mismatch (8/24 each) before trusting them.

1, 2 and 5 are unambiguous bug fixes. 3, 4 and 6 change which coordinate a cell
gets and therefore change published cell ROIs.

---

# Addendum 2 — 2026-08-27: how much is still guessed?

## Every electrode file states its own identity

Each Utah `.mat` carries `Fname`, the original acquisition path
(`D:\Data\UIC202311\Imaging\Registered\Electrodes.mat`), and `PatientIDStr`.
That is the file's own statement of which patient it belongs to, so coord-matching
against a hand-entered table is unnecessary. Reading it (`mat_patient_id`):

| folder | manifest says | file declares | |
|---|---|---|---|
| s01 … s41 (10 folders) | matching | matching | ✓ |
| **s47** | UT1-202302 | **202311** | ✗ holds s48's patient |
| s48 | UT1-202311 | 202311 | ✓ |
| s52 | UT202421 | 202421 | ✓ |
| **s53** | UT202422b | **202421** | ✗ duplicate of s52 |
| s54 | UT1-202418 | 202418 | ✓ |
| s55 | UT1-202503 | 202503 | ✓ |

So **UT1-202302** and **UT1-202422b** have no electrode file. s47's folder holds the
v7 export of patient 202311 — same 165 electrodes as s48's v7.3 export.

⚠ These files contain patient **names** in `PatientIDStr`. Use the numeric ID.

## The ordering assumption is removable

`build_micro_map` infers the label↔coordinate pairing by sorting microwires by
amplifier channel and matching that against `MicroElec` — validated on only s02 and
s06, and it fails on s23 (labels and coords misaligned; `mLHIP1-8` landed at x ≈ +8).

Every file also carries `MicroElecRaw` (1-based row indices), `ElecMapRaw` (col 0 =
label) and `ElecXYZMNIRaw` (coordinate). Indexing all three by the **same row**
gives label and coordinate together, with no ordering assumption at all
(`build_micro_label_map`). Validated on all 15 files: **352/360** microwires have
`sign(MNI_x)` consistent with the `mL`/`mR` in their own label — the 8 exceptions
are OFC contacts within 2.5 mm of the midline, where the sign is uninformative.
s23 goes from **8/24 to 24/24**.

For cells labelled `chanN`, the bridge is `ChannelMap1/2` → `(r,c)` → `LabelMap`
(a direct cell lookup), then label → coordinate. The v7.3 `Electrodes.mat` files
have no `ChannelMap`, but their folders carry a sibling `ChannelMap.mat` that does.

## Census: read vs guessed, all 984 cells

| n | category | |
|---:|---|---|
| 776 | **read directly** from the site's electrode file | Baylor 608 + Utah 168 |
| 140 | **verified identical** to the site file (≤ 0.5 mm) | UCLA — big-table number, corroborated |
| 61 | **derived** from the site file by a documented transform | Baylor: 35 MNI305→152, 26 macro + 3.15 mm |
| **7** | **guessed — no source file exists** | UT1-202302 (3), UT1-202422b (4) |

Utah specifically: **168/175 cells, 15/17 subjects fully readable** (was 97/175).

**Answer: 7 cells across 2 subjects are still guessed — 0.7 % of the dataset.**

## Status of fixes

Applied to `mc/analyse/anatomy_sources.py`:
- `_load_mat` now dereferences HDF5 object references, so MATLAB v7.3 files read
  (previously `LabelMap` came back as bare `None` and s48/s52/s54/s55 yielded zero
  microwires, silently falling back to the big table).
- `mat_text`, `mat_patient_id` — identity from the file's own `Fname`.
- `build_micro_label_map` — direct label↔coordinate read, no ordering assumption.

**Not yet wired into `cell_to_roi_july26.py`.** Doing so changes published cell
ROIs and should be done with a before/after diff.

---

# Addendum 3 — 2026-08-27: full rebuild from site files only

Every cell's coordinate re-derived from the recording site's own electrode file.
No coord-matching, no fallback to the hand-entered big table, no inference except
where explicitly tagged. Output:
`derivatives/ROI_assignment/coordinate_rebuild_2026-08-27/`.

## How each site's coordinate is now established

| site | rule |
|---|---|
| **Baylor** (669) | v2026 CSV `microwires` row (or `sEEG-micro` where no micro row), **MNI152 as shipped**. If the file fails its own MNI152-vs-MNI305 gate (≤ 8 mm mean), its MNI305 through the Fischl affine. |
| **UCLA** (140) | `sub-{NNN}_localizations.xlsx`, sheet `Sheet1`, rows with `isMicro == TRUE`, **MNI as shipped**. |
| **Utah** (175) | `Electrodes.mat`, resolved by the patient ID the file declares in its own `Fname`. Coordinate = `ElecXYZMNIRaw[MicroElecRaw[i]−1]`, label = `ElecMapRaw[MicroElecRaw[i]−1, 0]` — **same row index for both**, so no ordering assumption. `chanN` cells bridged via `ChannelMap1/2` → `LabelMap`. |

Resulting provenance across all 984 cells:

| n | tag | |
|---:|---|---|
| 608 | `baylor_file_micro` | as shipped |
| 168 | `utah_file_micro` | as shipped |
| 140 | `ucla_file_micro` | as shipped |
| 35 | `baylor_file_305to152` | BY2-YEN — same file's MNI305 + Fischl affine |
| 26 | `INFERRED_macro_plus_3.15mm` | BY2-YER — the only remaining inference |
| **7** | `NO_FILE` | UT1-202302 (3), UT1-202422b (4) |

**Baylor and UCLA coordinates do not move at all (0.00 mm).** The published table
already used file coordinates for both sites. Every change is Utah:

| subject | cells | median shift vs published |
|---|---:|---:|
| UT1-202503 | 8 | 52.8 mm |
| UT1-202418 | 6 | 42.8 mm |
| UT1-202421 | 6 | 42.7 mm |
| UT1_sj202308 | 16 | 17.5 mm |
| UT202314 | 10 | 2.7 mm |
| UT202413 | 39 | 2.7 mm |
| UT1-202311 | 9 | 1.3 mm |
| all other Utah subjects | 81 | 0.00 mm |

Note **UT1_sj202309 (s23) does not move**. Its published coordinate already equalled
the direct read — so the hand-entered big table was *right* for s23 and the old
ordering-based reconstruction was what was wrong. The old code correctly kept the
big table there, for the wrong reason.

## ROI consequences

**26 of 977 cells (2.7 %) change atlas ROI — all Utah.**

| n | transition |
|---:|---|
| 12 | mOFC → HC_anterior |
| 6 | mOFC → mPFC |
| 3 | EC → HC_anterior |
| 3 | mPFC → mOFC |
| 2 | mOFC → HC_mid |

Driven by UT1-202503 (7), UT1-202418 (6), UT1-202421 (6), UT1_sj202308 (3),
UT1-202217 (2), UT1-202311 (2).

## Three strictness tiers

| ROI | published | A | B | C |
|---|---:|---:|---:|---:|
| EC | 38 | 35 | 35 | 35 |
| HC_anterior | 276 | 270 | 288 | 291 |
| HC_mid | 231 | 218 | 226 | 233 |
| PCC | 61 | 61 | 61 | 61 |
| mOFC | **163** | **130** | **139** | **139** |
| mPFC | 155 | 142 | 142 | **158** |
| **cells kept** | 924 | 856 | 891 | **917** |

- **A** — as shipped only. Drops BY2-YEN (35) and BY2-YER (26).
- **B** — A + the MNI305→152 Fischl transform of BY2-YEN's own file.
- **C** — B + BY2-YER's macro + 3.15 mm reconstruction. Only the 7 `NO_FILE` cells excluded.

## Recommendation: tier C

The two things tier C keeps are **not** guesses of the kind that caused this audit:

- **BY2-YEN (35 cells)** — the coordinate is the site's own MNI305, put through the
  standard Fischl affine. Nothing is invented; only a coordinate-space conversion is
  applied, because that file's shipped MNI152 disagrees with its own MNI305.
- **BY2-YER (26 cells)** — that file ships no micro rows. The micro tip is placed
  3.15 mm beyond the deepest macro contact along the insertion axis. **The 3.15 mm
  constant is Baylor's own**: it is identical across all 119 bundles that *do* have
  `microwires` rows, and the reconstruction reproduces their supplied positions to
  0.25 mm median / 1.07 mm max. This is a validated reconstruction from the site's
  own geometry, not a hand-entered value.

Dropping BY2-YER costs **16 mPFC cells** (158 → 142), which matters for the mPFC
gradient analysis. Both remain individually flagged in `coord_source` so either can
be excluded in a sensitivity analysis.

What tier C does eliminate is every actual guess: the placeholder coordinate, the
coord-matched wrong-patient files, and the policy of preferring the hand-entered
table over the site's file.

## Manuscript figures that change

`alt_final_roi` counts feed Figure 2c and the per-ROI n's throughout:

- **mOFC 163 → 139** — the largest change. Affects Fig 2c, the mOFC state-RSA
  (β = 0.094), and the mOFC row of Fig 5a.
- **EC 38 → 35** — the sustained-cell fraction "12/38, 31.6 %" and the
  EC-versus-rest Fisher test must be recomputed.
- **HC_anterior 276 → 291**, **HC_mid 231 → 233**, **mPFC 155 → 158**, PCC unchanged.
- The mPFC gradient analysis (74/155 units inside the gradient mask, 42/32 median
  split) needs re-running: three mPFC cells are new and the Utah coordinates moved.
- Session counts per ROI change wherever a subject's cells changed region.
