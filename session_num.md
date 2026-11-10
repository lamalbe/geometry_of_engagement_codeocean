# Session counts (day-1 Familiar, \(N \ge 30\))

Use these \(n\) values in figure captions and methods. Do not invent other session counts.

## Cohort

Day-1 (`session_number==1`) Familiar sessions, image sets G and H.

- Total sessions in the default cohort: **48** (38 G, 10 H)
- Units: quality-filtered `df_units` (snr>1, isi_violations<1, firing_rate>0.1)
- Threshold: at least **30 neurons** in the named area (or in **both** areas of a pair)

## Area labels

Grouped labels are not always Allen acronyms:

| Label | Allen `structure_acronym`s |
|---|---|
| LGd | LGd |
| VISpl | VISp, VISrl, VISl, VISal |
| VISm | VISpm, VISam |
| SCMRN | MRN, SCig, SCiw |

Single cortical areas use the Allen acronym as-is: VISp, VISrl, VISl, VISal, VISpm, VISam.

## Single areas (\(N \ge 30\))

### Circuit (grouped)

| Area | Sessions |
|---|---|
| LGd | 16 |
| VISpl | 48 |
| VISm | 47 |
| SCMRN | 31 |

### Single cortical areas

| Area | Sessions |
|---|---|
| VISp | 46 |
| VISrl | 44 |
| VISl | 44 |
| VISal | 47 |
| VISpm | 47 |
| VISam | 44 |

## Channel pairs (\(N \ge 30\) in both areas)

| Pair | Sessions |
|---|---|
| LGd, VISpl | 16 |
| VISpl, VISm | 47 |
| VISm, SCMRN | 31 |

Image-identity decoding on set G is a subset of LGd → VISpl: **\(n=13\)**. VIS → SCm/MRN image-identity decoding (set G, \(N \ge 30\) in both areas): **\(n=26\)**. The additional VISpl → SCm/MRN paired analyses also retain **\(n=31\)**.

## Within-block unengagement

Sessions with at least 12 min of lick-rate-defined unengagement in the active block: **12**.

Among those sessions, quality-filtered units with \(N \ge 30\):

| Area | Sessions |
|---|---|
| VISp | 12 |
| VISrl | 10 |
| VISl | 10 |
| VISal | 12 |
| VISpm | 12 |
| VISam | 11 |
