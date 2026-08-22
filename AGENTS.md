# clean_scripts — agent context

Python `# %%` notebooks (run in Cursor/VS Code or as scripts). They analyze Allen Visual Behavior Neuropixels: how engagement (active vs passive) changes population geometry along visual cortex → midbrain.

Do not edit `data_analysis/` (legacy). Work here.

## Layout

| Path | Role |
|---|---|
| `code_gendata/` | Build pickles from raw sessions |
| `code_analysis/` | Plots/stats from those pickles |
| `code_cartoon/` | Paper schematics (not data analysis) |
| `code_supplementary/` | Sanity checks, unit-count histograms |
| `data/` | Derived tables (`df_cortex2mid.pk`, `df_decoding.pk`, stats, geometry) |
| `results/` | Figures (PDF) |

Raw per-session tables live outside the repo: `/data/allen_engagement_data/{day1,day2}/<session_id>/`. Allen cache: `/data/AllenNPXVisualBehavior`.

## Pipeline (run gendata before analysis)

1. `preprocess_data.py` → session pickles (`df_units`, `df_spikes`, `df_activity`, behavior). 50 ms bins.
2. Area-sliced activity / stats:
   - `generate_VIS2MID_activity.py` → `data/df_cortex2mid.pk`
   - `generate_decoding.py` → `data/df_decoding.pk` (LGd + VISpl + VISm + SCMRN, keeps `licks`)
   - `generate_poisson_stats.py` → `df_poisson_stats.pk`, `df_poisson_stats_subsampled.pk`
   - `generate_covar_stats.py` → `df_stats_area.pk`
   - `VISp_cloud_shape.py` → `visp_cloud_shape.pk`
   - `VISp_PC_cloud.py` → PCA cloud figures only (no table)
3. `generate_VIS2MID_geometry.py` → PCA/SVD basis and overlap tables (needs `df_cortex2mid.pk`)
4. `code_analysis/*` reads `data/*.pk` and writes `results/`

`raster_plot.py` is standalone (AllenSDK rasters); it does not use `df_activity`.

## Default cohort (most gendata)

Day-1 (`session_number==1`) Familiar sessions, image sets G and H. Viewing bins, `time_from_stimulus >= 0.1` s. Drop omitted, novel, and change presentations.

`active` is the task block (engaged vs passive viewing), not a spike feature.

## Area labels (grouped, not always Allen acronyms)

| Label | Allen `structure_acronym`s |
|---|---|
| VISp | VISp |
| VISpl | VISp, VISrl, VISl, VISal |
| VISm | VISpm, VISam |
| LGd | LGd |
| SCMRN | MRN, SCig, SCiw |

Channels used in geometry/MLR: `VISm → SCMRN`, `VISpl → SCMRN` (geometry); MLR also uses `LGd → VISpl`, `VISpl → VISm`. Image identity decoding also uses `LGd → VISpl` (not lick decoding).

## Neuron-order contract (do not break)

`df_units` is sorted by `probe_vertical_position` descending, then `reset_index()` so index is `0 … N-1`.

`df_spikes` is reordered with `.loc[df_units.id]`. Binning stacks in that row order, so **`df_activity.spikecounts[i]` is `df_units.iloc[i]`**.

Area slicing uses leftover index labels as positions:

```python
idx = df_units[df_units.structure_acronym.isin(acronyms)].index.values
spikecounts[idx]  # numpy positional indexing
```

Never `reset_index()` on a per-area `df_units` subset before taking `.index`. After changing preprocess, rerun the gendata chain and `code_supplementary/check_units_spikes_order.py` (expect `OK` for every session).

## Notebooks

### code_gendata

- **preprocess_data.py** — AllenSDK → per-session pickles; quality filter (snr>1, isi_violations<1, firing_rate>0.1); optotagging for non-wt; spike + baseline binning; align running/pupil/licks/rewards into bins.
- **generate_VIS2MID_activity.py** — Slice `spikecounts` into VISpl / VISm / SCMRN; write `df_cortex2mid.pk` (`spikecounts_area`, `N`, `area`).
- **generate_decoding.py** — Same cohort as VIS2MID activity, plus LGd; keeps per-bin `licks`. Writes `df_decoding.pk`. Do not reuse `df_cortex2mid.pk` for choice decoding. Lick decoding uses VISpl / VISm / SCMRN; image identity also uses LGd → VISpl.
- **generate_VIS2MID_geometry.py** — Mean-centered PCA (local) and SVD (cross-area) for VISm–SCMRN and VISpl–SCMRN. No z-score, correlation, train/test, or SVCA. Writes `df_basis_vectors.pk`, cosine and subspace-overlap tables. `TOP_K=10`, min 30 neurons/area.
- **generate_poisson_stats.py** — Per (area, session, active, image): PR, mean rates (Hz), count variances; plus 200-bin time chunks. Skip area if `<30` units.
- **generate_covar_stats.py** — Subsampled covariance stats (PR, mean counts, running/pupil) for fine and grouped areas → `df_stats_area.pk`.
- **VISp_cloud_shape.py** — VISp-only cloud geometry (PR, volume, extension, image separation) → `visp_cloud_shape.pk`.
- **VISp_PC_cloud.py** — 2D/3D PCA clouds for VISp image pairs (figures in `results/`).
- **raster_plot.py** — Example session raster from the Allen cache (hard-coded `SESSION_ID`).

### code_analysis

- **poisson_metrics.py** — Treves–Rolls sparsity, Fano-like sparsity, mean rate; Poisson diagnostics. Circuit plots use LGd / VISpl / VISm / SCMRN. VISp-only panel: a_sparsity, mean activity (`R_emp`), power-law and exponential gain. Part A: one complete four-area session; 20 example-neuron draws (`_choice_00`…`_19`). Part B: Fano histograms for every session with ≥50 neurons in all four circuit areas.
- **vis_circuit_metrics.py** — Circuit-wide metrics from `df_stats_area.pk` (PR, rates, active vs passive).
- **vis_primary_metrics.py** — VISp: merge `df_stats_area.pk` with `visp_cloud_shape.pk`.
- **multiple_linear_regression.py** — Per-channel MLR: predict target `rbar` / `PR_emp` from speed, pupil, source `rbar`, source `PR_emp`; active and passive fit separately.
- **cross_area_VIS2MID_channel.py** — Single-session VISm–SCMRN covariance / SVD heatmaps and trial clouds from `df_cortex2mid.pk` (hard-coded session/image).
- **alignment_VIS2MID_channel.py** — Plots for one channel (default `VISm_to_SCMRN`): SVD PR/spectrum (session × image; twin boxplot of SV ratios + PR ratio), VISm-only PC spectra (point plots + ratio boxplots), max cosine, subspace overlap. Filled boxes with t-test asterisks. Also expects `df_cross_covariance_in_local_subspace.pk` and `df_local_variance_in_cross_subspace.pk` if those cells are run.
- **decoding_analysis.py** — Active-trial lick vs no-lick logistic decoder (`N>=30`). Channels: VISm → SCMRN (main) and VISpl → SCMRN (supplement). Decode from full space, communication \(k\)-D, orthogonal complement, random \(k\)-D, local PCA, no-lick SVD, and passive communication / PCA axes. Also records \(\alpha_k=\|B_k^\top w\|^2\) (orientation of the full-space decoder; supplement). Labels are licks on **non-change** images, not change reports. Writes `df_decoding_{repeat,session,alpha_session}.pk` and `results/decoding_*.pdf` (VISpl figures use `_vispl_scmrn`).
- **image_decoding_analysis.py** — 8-way image identity (image set G) from flash-averaged active counts. Channels: VISm → SCMRN, VISpl → SCMRN, LGd → VISpl. Communication / PCA / random / complement, plus active flashes on passive communication / PCA axes (no no-lick SVD). Chance = \(1/8\). Writes `df_image_decoding_*.pk` and `results/image_decoding_*.pdf` (VISpl tag `_vispl_scmrn`; LGd tag `_lgd_vispl`).

### code_cartoon

- **decoding_schematic.py** — Two-panel cartoon (VISm vs SCMRN). Circle = directions in population space; cones from the origin = readout subspaces; red arrows = components of one decoder \(w\) (length \(\sim |e_\theta^\top w|\)). Writes `results/decoding_schematic.pdf`.

### code_supplementary

- **check_units_spikes_order.py** — `df_units.id` vs `df_spikes.unit` row order for all sessions.
- **unit_counts_by_area.py** — Pooled unit histograms by day (Familiar/Novel). Uses `df_units` only.
- **change_aligned_licks_rewards.py** — 20 random day-1 Familiar sessions; hit-change rasters (±2 s) with licks, rewards, and flash spans. Writes `results/change_aligned_licks_rewards_day1.pdf`.
- **suppl_behavior_session.py** — Day-1 Familiar running/pupil/licks/rewards traces, disengagement intervals, and session-level KDEs. Writes `data/df_behavior.pk`.
- **generate_covstats_behavior.py** — Cov stats by unengagement and by speed/pupil terciles, grouped by `(session_id, image_name)`. Builds `df_unengagement_summary.pk` / `df_view_terciles.pk` from `df_behavior.pk` if missing (`FORCE_REBUILD_BEHAVIOR_TABLES` to force). Writes `suppl_stats_*.pk`.
- **suppl_analysis_behavior.py** — Unengagement box/strip plots; speed/pupil point plots; PR~behavior MLR and channel MLR from `df_stats_area.pk`. Plot/MLR rows are session × image. Figures in `results/`.
- **suppl_PR_extrapolated.py** — PR(\(N\)) from covariance moments in `df_stats_area.pk` (average moments over `subsample_id` first). Figures `suppl_stats_PR_extrapolated_vs_N_*.pdf`.
- **suppl_PR_timewindows.py** — Within-block PR/\(N\) in five non-overlapping 11 min epochs after dropping the first 5 min of **each** block (active = block 0, passive = block 5). Last epoch absorbs leftover time. PR is per `(session_id, image_name)` inside an epoch (no `subsample_id`). Loads session `df_activity.pk`, not `df_stats_area.pk`. Cache: `suppl_stats_PR_timewindows.pk`. Figures `suppl_PR_timewindows_*.pdf`.
- **suppl_alignment_VISpl_SCMRN.py** — VISpl → SCMRN supplement of `alignment_VIS2MID_channel.py`: SVD PR/spectrum (twin boxplot + PR-ratio distribution), VISpl PC spectra, cosine across conditions, active↔passive overlap scatters. Session × image. Figures `suppl_alignment_VISpl_to_SCMRN_*.pdf`.

## Findings (keep in mind)

**PR time windows.** Engagement-dependent PR compression is not a slow drift through the recording. After discarding the first 5 min of each block, PR/\(N\) in 11 min epochs vs time-into-block stays separated (active vs passive). Window-averaged active/passive PR/\(N\) is below 1 in LGd, VISpl, VISm, and SCMRN (unit = session × image). Last 11 min of active vs first 11 min of kept passive still shows ratios below 1 in VISpl, VISm, and SCMRN (LGd is not significant).

**Lick decoding vs communication subspace (\(k=5\)).** \(\alpha_k\) is the fraction of the *full-space* decoder \(w\) that lies in the SVD modes. It is not a test that those modes carry lick information; it is a supplement. Main figures: schematic (`decoding_schematic.pdf`), readout boxes (full / comm / PCA / random / orthogonal), and communication vs PCA scatter. In SCMRN, \(\alpha_\text{active}>\alpha_\text{passive}\) can coexist with a flat performance-vs-\(\alpha_k\) relationship: alignment of \(w\) is not what explains session-to-session accuracy.

Where the information actually sits (`decoding_analysis.py`, session-mean balanced accuracy, \(n=31\)):

**VISm → SCMRN**

| Readout | SCMRN | VISm |
|---|---|---|
| Full space | 0.85 | 0.75 |
| Communication \(k\)-D (active) | 0.82 | **0.73** |
| Local PCA \(k\)-D (active) | 0.81 | 0.64 |
| Random \(k\)-D | 0.71 | 0.58 |
| Orthogonal to comm. | 0.72 | **0.56** (~chance) |
| Comm. \(k\)-D (passive axes) | 0.77 | 0.68 |
| PCA \(k\)-D (passive axes) | 0.74 | 0.63 |

**VISpl → SCMRN** (same midbrain neurons; \(V_k\) is now VISpl-defined)

| Readout | SCMRN | VISpl |
|---|---|---|
| Full space | 0.85 | 0.78 |
| Communication \(k\)-D (active) | 0.81 | **0.74** |
| Local PCA \(k\)-D (active) | 0.81 | 0.66 |
| Random \(k\)-D | 0.71 | 0.59 |
| Orthogonal to comm. | 0.72 | **0.58** (~chance) |
| Comm. \(k\)-D (passive axes) | 0.76 | 0.69 |
| PCA \(k\)-D (passive axes) | 0.74 | 0.65 |

VISm / VISpl: lick lives in the engaged communication subspace (\(U_k\) almost matches full space; the complement is chance; local PCs are not enough). Passive \(U_k\) still decodes above random, but below active \(U_k\). SCMRN: lick is a strong, distributed midbrain signal. Communication \(k\)-D \(\approx\) local PCs; a random 5-D slice and the complement of \(V_k\) still decode well. Passive \(V_k\) drops some (0.82→0.77) but remains well above chance. No-lick SVD \(\approx\) usual communication SVD in all three areas (the subspace is not a lick classifier fit on the same contrast). VISpl→SCMRN mirrors VISm→SCMRN.

\(\alpha_k\) (supplement): active \(>\) passive \(>\) random in both areas (VISm 31/31 sessions; SCMRN 30/31). Spearman(full-space accuracy, \(\alpha_\text{active}\)): VISm \(\rho=0.54\); SCMRN \(\rho=-0.20\) (flat). Alignment of \(w\) is not what explains midbrain accuracy.

**Decoding analysis (source for methods/results).** Script: `code_analysis/decoding_analysis.py`. Cartoon: `code_cartoon/decoding_schematic.py`. Activity table: `data/df_decoding.pk` from `generate_decoding.py` (keeps per-bin `licks`; do not use `df_cortex2mid.pk`). Cohort: day-1 (`session_number==1`) Familiar sessions, image sets G and H. Viewing bins with `time_from_stimulus >= 0.1` s; drop omitted, novel, and change presentations. Units: grouped areas with \(N \ge 30\). Labels: binary lick vs no-lick in the 50 ms bin (`licks` empty dict = no lick). These are licks on **non-change** images, not change reports or hits. Decoding uses **active** trials only.

Channels: VISm → SCMRN (main) and VISpl → SCMRN (supplement; \(V_k\) is then VISpl-defined). Communication subspace: mean-centered cross-covariance SVD, \(k=5\); \(U_k\) in cortex, \(V_k\) in SCMRN. Local geometry: top-\(k\) PCs of that area’s active counts. Decoder: L2-regularized logistic (`sklearn`, \(C=0.1\)) on mean-centered counts. Each session: keep all lick trials, undersample no-lick to match (10 repeats), 5-fold stratified CV. Metric plotted: balanced accuracy (equals accuracy on the 50/50 set); ROC-AUC is computed but not plotted. Session mean = mean over undersample repeats. Sessions need \(\ge 30\) lick trials; \(n=31\) for both channels.

Readouts (same balanced sets): (i) full space; (ii) projection onto active communication \(k\)-D; (iii) residual in the orthogonal complement of that subspace; (iv) random orthonormal \(k\)-D (5 draws/repeat; dimensionality control); (v) local PCA \(k\)-D. Supplement readouts: communication SVD on active **no-lick** trials only, then decode lick from that subspace (control: the channel is not a lick classifier); active activity projected onto **passive** communication \(U_k/V_k\) and onto **passive** PCs. \(\alpha_k=\|B_k^\top w\|^2\) is the fraction of the **full-space** unit decoder \(w\) in active / passive / random communication modes (50 random subspaces). \(\alpha_k\) is orientation of \(w\), not evidence that those modes carry lick information.

Main figures (VISm → SCMRN): `decoding_schematic.pdf` (cartoon), `decoding_readout_box.pdf` (full / comm / PCA / random / orthogonal), `decoding_comm_vs_pca.pdf`. Supplement: `decoding_passive_box.pdf` (active vs passive comm / PCA axes and no-lick SVD; no random \(k\)-D), other comm \(k\)-D scatters, \(\alpha_k\) boxes and paired active vs passive \(\alpha\), and the same set tagged `_vispl_scmrn`. Orthogonal = residual of communication \(U_k/V_k\), not of local PCs.

**Do not write:** that \(\alpha_k\) means “lick information in the subspace”; that SCMRN \(V_k\) is a unique lick bottleneck; that labels are change detections; that passive blocks are decoded (activity and labels are always active; only the axes can be passive).

**Image identity vs communication subspace (set G, 8-way, \(k=5\)).** Script: `image_decoding_analysis.py`. One sample = one flash (mean counts in \(0.1\)–\(0.25\) s); chance \(=1/8\). Whole active block; licks ignored as labels. SVD/PCA on flash-averaged matrices. Orthogonal = residual of communication \(U_k/V_k\), not of local PCs. Passive readouts: project **active** flashes onto passive \(U_k/V_k\) and passive PCs (do not decode passive activity). No no-lick SVD. Figures: `image_decoding_readout_box.pdf`, `image_decoding_passive_box.pdf`, comm scatters; tags `_vispl_scmrn` and `_lgd_vispl`.

VIS → SCMRN (\(n=26\)):

| Readout | SCMRN | VISm | VISpl |
|---|---|---|---|
| Full space | 0.33 | 0.96 | 0.99 |
| Communication \(k\)-D | 0.18 | 0.33 | 0.39 |
| Local PCA \(k\)-D | 0.17 | 0.53 | 0.76 |
| Random \(k\)-D | 0.18 | 0.34 | 0.41 |
| Orthogonal to comm. | 0.30 | **0.95** | **0.99** |
| Comm. (passive axes) | 0.19 | 0.43 | 0.56 |
| PCA (passive axes) | 0.18 | 0.57 | 0.76 |

Cortex: image identity is almost fully decodable and lives **outside** engaged \(U_k\) (complement \(\approx\) full; communication \(\approx\) random 5-D; local PCs do better than the channel). Passive \(U_k\) is **better** for image than engaged \(U_k\) (VISm 0.43 vs 0.33; VISpl 0.56 vs 0.39): engagement tilts the channel away from image. SCMRN: weak image signal overall; \(V_k\) is no better than a random 5-D slice. So the engaged VIS–SCMRN subspace is **not** the main visual-identity channel: in VISm it contains lick, not image. A 5-D communication window can still be above chance for image (0.33 vs 0.13) without being where the image code sits. VISpl → SCMRN matches. ~11% of active flashes have any in-window lick.

LGd → VISpl (\(n=13\); fewer sessions with \(N\ge30\) in LGd):

| Readout | LGd | VISpl |
|---|---|---|
| Full space | 0.84 | 0.99 |
| Communication \(k\)-D | 0.60 | 0.77 |
| Local PCA \(k\)-D | 0.54 | 0.73 |
| Random \(k\)-D | 0.35 | 0.39 |
| Orthogonal to comm. | 0.58 | 0.93 |
| Comm. (passive axes) | 0.54 | 0.78 |
| PCA (passive axes) | 0.51 | 0.72 |

Thalamo-cortical communication **does** carry image (comm \(\gg\) random in both areas; passive \(\approx\) active). VISpl still has leftover image in the complement. Do not use this channel for lick decoding unless asked.

**Do not write (image):** that the VIS–SCMRN channel is a visual identity code; that LGd was used for lick decoding; that passive blocks are decoded.

## Conventions

- Scripts use `Path(__file__).resolve().parent` with a `NameError` fallback for interactive `# %%` runs.
- `SAVE_PLOTS` is often `False` in analysis; set `True` to write PDFs.
- Population metrics do not depend on within-area neuron order; covariance/SVD heatmaps do (order is probe depth, probes mixed).
