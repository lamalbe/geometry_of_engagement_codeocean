# Task engagement reroutes vision-to-action hierarchical communication

Analysis code for the preprint. The files under `code_gendata/`, `code_analysis/`, `code_supplementary/`, and `code_theory/` are Python scripts with `# %%` cells. Run them with `python`, or run the whole sequence with `run_pipeline.py`.

Figures are written to `results/<script_name>/`. `SAVE_PLOTS` is on in every figure script.

## The Allen dataset is not included

The [Allen Visual Behavior Neuropixels](https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels) dataset is too large to distribute. Two modes:

**Reviewer / Code Ocean** (default, `RUN_FROM_RAW=0`). Skip preprocessing and per-session generation. Each analysis runs only when its derived table is already present. A missing input is reported and the pipeline continues.

```bash
python run_pipeline.py --dry-run   # print what would run
python run_pipeline.py
```

On Code Ocean, `./run` mounts the attached tables at `/data` and writes figures to `/results`.

**Full reproduction** (`RUN_FROM_RAW=1`). Download the Allen cache and point `ALLEN_RAW_DATA_DIR` at the directory that contains `manifests/`. Preprocessing writes one folder per session under `ALLEN_PREPROCESSED_DIR` (default `/data/allen_engagement_data`).

```bash
RUN_FROM_RAW=1 \
  ALLEN_RAW_DATA_DIR=/path/to/visual-behavior-neuropixels \
  python run_pipeline.py
```

`preprocess_data.py` needs AllenSDK. It is not in `environment.yml`: AllenSDK pins older numpy/pandas than these figure scripts. Install it only for a full run (`pip install allensdk`), in a separate environment if the pins conflict.

Paths live in `config.py`. Environment variables override them: `ALLEN_RAW_DATA_DIR`, `ALLEN_PREPROCESSED_DIR`, `DATA_DIR`, `RESULTS_DIR`, `RUN_FROM_RAW`.

## Tables to attach for a reviewer run

These files in `data/` are enough for the figures that are computed from summary tables. Together they are about 700 MB.

| File | About | Used by |
|---|---|---|
| `df_stats_area.pk` | 299 MB | circuit metrics, primary visual metrics, channel MLR, extrapolated PR, behavior MLR |
| `visp_cloud_shape.pk` | 8 MB | primary visual metrics |
| `df_poisson_stats.pk` | 10 MB | Poisson / sparsity figures |
| `df_poisson_stats_subsampled.pk` | 77 MB | Poisson / sparsity figures |
| `df_basis_vectors.pk` | 66 MB | alignment figures |
| `df_max_cosine_active_passive.pk` | 1 MB | alignment figures |
| `df_max_cosine_within_condition.pk` | 1 MB | VISm → SCMRN alignment |
| `df_subspace_overlap_pc.pk`, `df_subspace_overlap_svd.pk` | small | alignment figures |
| `df_decoding_repeat.pk`, `df_decoding_session.pk`, `df_decoding_alpha_session.pk` | 1 MB | lick-decoding figures (no refit) |
| `df_image_decoding_repeat.pk`, `df_image_decoding_session.pk` | < 1 MB | image-decoding figures (no refit) |
| `suppl_stats_unengagement.pk` | 33 MB | behavior supplement |
| `suppl_stats_speed_terciles.pk` | 55 MB | behavior supplement |
| `suppl_stats_pupil_terciles.pk` | 58 MB | behavior supplement |
| `suppl_stats_onset_window.pk` | 83 MB | onset-window channel MLR |
| `suppl_stats_PR_timewindows.pk` | 1 MB | within-block PR |

Leave these out of the capsule unless a full refit or an example cloud is required:

| File | About | Only needed for |
|---|---|---|
| `df_decoding.pk` | 7.3 GB | refitting decoders (`FORCE_REBUILD = True`) |
| `df_cortex2mid.pk` | 6.9 GB | `cross_area_VIS2MID_channel.py`, or rebuilding geometry |
| `df_behavior.pk` | 2.2 GB | session behavior traces in `suppl_behavior_session.py` |

`df_cross_covariance_in_local_subspace.pk` and `df_local_variance_in_cross_subspace.pk` are not read anymore.

Per-session Allen tables and the NWB cache stay outside the capsule. Without them the runner skips `preprocess_data.py`, the `generate_*.py` scripts, `circuit_cloud.py`, `cortex_cloud.py`, `unit_counts_by_area.py`, and `change_aligned_licks_rewards.py`.

## Environment

```bash
conda env create -f environment.yml
conda activate geometry_of_engagement
```

The derived tables were written with pandas 3. Use pandas 2.2 or newer. `suppl_analysis_behavior.py` can still open pandas 3 string columns if the installed pandas is older; the other scripts call `pd.read_pickle` directly.
