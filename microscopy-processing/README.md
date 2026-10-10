# Microscopy Tiling, Recovery, and Sampling on HPC

These scripts turn multi-scene Zeiss CZI scans of plant roots into fixed-size image tiles for classification. Processing runs as a Slurm array on a shared cluster.

## Problem

Each CZI file holds several scan regions, called scenes. A scene is split into halves, a spline is fitted along the root contour, and square 1152 × 1152 px patches are sampled along it and warped. Each patch is cut into four 576 × 576 tiles. A tile is kept if its root fraction (RF) is high enough. RF is the fraction of grayscale pixels darker than 175.

Two engineering problems follow from this:

1. Processing the whole collection must survive job failures and restarts without corrupting or duplicating output.
2. After production ran with a cutoff of RF ≥ 0.75, the effect of a lower cutoff had to be measured. The extra tiles also had to be added without re-exporting or touching any production tile.

## Scripts

### `batch_tile_mycorrhisee.py`: Slurm array driver

- `--discover-only` walks the input root in sorted order and writes a manifest atomically through a temp file and rename. Array task *i* processes manifest line *i* via `--index $SLURM_ARRAY_TASK_ID`, so task-to-file assignment stays fixed across resubmissions.
- Each input maps to an output directory that mirrors its path relative to the input root.
- A `.complete` marker containing the run's metadata makes completed files skip on rerun. `.failed` records the error. `--overwrite` reprocesses completed files. `--retry-failed` clears partial tiles before an incomplete file is reprocessed.
- Logging goes to a shared summary log, an error log with tracebacks, an append-only CSV run manifest, and a per-file log. Per-file logs capture the pipeline's stdout through a small tee class that flushes on every write, so Slurm output can be followed live.
- `--dry-run` prints the plan with complete or pending status and touches nothing.
- The tiling itself is done by `run_single_czi_test()` from `test_single_czi_tiling.py`, which this driver calls without modification. That module is not included.

### `recover.py`: additive recovery at RF ≥ 0.60

`recover(index)` reprocesses one scan from an inventory CSV and adds only tiles that production did not export. It writes them under a separate run directory with the prefix `yield_rf060__`.

- **No duplicate work.** Before warping a patch, the script checks whether all four of its tile names already exist in the production output. If so, the warp is skipped entirely.
- **Input pinning.** The scan's size and `mtime_ns` must match the inventory.
- **Resumability.** Each scene writes a JSON checkpoint with its tile list. On rerun, a scene whose listed tiles all exist is loaded from the checkpoint and not recomputed. A failed scene writes no checkpoint and is retried. `complete.json` is written last and only when no scene failed.
- **Idempotent writes.** Tiles are written to a temp file and renamed. If a recovered tile already exists, it is re-read and compared pixel by pixel with the recomputed tile. A mismatch fails that scene, so the scan is not marked complete.
- **Concurrency.** A non-blocking `flock` on a per-scan lock file stops two tasks from processing one scan.
- **Accounting.** Each tile is classified as already exported, below 0.60, in [0.60, 0.75), or ≥ 0.75 but missing from production. Per-scene and per-scan yields are written to `scene_yield.csv` and `summary.json`. The per-tile `tiles.csv` is read by the classifier to attach RF values.
- `--smoke` with `--max-scenes N` writes to a separate `smoke/` tree. Scene limits are rejected without `--smoke`.

### `run_pilot.py`: stratified pilot with reservoir sampling

The pilot measured how many tiles the lower cutoffs would add, and produced samples for visual review, before committing to a full rerun.

- **Scan selection.** It takes the known zero-yield scans, two seeded random low-yield scans from each of three blocks, and a fixed control set. The selection is written to `selected_scans.csv`. `--select-only` stops after this step.
- **Scene sampling.** Up to six scenes per scan, evenly spaced with `np.linspace`.
- The production geometry is reproduced for every candidate tile, and its RF is assigned to one of three disjoint half-open bands: [0.70, 0.75), [0.65, 0.70), and [0.60, 0.65).
- **Reservoir sampling.** Algorithm R keeps a uniform sample of 12 tiles per band per scan in one pass and O(12) memory. The first 12 tiles fill the reservoir. The n-th tile then replaces a random slot with probability 12/n. Each scan uses its own RNG seeded from a fixed base seed, so the samples are reproducible.
- The script writes per-scan yield tables, a tile manifest with fields for visual review, and a 4 × 3 contact sheet per band.

### `test_recovery.py`

This tests `recover.py` with synthetic tiles of known RF: 0.65, 0.80, 0.59, and 0.00. It checks the 0.60 boundary, that a fully exported patch is never re-warped, and the expected counts. It also checks that rerunning after completion, and resuming from scene checkpoints after `complete.json` is deleted, leave all bytes unchanged. Production tiles are verified unchanged throughout.

## Dependencies

- Python 3.9 or newer, numpy, opencv-python, aicsimageio with CZI support, and pillow for `run_pilot.py`.
- POSIX only, because of `fcntl`.

### Original research environment

None of these scripts runs outside the original environment. The following pieces are not included:

| Missing piece | Used by |
| --- | --- |
| `test_single_czi_tiling.py`: `run_single_czi_test` and `check_dependencies` | `batch_tile_mycorrhisee.py`, imported at module level |
| `spline_guided_tiling_utils_v2.py`: `preprocess_image`, `find_largest_contour_with_area`, `find_squares`, `warp_and_crop` | `recover.py`, `run_pilot.py`, `test_recovery.py` |
| `diagnose_empty_scans.py`: `SUSPECT`, `CONTROL`, `COORDS_1152_576`, `PADDING` | `run_pilot.py` |
| The CZI scan collection, the production tile tree, `input_inventory.csv`, and `tile_yield_all_scans.csv` | all |

Data paths are resolved from the `SCRATCH_ROOT` environment variable in all three scripts. `run_pilot.py` also reads `TILING_CODE_DIR`, the directory that contains the tiling utility modules. `batch_tile_mycorrhisee.py` accepts `--input-root`, `--output-root`, `--manifest`, and `--log-dir` to override individual paths. `recover.py` and `run_pilot.py` have no path options. Their paths are module constants, which `test_recovery.py` patches.

## Usage

```
# Batch tiling
python batch_tile_mycorrhisee.py --input-root IN --output-root OUT --manifest M --discover-only
python batch_tile_mycorrhisee.py --input-root IN --output-root OUT --manifest M --dry-run --max-files 10
sbatch --array=0-$((N-1)) job.sh   # job.sh runs: python batch_tile_mycorrhisee.py ... --index $SLURM_ARRAY_TASK_ID

# Recovery for one scan, smoke test first
python recover.py --index 0 --smoke --max-scenes 1
python recover.py --index 0

# Pilot
python run_pilot.py --select-only
python run_pilot.py
```

The Slurm job script is not included.

## Tests

```
cd microscopy-processing
python -m unittest test_recovery -v
```

The test fails at import in a clean environment, because it requires `aicsimageio` and `spline_guided_tiling_utils_v2`. It mocks every function from both modules, but they must still exist so that `unittest.mock.patch` can replace their attributes. The test was not run during this review.

## Limitations

- RF uses a fixed grayscale threshold of 175 and assumes a white background. Black padding pixels are recoloured to white before thresholding.
- The tile geometry is fixed: 1152 px patches, 576 px tiles, and 2304 px padding.

