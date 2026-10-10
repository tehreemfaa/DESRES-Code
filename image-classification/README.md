# Resumable Tile Classification

`classify.py` runs a trained four-class Swin-B model over every microscopy tile produced by the tiling pipeline. It records each tile's class probabilities and places a verified copy of the tile in a folder for its predicted class. The classes are `good`, `yellow`, `blurry`, and `opaque`.

## Problem

The source tiles are the only copy of the processed scans, so the classifier must never modify, move, or overwrite them. A run spans many GPU job allocations and can be killed at any point by a time limit, node failure, or out-of-memory error. Each restart must continue without duplicating work, losing predictions, or leaving copies that disagree with the results table. The final outputs must be checkable against the sources byte for byte.

## Design

### State machine

Every tile has one row in a SQLite table, `_state/progress.sqlite`. Its `state` column moves through these values:

```
pending ──> copy_pending ──> success
   │              └────────> copy_failed  (retryable with --retry-failures)
   └──────> failed                        (unreadable image or invalid prediction)
```

The full prediction record is stored as JSON in the same row. The SQLite database is the source of truth. `master_predictions.csv` is a derived view that is rebuilt from it whenever a run resumes.

### Commands

| Command | Hardware | Effect |
| --- | --- | --- |
| `preflight` | any | Validates source and output roots and prints model provenance. Writes nothing. |
| `prepare` | CPU | Walks the source roots, hashes every tile, and records the inventory. Interrupted inventories resume. A full audit then confirms no tile was added, removed, or changed. |
| `run` | CUDA GPU | Reconciles earlier work, then classifies pending tiles in windows of up to 4096 rows. |
| `finalize` | CPU | Checks CSV, database, copies, and sources against each other. Writes `summary.json` and two figures. |

### Copy protocol

1. The model returns float32 softmax probabilities. The record is checked for finite values, a sum within 1e-5 of 1, and consistent top-two ordering. Ties go to the lower class index.
2. The destination path is computed and the record is committed as `copy_pending` **before** any file is written. A crash after this point leaves a durable statement of intent.
3. The source is copied into `_state/staging/`. The copy is hashed and must match the inventoried SHA-256. The source's size and mtime are checked again, and the copy is fsynced.
4. The staging file is published with `os.link`. This fails if the destination exists, so a file is never overwritten. The staging name is then removed.
5. The row becomes `success`.

On resume, `reconcile()` revisits every non-pending row. If a destination exists and its hash matches the source, the row is complete. If the copy is missing, the copy is repeated. A file at the destination that no matching record owns raises `FileExistsError`. The tile is recorded as a failure and the existing file is left untouched.

### Safety guards

- Source roots may not overlap each other or the output root. Every write path goes through `Store.guard`, which rejects paths that leave the output root, touch a source, or cross a symlink.
- Symlinked source directories or tiles stop the run instead of being skipped silently.
- An exclusive `fcntl` lock on `_state/run.lock` prevents two runs from sharing one output directory.
- `model_adapter.describe()` records hashes of the checkpoint, the config, and the implementation files, plus library versions. A later run that sees different values refuses to continue in the same output directory.
- Unreadable images are recorded as `failed` with the exception text. No placeholder tensor is ever classified in their place.

### Finalisation checks

`finalize()` streams the database and the CSV in lockstep and requires identical records. Every `success` copy must exist with the source hash, and every file under the class folders must belong to exactly one `success` record. A full source audit then confirms that the source tree is unchanged. Confidence histograms use 20 bins and are accumulated while streaming, so predictions are never all held in memory.

## Files

| File | Purpose |
| --- | --- |
| `classify.py` | CLI, inventory, journal, inference loop, copy protocol, reconciliation, finalisation |
| `model_adapter.py` | Loads the trained model and its eval transforms from the original training repository. Records provenance. Computes probabilities under CUDA autocast. |
| `tests/test_safety.py` | Ten unit tests on synthetic PNG fixtures. Model loading and GPU inference are mocked. |

## Dependencies

- Python 3.10 or newer, because of `zip(..., strict=True)`. POSIX only, because of `fcntl`.
- numpy, torch, torchvision, pillow, matplotlib.
- The output root must be on a filesystem that supports hard links.

### Original research environment

`model_adapter.py` cannot run outside the original environment. It imports `utils.trainer` and `utils.model_utils` from the lab's training repository, which is not included. It also reads a results directory containing `config.json`, `best_model.pth`, `results.json`, and `test_predictions.csv`. `describe()` additionally opens the training, validation, and test image folders named in the config to confirm the class mapping. The default source and output paths in `classify.py` are resolved from the `SCRATCH_ROOT` environment variable. Pass `--source` and `--output` to override them. `model_adapter.py` locates the training repository through `CLASSIFICATION_REPO`.

## Usage

```
python classify.py preflight --source original=/path/to/tiles --output /path/to/out
python classify.py prepare   --source original=/path/to/tiles --output /path/to/out
python classify.py run       --source original=/path/to/tiles --output /path/to/out --batch-size 32 --workers 4
python classify.py finalize  --source original=/path/to/tiles --output /path/to/out
```

`--source alias=path` can be repeated. Aliases must match `[A-Za-z0-9_-]+`. `--limit N` processes at most N pending tiles and skips finalisation. `--retry-failures` returns `failed` rows to the pending queue and retries the copy step for `copy_failed` rows. `--defer-finalize` skips the automatic finalisation after a complete run.

Tiles are discovered only inside directories named `patchsize1152_tilesize576_200microns`, and only when the filename matches the tiling naming pattern.

## Tests

```
cd image-classification
python -m unittest discover -s tests -v
```

All 10 tests passed on macOS with Python 3.12 and torch 2.14.1. They cover the following:

- Path guards, including symlink escape.
- Class mapping, second-class ordering, margin, copy, and CSV recovery after truncation.
- Repair of a deleted output copy without loading the model.
- Refusal to overwrite an unowned destination file.
- Recovery after a crash between file publication and the `success` commit.
- Detection of modified sources.
- Rejection of probabilities that do not sum to 1, and deterministic tie-breaking.
- Unreadable images logged as failures.
- End-to-end batched inference with one corrupt input.
- Copy failures that preserve predictions and recover with `--retry-failures`.

The tests do not exercise real model loading, CUDA autocast, or multi-worker data loading.

## Limitations

- The class count is fixed at four. `new_prediction` and `valid_record` use `range(4)`.
- Every tile is hashed twice during `prepare`, once for the inventory and once for the audit, and again during `finalize`. This is deliberate, but it makes the cost I/O bound on large trees.
- Per-row SQLite commits during reconciliation favour durability over throughput.

