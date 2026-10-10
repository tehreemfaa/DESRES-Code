#!/usr/bin/env python
"""Batch driver for the MycorrhiSEE spline-guided CZI tiling pipeline.

Typical workflow
-----------------
1. Discover files and write a stable, sorted manifest:
     python batch_tile_mycorrhisee.py --discover-only

2. Dry-run to see what would happen (no files touched):
     python batch_tile_mycorrhisee.py --dry-run --max-files 10

3. Process one file (as a Slurm array task would):
     python batch_tile_mycorrhisee.py --index 0

4. Process a small pilot batch outside Slurm (testing only):
     python batch_tile_mycorrhisee.py --max-files 5 --max-scenes 1
"""

import io
import os
import sys
import csv
import json
import time
import socket
import shutil
import argparse
import traceback
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from test_single_czi_tiling import run_single_czi_test, check_dependencies

# [REDACTED for publication: HPC username removed from the default paths below.
#  Set SCRATCH_ROOT, or pass --input-root / --output-root / --manifest.]
# DEFAULT_INPUT_ROOT = "/scratch/<REDACTED_USER>/mycorrhisee/MycorrhiSEE"
SCRATCH_ROOT = os.environ.get("SCRATCH_ROOT", "/path/to/scratch")
DEFAULT_INPUT_ROOT = os.path.join(SCRATCH_ROOT, "mycorrhisee/MycorrhiSEE")
# [REDACTED for publication: HPC username and scratch paths removed from this note.]
# NOTE: <input parent directory> is root-owned (mode 755) -- <user>
# cannot create new files/dirs directly inside it. Output and manifests
# therefore live under <user scratch directory> directly, which is user-owned.
# DEFAULT_OUTPUT_ROOT = "/scratch/<REDACTED_USER>/mycorrhisee_tiling_output"
# DEFAULT_MANIFEST = "/scratch/<REDACTED_USER>/mycorrhisee_tiling_manifests/mycorrhisee_czi_manifest.txt"
DEFAULT_OUTPUT_ROOT = os.path.join(SCRATCH_ROOT, "mycorrhisee_tiling_output")
DEFAULT_MANIFEST = os.path.join(SCRATCH_ROOT, "mycorrhisee_tiling_manifests/mycorrhisee_czi_manifest.txt")
TILE_SUBDIR = "patchsize1152_tilesize576_200microns"
COMPLETE_MARKER = ".complete"
FAILED_MARKER = ".failed"

MANIFEST_FIELDS = [
    "abs_input_path", "rel_input_path", "output_dir", "status",
    "start_time", "end_time", "runtime_sec", "num_scenes_processed",
    "num_tiles_generated", "error_message", "slurm_job_id",
    "slurm_array_task_id", "hostname",
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_czi_files(input_root):
    """Recursively find all .czi files under input_root, sorted for stability."""
    found = []
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames.sort()
        for fn in filenames:
            if fn.lower().endswith(".czi"):
                found.append(os.path.join(dirpath, fn))
    found.sort()
    return found


def write_manifest(files, manifest_path):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        for p in files:
            f.write(p + "\n")
    os.replace(tmp_path, manifest_path)


def read_manifest(manifest_path):
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Path mapping / resumability
# ---------------------------------------------------------------------------

def relative_output_dir(filepath, input_root, output_root):
    rel = os.path.relpath(filepath, input_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(output_root, rel_no_ext)


def is_complete(out_dir):
    return os.path.isfile(os.path.join(out_dir, COMPLETE_MARKER))


def clean_partial_output(out_dir):
    """Remove partial tiles/markers from a previous incomplete attempt
    before reprocessing, so no stale PNGs linger alongside new ones."""
    tile_dir = os.path.join(out_dir, TILE_SUBDIR)
    if os.path.isdir(tile_dir):
        shutil.rmtree(tile_dir)
    for marker in (COMPLETE_MARKER, FAILED_MARKER):
        marker_path = os.path.join(out_dir, marker)
        if os.path.isfile(marker_path):
            os.remove(marker_path)


def mark_complete(out_dir, info):
    with open(os.path.join(out_dir, COMPLETE_MARKER), "w") as f:
        json.dump(info, f, indent=2)


def mark_failed(out_dir, info):
    with open(os.path.join(out_dir, FAILED_MARKER), "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_line(path, msg):
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")


def append_csv_row(manifest_csv_path, row):
    os.makedirs(os.path.dirname(manifest_csv_path), exist_ok=True)
    file_exists = os.path.isfile(manifest_csv_path)
    with open(manifest_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_scene_results_csv(out_dir, results):
    """Persist the per-scene result dicts returned by run_single_czi_test
    for provenance. Purely a serialization of data the pipeline already
    computed -- no processing logic here."""
    if not results:
        return
    path = os.path.join(out_dir, "scene_results.csv")
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


class _Tee:
    """Writes to all streams and flushes immediately after every write, so
    progress is visible in real time (Slurm .out file, per-file log) instead
    of sitting in a buffer until it fills or the process exits."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_one_file(filepath, input_root, output_root, max_scenes,
                      overwrite, retry_failed, summary_log, error_log,
                      manifest_csv, per_file_log_dir,
                      slurm_job_id, slurm_array_task_id, hostname):
    rel_path = os.path.relpath(filepath, input_root)
    out_dir = relative_output_dir(filepath, input_root, output_root)
    os.makedirs(out_dir, exist_ok=True)

    complete = is_complete(out_dir)
    if complete and not overwrite:
        log_line(summary_log, f"SKIP (already complete): {rel_path}")
        return "skipped"

    if not complete and (overwrite or retry_failed):
        clean_partial_output(out_dir)
    elif overwrite:
        clean_partial_output(out_dir)

    start_dt = datetime.now()
    row = {
        "abs_input_path": filepath,
        "rel_input_path": rel_path,
        "output_dir": out_dir,
        "status": "running",
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": "",
        "runtime_sec": "",
        "num_scenes_processed": "",
        "num_tiles_generated": "",
        "error_message": "",
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "hostname": hostname,
    }

    per_file_log = None
    if per_file_log_dir:
        os.makedirs(per_file_log_dir, exist_ok=True)
        safe_name = rel_path.replace(os.sep, "__")
        per_file_log = os.path.join(per_file_log_dir, f"{safe_name}.log")

    log_line(summary_log, f"START: {rel_path} -> {out_dir}")
    t0 = time.time()

    old_stdout = sys.stdout
    tee_fh = None
    try:
        if per_file_log:
            tee_fh = open(per_file_log, "a")
            sys.stdout = _Tee(old_stdout, tee_fh)

        try:
            results = run_single_czi_test(filepath, out_dir, max_scenes=max_scenes)
        finally:
            sys.stdout = old_stdout
            if tee_fh:
                tee_fh.close()

        num_scenes = len(results) if results else 0
        num_tiles = int(sum(r.get("NumTile75MoreRoot", 0) for r in results)) if results else 0
        runtime = time.time() - t0

        row.update({
            "status": "success",
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "runtime_sec": f"{runtime:.2f}",
            "num_scenes_processed": num_scenes,
            "num_tiles_generated": num_tiles,
        })
        write_scene_results_csv(out_dir, results)
        mark_complete(out_dir, row)
        log_line(summary_log,
                  f"DONE: {rel_path} ({num_scenes} scenes, {num_tiles} tiles, {runtime:.1f}s)")
        append_csv_row(manifest_csv, row)
        return "success"

    except Exception as e:
        sys.stdout = old_stdout
        if tee_fh:
            tee_fh.close()
        runtime = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()

        row.update({
            "status": "failed",
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "runtime_sec": f"{runtime:.2f}",
            "error_message": err_msg,
        })
        mark_failed(out_dir, row)
        log_line(summary_log, f"FAILED: {rel_path} ({err_msg})")
        log_line(error_log, f"FAILED: {rel_path}\n{tb}\n")
        append_csv_row(manifest_csv, row)
        return "failed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Line-buffer stdout/stderr so output (including the Slurm .out/.err
    # files) is visible in real time instead of sitting in a block buffer
    # until it fills or the process exits. Logging-only change.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Batch driver for MycorrhiSEE spline-guided CZI tiling. "
                     "Calls the existing, unmodified run_single_czi_test() "
                     "pipeline once per CZI file. Designed to be invoked once "
                     "per Slurm array task via --index."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT,
                         help="Root directory to recursively search for .czi files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                         help="Root directory to mirror input structure into for tile output.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                         help="Path to the stable, sorted list of CZI files "
                              "(written by --discover-only, read otherwise).")
    parser.add_argument("--file-list", default=None,
                         help="Explicit file list to use instead of --manifest "
                              "(one absolute .czi path per line).")
    parser.add_argument("--index", type=int, default=None,
                         help="Process only the file at this 0-based index in "
                              "the manifest/file-list. Intended for "
                              "$SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--max-files", type=int, default=None,
                         help="Process at most this many files this invocation "
                              "(ignored if --index is set).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the plan (which files, which output dirs, "
                              "complete/pending status) and exit without "
                              "processing anything.")
    parser.add_argument("--retry-failed", action="store_true",
                         help="For files without a .complete marker, clear any "
                              "partial output from a previous attempt before "
                              "reprocessing (avoids stale/duplicate PNGs). "
                              "Incomplete files are retried by default even "
                              "without this flag; this only adds cleanup.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess files even if already marked .complete "
                              "(clears prior output first).")
    parser.add_argument("--max-scenes", type=int, default=None,
                         help="TESTING ONLY: cap the number of scenes processed "
                              "per CZI file. Default (unset) processes ALL "
                              "scenes -- do not set this for production runs.")
    parser.add_argument("--discover-only", action="store_true",
                         help="Recursively scan --input-root for .czi files, "
                              "write the sorted manifest to --manifest, and exit. "
                              "Run this once before submitting any processing jobs.")
    parser.add_argument("--log-dir", default=None,
                         help="Directory for summary/error/per-file logs and the "
                              "CSV manifest. Default: <output-root>/_logs")
    args = parser.parse_args()

    # --- Discovery mode -----------------------------------------------
    if args.discover_only:
        print(f"Scanning {args.input_root} for .czi files...")
        files = discover_czi_files(args.input_root)
        write_manifest(files, args.manifest)
        print(f"Discovered {len(files)} .czi file(s).")
        print(f"Manifest written to: {args.manifest}")
        return

    manifest_path = args.file_list or args.manifest
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Run with --discover-only first to generate it, or pass --file-list.")
        sys.exit(1)

    all_files = read_manifest(manifest_path)
    if not all_files:
        print(f"ERROR: manifest at {manifest_path} is empty.")
        sys.exit(1)

    if args.index is not None:
        if args.index < 0 or args.index >= len(all_files):
            print(f"ERROR: --index {args.index} out of range "
                  f"(manifest has {len(all_files)} entries, valid: 0..{len(all_files)-1})")
            sys.exit(1)
        target_files = [all_files[args.index]]
    else:
        target_files = all_files
        if args.max_files is not None:
            target_files = target_files[:args.max_files]

    output_root = args.output_root
    log_dir = args.log_dir or os.path.join(output_root, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_log = os.path.join(log_dir, "summary.log")
    error_log = os.path.join(log_dir, "errors.log")
    manifest_csv = os.path.join(log_dir, "manifest.csv")
    per_file_log_dir = os.path.join(log_dir, "per_file")

    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    hostname = socket.gethostname()

    print("=" * 70)
    print("MycorrhiSEE batch tiling driver")
    print("=" * 70)
    print(f"Manifest:       {manifest_path} ({len(all_files)} total file(s))")
    print(f"Targeted now:   {len(target_files)} file(s)")
    print(f"Input root:     {args.input_root}")
    print(f"Output root:    {output_root}")
    print(f"Max scenes:     {'ALL (production default)' if args.max_scenes is None else f'{args.max_scenes} (TESTING ONLY)'}")
    print(f"Overwrite:      {args.overwrite}")
    print(f"Retry-failed:   {args.retry_failed}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Slurm job id:   {slurm_job_id or '(none)'}")
    print(f"Array task id:  {slurm_array_task_id or '(none)'}")
    print(f"Hostname:       {hostname}")
    print()

    if args.dry_run:
        for fp in target_files:
            rel = os.path.relpath(fp, args.input_root)
            out_dir = relative_output_dir(fp, args.input_root, output_root)
            status = "complete" if is_complete(out_dir) else "pending"
            print(f"  [{status}] {rel} -> {out_dir}")
        print(f"\nDry run complete. {len(target_files)} file(s) would be considered. "
              f"No files were processed.")
        return

    if not check_dependencies():
        sys.exit(1)

    n_success = n_skip = n_fail = 0
    for fp in target_files:
        result = process_one_file(
            fp, args.input_root, output_root, args.max_scenes,
            args.overwrite, args.retry_failed,
            summary_log, error_log, manifest_csv, per_file_log_dir,
            slurm_job_id, slurm_array_task_id, hostname,
        )
        if result == "success":
            n_success += 1
        elif result == "skipped":
            n_skip += 1
        else:
            n_fail += 1

    print()
    print(f"Batch run finished: {n_success} succeeded, {n_skip} skipped, {n_fail} failed.")
    print(f"Summary log:  {summary_log}")
    print(f"Error log:    {error_log}")
    print(f"CSV manifest: {manifest_csv}")

    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
