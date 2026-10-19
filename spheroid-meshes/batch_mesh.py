"""
Batch 3D mesh generator for multicellular spheroid probability maps.

Reads HDF5 probability maps produced by a segmentation model, binarises
each volume, detects internal cavities via flood-filling, extracts
isosurfaces with marching cubes, smooths the resulting meshes, and
exports STL files for downstream visualisation and measurement.

Dependencies
------------
numpy, scipy, scikit-image, pyvista, h5py

The HDF5 files and segmentation model are not included.
"""

import numpy as np
import pyvista as pv
from skimage import measure, morphology
from scipy import ndimage
import h5py
import os
import glob
import time
import json
from datetime import datetime


def process_sample(file_path, sample_name, output_dir,
                   min_object_size=1500, min_cavity_size=100,
                   smooth_iterations=100, relaxation_factor=0.1):
    """Process a single sample to generate smoothed cell and cavity meshes.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 probability map.
    sample_name : str
        Identifier used for output filenames.
    output_dir : str
        Directory for exported STL files.
    min_object_size : int
        Connected components smaller than this (in voxels) are removed
        from the binary volume before meshing.
    min_cavity_size : int
        Cavities smaller than this (in voxels) are discarded.
    smooth_iterations : int
        Number of Laplacian smoothing passes applied to each mesh.
    relaxation_factor : float
        Step size for each smoothing pass.

    Returns
    -------
    dict
        Processing results including mesh statistics and file paths.
    """

    print(f"\n{'=' * 60}")
    print(f"Processing: {sample_name}")
    print(f"{'=' * 60}")

    start_time = time.time()
    results = {
        "sample_name": sample_name,
        "success": False,
        "cell_mesh_created": False,
        "cavity_mesh_created": False,
        "processing_time": 0,
        "error": None,
        "stats": {},
    }

    try:
        print("Loading 3D image and converting to binary...")
        with h5py.File(file_path, "r") as f:
            data = f["exported_data"][:]
            binary = data[:, :, :, 0] <= 128

        total_voxels = np.prod(binary.shape)
        fill_ratio = np.sum(binary) / total_voxels
        print(f"  Shape: {binary.shape}")
        print(f"  True voxels: {np.sum(binary):,}")
        print(f"  Fill ratio: {fill_ratio:.3f} ({fill_ratio * 100:.1f}%)")

        print("Filtering small artifacts...")
        cleaned = morphology.remove_small_objects(
            binary, min_size=min_object_size, connectivity=3
        )
        artifacts_removed = int(np.sum(binary) - np.sum(cleaned))
        print(f"  Artifacts removed: {artifacts_removed:,} voxels")
        print(f"  Remaining voxels: {np.sum(cleaned):,}")

        print("Detecting inner cavities...")
        filled = ndimage.binary_fill_holes(cleaned)
        cavities = filled & (~cleaned)
        print(f"  Raw cavities: {np.sum(cavities):,} voxels")

        if np.sum(cavities) > 0:
            cavities = morphology.remove_small_objects(
                cavities, min_size=min_cavity_size, connectivity=3
            )
            print(f"  Filtered cavities: {np.sum(cavities):,} voxels")

        filled_structure = cleaned.copy()
        if np.sum(cavities) > 0:
            filled_structure[cavities] = True

        cell_voxels = int(np.sum(cleaned))
        cavity_voxels = int(np.sum(cavities))
        filled_voxels = int(np.sum(filled_structure))
        porosity = (cavity_voxels / filled_voxels * 100) if filled_voxels > 0 else 0

        results["stats"] = {
            "original_voxels": int(np.sum(binary)),
            "artifacts_removed": artifacts_removed,
            "cell_voxels": cell_voxels,
            "cavity_voxels": cavity_voxels,
            "filled_voxels": filled_voxels,
            "porosity_percent": round(porosity, 2),
            "fill_ratio": round(fill_ratio, 3),
            "volume_shape": list(binary.shape),
        }

        print(f"  Cell voxels: {cell_voxels:,}")
        print(f"  Cavity voxels: {cavity_voxels:,}")
        print(f"  Porosity: {porosity:.2f}%")

        print("Creating cell mesh...")
        try:
            verts, faces, _, _ = measure.marching_cubes(
                filled_structure.astype(float), level=0.5, spacing=(1.0, 1.0, 1.0)
            )
            pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
            mesh = pv.PolyData(verts, pv_faces)

            print("  Smoothing cell mesh...")
            mesh = mesh.smooth(
                n_iter=smooth_iterations, relaxation_factor=relaxation_factor
            )

            cell_path = os.path.join(output_dir, f"{sample_name}_cell.stl")
            mesh.save(cell_path)

            print(f"  Cell mesh: {len(verts):,} vertices, {len(faces):,} faces")
            print(f"  Saved: {cell_path}")

            results["cell_mesh_created"] = True
            results["stats"]["cell_vertices"] = len(verts)
            results["stats"]["cell_faces"] = len(faces)
            results["stats"]["cell_volume"] = float(mesh.volume)
            results["stats"]["cell_surface_area"] = float(mesh.area)

        except Exception as e:
            print(f"  Error creating cell mesh: {e}")
            results["error"] = f"Cell mesh error: {e}"

        if np.sum(cavities) > 0:
            print("Creating cavity mesh...")
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    cavities.astype(float), level=0.5, spacing=(1.0, 1.0, 1.0)
                )
                pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
                mesh = pv.PolyData(verts, pv_faces)

                print("  Smoothing cavity mesh...")
                mesh = mesh.smooth(
                    n_iter=smooth_iterations, relaxation_factor=relaxation_factor
                )

                cavity_path = os.path.join(output_dir, f"{sample_name}_cavities.stl")
                mesh.save(cavity_path)

                print(f"  Cavity mesh: {len(verts):,} vertices, {len(faces):,} faces")
                print(f"  Saved: {cavity_path}")

                results["cavity_mesh_created"] = True
                results["stats"]["cavity_vertices"] = len(verts)
                results["stats"]["cavity_faces"] = len(faces)
                results["stats"]["cavity_volume"] = float(mesh.volume)
                results["stats"]["cavity_surface_area"] = float(mesh.area)

            except Exception as e:
                print(f"  Error creating cavity mesh: {e}")
                err = f"Cavity mesh error: {e}"
                results["error"] = (
                    err if results["error"] is None else f"{results['error']}; {err}"
                )
        else:
            print("  No cavities detected")

        results["success"] = results["cell_mesh_created"]

    except Exception as e:
        print(f"  FAILED: {e}")
        results["error"] = str(e)

    results["processing_time"] = round(time.time() - start_time, 2)
    print(f"  Processing time: {results['processing_time']:.2f} seconds")
    return results


def batch_process(base_path, pattern="*.ome_Probabilities.h5",
                  min_object_size=1500, min_cavity_size=100):
    """Batch-process all matching HDF5 files under *base_path*.

    Parameters
    ----------
    base_path : str
        Directory containing a ``3-D 2-C Probabilities`` subdirectory
        with the HDF5 probability maps.
    pattern : str
        Glob pattern for input files.
    min_object_size : int
        Passed to :func:`process_sample`.
    min_cavity_size : int
        Passed to :func:`process_sample`.

    Returns
    -------
    list[dict]
        Per-sample result dictionaries.
    """

    prob_dir = os.path.join(base_path, "3-D 2-C Probabilities")
    output_dir = os.path.join(base_path, "meshes")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(prob_dir, pattern)))
    if not files:
        print(f"No files found matching {os.path.join(prob_dir, pattern)}")
        return []

    print(f"Found {len(files)} files to process")
    print(f"Output directory: {output_dir}\n")

    all_results = []
    start_time = time.time()

    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        sample_name = filename.replace(".ome_Probabilities.h5", "")
        print(f"[{i}/{len(files)}] {sample_name}")

        result = process_sample(
            file_path, sample_name, output_dir,
            min_object_size, min_cavity_size,
        )
        all_results.append(result)

    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r["success"])

    print(f"\n{'=' * 60}")
    print(f"Processed {len(files)} files in {total_time:.1f}s "
          f"({successful} succeeded, {len(files) - successful} failed)")
    print(f"{'=' * 60}")

    print(f"\n{'Sample':<20} {'Status':<8} {'Porosity':<10} {'Time':<8}")
    print("-" * 50)
    for r in all_results:
        status = "OK" if r["success"] else "FAIL"
        porosity = f"{r['stats'].get('porosity_percent', 0):.1f}%"
        print(f"{r['sample_name']:<20} {status:<8} {porosity:<10} {r['processing_time']:.1f}s")

    results_path = os.path.join(output_dir, "batch_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "base_path": base_path,
                "pattern": pattern,
                "min_object_size": min_object_size,
                "min_cavity_size": min_cavity_size,
            },
            "summary": {
                "total": len(files),
                "successful": successful,
                "failed": len(files) - successful,
                "total_time_seconds": round(total_time, 2),
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate smoothed 3D meshes from spheroid probability maps."
    )
    parser.add_argument("base_path", help="Directory containing the probability maps")
    parser.add_argument("--pattern", default="*.ome_Probabilities.h5",
                        help="Glob pattern for input files (default: %(default)s)")
    parser.add_argument("--min-object-size", type=int, default=1500,
                        help="Minimum cell-object size in voxels (default: %(default)s)")
    parser.add_argument("--min-cavity-size", type=int, default=100,
                        help="Minimum cavity size in voxels (default: %(default)s)")
    args = parser.parse_args()

    batch_process(
        args.base_path,
        pattern=args.pattern,
        min_object_size=args.min_object_size,
        min_cavity_size=args.min_cavity_size,
    )
