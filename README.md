# Scientific Computing Portfolio

This repository collects code from two microscopy research projects. The first processes whole-slide CZI microscopy scans into tiles, classifies tile quality with a fine-tuned vision transformer, and evaluates model behaviour with calibration analysis and embedding visualisation. The second reconstructs 3D meshes of multicellular spheroids and their internal cavities from probability maps produced by a segmentation model.

The files were extracted from larger research repositories. They are published here for review and are not a self-contained application. Each project README states which parts of the original research environment a script depends on.

## Projects

| Directory | Problem | Main techniques |
| --- | --- | --- |
| [microscopy-processing/](microscopy-processing/README.md) | Tile a large collection of multi-scene CZI scans on a Slurm cluster, then add tiles that fall between a lower and the production root-fraction cutoff | Embarrassingly parallel array jobs, idempotent resumable processing, per-scene checkpoints, reservoir sampling |
| [image-classification/](image-classification/README.md) | Classify every tile with a trained Swin-B model and sort copies into class folders without risking the read-only source data | SQLite write-ahead job journal, crash-consistent copy protocol, SHA-256 integrity auditing, batched GPU inference |
| [numerical-methods/](numerical-methods/README.md) | Evaluate a trained classifier beyond accuracy | Exact t-SNE in NumPy, SVD-based PCA, perplexity calibration by vectorised bisection, expected calibration error |
| [spheroid-meshes/](spheroid-meshes/README.md) | Reconstruct 3D surface meshes of multicellular spheroids and their internal cavities from segmentation probability maps | Marching cubes isosurface extraction, Laplacian mesh smoothing, binary hole-filling for cavity detection, batch processing with per-sample diagnostics |

The first three projects form one pipeline. The batch tiler writes PNG tiles into a tree that mirrors the input scans. The recovery script adds tiles under a lower root-fraction cutoff and records each tile's root fraction in a per-scan `tiles.csv`. The classifier reads both tile trees and that manifest, and its model adapter checks its class ordering against the predictions CSV written by the analysis module during training.

The spheroid mesh project is independent. It reads HDF5 probability maps, binarises each volume, detects cavities via flood-filling, extracts isosurfaces with marching cubes, smooths the resulting meshes, and exports STL files for downstream visualisation and measurement.

## Technical summary

- **Languages and libraries.** Python, NumPy, PyTorch, torchvision, OpenCV, Pillow, Matplotlib, SQLite, aicsimageio, scikit-image, SciPy, PyVista, h5py.
- **Numerical methods.** PCA via economy SVD. Exact t-SNE with Student-t kernel, early exaggeration, momentum, and adaptive gains. Per-point Gaussian bandwidths found by bisection on Shannon entropy. Reliability diagrams and expected calibration error. Marching cubes isosurface extraction. Laplacian mesh smoothing.
- **3D reconstruction.** Binary volume processing, connected-component filtering, flood-fill cavity detection, isosurface meshing, mesh smoothing and STL export, porosity quantification.
- **Data integrity.** SHA-256 content hashing, size and mtime_ns change detection, atomic temp-file-and-rename writes, no-clobber publication through hard links, symlink and path-escape guards.
- **Fault tolerance.** Resumable state machines persisted to SQLite or JSON checkpoints, file locks against concurrent runs, reconciliation passes that rebuild outputs from durable records after interruption.
- **HPC.** Slurm array task indexing, stable sorted manifests, per-task logs, line-buffered output for live job monitoring.
- **Sampling.** Algorithm R reservoir sampling within disjoint root-fraction bands, with fixed seeds for reproducibility.

## Repository structure

```
.
├── README.md
├── image-classification/
│   ├── README.md
│   ├── classify.py
│   ├── model_adapter.py
│   └── tests/
│       └── test_safety.py
├── microscopy-processing/
│   ├── README.md
│   ├── batch_tile_mycorrhisee.py
│   ├── recover.py
│   ├── run_pilot.py
│   └── test_recovery.py
├── numerical-methods/
│   ├── README.md
│   └── analysis.py
└── spheroid-meshes/
    ├── README.md
    └── batch_mesh.py
```

## Running the code

The code targets Python 3.10 or newer on Linux or macOS. Several modules use POSIX file locking, so Windows is not supported.

The test suite in `image-classification/` runs without the original research environment:

```
cd image-classification
python -m venv .venv
. .venv/bin/activate
pip install numpy torch torchvision pillow matplotlib
python -m unittest discover -s tests -v
```

The spheroid mesh generator requires `scikit-image`, `scipy`, `pyvista`, `h5py`, and `numpy`. Production runs need HDF5 probability maps from a trained segmentation model, which are not included.

Production runs of every project need private data, model checkpoints, or lab modules that are not included. The project READMEs list these dependencies.
