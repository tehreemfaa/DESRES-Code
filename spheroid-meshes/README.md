# Spheroid mesh generator

Reconstructs 3D surface meshes of multicellular spheroids and their
internal cavities from HDF5 probability maps produced by a trained
segmentation model (Ilastik).

This code was developed for a study of cavity formation induced by
hyaluronic acid synthase overexpression (HAS1, HAS2, HAS3) in HEK293T
spheroids imaged with light-sheet microscopy.

## What it does

1. **Binarise** the probability map by thresholding the foreground
   channel.
2. **Clean** the binary volume by removing connected components below a
   configurable size threshold.
3. **Detect cavities** by flood-filling the cleaned volume and
   subtracting the original mask. Small cavities are filtered out.
4. **Extract isosurfaces** with marching cubes (scikit-image) at the
   0.5 level set, producing triangulated meshes for both the outer cell
   structure and the internal cavities.
5. **Smooth** each mesh with Laplacian relaxation (PyVista).
6. **Export** STL files and a JSON summary with per-sample statistics
   (voxel counts, porosity, vertex and face counts, volume, surface
   area).

## Dependencies

- Python 3.10+
- numpy, scipy, scikit-image, pyvista, h5py

## Usage

```
python batch_mesh.py /path/to/data \
    --pattern "*.ome_Probabilities.h5" \
    --min-object-size 1500 \
    --min-cavity-size 100
```

The input directory must contain a `3-D 2-C Probabilities` subdirectory
with the HDF5 files. Output STL files and a `batch_results.json` summary
are written to a `meshes` subdirectory.

## Research environment

Production runs require HDF5 probability maps exported from a trained
Ilastik pixel-classification project. The segmentation model, raw
microscopy data, and Ilastik project files are not included.
