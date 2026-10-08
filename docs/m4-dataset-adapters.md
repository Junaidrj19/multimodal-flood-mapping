# M4 Dataset Adapters

The M4 adapter boundary is implemented in `src/floodmap/datasets/`. The
external corpora are never copied into this repository and no absolute local
dataset path is embedded in production Python code.

## External roots

Pass a root explicitly to an adapter, or set the corresponding environment
variable:

```text
FLOODMAP_KURO_SIWO_ROOT=/external/path/kuro_siwo
FLOODMAP_SEN1FLOODS11_ROOT=/external/path/senfloods11
```

Kuro Siwo is indexed from activation directories and its `01` partition only.
The adapter reads `catalogue.gkpg`, filters `exported == 1`, and verifies that
the exported source-level triplets and labelled grids agree. `SL1` and `MS1`
become the baseline pre/post pair. `SL2` remains in temporal metadata and is
reported in `dropped_sources`; it is never substituted for `SL1`.

Sen1Floods11 is indexed from the five hand-labelled directories:
`S1Hand`, `S2Hand`, `LabelHand`, `S1OtsuLabelHand` and `JRCWaterHand`. The four
official hand-labelled CSV manifests provide the split identity. A sample is
rejected if any component is missing or its CRS, transform or shape is not
aligned with the other four files.

## Canonical sample

`CanonicalSample` preserves source CRS, transform, shape, source spacing units,
temporal roles, native SAR representation, source label scheme, validity,
capabilities, preprocessing state and compact provenance. Optional fields are
absent rather than zero-filled: Sen1Floods11 has no pre-event SAR, DEM or
permanent-water/flood separation.

The typed interface exposes `optical` and `dem` as optional fields, while labels
and the canonical validity mask remain explicit fields for both hand-labelled
corpora.

The target-grid metadata derives a WGS84/UTM EPSG code from each sample
centroid and records a 10.0 m target resolution. It does not claim that 10 m is
native source resolution and does not perform reprojection or resampling.

For later raster preprocessing, the frozen policies remain type-specific:
SAR bilinear in linear power before dB conversion, optical bilinear, labels
nearest, validity masks nearest, and DEM bilinear. Output nodata is class-mask
255 and continuous -9999.0; Kuro's input ignore index 3 is not reused as
output nodata.

## Representation and validity

Kuro Siwo samples expose clipped linear sigma0 arrays (`max=0.15`) and record
that the clip was applied at the adapter. No additional speckle filter is
applied, and bundled `MK0_DEM` and `MK0_SLOPE` are excluded from the M4
SAR-only baseline. Kuro integrity checks require `MK0_MLU == 3`, `MK0_MNA == 0`
and SAR zero-valued pixels to agree.

Sen1Floods11 S1 is retained as VV/VH dB and S2 is retained as 13-band int16
TOA reflectance scaled by 10000. Its only declared validity mechanism is
`LabelHand == -1`; the adapter does not invent a pre-event image, change
features, DEM capability or permanent-water class.

The dataset capability loader attaches the declared validity-mechanism list to
the structured capability model. It is intentionally a list because Kuro has
three redundant mechanisms while Sen1Floods11 has one.
