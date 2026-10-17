"""Tests for the Sentinel-1 RTC VV/VH assembly adapter.

No test touches the network. Each builds small synthetic single-band rasters
with the shape and semantics the real Planetary Computer RTC assets have
(float32, nodata -32768, EPSG:32645, 10 m), so the adapter's contract is pinned
without depending on a signed URL or a 1.8 GB download.

The assembly exists only to satisfy M2's named-band gate. These tests therefore
pin both halves of that: that the names are written, and that the adapter
refuses inputs it must not silently reconcile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

from floodmap.preprocessing.s1_rtc_assembly import (  # noqa: E402
    BAND_NAMES,
    RTC_RADIOMETRIC_CONVENTION,
    AssembledBandPair,
    RtcAssemblyError,
    assemble_rtc_window,
)

# A small patch of UTM 45N at 10 m, in the same region as the Trishuli corridor
# so the lon/lat window transform exercises a realistic path.
_CRS = "EPSG:32645"
_RES = 10.0
_ORIGIN_X, _ORIGIN_Y = 300000.0, 3130000.0
_NODATA = -32768.0
_SIZE = 64


def _aoi_inside() -> tuple[float, float, float, float]:
    """A lon/lat box covering the middle half of the synthetic raster.

    Derived from the raster's own extent rather than hard-coded, so the window
    is guaranteed to intersect and the test does not depend on a coordinate
    guess being right.
    """
    from rasterio.warp import transform_bounds

    left, top = _ORIGIN_X, _ORIGIN_Y
    right, bottom = left + _SIZE * _RES, top - _SIZE * _RES
    qx, qy = (right - left) / 4.0, (top - bottom) / 4.0
    return transform_bounds(
        _CRS, "EPSG:4326", left + qx, bottom + qy, right - qx, top - qy
    )


def _write_single_band(
    path: Path,
    *,
    values: np.ndarray | None = None,
    crs: str = _CRS,
    res: float = _RES,
    origin: tuple[float, float] = (_ORIGIN_X, _ORIGIN_Y),
    nodata: float | None = _NODATA,
    size: int = _SIZE,
    dtype: str = "float32",
    description: str = "Sentinel-1 Calibrated and Terrain Corrected VV",
) -> Path:
    """One single-band raster shaped like a real RTC asset."""
    if values is None:
        values = np.full((size, size), 0.2, dtype=dtype)
    transform = Affine(res, 0.0, origin[0], 0.0, -res, origin[1])
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(dtype), 1)
        dst.set_band_description(1, description)
    return path


def _pair(tmp_path: Path, **vh_overrides) -> tuple[Path, Path]:
    vv = _write_single_band(tmp_path / "vv.tif")
    vh = _write_single_band(
        tmp_path / "vh.tif",
        values=np.full((_SIZE, _SIZE), 0.05, dtype="float32"),
        description="Sentinel-1 Calibrated and Terrain Corrected VH",
        **vh_overrides,
    )
    return vv, vh


# ---------------------------------------------------------------------------
# The contract M2 actually checks
# ---------------------------------------------------------------------------


def test_assembly_writes_exactly_the_band_names_m2_requires(tmp_path):
    """The whole purpose: descriptions must be literally 'VV' and 'VH'.

    M2 resolves bands by name and refuses to guess order, so a descriptive name
    like the provider's own 'Sentinel-1 Calibrated and Terrain Corrected VV' is
    unusable downstream.
    """
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    assert out.band_descriptions == ("VV", "VH") == BAND_NAMES
    with rasterio.open(out.path) as ds:
        assert ds.descriptions == ("VV", "VH")
        assert ds.count == 2


def test_assembly_satisfies_m2_named_band_validation(tmp_path):
    """End to end against M2's real validator, not a reimplementation."""
    from floodmap.preprocessing.validation import require_valid_raster

    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    metadata = require_valid_raster(out.path, min_bands=2, required_bands=("VV", "VH"))
    assert metadata.count == 2
    assert {d.strip() for d in metadata.descriptions if d} == {"VV", "VH"}


def test_band_order_is_vv_then_vh(tmp_path):
    """VV must be band 1 and VH band 2, verified by their distinct values."""
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    with rasterio.open(out.path) as ds:
        band1 = ds.read(1)
        band2 = ds.read(2)
    # VV was written as 0.2, VH as 0.05.
    assert np.allclose(band1[np.isfinite(band1)], 0.2)
    assert np.allclose(band2[np.isfinite(band2)], 0.05)


# ---------------------------------------------------------------------------
# Georeferencing and radiometry are preserved, not re-derived
# ---------------------------------------------------------------------------


def test_georeferencing_and_nodata_survive_assembly(tmp_path):
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    with rasterio.open(out.path) as ds:
        assert ds.crs.to_string() == _CRS
        assert ds.nodata == _NODATA
        assert ds.dtypes == ("float32", "float32")
        # No resampling: pixel size is unchanged.
        assert ds.res == (_RES, _RES)
        # Output origin sits on the source grid (integer pixel multiples).
        assert (ds.transform.c - _ORIGIN_X) % _RES == pytest.approx(0.0, abs=1e-6)
        assert (_ORIGIN_Y - ds.transform.f) % _RES == pytest.approx(0.0, abs=1e-6)


def test_window_is_a_subset_not_the_whole_raster(tmp_path):
    """Only the AOI window is read; this is the point of windowing."""
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    assert out.width < _SIZE or out.height < _SIZE
    assert out.width > 0 and out.height > 0


def test_radiometric_convention_is_recorded_as_gamma0_and_not_converted(tmp_path):
    """gamma0 must be labelled gamma0 and left numerically untouched."""
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    assert out.radiometric_convention == RTC_RADIOMETRIC_CONVENTION == "gamma0_linear_power"
    record = out.to_dict()
    assert "NOT sigma-nought" in record["radiometric_note"]
    # The values are passed through unchanged -- no cos(theta) applied anywhere.
    with rasterio.open(out.path) as ds:
        assert np.allclose(ds.read(1), 0.2)
        tags = ds.tags()
    assert tags["radiometric_convention"] == "gamma0_linear_power"


def test_statistics_report_valid_and_invalid_counts(tmp_path):
    values = np.full((_SIZE, _SIZE), 0.2, dtype="float32")
    values[:8, :] = _NODATA  # a fill stripe
    vv = _write_single_band(tmp_path / "vv.tif", values=values)
    vh = _write_single_band(
        tmp_path / "vh.tif",
        values=np.full((_SIZE, _SIZE), 0.05, dtype="float32"),
        description="... VH",
    )
    out = assemble_rtc_window(
        str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="synthetic_rtc"
    )
    stats = out.statistics["VV"]
    assert stats["pixels"] == stats["valid_pixels"] + stats["invalid_pixels"]
    assert stats["all_finite"] is True
    # nodata must never be counted as an observation.
    assert stats["min"] is None or stats["min"] > _NODATA


# ---------------------------------------------------------------------------
# Incompatible inputs are refused, never reconciled
# ---------------------------------------------------------------------------


def test_mismatched_crs_is_refused(tmp_path):
    vv, vh = _pair(tmp_path, crs="EPSG:32644")
    with pytest.raises(RtcAssemblyError, match="disagree on crs"):
        assemble_rtc_window(
            str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="x"
        )


def test_mismatched_resolution_is_refused(tmp_path):
    vv, vh = _pair(tmp_path, res=20.0)
    with pytest.raises(RtcAssemblyError, match="disagree on"):
        assemble_rtc_window(
            str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="x"
        )


def test_mismatched_origin_is_refused(tmp_path):
    """A shifted grid would stack VV and VH from different ground positions."""
    vv, vh = _pair(tmp_path, origin=(_ORIGIN_X + 30.0, _ORIGIN_Y))
    with pytest.raises(RtcAssemblyError, match="disagree on transform"):
        assemble_rtc_window(
            str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="x"
        )


def test_mismatched_dimensions_are_refused(tmp_path):
    """Differing raster sizes mean the two assets are not one gridded pair."""
    vv = _write_single_band(tmp_path / "vv.tif")
    vh = _write_single_band(
        tmp_path / "vh.tif",
        values=np.full((_SIZE // 2, _SIZE // 2), 0.05, dtype="float32"),
        description="... VH",
    )
    with pytest.raises(RtcAssemblyError, match="disagree on"):
        assemble_rtc_window(
            str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="x"
        )


def test_aoi_outside_projection_domain_is_refused(tmp_path):
    """A non-intersecting AOI is a clear error, not an empty raster or a NaN."""
    vv, vh = _pair(tmp_path)
    with pytest.raises(RtcAssemblyError, match="does not project into|does not intersect"):
        assemble_rtc_window(
            str(vv), str(vh), (0.0, 0.0, 0.1, 0.1), tmp_path / "pair.tif", source_item_id="x"
        )


def test_aoi_in_domain_but_off_raster_is_refused(tmp_path):
    """Inside UTM 45N but outside this raster's footprint."""
    vv, vh = _pair(tmp_path)
    # Same zone, far from the synthetic raster's small extent.
    with pytest.raises(RtcAssemblyError, match="does not intersect|does not project"):
        assemble_rtc_window(
            str(vv), str(vh), (86.5, 26.0, 86.6, 26.1), tmp_path / "pair.tif",
            source_item_id="x",
        )


def test_missing_nodata_is_refused(tmp_path):
    """An undeclared fill value would be read as a real observation."""
    vv, vh = _pair(tmp_path, nodata=None)
    # VV also needs it cleared for the pair to agree, so the nodata gate (not
    # the agreement gate) is what fires.
    vv = _write_single_band(tmp_path / "vv_nonodata.tif", nodata=None)
    with pytest.raises(RtcAssemblyError, match="no nodata"):
        assemble_rtc_window(
            str(vv), str(vh), _aoi_inside(), tmp_path / "pair.tif", source_item_id="x"
        )


def test_provenance_record_carries_item_and_source_profiles(tmp_path):
    vv, vh = _pair(tmp_path)
    out = assemble_rtc_window(
        str(vv),
        str(vh),
        _aoi_inside(),
        tmp_path / "pair.tif",
        source_item_id="S1D_example_rtc",
        source_assets={"vv": {"href": "https://example/vv.tif", "etag": "abc"},
                       "vh": {"href": "https://example/vh.tif", "etag": "def"}},
    )
    record = out.to_dict()
    assert record["sources"]["item_id"] == "S1D_example_rtc"
    assert record["sources"]["vv"]["etag"] == "abc"
    assert record["sources"]["vh"]["etag"] == "def"
    assert record["sources"]["vv_profile"]["crs"] == _CRS
    assert record["window_bounds"] and len(record["window_bounds"]) == 4
    assert isinstance(out, AssembledBandPair)
