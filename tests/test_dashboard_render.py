"""Render-helper tests: invalid pixels must never be painted as observations.

Synthetic arrays only. These check the presentation contract, not any scientific
interpretation of the values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from floodmap.dashboard.render import (
    DEFAULT_MAX_DIM,
    candidate_mask_rgba,
    change_rgba,
    decimated_shape,
    gray_rgba,
    read_display,
    valid_mask_rgba,
)


class TestCandidateMaskRgba:
    def test_not_observed_is_transparent(self):
        data = np.array([[1, 0, 255]], dtype="uint8")
        rgba = candidate_mask_rgba(data, nodata=255)
        assert rgba[0, 2, 3] == 0, "not-observed pixel must be fully transparent"

    def test_candidate_is_opaque_and_not_observed_is_not(self):
        data = np.array([[1, 0]], dtype="uint8")
        rgba = candidate_mask_rgba(data, nodata=255)
        assert rgba[0, 0, 3] == 215
        assert rgba[0, 0, 0] > rgba[0, 0, 1], "candidate should be red-dominant"
        assert rgba[0, 1, 3] < 100, "observed non-candidate is only a faint reference"

    def test_unexpected_class_is_transparent(self):
        data = np.array([[7]], dtype="uint8")
        rgba = candidate_mask_rgba(data, nodata=255)
        assert rgba[0, 0, 3] == 0

    def test_all_not_observed_gives_empty_overlay(self):
        data = np.full((4, 4), 255, dtype="uint8")
        assert candidate_mask_rgba(data, nodata=255)[..., 3].sum() == 0

    def test_valid_mask_overlay(self):
        data = np.array([[1, 0]], dtype="uint8")
        rgba = valid_mask_rgba(data)
        assert rgba[0, 0, 3] > 0
        assert rgba[0, 1, 3] == 0


class TestChangeRgba:
    def test_feature_nodata_is_transparent(self):
        data = np.array([[-6.0, -9999.0]], dtype="float32")
        rgba = change_rgba(data, nodata=-9999.0)
        assert rgba[0, 0, 3] > 0
        assert rgba[0, 1, 3] == 0

    def test_nan_is_transparent(self):
        data = np.array([[np.nan, 0.0]], dtype="float32")
        rgba = change_rgba(data, nodata=-9999.0)
        assert rgba[0, 0, 3] == 0
        assert rgba[0, 1, 3] > 0

    def test_darkening_and_brightening_are_distinguishable(self):
        data = np.array([[-8.0, 8.0]], dtype="float32")
        rgba = change_rgba(data, nodata=-9999.0, limit_db=8.0)
        darkening = rgba[0, 0]
        brightening = rgba[0, 1]
        assert darkening[2] > darkening[0], "darkening should be blue-dominant"
        assert brightening[0] > brightening[2], "brightening should be red-dominant"

    def test_zero_change_is_near_neutral(self):
        rgba = change_rgba(np.array([[0.0]], dtype="float32"), nodata=-9999.0)
        red, green, blue = rgba[0, 0, :3]
        assert red > 200 and green > 200 and blue > 200

    def test_no_nodata_declared_still_hides_nonfinite(self):
        rgba = change_rgba(np.array([[np.inf]], dtype="float32"), nodata=None)
        assert rgba[0, 0, 3] == 0


class TestGrayRgba:
    def test_nodata_is_transparent(self):
        data = np.array([[1.0, -9999.0]], dtype="float32")
        rgba = gray_rgba(data, nodata=-9999.0)
        assert rgba[0, 1, 3] == 0

    def test_constant_band_does_not_invent_contrast(self):
        rgba = gray_rgba(np.full((2, 2), 5.0, dtype="float32"), nodata=None)
        assert rgba[..., 3].sum() == 0


class TestDecimation:
    def test_long_side_is_capped(self):
        assert decimated_shape(4961, 3313, 1200) == (801, 1200)

    def test_small_grid_is_not_upscaled(self):
        assert decimated_shape(24, 24, 1200) == (24, 24)

    def test_default_cap_is_sane(self):
        assert 0 < DEFAULT_MAX_DIM <= 4096


class TestReadDisplay:
    def test_class_values_survive_nearest_decimation(self, tmp_path: Path):
        path = tmp_path / "mask.tif"
        data = np.zeros((1, 200, 200), dtype="uint8")
        data[0, :50, :50] = 1
        data[0, -10:, -10:] = 255
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=200,
            height=200,
            count=1,
            dtype="uint8",
            crs="EPSG:32645",
            transform=from_origin(500000, 3000000, 10, 10),
            nodata=255,
        ) as dst:
            dst.write(data)

        array, nodata = read_display(path, band=1, max_dim=64, resampling=Resampling.nearest)
        assert nodata == 255.0
        assert array.shape == (64, 64)
        assert set(np.unique(array)).issubset({0, 1, 255})
        assert array[0, 0] == 1

    def test_band_values_are_preserved(self, tmp_path: Path):
        path = tmp_path / "features.tif"
        data = np.stack(
            [np.full((10, 10), -6.0, dtype="float32"),
             np.full((10, 10), -7.0, dtype="float32")]
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=2,
            dtype="float32",
            crs="EPSG:32645",
            transform=from_origin(500000, 3000000, 10, 10),
            nodata=-9999.0,
        ) as dst:
            dst.write(data)

        band1, _ = read_display(path, band=1)
        band2, _ = read_display(path, band=2)
        assert float(band1.mean()) == pytest.approx(-6.0, abs=1e-6)
        assert float(band2.mean()) == pytest.approx(-7.0, abs=1e-6)
