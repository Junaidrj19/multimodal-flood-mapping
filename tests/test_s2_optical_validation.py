"""Sentinel-2 optical validity: SCL class handling and AOI availability statistics.

These tests exist because the three states this module keeps apart — *missing
data*, *masked*, and *valid observation* — support different scientific claims,
and a single "invalid" count would let a cloudy acquisition look like a granule
that never covered the AOI. They also pin the boundary of the class table: an
unrecognised SCL code must be reported, never absorbed into the valid mask.

No raster is opened here. The statistics are pure functions of arrays, so the
class handling can be tested without a product.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from floodmap.preprocessing.s2_optical import (
    SCL_CLASSES,
    SCL_REFERENCE,
    SclCategory,
    aoi_statistics,
    band_availability,
    classify_scl,
    observation_mask,
)


def test_every_published_scl_class_is_mapped_and_only_surfaces_are_observations():
    """The class table is complete and its observation rule is explicit."""
    assert sorted(SCL_CLASSES) == list(range(12))
    assert SCL_REFERENCE  # the table is cited, not asserted
    for code, entry in SCL_CLASSES.items():
        assert entry.code == code
        assert entry.name
        if entry.is_observation:
            assert entry.category is SclCategory.VALID
    # Water is an observation: the flood question is which water, not whether
    # water is visible.
    assert SCL_CLASSES[6].is_observation is True
    for masked in (2, 3, 8, 9, 10, 11):
        assert SCL_CLASSES[masked].is_observation is False


def test_cloud_cirrus_shadow_and_no_data_are_all_excluded_from_observations():
    scl = np.array(
        [
            [0, 1, 2, 3],  # no_data, saturated, cast shadow, cloud shadow
            [4, 5, 6, 7],  # vegetation, not vegetated, water, unclassified
            [8, 9, 10, 11],  # cloud med, cloud high, thin cirrus, snow
            [4, 6, 6, 5],
        ],
        dtype="int32",
    )
    mask = observation_mask(scl)
    assert mask[0].tolist() == [False, False, False, False]
    assert mask[1].tolist() == [True, True, True, True]
    assert mask[2].tolist() == [False, False, False, False]
    assert mask[3].tolist() == [True, True, True, True]
    summary = classify_scl(scl)
    assert summary.counts["cloud"] == 2
    assert summary.counts["cirrus"] == 1
    assert summary.counts["cloud_shadow"] == 2
    assert summary.counts["no_data"] == 2
    assert summary.counts["snow_ice"] == 1
    assert summary.counts["valid"] == 8
    assert summary.unexpected_codes == ()


def test_an_unrecognised_class_is_counted_and_never_treated_as_valid():
    """A changed encoding must surface as an anomaly, not widen the mask."""
    scl = np.array([[4, 4], [12, 4]], dtype="int32")
    summary = classify_scl(scl)
    assert summary.unexpected_codes == (12,)
    assert summary.counts["unexpected"] == 1
    assert observation_mask(scl).sum() == 3  # the code-12 pixel stays out


def test_float_scl_is_refused_rather_than_truncated():
    with pytest.raises(TypeError, match="integer-typed"):
        classify_scl(np.array([[4.0, 5.0]], dtype="float32"))


def test_aoi_statistics_are_over_the_aoi_not_the_raster():
    """A raster-wide denominator would describe land nobody asked about."""
    scl = np.array(
        [
            [4, 4, 4, 4],
            [4, 9, 4, 4],  # cloud
            [4, 4, 3, 4],  # cloud shadow
            [4, 4, 4, 0],  # no data
        ],
        dtype="int32",
    )
    aoi = np.zeros((4, 4), dtype=bool)
    aoi[1:4, 1:4] = True  # 9 pixels: 2 cloud/shadow/no-data + 6 valid
    stats = aoi_statistics(scl, aoi)
    assert stats.aoi_pixels == 9
    assert stats.valid_observations == 6
    assert stats.masked_cloud == 1
    assert stats.masked_cloud_shadow == 1
    assert stats.missing_data == 1
    assert stats.masked == 2 and stats.missing == 1
    assert stats.valid_fraction == pytest.approx(6 / 9)
    assert stats.is_accounted_for is True


def test_missing_and_masked_stay_distinguishable_when_the_raster_has_no_value():
    """A nodata pixel under an observable class is missing, not cloudy."""
    scl = np.full((2, 2), 5, dtype="int32")  # bare ground: observable
    aoi = np.ones((2, 2), dtype=bool)
    raster_valid = np.array([[True, True], [True, False]])
    stats = aoi_statistics(scl, aoi, raster_valid=raster_valid)
    assert stats.valid_observations == 3
    assert stats.missing_by_raster_mask == 1
    assert stats.masked == 0
    assert stats.missing == 1
    assert stats.is_accounted_for is True
    counts = stats.to_dict()["counts"]
    assert counts["missing_by_raster_mask"] == 1
    assert counts["masked_total"] == 0


def test_empty_aoi_is_an_error_not_a_data_availability_finding():
    with pytest.raises(ValueError, match="selects no pixels"):
        aoi_statistics(np.full((2, 2), 4, dtype="int32"), np.zeros((2, 2), dtype=bool))


def test_mismatched_grids_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        aoi_statistics(
            np.full((2, 2), 4, dtype="int32"), np.ones((3, 3), dtype=bool)
        )


def test_band_availability_reports_missing_bands_explicitly():
    availability = band_availability(["B02", "B03", "SCL"], ["B02", "B03", "B08"])
    assert availability.missing == ("B08",)
    assert availability.complete is False
    assert availability.to_dict()["missing"] == ["B08"]
    assert band_availability(["B02", "B03"], ["B02", "B03"]).complete is True


def test_aoi_mask_is_reprojected_onto_a_projected_grid_and_follows_location():
    """The AOI is 4326 and the production grid is projected, so the mask must be
    reprojected — and it must follow the geometry, not be all-True or all-False.

    An untested reprojection here would silently select the wrong pixels in
    production, where the CRSs always differ.
    """
    from affine import Affine

    from floodmap.acquisition.aoi import AreaOfInterest
    from floodmap.preprocessing.grid import AnalysisGrid
    from floodmap.preprocessing.pipeline import _aoi_geojson_on_grid
    from floodmap.preprocessing.s2_optical import aoi_mask_for_grid

    from pyproj import Transformer

    aoi = AreaOfInterest.from_bbox("synthetic-aoi", [0.0, 0.0, 4.0, 4.0], is_synthetic=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

    # A 4 km square anchored on the AOI's own centre, computed rather than
    # hard-coded: the point is the mask's placement, not my arithmetic. (An
    # anchoring mistake here is instructive — the first version of this test
    # anchored on the corner with a north-up transform and put the grid just
    # outside the AOI, which the mask correctly reported as all-False.)
    easting, northing = to_utm.transform(2.0, 2.0)
    on_aoi = AnalysisGrid(
        crs="EPSG:32631",
        transform=Affine.translation(easting, northing) @ Affine.scale(1000.0, -1000.0),
        width=4,
        height=4,
        resolution_m=1000.0,
    )
    geojson, reprojected = _aoi_geojson_on_grid(aoi, on_aoi, aoi.geometry)
    assert reprojected is True
    assert geojson["type"] == "Polygon"
    assert aoi_mask_for_grid(geojson, on_aoi).all(), "a grid inside the AOI must select every pixel"

    # The same grid size 20 degrees west: outside the AOI entirely.
    west_easting, _ = to_utm.transform(-20.0, 2.0)
    far_away = AnalysisGrid(
        crs="EPSG:32631",
        transform=Affine.translation(west_easting, northing) @ Affine.scale(1000.0, -1000.0),
        width=4,
        height=4,
        resolution_m=1000.0,
    )
    far_geojson, _ = _aoi_geojson_on_grid(aoi, far_away, aoi.geometry)
    assert not aoi_mask_for_grid(far_geojson, far_away).any()


def test_statistics_serialise_with_the_three_states_and_a_stated_denominator():
    scl = np.array([[4, 9], [0, 4]], dtype="int32")
    payload = aoi_statistics(scl, np.ones((2, 2), dtype=bool)).to_dict()
    assert payload["denominator"] == "pixels inside the AOI"
    assert payload["counts"]["valid_observations"] == 2
    assert payload["counts"]["masked_cloud"] == 1
    assert payload["counts"]["missing_no_data_class"] == 1
    assert payload["fully_accounted_for"] is True
    json.dumps(payload)  # machine-readable, not a formatted string
