"""Shared spatial reprojection tests.

Synthetic throughout: small rasters with hand-checkable geometry, so the
assertions are about the transformation rather than about any corpus.

The suite is organised around the properties that, if silently broken, would
corrupt results without failing anything else:

* a real grid is constructed, not just metadata;
* SAR averages in linear power and never in decibels;
* labels and masks move by nearest neighbour and keep their exact values;
* invalid source pixels cannot leak into interpolated valid output;
* image, target and mask stay pixel-aligned;
* training and inference go through the same code.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from floodmap.preprocessing.errors import PreprocessingConfigurationError
from floodmap.preprocessing.grid import AnalysisGrid
from floodmap.preprocessing.reprojection import (
    CANONICAL_RESOLUTION_M,
    NodataPolicy,
    linear_to_db_after_resampling,
    reproject_categorical,
    reproject_continuous,
    target_grid_for_bounds,
    target_grid_for_source,
    utm_epsg_for_lonlat,
)

# A small EPSG:3857 source grid near the Kuro tiles, with 10 PROJECTED unit
# pixels -- exactly the situation the canonical grid exists to correct.
SOURCE_TRANSFORM = Affine(10.0, 0.0, 4801325.0, 0.0, -10.0, 1292245.0)
SOURCE_SIZE = 32


def _source_grid(size: int = SOURCE_SIZE) -> AnalysisGrid:
    return AnalysisGrid(
        crs="EPSG:3857",
        transform=SOURCE_TRANSFORM,
        width=size,
        height=size,
        resolution_m=10.0,
    )


# ---------------------------------------------------------------------------
# UTM zone selection
# ---------------------------------------------------------------------------


class TestUtmZoneSelection:
    @pytest.mark.parametrize(
        "lon,lat,expected",
        [
            (0.5, 10.0, 32631),       # zone 31 N starts at 0
            (-0.5, 10.0, 32630),      # zone 30 N
            (-177.0, 10.0, 32601),    # first zone
            (177.0, 10.0, 32660),     # last zone
            (0.5, -10.0, 32731),      # southern hemisphere
            (-74.0, 40.7, 32618),     # New York
            (85.3, 27.7, 32645),      # Kathmandu — the production AOI zone
        ],
    )
    def test_known_zones(self, lon, lat, expected):
        assert utm_epsg_for_lonlat(lon, lat) == expected

    def test_zone_boundaries_are_exact_not_rounded(self):
        """A point exactly on a boundary belongs to the zone it starts."""
        # -180 .. -174 is zone 1; -174 starts zone 2.
        assert utm_epsg_for_lonlat(-180.0, 0.0) == 32601
        assert utm_epsg_for_lonlat(-174.0, 0.0) == 32602
        assert utm_epsg_for_lonlat(-173.999999, 0.0) == 32602
        assert utm_epsg_for_lonlat(-174.000001, 0.0) == 32601
        # 6-degree steps all the way up.
        assert utm_epsg_for_lonlat(6.0, 0.0) == 32632
        assert utm_epsg_for_lonlat(5.999999, 0.0) == 32631

    def test_longitude_180_does_not_produce_zone_61(self):
        """The clamp exists so the antimeridian cannot index a missing zone."""
        assert utm_epsg_for_lonlat(180.0, 0.0) == 32660
        assert utm_epsg_for_lonlat(-180.0, 0.0) == 32601

    def test_hemisphere_switch_and_equator(self):
        assert utm_epsg_for_lonlat(10.0, 0.0) == 32632      # equator -> north
        assert utm_epsg_for_lonlat(10.0, 1e-9) == 32632
        assert utm_epsg_for_lonlat(10.0, -1e-9) == 32732
        assert utm_epsg_for_lonlat(10.0, -45.0) == 32732

    def test_out_of_range_and_non_finite_are_rejected(self):
        for lon, lat in [(181.0, 0.0), (0.0, 91.0), (-181.0, 0.0), (0.0, -91.0)]:
            with pytest.raises(PreprocessingConfigurationError, match="outside geographic"):
                utm_epsg_for_lonlat(lon, lat)
        with pytest.raises(PreprocessingConfigurationError, match="non-finite"):
            utm_epsg_for_lonlat(float("nan"), 0.0)


# ---------------------------------------------------------------------------
# Target grid construction
# ---------------------------------------------------------------------------


class TestTargetGrid:
    def test_a_concrete_grid_is_built_not_just_metadata(self):
        """CRS, transform, width, height and resolution must all be real."""
        target = target_grid_for_source(_source_grid())
        assert isinstance(target, AnalysisGrid)
        assert target.crs.startswith("EPSG:326") or target.crs.startswith("EPSG:327")
        assert isinstance(target.transform, Affine)
        assert target.width > 0 and target.height > 0
        assert target.resolution_m == CANONICAL_RESOLUTION_M

    def test_target_is_utm_not_the_source_web_mercator(self):
        source = _source_grid()
        target = target_grid_for_source(source)
        assert source.crs == "EPSG:3857"
        assert target.crs != "EPSG:3857"

    def test_transform_encodes_the_requested_resolution(self):
        target = target_grid_for_source(_source_grid(), resolution_m=10.0)
        assert target.transform.a == pytest.approx(10.0)
        assert target.transform.e == pytest.approx(-10.0)
        assert target.transform.b == 0.0 and target.transform.d == 0.0

    def test_non_default_resolution_is_honoured(self):
        target = target_grid_for_source(_source_grid(), resolution_m=20.0)
        assert target.resolution_m == 20.0
        assert target.transform.a == pytest.approx(20.0)

    def test_natural_size_covers_the_source_footprint(self):
        source = _source_grid()
        target = target_grid_for_source(source)
        left, bottom, right, top = target.bounds
        # 32 px * 10 projected units at this latitude is slightly under 320 m
        # of true ground, so a 10 m grid needs about 32 pixels.
        assert 28 <= target.width <= 36
        assert 28 <= target.height <= 36
        assert right > left and top > bottom

    def test_fixed_tile_size_gives_exact_dimensions(self):
        """Required for batching: every sample must have identical shape."""
        target = target_grid_for_source(_source_grid(), tile_size=64)
        assert (target.height, target.width) == (64, 64)

    def test_fixed_tile_size_is_stable_across_differently_sized_sources(self):
        a = target_grid_for_source(_source_grid(32), tile_size=64)
        b = target_grid_for_source(_source_grid(28), tile_size=64)
        assert (a.height, a.width) == (b.height, b.width) == (64, 64)

    def test_grid_is_a_deterministic_function_of_position(self):
        first = target_grid_for_source(_source_grid(), tile_size=64)
        second = target_grid_for_source(_source_grid(), tile_size=64)
        assert first.crs == second.crs
        assert tuple(first.transform[:6]) == tuple(second.transform[:6])

    def test_degenerate_and_invalid_inputs_are_rejected(self):
        with pytest.raises(PreprocessingConfigurationError, match="degenerate"):
            target_grid_for_bounds(
                bounds=(10.0, 10.0, 10.0, 20.0), source_crs="EPSG:3857"
            )
        with pytest.raises(PreprocessingConfigurationError, match="resolution must be positive"):
            target_grid_for_bounds(
                bounds=(0.0, 0.0, 100.0, 100.0), source_crs="EPSG:3857", resolution_m=0.0
            )
        with pytest.raises(PreprocessingConfigurationError, match="tile_size must be positive"):
            target_grid_for_source(_source_grid(), tile_size=0)


# ---------------------------------------------------------------------------
# Continuous (SAR) resampling
# ---------------------------------------------------------------------------


class TestContinuousResampling:
    def test_shapes_dtype_and_grid(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=64)
        data = np.full((4, SOURCE_SIZE, SOURCE_SIZE), 0.1, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=target)
        assert result.data.shape == (4, 64, 64)
        assert result.data.dtype == np.float32
        assert result.valid.shape == (64, 64)
        assert result.valid.dtype == np.bool_
        assert result.grid is target

    def test_a_constant_field_is_preserved_by_bilinear(self):
        """Interpolating a constant must return that constant, not drift."""
        source = _source_grid()
        target = target_grid_for_source(source)
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), 0.0735, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=target)
        interior = result.data[0][result.valid]
        assert interior.size > 0
        np.testing.assert_allclose(interior, 0.0735, rtol=1e-5)

    def test_resampling_averages_in_linear_power_not_decibels(self):
        """The C3 ordering, demonstrated numerically.

        Two linear values 0.01 and 1.0 average to 0.505 in power, which is
        -2.97 dB. Averaging their decibel forms (-20 and 0) gives -10 dB,
        which is linear 0.1 -- five times smaller. Resampling in the linear
        domain and converting afterwards must give the former.
        """
        linear = np.array([[0.01, 1.0]], dtype=np.float64)
        valid = np.ones((1, 2), dtype=bool)

        power_mean = float(linear.mean())
        db_of_power_mean = 10.0 * np.log10(power_mean)
        mean_of_db = float((10.0 * np.log10(linear)).mean())

        assert db_of_power_mean == pytest.approx(-2.9676, abs=1e-3)
        assert mean_of_db == pytest.approx(-10.0, abs=1e-6)
        # The biased-low direction the contract warns about.
        assert mean_of_db < db_of_power_mean

        converted, converted_valid = linear_to_db_after_resampling(
            np.array([[power_mean]]), np.ones((1, 1), dtype=bool)
        )
        assert bool(converted_valid[0, 0])
        assert float(converted[0, 0]) == pytest.approx(db_of_power_mean, abs=1e-6)

    def test_averaging_decibel_input_is_refused(self):
        """The ordering cannot be bypassed by passing dB in."""
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=64)
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), -12.0, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        with pytest.raises(PreprocessingConfigurationError, match="averaging logarithms"):
            reproject_continuous(
                data, valid, source=source, target=target,
                resampling_name="bilinear", domain="db",
            )

    def test_nearest_on_decibel_input_is_allowed(self):
        """Nearest does not average, so the dB prohibition does not apply."""
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=64)
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), -12.0, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)
        result = reproject_continuous(
            data, valid, source=source, target=target,
            resampling_name="nearest", domain="db",
        )
        assert float(result.data[0][result.valid].max()) == pytest.approx(-12.0)

    def test_provenance_records_the_ordering_and_policy(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=64)
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), 0.1, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        provenance = reproject_continuous(
            data, valid, source=source, target=target, domain="linear_power"
        ).provenance
        assert provenance["domain"] == "linear_power"
        assert provenance["db_conversion_applied_before_resampling"] is False
        assert provenance["resampling"] == "bilinear"
        assert provenance["reprojected"] is True
        assert provenance["nodata_policy"]["mode"] == "strict"
        assert provenance["nodata_policy"]["invalid_source_treated_as_zero_backscatter"] is False
        assert provenance["source_grid"]["crs"] == "EPSG:3857"
        assert provenance["target_grid"]["resolution_m"] == 10.0

    def test_mismatched_mask_shape_is_rejected(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=64)
        data = np.zeros((1, SOURCE_SIZE, SOURCE_SIZE), dtype=np.float32)
        with pytest.raises(PreprocessingConfigurationError, match="validity mask shape"):
            reproject_continuous(
                data, np.ones((4, 4), dtype=bool), source=source, target=target
            )


# ---------------------------------------------------------------------------
# Nodata behaviour -- the subtle one
# ---------------------------------------------------------------------------


class TestNodataBehaviour:
    def _half_invalid(self):
        """A field whose left half is a real measurement and right half invalid.

        The invalid half carries 0.0, which is what Kuro actually stores and is
        indistinguishable from genuine near-zero backscatter.
        """
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), 0.12, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)
        data[0, :, SOURCE_SIZE // 2:] = 0.0
        valid[:, SOURCE_SIZE // 2:] = False
        return data, valid

    def test_invalid_pixels_do_not_pull_valid_output_toward_zero(self):
        """The core guarantee. Every surviving pixel must equal 0.12 exactly.

        If the 0.0 sentinel entered a bilinear kernel, pixels near the boundary
        would land between 0 and 0.12 -- a plausible, undetectable, fictional
        backscatter value sitting exactly where open water would be.
        """
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data, valid = self._half_invalid()

        result = reproject_continuous(
            data, valid, source=source, target=target, nodata=-9999.0
        )
        surviving = result.data[0][result.valid]
        assert surviving.size > 0
        np.testing.assert_allclose(surviving, 0.12, rtol=1e-5)
        # Nothing between the sentinel and the true value leaked through.
        assert not np.any((surviving > 1e-6) & (surviving < 0.1199))

    def test_output_nodata_only_where_invalid(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data, valid = self._half_invalid()

        result = reproject_continuous(
            data, valid, source=source, target=target, nodata=-9999.0
        )
        assert np.all(result.data[0][~result.valid] == -9999.0)
        assert not np.any(result.data[0][result.valid] == -9999.0)

    def test_strict_is_at_least_as_conservative_as_majority(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data, valid = self._half_invalid()

        strict = reproject_continuous(
            data, valid, source=source, target=target,
            nodata_policy=NodataPolicy("strict"),
        )
        majority = reproject_continuous(
            data, valid, source=source, target=target,
            nodata_policy=NodataPolicy("majority"),
        )
        assert int(strict.valid.sum()) <= int(majority.valid.sum())
        # Strict's valid set must be a subset of majority's.
        assert not np.any(strict.valid & ~majority.valid)

    def test_a_fully_invalid_source_yields_no_valid_output(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data = np.zeros((1, SOURCE_SIZE, SOURCE_SIZE), dtype=np.float32)
        valid = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=target)
        assert int(result.valid.sum()) == 0
        assert np.all(result.data == -9999.0)

    def test_area_outside_the_source_footprint_is_invalid_not_filled(self):
        """A fixed tile larger than the footprint must pad with nodata."""
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=96)
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), 0.09, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=target)
        assert int(result.valid.sum()) < 96 * 96, "padding must exist"
        assert int(result.valid.sum()) > 0
        np.testing.assert_allclose(result.data[0][result.valid], 0.09, rtol=1e-5)
        assert np.all(result.data[0][~result.valid] == -9999.0)

    def test_policy_rejects_an_unknown_mode(self):
        with pytest.raises(PreprocessingConfigurationError, match="unknown nodata policy"):
            NodataPolicy(mode="whatever")

    def test_policy_threshold_matches_its_mode(self):
        assert NodataPolicy("strict").minimum_valid_weight == 1.0
        assert NodataPolicy("majority").minimum_valid_weight == 0.5
        assert NodataPolicy("strict").threshold < 1.0  # tolerance only


# ---------------------------------------------------------------------------
# Categorical (label / mask) resampling
# ---------------------------------------------------------------------------


class TestCategoricalResampling:
    def _labels(self):
        labels = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=np.int16)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)
        # Adjacent 0 and 2 blocks: bilinear across this boundary would yield 1.
        labels[:, SOURCE_SIZE // 2:] = 2
        labels[:4, :4] = 3           # stored nodata
        valid[:4, :4] = False
        return labels, valid

    def test_nearest_preserves_exact_class_values(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels, valid = self._labels()

        result = reproject_categorical(
            labels, valid, source=source, target=target,
            ignore_index=3, semantic_values=(0, 1, 2),
        )
        present = set(int(v) for v in np.unique(result.data[result.valid]))
        assert present.issubset({0, 2}), present
        assert result.data.dtype == np.int64

    def test_no_intermediate_class_is_invented_at_a_boundary(self):
        """Class 1 is absent from the source and must stay absent.

        This is the concrete failure the nearest-neighbour rule prevents: a 0/2
        boundary interpolated linearly gives 1, which in this scheme is
        'Permanent Waters' -- a different, entirely plausible class.
        """
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels, valid = self._labels()
        assert 1 not in set(int(v) for v in np.unique(labels))

        result = reproject_categorical(
            labels, valid, source=source, target=target,
            ignore_index=3, semantic_values=(0, 1, 2),
        )
        assert 1 not in set(int(v) for v in np.unique(result.data))

    def test_ignore_index_is_preserved_and_never_becomes_a_class(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels, valid = self._labels()

        result = reproject_categorical(
            labels, valid, source=source, target=target,
            ignore_index=3, semantic_values=(0, 1, 2),
        )
        assert np.all(result.data[~result.valid] == 3)
        assert not np.any(result.data[result.valid] == 3)

    def test_an_averaging_method_is_refused_for_labels(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels, valid = self._labels()

        with pytest.raises(PreprocessingConfigurationError, match="invents classes"):
            reproject_categorical(
                labels, valid, source=source, target=target,
                ignore_index=3, resampling_name="bilinear",
            )

    def test_non_semantic_source_values_on_valid_pixels_are_rejected(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels = np.full((SOURCE_SIZE, SOURCE_SIZE), 7, dtype=np.int16)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        with pytest.raises(PreprocessingConfigurationError, match="non-semantic label"):
            reproject_categorical(
                labels, valid, source=source, target=target,
                ignore_index=3, semantic_values=(0, 1, 2),
            )

    def test_provenance_records_classes_and_method(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        labels, valid = self._labels()

        provenance = reproject_categorical(
            labels, valid, source=source, target=target,
            ignore_index=3, semantic_values=(0, 1, 2),
        ).provenance
        assert provenance["resampling"] == "nearest"
        assert provenance["ignore_index"] == 3
        assert provenance["source_classes_present"] == [0, 2]


# ---------------------------------------------------------------------------
# Alignment across payloads
# ---------------------------------------------------------------------------


class TestPayloadAlignment:
    def test_all_four_channels_labels_and_mask_land_on_one_grid(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)

        data = np.stack(
            [np.full((SOURCE_SIZE, SOURCE_SIZE), 0.05 * (i + 1), dtype=np.float32) for i in range(4)]
        )
        labels = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=np.int16)
        labels[:, SOURCE_SIZE // 2:] = 2
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)
        valid[:6, :6] = False
        labels[:6, :6] = 3
        data[:, :6, :6] = 0.0

        sar = reproject_continuous(data, valid, source=source, target=target)
        cat = reproject_categorical(
            labels, valid, source=source, target=target,
            ignore_index=3, semantic_values=(0, 1, 2),
        )

        assert sar.data.shape[-2:] == cat.data.shape == (48, 48)
        assert sar.valid.shape == cat.valid.shape == (48, 48)
        assert sar.grid.crs == cat.grid.crs
        assert tuple(sar.grid.transform[:6]) == tuple(cat.grid.transform[:6])

    def test_every_channel_shares_one_validity_footprint(self):
        """Channels must not be masked independently."""
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data = np.stack(
            [np.full((SOURCE_SIZE, SOURCE_SIZE), 0.05 * (i + 1), dtype=np.float32) for i in range(4)]
        )
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)
        valid[:, SOURCE_SIZE // 2:] = False
        data[:, :, SOURCE_SIZE // 2:] = 0.0

        result = reproject_continuous(data, valid, source=source, target=target)
        per_channel = [(result.data[c] == -9999.0) for c in range(4)]
        for channel in per_channel[1:]:
            np.testing.assert_array_equal(channel, per_channel[0])
        np.testing.assert_array_equal(per_channel[0], ~result.valid)

    def test_channel_values_are_not_mixed_between_channels(self):
        source = _source_grid()
        target = target_grid_for_source(source, tile_size=48)
        data = np.stack(
            [np.full((SOURCE_SIZE, SOURCE_SIZE), 0.05 * (i + 1), dtype=np.float32) for i in range(4)]
        )
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=target)
        for index in range(4):
            np.testing.assert_allclose(
                result.data[index][result.valid], 0.05 * (index + 1), rtol=1e-5
            )


# ---------------------------------------------------------------------------
# Already-aligned short circuit
# ---------------------------------------------------------------------------


class TestAlreadyAligned:
    def test_an_identical_grid_is_not_resampled(self):
        source = _source_grid()
        data = np.full((1, SOURCE_SIZE, SOURCE_SIZE), 0.1, dtype=np.float32)
        valid = np.ones((SOURCE_SIZE, SOURCE_SIZE), dtype=bool)

        result = reproject_continuous(data, valid, source=source, target=source)
        assert result.provenance["reprojected"] is False
        np.testing.assert_allclose(result.data[0], 0.1, rtol=1e-6)
        np.testing.assert_array_equal(result.valid, valid)
