"""Offline M3 feature-generation tests.

Everything here runs from tiny synthetic GeoTIFFs written through the real M2
artifact writer, so the tests exercise the actual M2/M3 interface rather than a
convenient stand-in. No test requires CDSE, network access, real Trishuli
imagery, or any validation-only product.

Numerical tests use analytically known arrays and assert exact expected values.
A feature formula that is "about right" is not verified.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pytest
import rasterio
import yaml
from affine import Affine
from rasterio.transform import from_origin

from floodmap.acquisition.aoi import AreaOfInterest
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import DiscoveryStatus
from floodmap.acquisition.providers.base import DiscoveryResult, SearchRequest
from floodmap.acquisition.scenes import Sentinel1Scene, Sentinel2Scene, SensorKind
from floodmap.acquisition.selection import select_sentinel1_pair, select_sentinel2_pair
from floodmap.acquisition.temporal import EventSpec, SearchWindow
from floodmap.features import numerics
from floodmap.features.config import FeatureConfig
from floodmap.features.errors import (
    FeatureBoundaryError,
    FeatureConfigurationError,
    FeatureGridError,
    FeatureInputError,
    FeatureRegistryError,
)
from floodmap.features.inputs import FeatureInputs, load_preprocessed_input
from floodmap.features.pipeline import build_feature_registry, generate_feature_set
from floodmap.features.registry import (
    FEATURE_TEMPLATES,
    ArtifactRole,
    BackscatterRepresentation,
    FeatureFamily,
    TransformKind,
    template_ids,
)
from floodmap.preprocessing.artifacts import write_raster_artifact
from floodmap.preprocessing.grid import AnalysisGrid
from floodmap.utils.config import load_config
from floodmap.utils.provenance import (
    ArtifactType,
    ProductionInput,
    ValidationOnlySource,
)
from tests.conftest import REPO_ROOT

# -----------------------------------------------------------------------------
# Synthetic fixtures
#
# A projected metre grid (UTM 45N covers Nepal) because terrain derivatives
# require one; 4x4 is the smallest grid with an interior pixel for the 3x3
# slope kernel plus a border to check that the border is invalidated.
# -----------------------------------------------------------------------------

GRID_CRS = "EPSG:32645"
RESOLUTION_M = 10.0
SIZE = 4
TRANSFORM = from_origin(300000.0, 3100000.0, RESOLUTION_M, RESOLUTION_M)
NODATA = -9999.0

AOI = AreaOfInterest.from_bbox("synthetic-m3", [85.0, 27.0, 85.1, 27.1], is_synthetic=True)
EVENT = EventSpec(
    event_date=date(2026, 8, 26),
    event_time_utc=datetime(2026, 8, 26, 6, tzinfo=timezone.utc),
)
PLAN = EVENT.plan(SearchWindow(before_days=12, after_days=12))

S1_BANDS = ("VV", "VH")
S2_BANDS = ("B03", "B04", "B08", "B11")
S2_ROLE_BINDING = {"green": "B03", "red": "B04", "nir": "B08", "swir16": "B11"}


def _grid(
    *,
    crs: str = GRID_CRS,
    transform: Any = TRANSFORM,
    resolution_m: float = RESOLUTION_M,
    size: int = SIZE,
) -> AnalysisGrid:
    return AnalysisGrid(
        crs=crs, transform=transform, width=size, height=size, resolution_m=resolution_m
    )


def _s1_scene(scene_id: str, acquired_at: datetime) -> Sentinel1Scene:
    return Sentinel1Scene(
        scene_id=scene_id,
        product_id=f"product-{scene_id}",
        acquired_at=acquired_at,
        product_type="IW_GRDH_1S",
        platform_short_name="SENTINEL-1",
        platform_unit="S1A",
        provider="cdse-odata",
        relative_orbit=19,
        orbit_direction="DESCENDING",
        polarisations=S1_BANDS,
    )


def _s2_scene(scene_id: str, acquired_at: datetime) -> Sentinel2Scene:
    return Sentinel2Scene(
        scene_id=scene_id,
        product_id=f"product-{scene_id}",
        acquired_at=acquired_at,
        product_type="S2MSI2A",
        platform_short_name="SENTINEL-2",
        platform_unit="S2A",
        provider="cdse-odata",
        scene_cloud_percent=31.0,
    )


def _manifest(sensor: SensorKind) -> AcquisitionManifest:
    """A complete M1 manifest, so the M2 artifacts M3 reads are traceable."""
    if sensor is SensorKind.SENTINEL1:
        before = _s1_scene("s1-before", datetime(2026, 8, 20, 6, tzinfo=timezone.utc))
        after = _s1_scene("s1-after", datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        selection = select_sentinel1_pair([before, after], PLAN)
    else:
        before = _s2_scene("s2-before", datetime(2026, 8, 19, 5, tzinfo=timezone.utc))
        after = _s2_scene("s2-after", datetime(2026, 8, 30, 5, tzinfo=timezone.utc))
        selection = select_sentinel2_pair([before, after], PLAN, strategy="nearest_in_time")
    request = SearchRequest(
        aoi=AOI, sensor=sensor, interval=PLAN.full_extent, product_type=before.product_type
    )
    discovery = DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        request=request,
        scenes=(before, after),
        provider="cdse-odata",
    )
    return AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={sensor.value: PLAN},
        requests={sensor.value: request},
        discoveries={sensor.value: discovery},
        selections={sensor.value: selection},
        provider="cdse-odata",
        config_version="m3-test-config",
    )


def _write_m2_artifact(
    output_dir: Path,
    stem: str,
    *,
    bands: Sequence[np.ndarray],
    band_names: Sequence[str],
    valid: Optional[np.ndarray] = None,
    manifest: AcquisitionManifest,
    sensor: Optional[SensorKind],
    production_input: Optional[ProductionInput] = None,
    grid: Optional[AnalysisGrid] = None,
    dem_version: Optional[str] = None,
    scene_ids: Sequence[str] = (),
    product_ids: Sequence[str] = (),
) -> Path:
    """Write one M2-compatible artifact through M2's own writer."""
    target_grid = grid or _grid()
    data = np.stack([np.asarray(band, dtype="float32") for band in bands])
    if valid is None:
        valid = np.ones((target_grid.height, target_grid.width), dtype=bool)
    data = np.where(valid[None, :, :], data, NODATA).astype("float32")
    artifact = write_raster_artifact(
        data=data,
        valid_mask=valid,
        grid=target_grid,
        output_dir=output_dir,
        stem=stem,
        manifest=manifest,
        sensor=sensor,
        production_input=production_input,
        source_product_ids=list(product_ids) or [f"product-{stem}"],
        source_scene_ids=list(scene_ids) or [stem],
        operations=["synthetic_m2_fixture"],
        config_version="m3-test-config",
        preprocessing_version="m2-test-pipeline",
        band_names=band_names,
        dem_version=dem_version,
        nodata=NODATA,
    )
    return artifact.artifact_path


def _s1_pair(
    tmp_path: Path,
    *,
    before_vv: np.ndarray,
    after_vv: np.ndarray,
    before_vh: Optional[np.ndarray] = None,
    after_vh: Optional[np.ndarray] = None,
    before_valid: Optional[np.ndarray] = None,
    after_valid: Optional[np.ndarray] = None,
    grid: Optional[AnalysisGrid] = None,
) -> dict[ArtifactRole, Path]:
    manifest = _manifest(SensorKind.SENTINEL1)
    ids = ["s1-before", "s1-after"]
    products = ["product-s1-before", "product-s1-after"]
    return {
        ArtifactRole.SENTINEL1_BEFORE: _write_m2_artifact(
            tmp_path / "m2" / "sentinel1",
            "before_s1-before",
            bands=[before_vv, before_vh if before_vh is not None else before_vv],
            band_names=S1_BANDS,
            valid=before_valid,
            manifest=manifest,
            sensor=SensorKind.SENTINEL1,
            grid=grid,
            scene_ids=ids,
            product_ids=products,
        ),
        ArtifactRole.SENTINEL1_AFTER: _write_m2_artifact(
            tmp_path / "m2" / "sentinel1",
            "after_s1-after",
            bands=[after_vv, after_vh if after_vh is not None else after_vv],
            band_names=S1_BANDS,
            valid=after_valid,
            manifest=manifest,
            sensor=SensorKind.SENTINEL1,
            grid=grid,
            scene_ids=ids,
            product_ids=products,
        ),
    }


def _s2_pair(
    tmp_path: Path,
    *,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    before_valid: Optional[np.ndarray] = None,
    after_valid: Optional[np.ndarray] = None,
) -> dict[ArtifactRole, Path]:
    manifest = _manifest(SensorKind.SENTINEL2)
    ids = ["s2-before", "s2-after"]
    products = ["product-s2-before", "product-s2-after"]
    return {
        ArtifactRole.SENTINEL2_BEFORE: _write_m2_artifact(
            tmp_path / "m2" / "sentinel2",
            "before_s2-before",
            bands=[before[name] for name in S2_BANDS],
            band_names=S2_BANDS,
            valid=before_valid,
            manifest=manifest,
            sensor=SensorKind.SENTINEL2,
            scene_ids=ids,
            product_ids=products,
        ),
        ArtifactRole.SENTINEL2_AFTER: _write_m2_artifact(
            tmp_path / "m2" / "sentinel2",
            "after_s2-after",
            bands=[after[name] for name in S2_BANDS],
            band_names=S2_BANDS,
            valid=after_valid,
            manifest=manifest,
            sensor=SensorKind.SENTINEL2,
            scene_ids=ids,
            product_ids=products,
        ),
    }


def _dem(
    tmp_path: Path,
    elevation: np.ndarray,
    *,
    valid: Optional[np.ndarray] = None,
    grid: Optional[AnalysisGrid] = None,
) -> dict[ArtifactRole, Path]:
    return {
        ArtifactRole.DEM: _write_m2_artifact(
            tmp_path / "m2" / "dem",
            "elevation",
            bands=[elevation],
            band_names=("elevation",),
            valid=valid,
            manifest=_manifest(SensorKind.SENTINEL1),
            sensor=None,
            production_input=ProductionInput.COPERNICUS_DEM,
            grid=grid,
            dem_version="synthetic-worlddem30-version",
        )
    }


def _config(**overrides: Any) -> FeatureConfig:
    mapping: dict[str, Any] = {
        "version": "m3-test-config",
        "pipeline_version": "m3-test-pipeline",
        "registry_version": "1.0.0",
        "feature_set_version": "1.0.0",
        "enabled_templates": list(template_ids()),
        "allowed_sources": ["sentinel-1", "sentinel-2", "copernicus-dem"],
        "sentinel1": {
            "polarisation_roles": ["vv", "vh"],
            "backscatter_representation": "decibel",
        },
        "sentinel2": {"spectral_band_roles": ["green", "red", "nir", "swir16"]},
        "terrain": {"elevation_unit": "m", "require_projected_grid": True},
        "band_roles": {
            "sentinel1_before": {"vv": "VV", "vh": "VH"},
            "sentinel1_after": {"vv": "VV", "vh": "VH"},
            "sentinel2_before": dict(S2_ROLE_BINDING),
            "sentinel2_after": dict(S2_ROLE_BINDING),
            "dem": {"elevation": "elevation"},
        },
        "numerics": {"output_nodata": NODATA},
        "grid": {"transform_tolerance": 0.0},
    }
    mapping.update(overrides)
    return FeatureConfig.from_mapping(mapping)


def _full_inputs(tmp_path: Path) -> dict[ArtifactRole, Path]:
    ones = np.full((SIZE, SIZE), 1.0, dtype="float32")
    paths: dict[ArtifactRole, Path] = {}
    paths.update(_s1_pair(tmp_path, before_vv=-8.0 * ones, after_vv=-16.0 * ones))
    paths.update(
        _s2_pair(
            tmp_path,
            before={
                "B03": 0.30 * ones,
                "B04": 0.20 * ones,
                "B08": 0.10 * ones,
                "B11": 0.05 * ones,
            },
            after={
                "B03": 0.32 * ones,
                "B04": 0.18 * ones,
                "B08": 0.04 * ones,
                "B11": 0.02 * ones,
            },
        )
    )
    paths.update(_dem(tmp_path, np.zeros((SIZE, SIZE), dtype="float32")))
    return paths


def _read_feature(artifact, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = list(artifact.registry.names())
    index = names.index(name) + 1
    with rasterio.open(artifact.features_path) as dataset:
        values = dataset.read(index)
    with rasterio.open(artifact.valid_mask_path) as dataset:
        valid = dataset.read(index) > 0
    return values, valid


# =============================================================================
# 1. Feature formulas, verified against hand calculations
# =============================================================================


class TestNumericalFormulas:
    def test_normalized_difference_exact(self):
        first = np.array([[0.3]])
        second = np.array([[0.1]])
        ok = np.array([[True]])
        values, valid = numerics.normalized_difference(first, second, ok, ok)
        # (0.3 - 0.1) / (0.3 + 0.1) = 0.2 / 0.4 = 0.5
        assert valid[0, 0]
        assert values[0, 0] == pytest.approx(0.5, abs=1e-12)

    def test_normalized_difference_is_antisymmetric(self):
        a, b = np.array([[0.4]]), np.array([[0.1]])
        ok = np.array([[True]])
        forward, _ = numerics.normalized_difference(a, b, ok, ok)
        reverse, _ = numerics.normalized_difference(b, a, ok, ok)
        assert forward[0, 0] == pytest.approx(-reverse[0, 0], abs=1e-12)

    def test_difference_exact(self):
        post = np.array([[-16.0]])
        pre = np.array([[-8.0]])
        ok = np.array([[True]])
        values, valid = numerics.difference(post, pre, ok, ok)
        assert valid[0, 0]
        assert values[0, 0] == pytest.approx(-8.0, abs=1e-12)

    def test_log_ratio_db_exact(self):
        post = np.array([[4.0]])
        pre = np.array([[2.0]])
        ok = np.array([[True]])
        values, valid = numerics.log_ratio_db(post, pre, ok, ok)
        # 10 * log10(4/2) = 10 * log10(2) = 3.0102999566398...
        assert valid[0, 0]
        assert values[0, 0] == pytest.approx(3.010299956639812, abs=1e-12)

    def test_log_ratio_db_is_zero_for_no_change(self):
        same = np.array([[0.7]])
        ok = np.array([[True]])
        values, valid = numerics.log_ratio_db(same, same, ok, ok)
        assert valid[0, 0]
        assert values[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_log_ratio_db_is_symmetric_for_reciprocal_change(self):
        """The decibel form must weight a halving and a doubling equally."""
        ok = np.array([[True]])
        doubled, _ = numerics.log_ratio_db(np.array([[2.0]]), np.array([[1.0]]), ok, ok)
        halved, _ = numerics.log_ratio_db(np.array([[1.0]]), np.array([[2.0]]), ok, ok)
        assert doubled[0, 0] == pytest.approx(-halved[0, 0], abs=1e-12)

    def test_db_difference_equals_linear_log_ratio(self):
        """The representation gate is only correct if the two agree numerically.

        A decibel difference and a linear log-ratio are the same physical
        quantity. This pins that equivalence, which is the entire justification
        for selecting the transform by declared representation.
        """
        pre_linear = np.array([[0.05]])
        post_linear = np.array([[0.01]])
        ok = np.array([[True]])
        pre_db, _ = numerics.linear_to_db(pre_linear, ok)
        post_db, _ = numerics.linear_to_db(post_linear, ok)
        from_db, _ = numerics.difference(post_db, pre_db, ok, ok)
        from_linear, _ = numerics.log_ratio_db(post_linear, pre_linear, ok, ok)
        assert from_db[0, 0] == pytest.approx(from_linear[0, 0], abs=1e-9)

    def test_linear_to_db_exact(self):
        ok = np.array([[True]])
        values, valid = numerics.linear_to_db(np.array([[0.1]]), ok)
        assert valid[0, 0]
        assert values[0, 0] == pytest.approx(-10.0, abs=1e-12)


class TestSlopeFormula:
    def test_planar_ramp_gives_exact_expected_slope(self):
        """A 10 m rise per 10 m pixel is a gradient of 1, i.e. exactly 45 degrees."""
        columns = np.arange(SIZE, dtype="float64")
        elevation = np.tile(columns * RESOLUTION_M, (SIZE, 1))
        ok = np.ones_like(elevation, dtype=bool)
        values, valid = numerics.slope_horn_degrees(elevation, ok, resolution_m=RESOLUTION_M)
        assert valid[1, 1] and valid[1, 2]
        assert values[1, 1] == pytest.approx(45.0, abs=1e-9)
        assert values[2, 2] == pytest.approx(45.0, abs=1e-9)

    def test_flat_terrain_gives_zero_slope(self):
        elevation = np.full((SIZE, SIZE), 1234.5)
        ok = np.ones_like(elevation, dtype=bool)
        values, valid = numerics.slope_horn_degrees(elevation, ok, resolution_m=RESOLUTION_M)
        assert valid[1, 1]
        assert values[1, 1] == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_gradient_combines_both_partials(self):
        """A 1:1 rise in both axes gives atan(sqrt(2)), not atan(1)."""
        rows = np.arange(SIZE, dtype="float64")[:, None] * RESOLUTION_M
        columns = np.arange(SIZE, dtype="float64")[None, :] * RESOLUTION_M
        elevation = rows + columns
        ok = np.ones_like(elevation, dtype=bool)
        values, _ = numerics.slope_horn_degrees(elevation, ok, resolution_m=RESOLUTION_M)
        expected = np.degrees(np.arctan(np.sqrt(2.0)))
        assert values[1, 1] == pytest.approx(expected, abs=1e-9)

    def test_border_is_invalid_because_no_window_exists(self):
        elevation = np.zeros((SIZE, SIZE))
        ok = np.ones_like(elevation, dtype=bool)
        _, valid = numerics.slope_horn_degrees(elevation, ok, resolution_m=RESOLUTION_M)
        assert not valid[0, :].any()
        assert not valid[-1, :].any()
        assert not valid[:, 0].any()
        assert not valid[:, -1].any()
        assert valid[1:-1, 1:-1].all()

    def test_one_invalid_neighbour_invalidates_the_centre(self):
        elevation = np.zeros((SIZE, SIZE))
        ok = np.ones_like(elevation, dtype=bool)
        ok[0, 0] = False
        _, valid = numerics.slope_horn_degrees(elevation, ok, resolution_m=RESOLUTION_M)
        assert not valid[1, 1], "a slope computed across a void is not a measurement"
        assert valid[2, 2]

    def test_grid_too_small_for_a_window_fails_explicitly(self):
        with pytest.raises(FeatureConfigurationError, match="3x3"):
            numerics.slope_horn_degrees(
                np.zeros((2, 2)), np.ones((2, 2), dtype=bool), resolution_m=RESOLUTION_M
            )

    def test_non_positive_resolution_fails(self):
        with pytest.raises(FeatureConfigurationError, match="positive grid resolution"):
            numerics.slope_horn_degrees(
                np.zeros((3, 3)), np.ones((3, 3), dtype=bool), resolution_m=0.0
            )


# =============================================================================
# 2. Numerical edge cases: zero, negative, NaN, infinity, nodata
# =============================================================================


class TestRatioEdgeCases:
    @pytest.mark.parametrize("pre", [0.0, -1.0, -0.001])
    def test_log_ratio_rejects_non_positive_denominator(self, pre: float):
        values, valid = numerics.log_ratio_db(
            np.array([[1.0]]), np.array([[pre]]), np.array([[True]]), np.array([[True]])
        )
        assert not valid[0, 0], "no epsilon may rescue a zero or negative denominator"

    @pytest.mark.parametrize("post", [0.0, -1.0])
    def test_log_ratio_rejects_non_positive_numerator(self, post: float):
        _, valid = numerics.log_ratio_db(
            np.array([[post]]), np.array([[1.0]]), np.array([[True]]), np.array([[True]])
        )
        assert not valid[0, 0]

    def test_log_ratio_does_not_fabricate_a_dark_value_for_zero_backscatter(self):
        """Flooring zero at a tiny positive number would manufacture a water-like value."""
        values, valid = numerics.log_ratio_db(
            np.array([[0.0]]), np.array([[0.5]]), np.array([[True]]), np.array([[True]])
        )
        assert not valid[0, 0]
        assert not np.isfinite(values[0, 0]) or values[0, 0] != pytest.approx(0.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_inputs_are_invalid_everywhere(self, bad: float):
        ok = np.array([[True]])
        arrays = (np.array([[bad]]), np.array([[1.0]]))
        for transform in (
            numerics.difference,
            numerics.log_ratio_db,
            numerics.normalized_difference,
        ):
            _, valid = transform(arrays[0], arrays[1], ok, ok)
            assert not valid[0, 0], f"{transform.__name__} accepted {bad}"
            _, valid = transform(arrays[1], arrays[0], ok, ok)
            assert not valid[0, 0], f"{transform.__name__} accepted {bad}"

    def test_normalized_difference_rejects_zero_denominator(self):
        values, valid = numerics.normalized_difference(
            np.array([[0.0]]), np.array([[0.0]]), np.array([[True]]), np.array([[True]])
        )
        assert not valid[0, 0], "0/0 must not become a plausible index value"

    def test_normalized_difference_rejects_cancelling_denominator(self):
        _, valid = numerics.normalized_difference(
            np.array([[0.5]]), np.array([[-0.5]]), np.array([[True]]), np.array([[True]])
        )
        assert not valid[0, 0]

    def test_incoming_invalid_mask_is_respected_even_when_values_are_finite(self):
        """Nodata that happens to be finite must not become a valid observation."""
        values, valid = numerics.difference(
            np.array([[NODATA]]),
            np.array([[1.0]]),
            np.array([[False]]),
            np.array([[True]]),
        )
        assert not valid[0, 0]

    def test_apply_nodata_writes_the_sentinel_into_invalid_pixels(self):
        values = np.array([[1.0, 2.0]])
        valid = np.array([[True, False]])
        result = numerics.apply_nodata(values, valid, NODATA)
        assert result[0, 0] == 1.0
        assert result[0, 1] == NODATA

    def test_mask_shape_mismatch_fails_rather_than_broadcasting(self):
        with pytest.raises(FeatureConfigurationError, match="does not match data shape"):
            numerics.finite_and_valid(np.zeros((2, 2)), np.ones((3, 3), dtype=bool))


# =============================================================================
# 3. Registry completeness and the band-role indirection
# =============================================================================


class TestFeatureRegistry:
    def test_every_enabled_feature_has_complete_metadata(self):
        registry = build_feature_registry(_config())
        assert registry.features
        for feature in registry.features:
            assert feature.name
            assert feature.template_id
            assert isinstance(feature.family, FeatureFamily)
            assert isinstance(feature.source, ProductionInput)
            assert isinstance(feature.transform, TransformKind)
            assert feature.definition and "{role}" not in feature.definition
            assert feature.units
            assert feature.dtype
            assert feature.nodata_policy
            assert len(feature.rationale) > 60, f"{feature.name} has no real rationale"
            assert feature.feature_version
            assert feature.inputs or feature.derived_from, feature.name
            if feature.valid_range is not None:
                low, high = feature.valid_range
                assert low < high
                assert feature.valid_range_condition

    def test_every_input_resolves_to_a_concrete_band_name(self):
        registry = build_feature_registry(_config())
        for feature in registry.features:
            for item in feature.inputs:
                assert item.band_name, f"{feature.name}: unresolved band for {item.band_role}"
                assert "{role}" not in item.band_role

    def test_registry_is_serializable_and_round_trips(self):
        registry = build_feature_registry(_config())
        text = json.dumps(registry.to_dict(), sort_keys=True)
        assert json.loads(text)["features"][0]["name"] == registry.names()[0]

    def test_feature_names_are_unique(self):
        names = build_feature_registry(_config()).names()
        assert len(names) == len(set(names))

    def test_derived_features_follow_their_base_features(self):
        registry = build_feature_registry(_config())
        order = {name: index for index, name in enumerate(registry.names())}
        for feature in registry.features:
            for base in feature.derived_from:
                assert order[base] < order[feature.name], feature.name

    def test_catalogue_covers_every_required_feature_family(self):
        families = {template.family for template in FEATURE_TEMPLATES}
        assert families == set(FeatureFamily)

    def test_unknown_template_is_rejected_not_ignored(self):
        with pytest.raises(FeatureRegistryError, match="unknown feature template"):
            build_feature_registry(_config(enabled_templates=["not_a_feature"]))

    def test_unbound_band_role_fails_by_name_and_never_guesses(self):
        config = _config(
            band_roles={
                "sentinel1_before": {"vv": "VV", "vh": "VH"},
                "sentinel1_after": {"vv": "VV", "vh": "VH"},
                # 'nir' deliberately absent
                "sentinel2_before": {"green": "B03", "red": "B04", "swir16": "B11"},
                "sentinel2_after": dict(S2_ROLE_BINDING),
                "dem": {"elevation": "elevation"},
            },
            enabled_templates=["s2_ndwi_pre"],
            sentinel2={"spectral_band_roles": ["green", "nir"]},
        )
        with pytest.raises(FeatureConfigurationError, match="band role 'nir' is not bound"):
            build_feature_registry(config)

    def test_unresolved_polarisation_roles_are_not_defaulted(self):
        config = _config(
            enabled_templates=["s1_change_db"],
            sentinel1={"backscatter_representation": "decibel"},
        )
        with pytest.raises(FeatureConfigurationError, match="polarisation_roles is unresolved"):
            build_feature_registry(config)

    def test_unresolved_spectral_band_roles_are_not_defaulted(self):
        config = _config(enabled_templates=["s2_reflectance_pre"], sentinel2={})
        with pytest.raises(FeatureConfigurationError, match="spectral_band_roles is unresolved"):
            build_feature_registry(config)

    def test_derived_feature_without_its_base_features_is_rejected(self):
        with pytest.raises(FeatureConfigurationError, match="derived from"):
            build_feature_registry(_config(enabled_templates=["s2_ndwi_change"]))

    def test_single_polarisation_product_does_not_get_a_cross_pol_feature(self):
        """Dual polarisation must never be assumed to exist."""
        config = _config(
            enabled_templates=["s1_backscatter_pre", "s1_co_cross_ratio_db_pre"],
            sentinel1={
                "polarisation_roles": ["vv"],
                "backscatter_representation": "decibel",
            },
            band_roles={
                "sentinel1_before": {"vv": "VV"},
                "sentinel1_after": {"vv": "VV"},
            },
        )
        with pytest.raises(FeatureConfigurationError, match="band role 'vh' is not bound"):
            build_feature_registry(config)


class TestBackscatterRepresentationGate:
    def test_representation_must_be_declared(self):
        config = _config(
            enabled_templates=["s1_change_db"],
            sentinel1={"polarisation_roles": ["vv"]},
        )
        with pytest.raises(FeatureConfigurationError, match="backscatter_representation"):
            build_feature_registry(config)

    def test_decibel_input_uses_a_plain_difference(self):
        registry = build_feature_registry(
            _config(
                enabled_templates=["s1_change_db"],
                sentinel1={
                    "polarisation_roles": ["vv"],
                    "backscatter_representation": "decibel",
                },
            )
        )
        feature = registry.get("s1_vv_change_db")
        assert feature.transform is TransformKind.DIFFERENCE
        assert feature.units == "dB"

    def test_linear_input_uses_the_log_ratio(self):
        registry = build_feature_registry(
            _config(
                enabled_templates=["s1_change_db", "s1_backscatter_pre"],
                sentinel1={
                    "polarisation_roles": ["vv"],
                    "backscatter_representation": "linear",
                },
            )
        )
        assert registry.get("s1_vv_change_db").transform is TransformKind.LOG_RATIO_DB
        # The level feature changes units with the representation; the change
        # feature stays in dB because the log-ratio produces decibels.
        assert registry.get("s1_vv_pre").units == "linear power (sigma-nought)"
        assert registry.get("s1_vv_change_db").units == "dB"

    def test_representation_is_recorded_on_the_registry(self):
        registry = build_feature_registry(_config())
        assert registry.backscatter_representation is BackscatterRepresentation.DECIBEL


# =============================================================================
# 4. Change features preserve pre, post and change
# =============================================================================


class TestChangeFeatures:
    def test_pre_post_and_change_are_all_retained(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="change-set"
        )
        names = set(artifact.registry.names())
        for expected in (
            "s1_vv_pre",
            "s1_vv_post",
            "s1_vv_change_db",
            "s2_nir_pre",
            "s2_nir_post",
            "s2_nir_change",
            "s2_ndwi_pre",
            "s2_ndwi_post",
            "s2_ndwi_change",
        ):
            assert expected in names, f"{expected} was discarded"

    def test_sentinel1_change_matches_the_hand_calculation(self, tmp_path: Path):
        ones = np.ones((SIZE, SIZE), dtype="float32")
        paths = _full_inputs(tmp_path)
        paths.update(_s1_pair(tmp_path, before_vv=-8.0 * ones, after_vv=-16.0 * ones))
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="s1-change"
        )
        values, valid = _read_feature(artifact, "s1_vv_change_db")
        assert valid.all()
        # -16 dB - (-8 dB) = -8 dB
        assert values == pytest.approx(np.full((SIZE, SIZE), -8.0), abs=1e-5)

    def test_spectral_indices_match_hand_calculations(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="indices",
        )
        # pre: green=0.30, red=0.20, nir=0.10, swir16=0.05
        ndwi_pre, _ = _read_feature(artifact, "s2_ndwi_pre")
        assert ndwi_pre[0, 0] == pytest.approx((0.30 - 0.10) / (0.30 + 0.10), abs=1e-6)
        mndwi_pre, _ = _read_feature(artifact, "s2_mndwi_pre")
        assert mndwi_pre[0, 0] == pytest.approx((0.30 - 0.05) / (0.30 + 0.05), abs=1e-6)
        ndvi_pre, _ = _read_feature(artifact, "s2_ndvi_pre")
        assert ndvi_pre[0, 0] == pytest.approx((0.10 - 0.20) / (0.10 + 0.20), abs=1e-6)
        # post: green=0.32, nir=0.04
        ndwi_post, _ = _read_feature(artifact, "s2_ndwi_post")
        expected_post = (0.32 - 0.04) / (0.32 + 0.04)
        assert ndwi_post[0, 0] == pytest.approx(expected_post, abs=1e-6)
        change, _ = _read_feature(artifact, "s2_ndwi_change")
        expected_pre = (0.30 - 0.10) / (0.30 + 0.10)
        assert change[0, 0] == pytest.approx(expected_post - expected_pre, abs=1e-6)

    def test_cross_polarisation_ratio_matches_the_hand_calculation(self, tmp_path: Path):
        ones = np.ones((SIZE, SIZE), dtype="float32")
        paths = _full_inputs(tmp_path)
        paths.update(
            _s1_pair(
                tmp_path,
                before_vv=-8.0 * ones,
                before_vh=-14.0 * ones,
                after_vv=-16.0 * ones,
                after_vh=-20.0 * ones,
            )
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="crosspol"
        )
        pre, _ = _read_feature(artifact, "s1_vv_vh_ratio_db_pre")
        post, _ = _read_feature(artifact, "s1_vv_vh_ratio_db_post")
        assert pre[0, 0] == pytest.approx(-8.0 - (-14.0), abs=1e-5)
        assert post[0, 0] == pytest.approx(-16.0 - (-20.0), abs=1e-5)

    def test_reflectance_change_matches_the_hand_calculation(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="reflectance-change",
        )
        values, _ = _read_feature(artifact, "s2_nir_change")
        assert values[0, 0] == pytest.approx(0.04 - 0.10, abs=1e-6)


# =============================================================================
# 5. Mask propagation
# =============================================================================


class TestMaskPropagation:
    def test_one_invalid_source_pixel_invalidates_the_derived_feature(self, tmp_path: Path):
        ones = np.ones((SIZE, SIZE), dtype="float32")
        before_valid = np.ones((SIZE, SIZE), dtype=bool)
        before_valid[0, 0] = False
        paths = _full_inputs(tmp_path)
        paths.update(
            _s1_pair(
                tmp_path,
                before_vv=-8.0 * ones,
                after_vv=-16.0 * ones,
                before_valid=before_valid,
            )
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="mask-propagation"
        )

        change_values, change_valid = _read_feature(artifact, "s1_vv_change_db")
        assert not change_valid[0, 0], "an invalid pre-event pixel must invalidate the change"
        assert change_values[0, 0] == pytest.approx(NODATA)
        assert change_valid[1:, 1:].all()

        _, pre_valid = _read_feature(artifact, "s1_vv_pre")
        assert not pre_valid[0, 0]

        # The post-event artifact was fully valid, so its own feature stays valid
        # there. Validity is per feature, never collapsed across the stack.
        _, post_valid = _read_feature(artifact, "s1_vv_post")
        assert post_valid[0, 0]

    def test_a_feature_never_becomes_valid_through_substitution(self, tmp_path: Path):
        """The invalid pixel must not acquire a plausible value from anywhere."""
        ones = np.ones((SIZE, SIZE), dtype="float32")
        valid = np.ones((SIZE, SIZE), dtype=bool)
        valid[2, 2] = False
        paths = _full_inputs(tmp_path)
        paths.update(
            _s2_pair(
                tmp_path,
                before={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": 0.1 * ones,
                    "B11": 0.05 * ones,
                },
                after={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": 0.1 * ones,
                    "B11": 0.05 * ones,
                },
                after_valid=valid,
            )
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="no-substitution"
        )
        for name in ("s2_ndwi_post", "s2_ndwi_change", "s2_nir_change", "s2_mndwi_post"):
            values, feature_valid = _read_feature(artifact, name)
            assert not feature_valid[2, 2], name
            assert values[2, 2] == pytest.approx(NODATA), name

    def test_features_sharing_validity_conditions_get_identical_masks(self, tmp_path: Path):
        """Inconsistent masks for identical conditions would be a silent defect."""
        ones = np.ones((SIZE, SIZE), dtype="float32")
        valid = np.ones((SIZE, SIZE), dtype=bool)
        valid[1, 2] = False
        paths = _full_inputs(tmp_path)
        paths.update(
            _s2_pair(
                tmp_path,
                before={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": 0.1 * ones,
                    "B11": 0.05 * ones,
                },
                after={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": 0.1 * ones,
                    "B11": 0.05 * ones,
                },
                before_valid=valid,
            )
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="consistent-masks"
        )
        # NDWI-pre and MNDWI-pre both depend only on the pre-event artifact's
        # validity here, so their masks must agree exactly.
        _, ndwi = _read_feature(artifact, "s2_ndwi_pre")
        _, mndwi = _read_feature(artifact, "s2_mndwi_pre")
        assert np.array_equal(ndwi, mndwi)

    def test_per_feature_masks_are_written_not_a_single_plane(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="mask-bands",
        )
        with rasterio.open(artifact.valid_mask_path) as dataset:
            assert dataset.count == len(artifact.registry.names())
            assert list(dataset.descriptions) == [
                f"{name}_valid" for name in artifact.registry.names()
            ]

    def test_mask_semantics_are_recorded_and_not_evidence_of_absence(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="mask-semantics",
        )
        semantics = artifact.quality["mask_semantics"]
        assert "1=valid" in semantics["per_feature"]
        assert "not evidence of absence" in semantics["note"]

    def test_invalid_counts_are_reported_per_feature(self, tmp_path: Path):
        ones = np.ones((SIZE, SIZE), dtype="float32")
        valid = np.ones((SIZE, SIZE), dtype=bool)
        valid[0, 0] = False
        paths = _full_inputs(tmp_path)
        paths.update(
            _s1_pair(tmp_path, before_vv=-8.0 * ones, after_vv=-16.0 * ones, before_valid=valid)
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="counts"
        )
        report = artifact.quality["features"]["s1_vv_change_db"]
        assert report["invalid_pixel_count"] == 1
        assert report["valid_pixel_count"] == SIZE * SIZE - 1


# =============================================================================
# 6. Grid enforcement — M3 never resamples
# =============================================================================


class TestGridEnforcement:
    def test_incompatible_grids_fail_explicitly(self, tmp_path: Path):
        ones = np.ones((SIZE, SIZE), dtype="float32")
        paths = _full_inputs(tmp_path)
        shifted = AnalysisGrid(
            crs=GRID_CRS,
            transform=from_origin(400000.0, 3100000.0, RESOLUTION_M, RESOLUTION_M),
            width=SIZE,
            height=SIZE,
            resolution_m=RESOLUTION_M,
        )
        paths.update(_s1_pair(tmp_path, before_vv=-8.0 * ones, after_vv=-16.0 * ones, grid=shifted))
        with pytest.raises(FeatureGridError, match="does not share the analysis grid"):
            generate_feature_set(
                paths, tmp_path / "out", config=_config(), feature_set_id="misaligned"
            )

    def test_mismatched_crs_fails(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        other = AnalysisGrid(
            crs="EPSG:32644",
            transform=TRANSFORM,
            width=SIZE,
            height=SIZE,
            resolution_m=RESOLUTION_M,
        )
        paths.update(_dem(tmp_path, np.zeros((SIZE, SIZE), dtype="float32"), grid=other))
        with pytest.raises(FeatureGridError):
            generate_feature_set(
                paths, tmp_path / "out", config=_config(), feature_set_id="crs-mismatch"
            )

    def test_mismatched_resolution_fails(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        coarse = AnalysisGrid(
            crs=GRID_CRS,
            transform=from_origin(300000.0, 3100000.0, 20.0, 20.0),
            width=SIZE,
            height=SIZE,
            resolution_m=20.0,
        )
        paths.update(_dem(tmp_path, np.zeros((SIZE, SIZE), dtype="float32"), grid=coarse))
        with pytest.raises(FeatureGridError):
            generate_feature_set(
                paths, tmp_path / "out", config=_config(), feature_set_id="res-mismatch"
            )

    def test_resampling_cannot_be_enabled(self):
        with pytest.raises(FeatureConfigurationError, match="allow_resampling must be false"):
            _config(grid={"allow_resampling": True})

    def test_alignment_report_is_recorded_in_quality(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="alignment",
        )
        alignment = artifact.quality["alignment"]
        assert alignment["reference_grid"]["crs"] == GRID_CRS
        for role, report in alignment["per_artifact"].items():
            assert report["aligned"] is True, role


class TestTerrainGridRequirements:
    def test_geographic_grid_is_rejected_for_terrain_derivatives(self, tmp_path: Path):
        geographic = AnalysisGrid(
            crs="EPSG:4326",
            transform=from_origin(85.0, 27.1, 0.001, 0.001),
            width=SIZE,
            height=SIZE,
            resolution_m=0.001,
        )
        ones = np.ones((SIZE, SIZE), dtype="float32")
        paths: dict[ArtifactRole, Path] = {}
        paths.update(
            _s1_pair(tmp_path, before_vv=-8.0 * ones, after_vv=-16.0 * ones, grid=geographic)
        )
        paths.update(_dem(tmp_path, np.zeros((SIZE, SIZE), dtype="float32"), grid=geographic))
        config = _config(
            enabled_templates=["s1_backscatter_pre", "dem_slope_degrees"],
            sentinel1={
                "polarisation_roles": ["vv"],
                "backscatter_representation": "decibel",
            },
        )
        with pytest.raises(FeatureConfigurationError, match="projected analysis grid"):
            generate_feature_set(
                paths, tmp_path / "out", config=config, feature_set_id="geographic"
            )

    def test_unresolved_elevation_unit_blocks_terrain_features(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        config = _config(enabled_templates=["dem_elevation", "dem_slope_degrees"], terrain={})
        with pytest.raises(FeatureConfigurationError, match="elevation_unit is unresolved"):
            generate_feature_set(paths, tmp_path / "out", config=config, feature_set_id="unit")

    def test_non_metre_elevation_unit_is_rejected(self):
        with pytest.raises(FeatureConfigurationError, match="elevation_unit"):
            _config(terrain={"elevation_unit": "ft"})


# =============================================================================
# 7. Input contract — M2 artifacts only
# =============================================================================


class TestInputContract:
    def test_missing_sibling_files_are_rejected(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        provenance = paths[ArtifactRole.DEM].with_suffix("").with_name("elevation_provenance.yaml")
        provenance.unlink()
        with pytest.raises(FeatureInputError, match="provenance record"):
            load_preprocessed_input(
                ArtifactRole.DEM,
                paths[ArtifactRole.DEM],
                allowed_sources=[ProductionInput.COPERNICUS_DEM],
            )

    def test_non_preprocessed_artifact_is_rejected(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        path = paths[ArtifactRole.DEM]
        provenance_path = path.parent / "elevation_provenance.yaml"
        record = yaml.safe_load(provenance_path.read_text())
        record["artifact_type"] = ArtifactType.ACQUISITION_MANIFEST.value
        provenance_path.write_text(yaml.safe_dump(record))
        with pytest.raises(FeatureInputError, match="artifact_type"):
            load_preprocessed_input(
                ArtifactRole.DEM, path, allowed_sources=[ProductionInput.COPERNICUS_DEM]
            )

    def test_artifact_without_manifest_link_is_rejected(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        path = paths[ArtifactRole.DEM]
        provenance_path = path.parent / "elevation_provenance.yaml"
        record = yaml.safe_load(provenance_path.read_text())
        record["acquisition_manifest_id"] = None
        provenance_path.write_text(yaml.safe_dump(record))
        with pytest.raises(FeatureInputError, match="acquisition_manifest_id"):
            load_preprocessed_input(
                ArtifactRole.DEM, path, allowed_sources=[ProductionInput.COPERNICUS_DEM]
            )

    def test_missing_required_artifact_is_named_in_the_error(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        del paths[ArtifactRole.DEM]
        with pytest.raises(FeatureInputError, match="dem"):
            generate_feature_set(
                paths, tmp_path / "out", config=_config(), feature_set_id="missing-dem"
            )

    def test_bands_are_addressed_by_name_not_position(self, tmp_path: Path):
        """A reordered artifact must not silently feed the wrong channel."""
        ones = np.ones((SIZE, SIZE), dtype="float32")
        manifest = _manifest(SensorKind.SENTINEL2)
        # Same values, band names in reversed order.
        forward = _write_m2_artifact(
            tmp_path / "forward",
            "before_s2-before",
            bands=[0.3 * ones, 0.2 * ones, 0.1 * ones, 0.05 * ones],
            band_names=S2_BANDS,
            manifest=manifest,
            sensor=SensorKind.SENTINEL2,
        )
        reversed_names = tuple(reversed(S2_BANDS))
        reverse = _write_m2_artifact(
            tmp_path / "reverse",
            "before_s2-before",
            bands=[0.05 * ones, 0.1 * ones, 0.2 * ones, 0.3 * ones],
            band_names=reversed_names,
            manifest=manifest,
            sensor=SensorKind.SENTINEL2,
        )
        allowed = [ProductionInput.SENTINEL2]
        first = load_preprocessed_input(
            ArtifactRole.SENTINEL2_BEFORE, forward, allowed_sources=allowed
        )
        second = load_preprocessed_input(
            ArtifactRole.SENTINEL2_BEFORE, reverse, allowed_sources=allowed
        )
        # Positions differ, names agree, so name-addressed reads must agree.
        assert first.band_index("B08") != second.band_index("B08")
        assert first.read_band("B08")[0][0, 0] == pytest.approx(second.read_band("B08")[0][0, 0])

    def test_unknown_band_name_fails_with_the_available_names(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        loaded = load_preprocessed_input(
            ArtifactRole.DEM,
            paths[ArtifactRole.DEM],
            allowed_sources=[ProductionInput.COPERNICUS_DEM],
        )
        with pytest.raises(FeatureInputError, match="available band descriptions"):
            loaded.read_band("not_a_band")

    def test_m2_preserves_band_names_into_the_artifact(self, tmp_path: Path):
        """The M2/M3 boundary must not lose band identity."""
        paths = _full_inputs(tmp_path)
        loaded = load_preprocessed_input(
            ArtifactRole.SENTINEL1_BEFORE,
            paths[ArtifactRole.SENTINEL1_BEFORE],
            allowed_sources=[ProductionInput.SENTINEL1],
        )
        assert loaded.band_names == S1_BANDS

    def test_empty_input_set_is_rejected(self):
        with pytest.raises(FeatureInputError, match="no M2 artifacts"):
            FeatureInputs.load({}, allowed_sources=[ProductionInput.SENTINEL1])


# =============================================================================
# 8. Provenance chain
# =============================================================================


class TestProvenanceChain:
    def test_feature_artifact_traces_to_manifest_and_source_scenes(self, tmp_path: Path):
        manifest = _manifest(SensorKind.SENTINEL1)
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="provenance",
        )
        provenance = artifact.provenance
        assert provenance.artifact_type is ArtifactType.FEATURE_STACK

        # feature artifact -> M2 artifact -> acquisition manifest
        assert manifest.artifact_id in artifact.quality["acquisition_manifest_ids"]

        # -> original source scenes
        assert provenance.sentinel1 is not None
        assert provenance.sentinel1.before is not None
        assert provenance.sentinel1.before.scene_id == "s1-before"
        assert provenance.sentinel1.after.scene_id == "s1-after"
        assert provenance.sentinel1.same_relative_orbit is True
        assert provenance.sentinel2 is not None
        assert provenance.sentinel2.before.scene_id == "s2-before"

        # -> source product IDs and DEM version
        assert "product-s1-before" in provenance.source_product_ids
        assert "product-s2-after" in provenance.source_product_ids
        assert provenance.dem_version == "synthetic-worlddem30-version"

    def test_provenance_is_written_and_reparses(self, tmp_path: Path):
        from floodmap.utils.provenance import ArtifactProvenance

        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="reparse",
        )
        reloaded = ArtifactProvenance.from_yaml(
            artifact.provenance_path.read_text(encoding="utf-8")
        )
        assert reloaded.artifact_type is ArtifactType.FEATURE_STACK
        assert reloaded.preprocessing_version == "m3-test-pipeline"
        assert reloaded.config_version == "m3-test-config"

    def test_required_attributions_are_attached(self, tmp_path: Path):
        from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="attribution",
        )
        assert artifact.provenance.attribution[:3] == list(REQUIRED_ATTRIBUTIONS)

    def test_production_inputs_are_recorded_and_permitted(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="sources",
        )
        recorded = set(artifact.provenance.production_inputs)
        assert recorded == {
            ProductionInput.SENTINEL1,
            ProductionInput.SENTINEL2,
            ProductionInput.COPERNICUS_DEM,
        }

    def test_limitations_state_that_this_is_not_a_prediction(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="limitations",
        )
        text = " ".join(artifact.provenance.limitations).lower()
        assert "not a flood/debris prediction" in text
        assert "evidence" in text


# =============================================================================
# 9. Production / validation boundary — no label leakage
# =============================================================================


class TestLeakageBoundary:
    @pytest.mark.parametrize("source", [item.value for item in ValidationOnlySource])
    def test_validation_only_source_cannot_be_configured(self, source: str):
        with pytest.raises(FeatureBoundaryError, match="validation-only source"):
            _config(allowed_sources=[source])

    def test_unrecognised_source_is_rejected(self):
        with pytest.raises(FeatureBoundaryError, match="not a permitted production input"):
            _config(allowed_sources=["some-other-damage-map"])

    def test_artifact_declaring_a_disallowed_source_is_rejected(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        with pytest.raises(FeatureBoundaryError, match="allowed_sources"):
            FeatureInputs.load(
                {ArtifactRole.DEM: paths[ArtifactRole.DEM]},
                allowed_sources=[ProductionInput.SENTINEL1],
            )

    def test_provenance_naming_a_validation_only_source_cannot_be_loaded(self, tmp_path: Path):
        """A hand-edited artifact must not be able to smuggle a forbidden source in."""
        paths = _full_inputs(tmp_path)
        path = paths[ArtifactRole.DEM]
        provenance_path = path.parent / "elevation_provenance.yaml"
        record = yaml.safe_load(provenance_path.read_text())
        record["production_inputs"] = [ValidationOnlySource.EMSR927.value]
        provenance_path.write_text(yaml.safe_dump(record))
        with pytest.raises((FeatureBoundaryError, FeatureInputError)):
            load_preprocessed_input(
                ArtifactRole.DEM, path, allowed_sources=[ProductionInput.COPERNICUS_DEM]
            )

    def test_shipped_config_allows_only_permitted_production_inputs(self):
        permitted = {item.value for item in ProductionInput}
        forbidden = {item.value for item in ValidationOnlySource}
        declared = set(load_config("features")["allowed_sources"])
        assert declared <= permitted
        assert not declared & forbidden

    def test_every_catalogue_feature_comes_from_a_permitted_source(self):
        forbidden = {item.value for item in ValidationOnlySource}
        for template in FEATURE_TEMPLATES:
            assert isinstance(template.source, ProductionInput)
            assert template.source.value not in forbidden

    def test_feature_modules_are_covered_by_the_boundary_scan(self):
        """The AGENTS.md §3 textual guard must actually reach the new package."""
        from tests.test_data_boundary import EXEMPT_PATHS, _production_python_files

        scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _production_python_files()}
        feature_modules = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "src" / "floodmap" / "features").rglob("*.py")
        }
        assert feature_modules, "no feature modules found"
        assert feature_modules <= scanned
        assert not any(item.startswith("src/floodmap/features") for item in EXEMPT_PATHS)


class TestNormalisationIsLeakageSafe:
    def test_normalisation_is_disabled_by_default(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="no-normalisation",
        )
        record = artifact.quality["normalisation"]
        assert record["applied"] is False
        assert record["fitted_in_m3"] is False

    def test_shipped_config_does_not_normalise_and_declares_training_only(self):
        normalisation = load_config("features")["normalisation"]
        assert normalisation["method"] is None
        assert normalisation["statistics_source"] == "training_split_only"

    def test_statistics_source_outside_training_is_rejected(self, tmp_path: Path):
        statistics = tmp_path / "stats.json"
        statistics.write_text(json.dumps({"s1_vv_pre": {"mean": 0.0, "std": 1.0}}))
        config = _config(
            normalisation={
                "method": "zscore",
                "statistics_source": "unseen_himalaya_test",
                "statistics_path": str(statistics),
                "version": "1.0.0",
            }
        )
        with pytest.raises(FeatureBoundaryError, match="training split only"):
            config.normalisation.require_statistics_path()

    def test_enabled_normalisation_requires_an_explicit_statistics_file(self):
        config = _config(normalisation={"method": "zscore", "version": "1.0.0"})
        with pytest.raises(FeatureConfigurationError, match="statistics_path is unresolved"):
            config.normalisation.require_statistics_path()

    def test_enabled_normalisation_requires_a_version(self, tmp_path: Path):
        statistics = tmp_path / "stats.json"
        statistics.write_text("{}")
        config = _config(normalisation={"method": "zscore", "statistics_path": str(statistics)})
        with pytest.raises(FeatureConfigurationError, match="version is unresolved"):
            config.normalisation.require_statistics_path()

    def test_applied_normalisation_uses_supplied_parameters_and_records_them(self, tmp_path: Path):
        config_base = _config()
        registry = build_feature_registry(config_base)
        statistics = {name: {"mean": 2.0, "std": 4.0} for name in registry.names()}
        statistics_path = tmp_path / "stats.json"
        statistics_path.write_text(json.dumps(statistics))
        config = _config(
            normalisation={
                "method": "zscore",
                "statistics_source": "training_split_only",
                "statistics_path": str(statistics_path),
                "version": "stats-1.0.0",
            }
        )
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=config,
            feature_set_id="normalised",
        )
        record = artifact.quality["normalisation"]
        assert record["applied"] is True
        assert record["fitted_in_m3"] is False
        assert record["version"] == "stats-1.0.0"
        assert record["parameters"]["s1_vv_pre"] == {
            "method": "zscore",
            "mean": 2.0,
            "std": 4.0,
        }
        values, _ = _read_feature(artifact, "s1_vv_pre")
        # (-8 - 2) / 4 = -2.5
        assert values[0, 0] == pytest.approx(-2.5, abs=1e-5)

    def test_partial_statistics_are_rejected(self, tmp_path: Path):
        statistics_path = tmp_path / "stats.json"
        statistics_path.write_text(json.dumps({"s1_vv_pre": {"mean": 0.0, "std": 1.0}}))
        config = _config(
            normalisation={
                "method": "zscore",
                "statistics_path": str(statistics_path),
                "version": "stats-1.0.0",
            }
        )
        with pytest.raises(FeatureConfigurationError, match="no entry for"):
            generate_feature_set(
                _full_inputs(tmp_path),
                tmp_path / "out",
                config=config,
                feature_set_id="partial-stats",
            )


# =============================================================================
# 10. M3 does not classify, and does not implement M4
# =============================================================================


class TestNoClassification:
    def test_quality_record_states_no_classification_was_performed(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="no-classification",
        )
        assert artifact.quality["classification_performed"] is False

    def test_no_feature_is_a_mask_or_a_class_label(self):
        registry = build_feature_registry(_config())
        for name in registry.names():
            lowered = name.lower()
            assert "mask" not in lowered, name
            assert "class" not in lowered, name
            assert not lowered.startswith("flood"), name
            assert not lowered.startswith("is_"), name

    def test_no_feature_claims_to_detect_flooding(self):
        registry = build_feature_registry(_config())
        for feature in registry.features:
            text = f"{feature.definition} {feature.rationale}".lower()
            for overclaim in ("detects flood", "detects flooding", "identifies flood"):
                assert overclaim not in text, f"{feature.name} overclaims: {overclaim}"

    def test_configuration_defines_no_decision_threshold(self):
        text = json.dumps(load_config("features")).lower()
        for forbidden in ("threshold", "probability", "num_classes", "class_names"):
            assert forbidden not in text, f"configs/features.yaml must not define {forbidden}"

    def test_no_clipping_is_applied_by_default(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="no-clipping",
        )
        assert artifact.quality["clipping_applied"] is False
        for report in artifact.quality["features"].values():
            if report["declared_range_report"] is not None:
                assert report["declared_range_report"]["clipped"] is False

    def test_clipping_requires_a_written_justification(self):
        config = _config(numerics={"output_nodata": NODATA, "clip_to_valid_range": True})
        with pytest.raises(FeatureConfigurationError, match="without a justification"):
            config.numerics.require_clip_policy()

    def test_out_of_range_values_are_reported_not_corrected(self, tmp_path: Path):
        """A negative reflectance is information, not something to silently fix."""
        ones = np.ones((SIZE, SIZE), dtype="float32")
        negative = -0.02 * ones
        paths = _full_inputs(tmp_path)
        paths.update(
            _s2_pair(
                tmp_path,
                before={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": negative,
                    "B11": 0.05 * ones,
                },
                after={
                    "B03": 0.3 * ones,
                    "B04": 0.2 * ones,
                    "B08": 0.1 * ones,
                    "B11": 0.05 * ones,
                },
            )
        )
        artifact = generate_feature_set(
            paths, tmp_path / "out", config=_config(), feature_set_id="out-of-range"
        )
        report = artifact.quality["features"]["s2_nir_pre"]["declared_range_report"]
        assert report["valid_pixels_outside_range"] == SIZE * SIZE
        assert report["clipped"] is False
        values, valid = _read_feature(artifact, "s2_nir_pre")
        assert valid.all()
        assert values[0, 0] == pytest.approx(-0.02, abs=1e-6)

    def test_no_terrain_routing_or_network_feature_exists(self):
        """Flood routing, tracing and connectivity belong to later milestones."""
        for template in FEATURE_TEMPLATES:
            text = f"{template.template_id} {template.definition}".lower()
            for forbidden in (
                "flow_direction",
                "flow_accumulation",
                "watershed",
                "routing",
                "downstream",
                "connectivity",
                "road",
            ):
                assert forbidden not in text, f"{template.template_id} leaks M5+ scope"


# =============================================================================
# 11. Artifact format
# =============================================================================


class TestArtifactFormat:
    def test_expected_files_are_written(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="layout",
        )
        base = tmp_path / "out" / "features" / "layout"
        for expected in (
            "features.tif",
            "features_valid_mask.tif",
            "feature_registry.json",
            "features_quality.json",
            "features_provenance.yaml",
        ):
            assert (base / expected).is_file(), expected

    def test_band_descriptions_are_the_feature_names_in_registry_order(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="band-names",
        )
        with rasterio.open(artifact.features_path) as dataset:
            assert list(dataset.descriptions) == list(artifact.registry.names())
            assert dataset.count == len(artifact.registry.names())

    def test_analysis_grid_is_preserved_exactly(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="grid-preserved",
        )
        with rasterio.open(artifact.features_path) as dataset:
            assert dataset.crs.to_string() == GRID_CRS
            assert dataset.transform == TRANSFORM
            assert (dataset.width, dataset.height) == (SIZE, SIZE)
            assert dataset.nodata == NODATA

    def test_registry_on_disk_describes_every_written_band(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="registry-on-disk",
        )
        written = json.loads(artifact.registry_path.read_text(encoding="utf-8"))
        with rasterio.open(artifact.features_path) as dataset:
            bands = list(dataset.descriptions)
        assert [feature["name"] for feature in written["features"]] == bands
        assert written["feature_set_version"] == "1.0.0"
        assert written["config_version"] == "m3-test-config"

    def test_versions_are_recorded_on_the_raster_tags(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="tags",
        )
        with rasterio.open(artifact.features_path) as dataset:
            tags = dataset.tags()
        assert tags["feature_set_id"] == "tags"
        assert tags["feature_pipeline_version"] == "m3-test-pipeline"
        assert tags["feature_set_version"] == "1.0.0"
        assert "s1_vv_pre" in tags["feature_names"]

    def test_quality_record_carries_the_source_artifact_ids(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="source-ids",
        )
        assert "product-s1-before" in artifact.quality["source_artifact_ids"]
        assert artifact.quality["feature_count"] == len(artifact.registry.names())


# =============================================================================
# 12. Determinism
# =============================================================================


class TestDeterminism:
    def test_identical_inputs_and_config_give_identical_features(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        first = generate_feature_set(
            paths, tmp_path / "run-a", config=_config(), feature_set_id="determinism"
        )
        second = generate_feature_set(
            paths, tmp_path / "run-b", config=_config(), feature_set_id="determinism"
        )
        with rasterio.open(first.features_path) as a, rasterio.open(second.features_path) as b:
            assert np.array_equal(a.read(), b.read())
            assert list(a.descriptions) == list(b.descriptions)
        with rasterio.open(first.valid_mask_path) as a, rasterio.open(second.valid_mask_path) as b:
            assert np.array_equal(a.read(), b.read())

    def test_registry_and_quality_metadata_are_byte_identical(self, tmp_path: Path):
        paths = _full_inputs(tmp_path)
        first = generate_feature_set(
            paths, tmp_path / "run-a", config=_config(), feature_set_id="determinism"
        )
        second = generate_feature_set(
            paths, tmp_path / "run-b", config=_config(), feature_set_id="determinism"
        )
        assert first.registry_path.read_text() == second.registry_path.read_text()

        def _normalise(path: Path) -> dict[str, Any]:
            record = json.loads(path.read_text(encoding="utf-8"))
            # Input paths differ only by the per-run output directory.
            record.pop("inputs", None)
            return record

        assert _normalise(first.quality_path) == _normalise(second.quality_path)

    def test_feature_order_is_independent_of_configuration_order(self):
        forward = build_feature_registry(_config(enabled_templates=list(template_ids())))
        shuffled = build_feature_registry(
            _config(enabled_templates=list(reversed(list(template_ids()))))
        )
        assert forward.names() == shuffled.names()

    def test_no_timestamp_appears_in_the_registry_or_quality_record(self, tmp_path: Path):
        artifact = generate_feature_set(
            _full_inputs(tmp_path),
            tmp_path / "out",
            config=_config(),
            feature_set_id="no-timestamp",
        )
        for path in (artifact.registry_path, artifact.quality_path):
            text = path.read_text(encoding="utf-8").lower()
            assert "generated_at" not in text, path.name


# =============================================================================
# 13. Configuration contract
# =============================================================================


class TestShippedConfiguration:
    def test_config_parses_and_declares_a_version(self):
        config = load_config("features")
        assert config["version"]
        assert config["feature_set_version"]

    def test_config_sections_exist(self):
        config = load_config("features")
        for key in (
            "allowed_sources",
            "enabled_templates",
            "sentinel1",
            "sentinel2",
            "terrain",
            "band_roles",
            "numerics",
            "grid",
            "normalisation",
        ):
            assert key in config, f"configs/features.yaml must define {key}"

    def test_every_enabled_template_exists_in_the_catalogue(self):
        enabled = load_config("features")["enabled_templates"]
        assert enabled
        unknown = [item for item in enabled if item not in template_ids()]
        assert not unknown, f"configs/features.yaml enables unknown templates: {unknown}"

    def test_unverified_bindings_remain_unresolved_rather_than_guessed(self):
        """These cannot be known before a product is inspected; a default would be invented."""
        config = load_config("features")
        assert config["sentinel1"]["polarisation_roles"] is None
        assert config["sentinel1"]["backscatter_representation"] is None
        assert config["sentinel2"]["spectral_band_roles"] is None
        assert config["terrain"]["elevation_unit"] is None
        assert config["numerics"]["output_nodata"] is None
        for artifact_role, binding in config["band_roles"].items():
            assert binding is None, f"band_roles.{artifact_role} was guessed"

    def test_resampling_and_clipping_are_disabled_in_the_shipped_config(self):
        config = load_config("features")
        assert config["grid"]["allow_resampling"] is False
        assert config["numerics"]["clip_to_valid_range"] is False
        assert config["terrain"]["require_projected_grid"] is True

    def test_shipped_config_cannot_run_a_production_feature_set(self):
        """An unresolved production run must fail, not silently use defaults."""
        config = FeatureConfig.from_mapping(load_config("features"))
        with pytest.raises(FeatureConfigurationError):
            build_feature_registry(config)

    def test_feature_set_id_is_required(self, tmp_path: Path):
        with pytest.raises(FeatureConfigurationError, match="feature_set_id is required"):
            generate_feature_set(
                _full_inputs(tmp_path), tmp_path / "out", config=_config(), feature_set_id="  "
            )

    def test_output_nodata_is_not_defaulted(self):
        config = _config(numerics={})
        with pytest.raises(FeatureConfigurationError, match="output_nodata is unresolved"):
            config.numerics.require_nodata()

    def test_feature_set_version_is_not_defaulted(self):
        config = _config(feature_set_version=None)
        with pytest.raises(FeatureConfigurationError, match="feature_set_version is unresolved"):
            config.require_feature_set_version()

    def test_empty_enabled_templates_is_rejected(self):
        config = _config(enabled_templates=[])
        with pytest.raises(FeatureConfigurationError, match="no feature family has been enabled"):
            config.require_enabled_templates()

    def test_empty_allowed_sources_is_rejected(self):
        config = _config(allowed_sources=[])
        with pytest.raises(FeatureConfigurationError, match="allowed_sources is empty"):
            config.require_allowed_sources()

    def test_unknown_configuration_key_is_rejected(self):
        with pytest.raises(FeatureConfigurationError):
            _config(not_a_setting=True)

    def test_unknown_artifact_role_in_band_roles_is_rejected(self):
        config = _config(band_roles={"sentinel3_before": {"vv": "VV"}})
        with pytest.raises(FeatureConfigurationError, match="not a known artifact role"):
            config.resolved_band_roles()


# =============================================================================
# 14. Training-data adapter boundary
# =============================================================================


class TestTrainingDataAdapterBoundary:
    def test_permitted_training_datasets_are_not_a_production_feature_source(self):
        """The production generator must not contain dataset-specific adaptation.

        Training corpora differ in band naming, resolution, label conventions and
        sensor representation. Adapting them inside this generator would make the
        scientific definition of a feature depend on which corpus supplied it.
        The adapter belongs in the segmentation milestone; see
        docs/data-contract.md §0.6.
        """
        declared = set(load_config("features")["allowed_sources"])
        assert ProductionInput.PERMITTED_TRAINING_DATASET.value not in declared

    def test_no_dataset_specific_identifier_appears_in_the_feature_package(self):
        markers = ("kuro_siwo", "kurosiwo", "sen1floods11", "sen1floods")
        offenders: dict[str, list[str]] = {}
        for path in (REPO_ROOT / "src" / "floodmap" / "features").rglob("*.py"):
            lowered = path.read_text(encoding="utf-8").lower()
            found = [marker for marker in markers if marker in lowered]
            if found:
                offenders[path.name] = found
        assert not offenders, (
            "the production feature generator must not special-case a training "
            f"dataset: {offenders}"
        )

    def test_registry_is_sufficient_for_an_adapter_to_target(self):
        """An adapter must be able to reproduce each feature from the registry alone."""
        registry = build_feature_registry(_config())
        for feature in registry.features:
            assert feature.transform.value in {item.value for item in TransformKind}
            assert feature.definition
            assert feature.units
            assert feature.inputs or feature.derived_from
