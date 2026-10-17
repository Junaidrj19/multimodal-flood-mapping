"""Tests for the real-RTC change-candidate stage.

The fixtures model the assembled MPC RTC contract: two bands named VV/VH,
gamma0 linear power, explicit nodata and a georeferenced projected grid.  No
network or real imagery is used by these unit tests.

The AOI bindings here are synthetic. They are built directly rather than read
from the project configuration so that these tests exercise the scoping logic
without asserting anything about the real approved Trishuli extent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
import yaml
from affine import Affine
from pyproj import Transformer

from floodmap.baseline.rtc_change_candidates import (
    APPROVED_COVERAGE_DISCLOSURE,
    CHANGE_CANDIDATE_NODATA,
    CHANGE_FEATURE_NODATA,
    RTC_CONVENTION,
    AoiBinding,
    RtcChangeError,
    RtcPairMetadata,
    compute_rtc_change_candidates,
    load_configured_aoi_binding,
    load_rtc_change_config,
    write_rtc_change_artifacts,
)
from floodmap.preprocessing.errors import AlignmentError


CRS = "EPSG:32645"
TRANSFORM = Affine(10.0, 0.0, 300_000.0, 0.0, -10.0, 3_100_000.0)
NODATA = -32768.0
#: The 5x6 fixture grid spans x 300000..300060, y 3099950..3100000.
GRID_BOUNDS = (300_000.0, 3_099_950.0, 300_060.0, 3_100_000.0)
FIXTURE_SHA256 = "a" * 64


def _write(path: Path, values: np.ndarray, *, transform: Affine = TRANSFORM, convention: str = RTC_CONVENTION) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=2,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=NODATA,
    ) as destination:
        destination.write(values.astype("float32"))
        destination.set_band_description(1, "VV")
        destination.set_band_description(2, "VH")
        destination.update_tags(radiometric_convention=convention, source_item_id=path.stem)
    return path


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    pre = np.full((2, 5, 6), 0.2, dtype="float32")
    post = pre.copy()
    # A pair of pixels darkens by 5 dB in both polarisations and is a candidate.
    post[:, 1, 1] *= 10 ** (-5.0 / 10.0)
    post[:, 1, 2] *= 10 ** (-5.0 / 10.0)
    # This pixel is invalid in both dates; it must not become a candidate.
    pre[:, 3, 3] = NODATA
    post[:, 3, 3] = NODATA
    return _write(tmp_path / "pre.tif", pre), _write(tmp_path / "post.tif", post)


def _to_4326(geometry):
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    return shapely.transform(
        geometry,
        lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1])),
    )


def _aoi_binding(
    bounds: tuple[float, float, float, float] = GRID_BOUNDS,
    *,
    aoi_id: str = "synthetic-rtc-change-fixture",
    coverage: dict | None = None,
) -> AoiBinding:
    """A synthetic binding whose extent is given in the fixture's UTM bounds."""
    return AoiBinding(
        aoi_id=aoi_id,
        source_sha256=FIXTURE_SHA256,
        geojson_path=Path("configs/aoi/SYNTHETIC-fixture.geojson"),
        provenance_status="unverified",
        geometry_4326=_to_4326(shapely.box(*bounds)),
        name="SYNTHETIC fixture extent",
        coverage=coverage if coverage is not None else {},
    )


def _metadata() -> RtcPairMetadata:
    return RtcPairMetadata.from_binding(
        _aoi_binding(),
        event_date="2026-08-26",
        pre_scene_id="pre_rtc",
        post_scene_id="post_rtc",
        pre_acquired_at="2026-08-16T12:21:53.557589Z",
        post_acquired_at="2026-08-28T12:21:54.261291Z",
        relative_orbit=85,
        orbit_direction="ASCENDING",
    )


def test_change_features_are_log_ratios_and_candidate_requires_both_polarisations(tmp_path):
    pre, post = _pair(tmp_path)
    result = compute_rtc_change_candidates(
        pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
    )

    assert result.valid[1, 1]
    assert result.vv_change_db[1, 1] == pytest.approx(-5.0, abs=1e-5)
    assert result.vh_change_db[1, 1] == pytest.approx(-5.0, abs=1e-5)
    assert result.candidates[1, 1]
    assert result.candidates.sum() == 2
    assert not result.valid[3, 3]
    assert result.vv_change_db[3, 3] == CHANGE_FEATURE_NODATA


def test_nonpositive_power_is_invalid_not_regularised(tmp_path):
    pre, post = _pair(tmp_path)
    with rasterio.open(pre, "r+") as destination:
        values = destination.read()
        values[:, 0, 0] = 0.0
        destination.write(values)
    result = compute_rtc_change_candidates(
        pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
    )
    assert not result.valid[0, 0]
    assert result.candidates[0, 0] == 0


def test_sigma0_or_missing_convention_is_refused(tmp_path):
    pre, post = _pair(tmp_path)
    _write(tmp_path / "bad.tif", np.ones((2, 5, 6), dtype="float32"), convention="linear_sigma0")
    with pytest.raises(RtcChangeError, match="gamma0_linear_power"):
        compute_rtc_change_candidates(
            tmp_path / "bad.tif",
            post,
            aoi=_aoi_binding(),
            vv_threshold_db=-3,
            vh_threshold_db=-3,
        )


def test_candidate_config_requires_gamma0_and_alignment(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "version: 0.1.0\n"
        "inputs:\n"
        "  radiometric_convention: linear_sigma0\n"
        "  require_same_grid: false\n"
        "candidate_rule:\n"
        "  vv_change_db_max: -3\n"
        "  vh_change_db_max: -3\n",
        encoding="utf-8",
    )
    with pytest.raises(RtcChangeError, match="gamma0_linear_power"):
        load_rtc_change_config(path)


def test_grid_mismatch_is_refused(tmp_path):
    pre, post = _pair(tmp_path)
    values = np.ones((2, 5, 6), dtype="float32")
    shifted = _write(
        tmp_path / "shifted.tif",
        values,
        transform=Affine(10.0, 0.0, 300_010.0, 0.0, -10.0, 3_100_000.0),
    )
    with pytest.raises(AlignmentError, match="aligned|transform|extent"):
        compute_rtc_change_candidates(
            pre, shifted, aoi=_aoi_binding(), vv_threshold_db=-3, vh_threshold_db=-3
        )


def test_writer_preserves_georeferencing_and_records_unverified_provenance(tmp_path):
    pre, post = _pair(tmp_path)
    result = compute_rtc_change_candidates(
        pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
    )
    paths = write_rtc_change_artifacts(
        result,
        tmp_path / "out",
        pair=_metadata(),
        vv_threshold_db=-3.0,
        vh_threshold_db=-3.0,
        config_version="0.1.0",
    )

    with rasterio.open(paths["features"]) as dataset:
        assert dataset.crs.to_string() == CRS
        assert dataset.transform == TRANSFORM
        assert dataset.descriptions == ("vv_change_db", "vh_change_db")
        assert dataset.nodata == CHANGE_FEATURE_NODATA
    with rasterio.open(paths["candidate_mask"]) as dataset:
        assert dataset.descriptions == ("change_candidate",)
        assert dataset.nodata == CHANGE_CANDIDATE_NODATA
        assert set(np.unique(dataset.read(1)).tolist()) <= {0, 1, CHANGE_CANDIDATE_NODATA}

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["status"] == "candidate_evidence_only"
    assert summary["radiometric_convention"] == RTC_CONVENTION
    assert summary["aoi_provenance_status"] == "UNVERIFIED"
    assert "confirmed flood" in summary["interpretation"]


# --------------------------------------------------------------------------
# AOI binding and scoping
# --------------------------------------------------------------------------


class TestAoiScoping:
    """The configured extent must bound every candidate and every denominator."""

    def _half_grid_binding(self) -> AoiBinding:
        """An extent covering only the left half of the fixture grid."""
        left, bottom, _, top = GRID_BOUNDS
        return _aoi_binding((left, bottom, left + 30.0, top))

    def test_aoi_mask_marks_only_pixel_centres_inside_the_extent(self, tmp_path):
        pre, post = _pair(tmp_path)
        result = compute_rtc_change_candidates(
            pre, post, aoi=self._half_grid_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
        )
        # Columns 0-2 are inside the 30 m-wide extent, columns 3-5 are outside.
        assert result.aoi_mask[:, :3].all()
        assert not result.aoi_mask[:, 3:].any()
        assert int(result.aoi_mask.sum()) == 15

    def test_out_of_extent_darkening_is_not_a_candidate(self, tmp_path):
        pre = np.full((2, 5, 6), 0.2, dtype="float32")
        post = pre.copy()
        # One darkening pixel inside the extent, one outside it.
        post[:, 1, 1] *= 10 ** (-5.0 / 10.0)
        post[:, 1, 4] *= 10 ** (-5.0 / 10.0)
        pre_path = _write(tmp_path / "pre.tif", pre)
        post_path = _write(tmp_path / "post.tif", post)

        result = compute_rtc_change_candidates(
            pre_path,
            post_path,
            aoi=self._half_grid_binding(),
            vv_threshold_db=-3.0,
            vh_threshold_db=-3.0,
        )
        assert result.candidates[1, 1]
        assert not result.candidates[1, 4], "darkening outside the extent must not be a candidate"
        assert int(result.candidates.sum()) == 1

    def test_candidates_are_a_subset_of_valid_which_is_a_subset_of_the_extent(self, tmp_path):
        pre, post = _pair(tmp_path)
        result = compute_rtc_change_candidates(
            pre, post, aoi=self._half_grid_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
        )
        assert not (result.candidates & ~result.valid).any()
        assert not (result.valid & ~result.aoi_mask).any()
        assert not (result.candidates & ~result.aoi_mask).any()

    def test_nodata_pixels_are_excluded_from_the_valid_support(self, tmp_path):
        pre, post = _pair(tmp_path)
        result = compute_rtc_change_candidates(
            pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
        )
        # (3, 3) is nodata in both dates but lies inside the extent, so it is a
        # delivered-but-invalid pixel: in the AOI mask, absent from the valid mask.
        assert result.aoi_mask[3, 3]
        assert not result.valid[3, 3]
        scope = result.aoi_scope
        assert scope["aoi_delivered_but_invalid_pixels"] == 1
        assert scope["aoi_valid_pixels"] == 29
        assert scope["aoi_pixel_centers_in_window"] == 30

    def test_an_extent_that_misses_the_window_is_refused(self, tmp_path):
        pre, post = _pair(tmp_path)
        far_away = _aoi_binding((400_000.0, 3_200_000.0, 400_100.0, 3_200_100.0))
        with pytest.raises(RtcChangeError, match="no pixel centre inside"):
            compute_rtc_change_candidates(
                pre, post, aoi=far_away, vv_threshold_db=-3.0, vh_threshold_db=-3.0
            )


class TestDenominators:
    """Every reported rate must name the support it was divided by."""

    def test_full_extent_area_is_never_the_observed_area(self, tmp_path):
        pre, post = _pair(tmp_path)
        # An extent twice as tall as the delivered window: half of it is unobserved.
        left, bottom, right, top = GRID_BOUNDS
        binding = _aoi_binding((left, bottom - (top - bottom), right, top))
        result = compute_rtc_change_candidates(
            pre, post, aoi=binding, vv_threshold_db=-3.0, vh_threshold_db=-3.0
        )
        scope = result.aoi_scope

        assert scope["aoi_geometric_area_km2"] > scope["aoi_delivered_area_km2"]
        assert scope["aoi_valid_area_km2"] < scope["aoi_geometric_area_km2"]
        assert scope["aoi_delivered_fraction"] == pytest.approx(0.5, abs=1e-3)
        assert scope["aoi_unobserved_fraction"] == pytest.approx(0.5, abs=1e-3)
        assert scope["aoi_unobserved_area_km2"] > 0.0

    def test_candidate_rate_over_the_full_extent_is_lower_than_over_valid_support(self, tmp_path):
        pre, post = _pair(tmp_path)
        left, bottom, right, top = GRID_BOUNDS
        binding = _aoi_binding((left, bottom - (top - bottom), right, top))
        scope = compute_rtc_change_candidates(
            pre, post, aoi=binding, vv_threshold_db=-3.0, vh_threshold_db=-3.0
        ).aoi_scope
        assert (
            scope["candidate_fraction_of_aoi_geometric_area"]
            < scope["candidate_fraction_of_aoi_valid_pixels"]
        )

    def test_window_scope_is_labelled_and_separate_from_aoi_scope(self, tmp_path):
        pre, post = _pair(tmp_path)
        binding = _aoi_binding((GRID_BOUNDS[0], GRID_BOUNDS[1], GRID_BOUNDS[0] + 30.0, GRID_BOUNDS[3]))
        summary = compute_rtc_change_candidates(
            pre, post, aoi=binding, vv_threshold_db=-3.0, vh_threshold_db=-3.0
        ).summary(vv_threshold_db=-3.0, vh_threshold_db=-3.0)

        assert summary["scope"] == "delivered_raster_window_array_counts"
        assert "NOT the AOI" in summary["scope_note"]
        assert summary["total_pixels"] == 30
        assert summary["aoi_scope"]["aoi_pixel_centers_in_window"] == 15
        assert summary["aoi_scope"]["window_pixels_outside_aoi"] == 15

    def test_every_reported_denominator_is_defined(self, tmp_path):
        pre, post = _pair(tmp_path)
        scope = compute_rtc_change_candidates(
            pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
        ).aoi_scope
        definitions = scope["denominator_definitions"]
        for key in (
            "aoi_geometric_area_km2",
            "aoi_delivered_area_km2",
            "aoi_unobserved_area_km2",
            "aoi_valid_area_km2",
            "candidate_fraction_of_aoi_valid_pixels",
            "candidate_fraction_of_aoi_geometric_area",
            "window_pixels_outside_aoi",
        ):
            assert key in definitions and definitions[key].strip()
        assert "NOT an observed area" in definitions["aoi_geometric_area_km2"]


class TestWrittenAoiArtifacts:
    """The AOI binding and disclosure must survive into the written artifacts."""

    def _write(self, tmp_path: Path):
        pre, post = _pair(tmp_path)
        result = compute_rtc_change_candidates(
            pre, post, aoi=_aoi_binding(), vv_threshold_db=-3.0, vh_threshold_db=-3.0
        )
        return result, write_rtc_change_artifacts(
            result,
            tmp_path / "out",
            pair=_metadata(),
            vv_threshold_db=-3.0,
            vh_threshold_db=-3.0,
            config_version="0.1.0",
        )

    def test_aoi_mask_raster_is_written_with_declared_semantics(self, tmp_path):
        result, paths = self._write(tmp_path)
        with rasterio.open(paths["aoi_mask"]) as dataset:
            assert dataset.descriptions == ("inside_configured_aoi",)
            assert dataset.nodata == 0
            assert set(np.unique(dataset.read(1)).tolist()) <= {0, 1}
            tags = dataset.tags()
            assert tags["aoi_id"] == "synthetic-rtc-change-fixture"
            assert tags["aoi_source_sha256"] == FIXTURE_SHA256
            assert tags["rasterization"] == "center"
            assert tags["all_touched"] == "false"
            assert tags["resampling"] == "none"
            np.testing.assert_array_equal(dataset.read(1), result.aoi_mask.astype("uint8"))

    def test_summary_carries_the_aoi_identity_hash_and_disclosure(self, tmp_path):
        _, paths = self._write(tmp_path)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        assert summary["aoi_id"] == "synthetic-rtc-change-fixture"
        assert summary["aoi_source_sha256"] == FIXTURE_SHA256
        assert summary["pair"]["aoi_source_sha256"] == FIXTURE_SHA256
        # No sidecar coverage was supplied, so the approved text is the fallback.
        assert summary["coverage_disclosure"] == APPROVED_COVERAGE_DISCLOSURE
        assert summary["aoi_scope"]["disclosure_matches_m614_approved_text"] is True

    def test_masks_declare_how_to_separate_outside_extent_from_unobserved(self, tmp_path):
        _, paths = self._write(tmp_path)
        with rasterio.open(paths["candidate_mask"]) as dataset:
            tags = dataset.tags()
        assert "rtc_aoi_mask.tif" in tags["not_observed_meaning"]
        assert tags["analysis_scope"] == "configured_aoi_valid_pixels_only"

    def test_provenance_records_partial_observation_and_claim_limits(self, tmp_path):
        _, paths = self._write(tmp_path)
        record = yaml.safe_load(paths["candidate_provenance"].read_text(encoding="utf-8"))
        limitations = " ".join(record["limitations"])
        assert "not confirmed flood or damage" in limitations
        assert FIXTURE_SHA256 in limitations
        assert "Partial observation" in limitations
        assert "must not be interpreted as unaffected" in limitations
        assert "strict subset" in record["notes"]

    def test_writer_refuses_candidates_outside_the_valid_mask(self, tmp_path):
        result, _ = self._write(tmp_path)
        tampered = type(result)(
            pre_valid=result.pre_valid,
            post_valid=result.post_valid,
            valid=np.zeros_like(result.valid),
            vv_change_db=result.vv_change_db,
            vh_change_db=result.vh_change_db,
            candidates=result.candidates,
            alignment=result.alignment,
            input_metadata=result.input_metadata,
            aoi_mask=result.aoi_mask,
            aoi_scope=result.aoi_scope,
        )
        with pytest.raises(RtcChangeError, match="outside the valid mask"):
            write_rtc_change_artifacts(
                tampered,
                tmp_path / "tampered",
                pair=_metadata(),
                vv_threshold_db=-3.0,
                vh_threshold_db=-3.0,
                config_version="0.1.0",
            )

    def test_writer_refuses_candidates_outside_the_configured_extent(self, tmp_path):
        result, _ = self._write(tmp_path)
        tampered = type(result)(
            pre_valid=result.pre_valid,
            post_valid=result.post_valid,
            valid=result.valid,
            vv_change_db=result.vv_change_db,
            vh_change_db=result.vh_change_db,
            candidates=result.candidates,
            alignment=result.alignment,
            input_metadata=result.input_metadata,
            aoi_mask=np.zeros_like(result.aoi_mask),
            aoi_scope=result.aoi_scope,
        )
        with pytest.raises(RtcChangeError, match="outside the configured AOI"):
            write_rtc_change_artifacts(
                tampered,
                tmp_path / "tampered",
                pair=_metadata(),
                vv_threshold_db=-3.0,
                vh_threshold_db=-3.0,
                config_version="0.1.0",
            )


class TestConfiguredAoiBinding:
    """The extent comes from configuration and is pinned by digest, not by label."""

    def _workspace(self, tmp_path: Path, *, digest: str | None = None) -> Path:
        geometry = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": shapely.geometry.mapping(_to_4326(shapely.box(*GRID_BOUNDS))),
                }
            ],
        }
        aoi_dir = tmp_path / "configs" / "aoi"
        aoi_dir.mkdir(parents=True, exist_ok=True)
        geometry_path = aoi_dir / "fixture.geojson"
        payload = json.dumps(geometry, indent=2) + "\n"
        geometry_path.write_text(payload, encoding="utf-8")
        real_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        (aoi_dir / "fixture.provenance.yaml").write_text(
            yaml.safe_dump(
                {
                    "aoi_id": "fixture-aoi-SYNTHETIC",
                    "geometry": {"sha256": real_digest},
                    "coverage": {"disclosure": "SYNTHETIC disclosure", "delivered_fraction": 0.5},
                }
            ),
            encoding="utf-8",
        )
        config = tmp_path / "configs" / "data.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "acquisition": {
                        "aoi": {
                            "aoi_id": "fixture-aoi-SYNTHETIC",
                            "name": "SYNTHETIC",
                            "geojson_path": "configs/aoi/fixture.geojson",
                            "sha256": digest if digest is not None else real_digest,
                            "provenance_status": "unverified",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return config

    def test_binding_reads_identity_digest_and_sidecar_coverage(self, tmp_path):
        config = self._workspace(tmp_path)
        binding = load_configured_aoi_binding(config, project_root=tmp_path)
        assert binding.aoi_id == "fixture-aoi-SYNTHETIC"
        assert binding.provenance_status == "unverified"
        assert binding.coverage["delivered_fraction"] == 0.5
        assert binding.disclosure == "SYNTHETIC disclosure"
        assert binding.provenance_sha256 is not None
        assert binding.geometry_4326.is_valid

    def test_a_mismatched_digest_refuses_to_load(self, tmp_path):
        config = self._workspace(tmp_path, digest="0" * 64)
        with pytest.raises(ValueError, match="was pinned|sha256"):
            load_configured_aoi_binding(config, project_root=tmp_path)

    def test_a_malformed_digest_is_refused(self, tmp_path):
        config = self._workspace(tmp_path, digest="not-a-digest")
        with pytest.raises(RtcChangeError, match="64-character hex digest"):
            load_configured_aoi_binding(config, project_root=tmp_path)

    def test_a_sidecar_for_a_different_extent_is_refused(self, tmp_path):
        config = self._workspace(tmp_path)
        sidecar = tmp_path / "configs" / "aoi" / "fixture.provenance.yaml"
        record = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        record["aoi_id"] = "some-other-extent"
        sidecar.write_text(yaml.safe_dump(record), encoding="utf-8")
        with pytest.raises(RtcChangeError, match="does not match"):
            load_configured_aoi_binding(config, project_root=tmp_path)

    def test_pair_metadata_requires_a_real_digest_when_supplied(self):
        with pytest.raises(RtcChangeError, match="64-character hex digest"):
            RtcPairMetadata(
                aoi_id="x",
                event_date="2026-08-26",
                pre_scene_id="a",
                post_scene_id="b",
                pre_acquired_at="2026-08-16T12:21:53Z",
                post_acquired_at="2026-08-28T12:21:54Z",
                relative_orbit=85,
                orbit_direction="ASCENDING",
                aoi_source_sha256="short",
            )

    def test_pair_metadata_preserves_unverified_provenance_only(self):
        binding = AoiBinding(
            aoi_id="x",
            source_sha256=FIXTURE_SHA256,
            geojson_path=Path("x.geojson"),
            provenance_status="verified",
            geometry_4326=_to_4326(shapely.box(*GRID_BOUNDS)),
        )
        with pytest.raises(RtcChangeError, match="preserves AOI provenance as UNVERIFIED"):
            RtcPairMetadata.from_binding(
                binding,
                event_date="2026-08-26",
                pre_scene_id="a",
                post_scene_id="b",
                pre_acquired_at="2026-08-16T12:21:53Z",
                post_acquired_at="2026-08-28T12:21:54Z",
                relative_orbit=85,
                orbit_direction="ASCENDING",
            )
