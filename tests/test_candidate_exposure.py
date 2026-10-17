"""Tests for the M6.4 candidate infrastructure-exposure stage.

Fixtures model the M6.3 output contract (single-band uint8, ``0``/``1``/``255``,
stage tag, projected metric CRS) and the on-disk pre-event OSM layout written by
``scripts/impact.py``. Nothing here touches the network or real imagery, and no
test asserts a flood, damage or accessibility conclusion — only that the
measured spatial intersection and the refusals behave as documented.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import LineString, Point, Polygon, box

from floodmap.infrastructure.candidate_exposure import (
    CANDIDATE_INTERSECTING,
    CANDIDATE_STATUSES,
    LAYER_FILENAMES,
    NO_CANDIDATE_INTERSECTION,
    OUTSIDE_CANDIDATE_GRID,
    PROVENANCE_SIDECAR,
    UNOBSERVED,
    CandidateExposureError,
    IntersectionRules,
    MissingInfrastructureLayers,
    assess_candidate_intersections,
    assess_lines,
    assess_polygons,
    assess_settlements,
    blocker_report,
    candidate_exposure_provenance,
    check_layers_overlap_mask,
    load_candidate_pair_metadata,
    load_local_osm_layers,
    summarize_candidate_exposure,
    write_blocker_report,
    write_candidate_exposure_artifacts,
)
from floodmap.infrastructure.candidate_mask import (
    CANDIDATE,
    CANDIDATE_SEMANTIC_VALUES_TAG,
    NOT_CANDIDATE,
    NOT_OBSERVED_VALUE,
    VALID_MASK_SEMANTICS_TAG,
    CandidateMaskError,
    candidate_semantics,
    load_candidate_mask,
)
from floodmap.infrastructure.mask import MaskSemantics
from floodmap.preprocessing.errors import AlignmentError
from floodmap.utils.config import load_config
from floodmap.utils.provenance import ArtifactType, ProductionInput
from tests.conftest import REPO_ROOT

# A small 10 m UTM 45N grid, 6 x 5 pixels, origin chosen so the AOI falls near
# the Trishuli corridor's UTM zone without reusing any real coordinate.
CRS = "EPSG:32645"
PIXEL = 10.0
ORIGIN_X = 300_000.0
ORIGIN_Y = 3_100_000.0
TRANSFORM = Affine(PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y)
WIDTH, HEIGHT = 6, 5

M63_STAGE = "m6.3-rtc-change-candidates-0.2.0"
EVENT_DATE = "2026-08-26"
OSM_SNAPSHOT = "2026-07-27T09:00:00Z"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pixel_centre(row: int, col: int) -> tuple[float, float]:
    x, y = TRANSFORM @ (col + 0.5, row + 0.5)
    return float(x), float(y)


def _classes() -> np.ndarray:
    """Three populations: candidate (row 1), not observed (row 3), rest observed."""
    classes = np.full((HEIGHT, WIDTH), NOT_CANDIDATE, dtype="uint8")
    classes[1, :] = CANDIDATE
    classes[3, :] = NOT_OBSERVED_VALUE
    return classes


def _write_candidate_mask(
    path: Path,
    classes: np.ndarray | None = None,
    *,
    transform: Affine = TRANSFORM,
    stage: str | None = M63_STAGE,
    semantic_values: str | None = CANDIDATE_SEMANTIC_VALUES_TAG,
    nodata: float | None = float(NOT_OBSERVED_VALUE),
) -> Path:
    classes = _classes() if classes is None else classes
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=classes.shape[1],
        height=classes.shape[0],
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(classes, 1)
        dst.set_band_description(1, "change_candidate")
        tags = {}
        if stage is not None:
            tags["artifact_stage"] = stage
        if semantic_values is not None:
            tags["semantic_values"] = semantic_values
        if tags:
            dst.update_tags(**tags)
    return path


def _write_valid_mask(
    path: Path,
    valid: np.ndarray | None = None,
    *,
    transform: Affine = TRANSFORM,
    semantics: str | None = VALID_MASK_SEMANTICS_TAG,
) -> Path:
    valid = (_classes() != NOT_OBSERVED_VALUE) if valid is None else valid
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=valid.shape[1],
        height=valid.shape[0],
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(valid.astype("uint8"), 1)
        dst.set_band_description(1, "valid_change_observation")
        if semantics is not None:
            dst.update_tags(mask_semantics=semantics)
    return path


def _m63_summary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "stage": M63_STAGE,
                "radiometric_convention": "gamma0_linear_power",
                "candidate_rule": {"vv_change_db_max": -3.0, "vh_change_db_max": -3.0},
                "pair": {
                    "aoi_id": "synthetic-candidate-exposure-fixture",
                    "event_date": EVENT_DATE,
                    "aoi_provenance_status": "UNVERIFIED",
                    "pre": {
                        "scene_id": "pre_rtc",
                        "acquired_at": "2026-08-16T12:21:53.557589Z",
                        "relative_orbit": 85,
                        "orbit_direction": "ASCENDING",
                    },
                    "post": {
                        "scene_id": "post_rtc",
                        "acquired_at": "2026-08-28T12:21:54.261291Z",
                        "relative_orbit": 85,
                        "orbit_direction": "ASCENDING",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _mask(tmp_path: Path, **kwargs):
    mask_path = _write_candidate_mask(tmp_path / "rtc_change_candidates.tif", **kwargs)
    valid_path = _write_valid_mask(tmp_path / "rtc_valid_mask.tif")
    return load_candidate_mask(mask_path, valid_path)


def _to_wgs84(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return frame.to_crs("EPSG:4326")


def _roads_layer() -> gpd.GeoDataFrame:
    """One road along the candidate row, one along an observed non-candidate row,
    one along the not-observed row, and one entirely off the grid."""
    rows = {"candidate": 1, "clear": 2, "unobserved": 3}
    records = []
    for label, row in rows.items():
        y = _pixel_centre(row, 0)[1]
        records.append(
            {
                "osm_id": f"way/{label}",
                "name": f"{label} road",
                "highway": "tertiary",
                "geometry": LineString(
                    [(ORIGIN_X + 1.0, y), (ORIGIN_X + WIDTH * PIXEL - 1.0, y)]
                ),
            }
        )
    far_y = ORIGIN_Y + 10_000.0
    records.append(
        {
            "osm_id": "way/offgrid",
            "name": "off grid road",
            "highway": "track",
            "geometry": LineString([(ORIGIN_X, far_y), (ORIGIN_X + 50.0, far_y)]),
        }
    )
    return gpd.GeoDataFrame(records, crs=CRS)


def _buildings_layer() -> gpd.GeoDataFrame:
    def square(row: int, col: int) -> Polygon:
        x, y = TRANSFORM @ (col, row)
        return box(x + 1.0, y - PIXEL + 1.0, x + PIXEL - 1.0, y - 1.0)

    return gpd.GeoDataFrame(
        [
            {"osm_id": "way/b-candidate", "building": "yes", "geometry": square(1, 2)},
            {"osm_id": "way/b-clear", "building": "house", "geometry": square(2, 2)},
            {"osm_id": "way/b-unobserved", "building": "yes", "geometry": square(3, 2)},
        ],
        crs=CRS,
    )


def _settlements_layer() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "osm_id": "node/s-candidate",
                "name": "Candidate village",
                "place": "village",
                "geometry": Point(*_pixel_centre(1, 3)),
            },
            {
                "osm_id": "node/s-clear",
                "name": "Clear hamlet",
                "place": "hamlet",
                "geometry": Point(*_pixel_centre(2, 3)),
            },
            {
                "osm_id": "node/s-unobserved",
                "name": "Unobserved village",
                "place": "village",
                "geometry": Point(*_pixel_centre(3, 3)),
            },
            {
                "osm_id": "node/s-offgrid",
                "name": "Off grid village",
                "place": "village",
                "geometry": Point(ORIGIN_X - 5_000.0, ORIGIN_Y + 5_000.0),
            },
        ],
        crs=CRS,
    )


def _write_osm_dir(
    directory: Path,
    *,
    layers: dict[str, gpd.GeoDataFrame] | None = None,
    sidecar: dict | None = None,
    write_sidecar: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    layers = (
        layers
        if layers is not None
        else {
            "roads": _roads_layer(),
            "buildings": _buildings_layer(),
            "settlements": _settlements_layer(),
        }
    )
    for name, frame in layers.items():
        (directory / LAYER_FILENAMES[name]).write_text(
            _to_wgs84(frame).to_json(drop_id=True), encoding="utf-8"
        )
    if write_sidecar:
        aoi = _to_wgs84(gpd.GeoDataFrame(geometry=[box(*_grid_bounds())], crs=CRS))
        default = {
            "source": "OpenStreetMap via the ohsome API",
            "snapshot": OSM_SNAPSHOT,
            "event_date": EVENT_DATE,
            "aoi_bbox": [float(v) for v in aoi.total_bounds],
            "attribution": "© OpenStreetMap contributors.",
            "license": "ODbL 1.0",
        }
        default.update(sidecar or {})
        (directory / PROVENANCE_SIDECAR).write_text(json.dumps(default), encoding="utf-8")
    return directory


def _grid_bounds() -> tuple[float, float, float, float]:
    return (
        ORIGIN_X,
        ORIGIN_Y - HEIGHT * PIXEL,
        ORIGIN_X + WIDTH * PIXEL,
        ORIGIN_Y,
    )


@pytest.fixture
def rules() -> IntersectionRules:
    return IntersectionRules(
        line_sample_step_m=5.0,
        line_min_candidate_m=10.0,
        polygon_min_candidate_px=1,
        settlement_buffer_m=0.0,
    )


# ---------------------------------------------------------------------------
# Candidate mask loading
# ---------------------------------------------------------------------------


class TestCandidateMaskLoading:
    def test_loads_and_keeps_three_pixel_populations_distinct(self, tmp_path):
        mask = _mask(tmp_path)
        counts = mask.pixel_counts()
        assert counts["candidate"] == WIDTH
        assert counts["not_observed"] == WIDTH
        assert counts["not_candidate"] == WIDTH * HEIGHT - 2 * WIDTH
        # not_observed must not be folded into not_candidate.
        assert counts["candidate"] + counts["not_candidate"] + counts["not_observed"] == (
            WIDTH * HEIGHT
        )
        assert mask.valid.sum() == WIDTH * HEIGHT - WIDTH
        description = mask.describe()
        assert description["stage"] == M63_STAGE
        assert "not confirmed flood" in description["interpretation"]
        assert description["candidate_grid_area_km2"] == pytest.approx(
            WIDTH * PIXEL * PIXEL / 1e6, abs=1e-9
        )

    def test_candidate_semantics_are_not_the_segmentation_contract(self):
        candidate = candidate_semantics()
        segmentation = MaskSemantics.from_config()
        assert candidate.flood == CANDIDATE == 1
        assert candidate != segmentation
        # The segmentation flood id must not be reinterpreted as a candidate.
        assert segmentation.flood != candidate.flood

    def test_three_class_segmentation_mask_is_refused(self, tmp_path):
        """A mask carrying the segmentation 'floods' id (2) must not be accepted."""
        classes = _classes()
        classes[0, 0] = 2
        path = _write_candidate_mask(tmp_path / "seg.tif", classes)
        with pytest.raises(CandidateMaskError, match="three-class segmentation mask"):
            load_candidate_mask(path)

    def test_unknown_producer_is_refused(self, tmp_path):
        path = _write_candidate_mask(tmp_path / "other.tif", stage="some-other-stage-9.9.9")
        with pytest.raises(CandidateMaskError, match="artifact_stage"):
            load_candidate_mask(path)

    def test_changed_semantic_values_tag_is_refused(self, tmp_path):
        path = _write_candidate_mask(
            tmp_path / "moved.tif", semantic_values="0=dry,1=flood,255=nodata"
        )
        with pytest.raises(CandidateMaskError, match="semantic_values"):
            load_candidate_mask(path)

    def test_geographic_crs_is_refused(self, tmp_path):
        path = tmp_path / "geographic.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=Affine(0.0001, 0.0, 85.0, 0.0, -0.0001, 28.0),
            nodata=NOT_OBSERVED_VALUE,
        ) as dst:
            dst.write(_classes(), 1)
            dst.update_tags(
                artifact_stage=M63_STAGE, semantic_values=CANDIDATE_SEMANTIC_VALUES_TAG
            )
        with pytest.raises(CandidateMaskError, match="geographic"):
            load_candidate_mask(path)

    def test_misaligned_valid_mask_is_refused(self, tmp_path):
        mask_path = _write_candidate_mask(tmp_path / "mask.tif")
        shifted = _write_valid_mask(
            tmp_path / "shifted_valid.tif",
            transform=Affine(PIXEL, 0.0, ORIGIN_X + PIXEL, 0.0, -PIXEL, ORIGIN_Y),
        )
        with pytest.raises(AlignmentError):
            load_candidate_mask(mask_path, shifted)

    def test_valid_mask_disagreement_is_refused(self, tmp_path):
        mask_path = _write_candidate_mask(tmp_path / "mask.tif")
        # Claim every pixel was observed while the class raster says row 3 was not.
        valid_path = _write_valid_mask(
            tmp_path / "valid.tif", np.ones((HEIGHT, WIDTH), dtype="uint8")
        )
        with pytest.raises(CandidateMaskError, match="disagree"):
            load_candidate_mask(mask_path, valid_path)

    def test_wrong_nodata_declaration_is_refused(self, tmp_path):
        path = _write_candidate_mask(tmp_path / "mask.tif", nodata=0.0)
        with pytest.raises(CandidateMaskError, match="nodata"):
            load_candidate_mask(path)


# ---------------------------------------------------------------------------
# Spatial intersection
# ---------------------------------------------------------------------------


class TestLineIntersection:
    def test_statuses_separate_candidate_clear_unobserved_and_offgrid(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_lines(_roads_layer(), mask, rules)
        status = dict(zip(assessed["osm_id"], assessed["candidate_status"]))
        assert status["way/candidate"] == CANDIDATE_INTERSECTING
        assert status["way/clear"] == NO_CANDIDATE_INTERSECTION
        assert status["way/unobserved"] == UNOBSERVED
        assert status["way/offgrid"] == OUTSIDE_CANDIDATE_GRID

    def test_candidate_length_is_measured_in_metres_not_pixels(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_lines(_roads_layer(), mask, rules).set_index("osm_id")
        road = assessed.loc["way/candidate"]
        # The line spans the full 6-pixel candidate row minus a 1 m margin at
        # each end, and every sample falls on a candidate pixel.
        assert road["length_m"] == pytest.approx(WIDTH * PIXEL - 2.0, abs=1e-6)
        assert road["candidate_m"] == pytest.approx(road["length_m"], abs=1e-6)
        assert road["not_observed_m"] == pytest.approx(0.0, abs=1e-6)
        assert assessed.loc["way/unobserved", "not_observed_m"] == pytest.approx(
            WIDTH * PIXEL - 2.0, abs=1e-6
        )

    def test_length_components_sum_to_total_length(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_lines(_roads_layer(), mask, rules)
        parts = (
            assessed["candidate_m"]
            + assessed["not_observed_m"]
            + assessed["outside_grid_m"]
            + assessed["not_candidate_m"]
        )
        np.testing.assert_allclose(parts.to_numpy(), assessed["length_m"].to_numpy(), atol=1e-3)

    def test_no_output_column_or_status_mentions_flood(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_lines(_roads_layer(), mask, rules)
        blob = " ".join(assessed.columns) + " " + " ".join(assessed["candidate_status"])
        assert "flood" not in blob.lower()
        assert "permanent_water" not in blob.lower()


class TestPolygonIntersection:
    def test_building_statuses(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_polygons(_buildings_layer(), mask, rules).set_index("osm_id")
        assert assessed.loc["way/b-candidate", "candidate_status"] == CANDIDATE_INTERSECTING
        assert assessed.loc["way/b-clear", "candidate_status"] == NO_CANDIDATE_INTERSECTION
        assert assessed.loc["way/b-unobserved", "candidate_status"] == UNOBSERVED
        assert assessed.loc["way/b-candidate", "candidate_px"] >= 1
        assert assessed.loc["way/b-unobserved", "not_observed_px"] >= 1

    def test_threshold_is_respected(self, tmp_path):
        """Raising the pixel threshold above what a footprint touches clears it."""
        mask = _mask(tmp_path)
        strict = IntersectionRules(polygon_min_candidate_px=10_000)
        assessed = assess_polygons(_buildings_layer(), mask, strict).set_index("osm_id")
        assert assessed.loc["way/b-candidate", "candidate_status"] != CANDIDATE_INTERSECTING
        # The measurement itself is unchanged; only the verdict moves.
        assert assessed.loc["way/b-candidate", "candidate_px"] >= 1


class TestSettlementIntersection:
    def test_point_statuses_use_the_containing_pixel(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_settlements(_settlements_layer(), mask, rules).set_index("osm_id")
        assert assessed.loc["node/s-candidate", "candidate_status"] == CANDIDATE_INTERSECTING
        assert assessed.loc["node/s-clear", "candidate_status"] == NO_CANDIDATE_INTERSECTION
        assert assessed.loc["node/s-unobserved", "candidate_status"] == UNOBSERVED
        assert assessed.loc["node/s-offgrid", "candidate_status"] == OUTSIDE_CANDIDATE_GRID
        assert (assessed["settlement_buffer_m"] == 0.0).all()

    def test_positive_buffer_is_recorded_and_widens_the_test(self, tmp_path):
        mask = _mask(tmp_path)
        buffered = IntersectionRules(settlement_buffer_m=15.0)
        assessed = assess_settlements(_settlements_layer(), mask, buffered).set_index("osm_id")
        assert (assessed["settlement_buffer_m"] == 15.0).all()
        # A 15 m disc around the row-2 point reaches the row-1 candidate pixels.
        assert assessed.loc["node/s-clear", "candidate_status"] == CANDIDATE_INTERSECTING
        assert assessed.loc["node/s-clear", "pixels_touched"] > 1


class TestCrsHandling:
    def test_layers_in_a_different_crs_give_the_same_result(self, tmp_path, rules):
        """Reprojection, not coordinate reinterpretation."""
        mask = _mask(tmp_path)
        native = assess_lines(_roads_layer(), mask, rules).set_index("osm_id")
        wgs84 = assess_lines(_to_wgs84(_roads_layer()), mask, rules).set_index("osm_id")
        assert list(native["candidate_status"]) == list(wgs84["candidate_status"])
        # Geodesic reprojection of a short segment keeps the length within a
        # fraction of a pixel.
        np.testing.assert_allclose(
            native["candidate_m"].to_numpy(), wgs84["candidate_m"].to_numpy(), atol=1.0
        )

    def test_output_geojson_is_written_in_epsg4326(self, tmp_path, rules):
        mask = _mask(tmp_path)
        assessed = assess_candidate_intersections(
            {"roads": _roads_layer(), "buildings": _buildings_layer()}, mask, rules
        )
        summary = summarize_candidate_exposure(assessed, mask, rules)
        paths = write_candidate_exposure_artifacts(assessed, summary, tmp_path / "out")
        written = gpd.read_file(paths["roads"])
        assert written.crs.to_epsg() == 4326
        assert len(written) == len(_roads_layer())


class TestEmptyInputs:
    @pytest.mark.parametrize("layer", ["roads", "buildings", "settlements"])
    def test_empty_layer_produces_empty_result_not_an_error(self, tmp_path, rules, layer):
        mask = _mask(tmp_path)
        template = {
            "roads": _roads_layer(),
            "buildings": _buildings_layer(),
            "settlements": _settlements_layer(),
        }[layer]
        empty = template.iloc[0:0].copy()
        assessed = assess_candidate_intersections({layer: empty}, mask, rules)
        frame = assessed[layer]
        assert len(frame) == 0
        assert "candidate_status" in frame.columns
        summary = summarize_candidate_exposure(assessed, mask, rules)
        entry = summary["layers"][layer]
        assert entry["features_total"] == 0
        assert all(entry["count_by_status"][s] == 0 for s in CANDIDATE_STATUSES)

    def test_fully_unobserved_mask_yields_no_candidate_intersections(self, tmp_path, rules):
        classes = np.full((HEIGHT, WIDTH), NOT_OBSERVED_VALUE, dtype="uint8")
        mask_path = _write_candidate_mask(tmp_path / "unobserved.tif", classes)
        valid_path = _write_valid_mask(
            tmp_path / "unobserved_valid.tif", np.zeros((HEIGHT, WIDTH), dtype="uint8")
        )
        mask = load_candidate_mask(mask_path, valid_path)
        assert mask.pixel_counts()["candidate"] == 0
        assessed = assess_lines(_roads_layer(), mask, rules).set_index("osm_id")
        assert assessed.loc["way/candidate", "candidate_status"] == UNOBSERVED
        assert assessed.loc["way/offgrid", "candidate_status"] == OUTSIDE_CANDIDATE_GRID
        assert CANDIDATE_INTERSECTING not in set(assessed["candidate_status"])


# ---------------------------------------------------------------------------
# Local OSM input
# ---------------------------------------------------------------------------


class TestLocalOsmLayers:
    def test_reads_layers_and_records_snapshot(self, tmp_path):
        directory = _write_osm_dir(tmp_path / "osm")
        osm = load_local_osm_layers(directory, event_date=EVENT_DATE)
        assert set(osm.layers) == {"roads", "buildings", "settlements"}
        assert osm.snapshot == datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
        assert osm.missing_optional == ["bridges", "hubs"]
        assert "© OpenStreetMap contributors." in osm.attribution()
        report = osm.layer_report()
        assert report["roads"]["features"] == len(_roads_layer())

    def test_missing_directory_raises_missing_layers(self, tmp_path):
        with pytest.raises(MissingInfrastructureLayers) as excinfo:
            load_local_osm_layers(tmp_path / "absent")
        assert set(excinfo.value.missing) == {"roads", "buildings", "settlements"}

    def test_missing_required_layer_raises_missing_layers(self, tmp_path):
        directory = _write_osm_dir(tmp_path / "osm")
        (directory / LAYER_FILENAMES["buildings"]).unlink()
        with pytest.raises(MissingInfrastructureLayers) as excinfo:
            load_local_osm_layers(directory, event_date=EVENT_DATE)
        assert excinfo.value.missing == ["buildings"]

    def test_missing_provenance_sidecar_is_refused_by_default(self, tmp_path):
        directory = _write_osm_dir(tmp_path / "osm", write_sidecar=False)
        with pytest.raises(CandidateExposureError, match="osm_provenance.json"):
            load_local_osm_layers(directory, event_date=EVENT_DATE)

    def test_missing_sidecar_can_be_waived_but_is_noted(self, tmp_path):
        directory = _write_osm_dir(tmp_path / "osm", write_sidecar=False)
        osm = load_local_osm_layers(
            directory, event_date=EVENT_DATE, require_provenance_sidecar=False
        )
        assert osm.snapshot is None
        assert any("unrecorded" in note for note in osm.notes)

    def test_post_event_snapshot_is_refused(self, tmp_path):
        directory = _write_osm_dir(
            tmp_path / "osm", sidecar={"snapshot": "2026-08-27T00:00:00Z"}
        )
        with pytest.raises(CandidateExposureError, match="validation-only"):
            load_local_osm_layers(directory, event_date=EVENT_DATE)

    def test_event_date_mismatch_is_refused(self, tmp_path):
        directory = _write_osm_dir(tmp_path / "osm", sidecar={"event_date": "2025-01-01"})
        with pytest.raises(CandidateExposureError, match="temporal windows"):
            load_local_osm_layers(directory, event_date=EVENT_DATE)

    def test_optional_layers_are_read_when_present(self, tmp_path):
        directory = _write_osm_dir(
            tmp_path / "osm",
            layers={
                "roads": _roads_layer(),
                "buildings": _buildings_layer(),
                "settlements": _settlements_layer(),
                "bridges": _roads_layer().iloc[:1].copy(),
            },
        )
        osm = load_local_osm_layers(directory, event_date=EVENT_DATE)
        assert "bridges" in osm.layers
        assert osm.missing_optional == ["hubs"]


class TestAoiAssociation:
    def test_overlapping_inputs_are_reported(self, tmp_path):
        mask = _mask(tmp_path)
        osm = load_local_osm_layers(_write_osm_dir(tmp_path / "osm"), event_date=EVENT_DATE)
        report = check_layers_overlap_mask(osm, mask)
        assert report["overlaps_recorded_aoi"] is True
        assert report["layers"]["roads"]["features"] == len(_roads_layer())
        # The off-grid road does not intersect the raster footprint.
        assert report["layers"]["roads"]["intersecting_mask_extent"] == len(_roads_layer()) - 1

    def test_disjoint_aoi_is_refused(self, tmp_path):
        mask = _mask(tmp_path)
        directory = _write_osm_dir(
            tmp_path / "osm", sidecar={"aoi_bbox": [10.0, 10.0, 10.1, 10.1]}
        )
        osm = load_local_osm_layers(directory, event_date=EVENT_DATE)
        with pytest.raises(CandidateExposureError, match="different areas"):
            check_layers_overlap_mask(osm, mask)


# ---------------------------------------------------------------------------
# Summary, provenance and blocker
# ---------------------------------------------------------------------------


class TestSummaryAndProvenance:
    def test_summary_reports_counts_lengths_and_coverage(self, tmp_path, rules):
        mask = _mask(tmp_path)
        osm = load_local_osm_layers(_write_osm_dir(tmp_path / "osm"), event_date=EVENT_DATE)
        pair = load_candidate_pair_metadata(_m63_summary(tmp_path / "rtc_change_summary.json"))
        aoi_report = check_layers_overlap_mask(osm, mask)
        assessed = assess_candidate_intersections(osm.layers, mask, rules)
        summary = summarize_candidate_exposure(
            assessed, mask, rules, osm=osm, aoi_report=aoi_report, pair=pair
        )

        roads = summary["layers"]["roads"]
        assert roads["count_by_status"][CANDIDATE_INTERSECTING] == 1
        assert roads["count_by_status"][UNOBSERVED] == 1
        assert roads["count_by_status"][OUTSIDE_CANDIDATE_GRID] == 1
        assert roads["candidate_intersecting_length_km"] > 0
        assert "tertiary" in roads["candidate_intersecting_km_by_highway"]
        assert "osm_id" in roads["preserved_attributes"]

        assert summary["layers"]["buildings"]["count_by_status"][CANDIDATE_INTERSECTING] == 1
        settlements = summary["layers"]["settlements"]
        assert settlements["candidate_intersecting"] == [
            {
                "osm_id": "node/s-candidate",
                "name": "Candidate village",
                "place": "village",
            }
        ]

        coverage = summary["data_coverage"]
        assert coverage["candidate_mask"]["pixel_counts"]["not_observed"] == WIDTH
        assert coverage["optional_layers_absent"] == ["bridges", "hubs"]
        assert coverage["aoi_association"]["overlaps_recorded_aoi"] is True
        assert summary["osm"]["retrieval_performed"] is False
        assert summary["aoi_provenance_status"] == "UNVERIFIED"
        assert summary["status"] == "candidate_exposure_only"
        assert "not confirmed flood extents" in summary["interpretation"]
        assert any("not confirmed flood" in limit for limit in summary["limitations"])

    def test_provenance_records_inputs_osm_snapshot_and_limitations(self, tmp_path, rules):
        mask = _mask(tmp_path)
        osm = load_local_osm_layers(_write_osm_dir(tmp_path / "osm"), event_date=EVENT_DATE)
        pair = load_candidate_pair_metadata(_m63_summary(tmp_path / "rtc_change_summary.json"))
        assessed = assess_candidate_intersections(osm.layers, mask, rules)
        summary = summarize_candidate_exposure(assessed, mask, rules, osm=osm, pair=pair)
        provenance = candidate_exposure_provenance(
            mask=mask, osm=osm, summary=summary, pair=pair, config_version="0.1.0"
        )

        assert provenance.artifact_type is ArtifactType.INFRASTRUCTURE_EXPOSURE
        assert set(provenance.production_inputs) == {
            ProductionInput.SENTINEL1,
            ProductionInput.OSM_PRE_EVENT,
        }
        assert provenance.source_product_ids == ["pre_rtc", "post_rtc"]
        assert provenance.osm_snapshot_date.startswith("2026-07-27")
        assert provenance.event_date == EVENT_DATE
        assert provenance.sentinel1.same_relative_orbit is True
        assert provenance.sentinel1.before.acquired_at == "2026-08-16T12:21:53.557589Z"
        assert any("UNVERIFIED" in limit for limit in provenance.limitations)
        assert "No download was performed" in provenance.notes
        # Required challenge attributions are attached automatically.
        assert "Contains modified Copernicus Sentinel data 2026." in provenance.attribution
        assert "© OpenStreetMap contributors." in provenance.attribution

    def test_osm_snapshot_must_predate_event_in_provenance_chain(self, tmp_path):
        """The pre-event rule is enforced at load time, so provenance cannot
        record a post-event snapshot."""
        directory = _write_osm_dir(
            tmp_path / "osm", sidecar={"snapshot": "2026-08-26T00:00:00Z"}
        )
        with pytest.raises(CandidateExposureError):
            load_local_osm_layers(directory, event_date=EVENT_DATE)

    def test_writer_emits_geojson_summary_and_provenance(self, tmp_path, rules):
        mask = _mask(tmp_path)
        osm = load_local_osm_layers(_write_osm_dir(tmp_path / "osm"), event_date=EVENT_DATE)
        pair = load_candidate_pair_metadata(_m63_summary(tmp_path / "rtc_change_summary.json"))
        assessed = assess_candidate_intersections(osm.layers, mask, rules)
        summary = summarize_candidate_exposure(assessed, mask, rules, osm=osm, pair=pair)
        provenance = candidate_exposure_provenance(
            mask=mask, osm=osm, summary=summary, pair=pair, config_version="0.1.0"
        )
        paths = write_candidate_exposure_artifacts(
            assessed, summary, tmp_path / "out", provenance=provenance
        )
        assert set(paths) >= {"roads", "buildings", "settlements", "summary", "provenance"}
        reloaded = json.loads(paths["summary"].read_text(encoding="utf-8"))
        assert reloaded["stage"].startswith("m6.4-")
        assert "osm_id" in gpd.read_file(paths["roads"]).columns
        assert "candidate_status" in gpd.read_file(paths["settlements"]).columns
        assert "infrastructure_exposure" in paths["provenance"].read_text(encoding="utf-8")


class TestBlocker:
    def test_blocker_report_states_gate_available_inputs_and_next_step(self, tmp_path):
        mask = _mask(tmp_path)
        pair = load_candidate_pair_metadata(_m63_summary(tmp_path / "rtc_change_summary.json"))
        report = blocker_report(
            reason="required pre-event OSM layers are not present on disk",
            missing=["roads", "buildings", "settlements"],
            searched=tmp_path / "data/raw/osm",
            mask=mask,
            pair=pair,
            next_step=("establish AOI provenance", "fetch pre-event OSM", "re-run"),
        )
        assert report["status"] == "blocked"
        assert report["retrieval_performed"] is False
        assert report["authorisation"]["authorises_download"] is False
        assert report["authorisation"]["aoi_provenance_status"] == "UNVERIFIED"
        assert report["expected_filenames"]["roads"] == "osm_roads.geojson"
        # The real inputs that WERE available are still reported.
        assert report["inputs_available"]["candidate_mask"]["stage"] == M63_STAGE
        assert report["inputs_available"]["source_pair"]["event_date"] == EVENT_DATE
        assert len(report["next_step"]) == 3
        assert any("synthesised" in line for line in report["not_done"])

        path = write_blocker_report(report, tmp_path / "out")
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "blocked"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_shipped_config_loads_into_rules(self):
        config = load_config("candidate_exposure")
        rules = IntersectionRules.from_config(config)
        assert rules.line_sample_step_m == 5.0
        assert rules.line_min_candidate_m == 10.0
        assert rules.polygon_min_candidate_px == 1
        assert rules.settlement_buffer_m == 0.0

    def test_config_disables_retrieval_and_pins_the_m63_stage(self):
        inputs = load_config("candidate_exposure")["inputs"]
        assert inputs["candidate_mask_stage"] == M63_STAGE
        assert inputs["osm"]["allow_download"] is False
        assert inputs["osm"]["require_pre_event_snapshot"] is True
        assert inputs["osm"]["layer_dir"] is None

    def test_thresholds_match_the_flood_class_exposure_config(self):
        """M6.4 restates impact.yaml's thresholds so the two are comparable."""
        candidate = load_config("candidate_exposure")["intersection"]
        flood_class = load_config("impact")["exposure"]
        assert candidate["line_sample_step_m"] == flood_class["line_sample_step_m"]
        assert candidate["line_min_candidate_m"] == flood_class["line_min_flooded_m"]
        assert candidate["polygon_min_candidate_px"] == flood_class["building_min_flood_px"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"line_sample_step_m": 0.0},
            {"line_min_candidate_m": -1.0},
            {"polygon_min_candidate_px": 0},
            {"settlement_buffer_m": -5.0},
        ],
    )
    def test_invalid_rules_are_refused(self, kwargs):
        with pytest.raises(CandidateExposureError):
            IntersectionRules(**kwargs)


# ---------------------------------------------------------------------------
# Real M6.3 artifacts
#
# These exercise the real local Sentinel-1 RTC candidate mask when it is
# present. The artifacts are git-ignored, so the tests skip in a fresh
# checkout rather than failing. They assert the real raster satisfies the M6.4
# input contract; they assert nothing about flooding or damage, and no real
# infrastructure exposure is claimed here, because no permitted pre-event OSM
# layer exists locally.
# ---------------------------------------------------------------------------

_REAL_M63_DIR = REPO_ROOT / "artifacts" / "m6_3" / "rtc_change_candidates"
_REAL_MASK = _REAL_M63_DIR / "rtc_change_candidates.tif"
_REAL_VALID = _REAL_M63_DIR / "rtc_valid_mask.tif"
_REAL_SUMMARY = _REAL_M63_DIR / "rtc_change_summary.json"

_real_artifacts = pytest.mark.skipif(
    not (_REAL_MASK.is_file() and _REAL_VALID.is_file() and _REAL_SUMMARY.is_file()),
    reason="M6.3 artifacts have not been generated in this environment (git-ignored)",
)


@_real_artifacts
class TestRealCandidateMask:
    def test_real_mask_satisfies_the_m64_input_contract(self):
        config = load_config("candidate_exposure")["inputs"]
        mask = load_candidate_mask(
            _REAL_MASK,
            _REAL_VALID,
            expected_stage=str(config["candidate_mask_stage"]),
            alignment_tolerance=float(config["alignment_tolerance"]),
        )
        assert mask.crs.is_projected
        assert mask.alignment["aligned"] is True
        counts = mask.pixel_counts()
        assert sum(counts.values()) == mask.classes.size
        assert counts["candidate"] > 0
        # The three populations stay separable on the real raster.
        assert set(np.unique(mask.classes).tolist()) <= {
            NOT_CANDIDATE,
            CANDIDATE,
            NOT_OBSERVED_VALUE,
        }

    def test_real_pair_metadata_carries_scene_ids_and_timestamps(self):
        pair = load_candidate_pair_metadata(_REAL_SUMMARY)
        assert pair["aoi_provenance_status"] == "UNVERIFIED"
        for side in ("pre", "post"):
            assert pair[side]["scene_id"]
            assert pair[side]["acquired_at"]
            assert isinstance(pair[side]["relative_orbit"], int)

    def test_real_blocker_retains_the_available_real_inputs(self, tmp_path):
        """With no permitted OSM layer on disk, the stage reports a blocker that
        still records exactly which real inputs were available."""
        mask = load_candidate_mask(_REAL_MASK, _REAL_VALID)
        pair = load_candidate_pair_metadata(_REAL_SUMMARY)
        with pytest.raises(MissingInfrastructureLayers) as excinfo:
            load_local_osm_layers(tmp_path / "no-osm-here", event_date=pair["event_date"])
        report = blocker_report(
            reason=str(excinfo.value),
            missing=excinfo.value.missing,
            searched=excinfo.value.searched,
            mask=mask,
            pair=pair,
        )
        assert report["status"] == "blocked"
        assert report["retrieval_performed"] is False
        described = report["inputs_available"]["candidate_mask"]
        assert described["crs"].startswith("EPSG:")
        assert described["pixel_counts"]["candidate"] > 0
        assert report["inputs_available"]["source_pair"]["aoi_provenance_status"] == "UNVERIFIED"
