"""M6.10 — the derived HydroRIVERS corridor: provenance, pin, coverage, gates.

What these tests protect
------------------------
The replacement study extent is only an improvement if three independent claims
hold, and each can fail on its own:

1. **the artifact is the one recorded** — bytes pinned, geometry valid, area and
   bounds agree with the record rather than with a recollection of it;
2. **the record explains where the geometry came from** — source, version,
   licence, citation, selected reaches, reference point, buffer, CRS and
   coverage are all present and machine-readable;
3. **the artifact authorises nothing** — it is unverified provenance, the
   acquisition configuration is untouched, and both download gates stay closed.

Point 3 is the one most likely to rot. A future change that promoted this AOI to
`verified` because its geometry is now traceable would silently open the
acquisition gate, and "the geometry is documented" is not the same claim as
"an operator adopted this extent and ratified the buffer". These tests assert
the distinct claims separately.

Most tests here read the committed artifact and need no source data. The two that
rebuild or measure coverage are skipped when their inputs are absent, because
those inputs are git-ignored by design (AGENTS.md §15) and their absence is not
a test failure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import shapely
import yaml
from shapely.geometry import LineString, Point, Polygon, box

from floodmap.acquisition.aoi import AoiProvenanceStatus, AreaOfInterest
from floodmap.hydrology.hydrorivers import (
    DELIVERY_CRS,
    STUDY_AREA_CRS,
    HydroRiversNetwork,
    Reach,
    buffer_corridor,
    build_corridor_line,
    choose_upstream_branch,
    drop_repeated_vertices,
    reproject_geometry,
    select_coverage_bounded_corridor,
    trim_line_to_point,
)
from floodmap.utils.config import load_config, project_root
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

CONFIG_NAME = "aoi/trishuli-corridor-hydrorivers-v10"
OLD_AOI = Path("configs/aoi/trishuli-corridor-UNVERIFIED.geojson")

#: The M6.7 record of the artifact this milestone replaces. Pinned here so a
#: rewrite of the old corridor fails loudly instead of going unnoticed.
OLD_AOI_SHA256 = "9035eeee025f311be7f16af1d23737a46caee20eb0b2a8dfa144d111b9362610"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config() -> dict:
    return load_config(CONFIG_NAME)


def _root() -> Path:
    return project_root()


def _geojson_path() -> Path:
    return _root() / _config()["geojson_path"]


def _provenance() -> dict:
    return yaml.safe_load((_root() / _config()["provenance_path"]).read_text(encoding="utf-8"))


def _document() -> dict:
    return json.loads(_geojson_path().read_text(encoding="utf-8"))


def _polygon_wgs84() -> Polygon:
    document = _document()
    return shapely.geometry.shape(document["features"][0]["geometry"])


def _polygon_utm() -> Polygon:
    return reproject_geometry(_polygon_wgs84())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_builder():
    """Import ``scripts/build_corridor_aoi.py`` as a module.

    The scripts directory is not a package, and importing through importlib is
    the pattern the acquisition tests already use, so a script's behaviour can be
    tested without adding a production import path for it.
    """
    path = _root() / "scripts" / "build_corridor_aoi.py"
    spec = importlib.util.spec_from_file_location("build_corridor_aoi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _reach(
    hyriv_id: int,
    next_down: int = 0,
    *,
    upland: float = 100.0,
    ord_stra: int = 2,
    ord_clas: int = 5,
    length: float = 1.0,
    endorheic: int = 0,
) -> Reach:
    return Reach(
        hyriv_id=hyriv_id,
        next_down=next_down,
        upland_skm=upland,
        ord_stra=ord_stra,
        ord_clas=ord_clas,
        endorheic=endorheic,
        length_km=length,
        dis_av_cms=10.0,
    )


# ---------------------------------------------------------------------------
# 1. Artifact schema and geometry
# ---------------------------------------------------------------------------


class TestArtifactGeometry:
    def test_the_artifact_is_a_single_polygon_feature_collection(self):
        """The AOI contract accepts exactly one polygon and nothing weaker."""
        document = _document()
        assert document["type"] == "FeatureCollection"
        assert len(document["features"]) == 1
        geometry = document["features"][0]["geometry"]
        assert geometry["type"] == "Polygon"
        rings = geometry["coordinates"]
        assert len(rings) == 1, "a corridor has no holes; a hole would need explaining"
        ring = rings[0]
        assert len(ring) >= 4
        assert ring[0] == ring[-1], "the exterior ring must be closed"

    def test_coordinates_are_longitude_latitude_in_range(self):
        """Stored as EPSG:4326, which is the only CRS the loader accepts."""
        for lon, lat in _polygon_wgs84().exterior.coords:
            assert 80.0 <= lon <= 90.0, lon
            assert 26.0 <= lat <= 30.0, lat

    def test_geometry_is_valid_and_the_record_says_so(self):
        polygon = _polygon_wgs84()
        assert polygon.is_valid
        record = _provenance()["geometry"]
        assert record["is_valid"] is True
        assert record["validity_report"] == "Valid Geometry"

    def test_area_matches_the_record(self):
        """Recomputed in metres, not read back from the file it is checking."""
        area_km2 = _polygon_utm().area / 1e6
        recorded = _provenance()["geometry"]["area_km2"]
        assert area_km2 == pytest.approx(recorded, rel=1e-6)

    def test_bounds_match_the_record(self):
        bounds = [round(v, 7) for v in _polygon_wgs84().bounds]
        assert bounds == _provenance()["geometry"]["bounds_4326"]

    def test_vertex_count_matches_the_record(self):
        coords = len(list(_polygon_wgs84().exterior.coords))
        assert coords == _provenance()["geometry"]["vertex_count"]

    def test_the_polygon_is_not_the_bounding_box(self):
        """The envelope would be several times the corridor and is never a proxy.

        This is the failure `docs/dataset-registry.md` §1.6.2 records: a bbox can
        both falsely accept and falsely reject a product, so substituting one
        must not look like a corridor.
        """
        polygon = _polygon_wgs84()
        envelope = box(*polygon.bounds)
        assert envelope.area > 3.0 * polygon.area

    def test_area_is_consistent_with_the_declared_buffer_and_length(self):
        """A flat-capped corridor of half-width w and length L is ~2wL.

        The tolerance is wide because the ratio is a sanity check on the
        construction order (buffer in metres, not degrees), not a measurement:
        self-overlap at tight bends makes the true polygon smaller than 2wL, and
        buffering in degrees would be off by a factor of ~1.1 in longitude.
        """
        record = _provenance()["geography"]["reach_selection"]
        half_width_m = float(record["buffer_m"])
        length_m = float(record["centreline_length_m"])
        expected = 2.0 * half_width_m * length_m / 1e6
        assert _polygon_utm().area / 1e6 == pytest.approx(expected, rel=0.10)

    def test_geometry_carries_no_wgs84_degree_scale_buffer(self):
        """Guards the specific error of buffering before reprojecting.

        A 1 km buffer applied in degrees would widen the corridor to ~1 degree,
        which the bounds check above would catch; this states the reason.
        """
        polygon = _polygon_wgs84()
        width_deg = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1])
        assert width_deg < 0.5


# ---------------------------------------------------------------------------
# 2. Provenance record
# ---------------------------------------------------------------------------


class TestProvenanceRecord:
    def test_record_identifies_itself_and_its_generator(self):
        record = _provenance()
        assert record["artifact_type"] == "study_area_definition"
        assert record["generator"] == "scripts/build_corridor_aoi.py"
        assert record["code_version"], "a rebuild must be attributable to a revision"
        assert record["created_utc"]

    def test_source_is_identified_with_version_and_tile(self):
        source = _provenance()["source"]
        assert source["product"] == "HydroRIVERS"
        assert source["version"] == "1.0"
        assert source["tile"] == "HydroRIVERS_v10_as"
        assert source["delivery_crs"] == DELIVERY_CRS
        assert source["layer"] == "HydroRIVERS_v10_as"
        # The local file is hashed, so a silently replaced source is detectable.
        assert len(source["local_sha256"]) == 64
        assert source["local_part_count"] > 0

    def test_licence_and_citation_are_recorded_verbatim(self):
        """Required by the HydroSHEDS licence (§4.1) and §4.2's citation request."""
        source = _provenance()["source"]
        assert "free for commercial and non-commercial use" in source["licence"]
        assert "HydroSHEDS" in source["licence"]
        assert "Lehner" in source["citation"]
        assert "2013" in source["citation"]
        assert "2171-2186" in source["citation"]

    def test_reference_point_is_recorded_with_its_evidence(self):
        reference = _provenance()["geography"]["reference"]
        assert reference["lat"] == pytest.approx(28.27778, abs=1e-9)
        assert reference["lon"] == pytest.approx(85.37778, abs=1e-9)
        assert "Rasuwagadhi" in reference["name"] or "Rasuwa" in reference["name"]
        assert reference["source"], "a coordinate without a source is not evidence"
        assert reference["uncertainty_m"] > 0

    def test_selected_reaches_are_recorded_and_internally_consistent(self):
        selection = _provenance()["geography"]["reach_selection"]
        reaches = selection["selected_reaches"]
        assert reaches["reach_count"] == len(reaches["reach_ids"])
        assert reaches["reach_count"] > 1
        assert reaches["start_reach_id"] == reaches["reach_ids"][0]
        assert reaches["end_reach_id"] == reaches["reach_ids"][-1]
        assert selection["terminus_reach_id"] == reaches["end_reach_id"]
        # Every id is an Asia-region HydroRIVERS identifier (leading digit 4).
        assert all(str(hyriv_id).startswith("4") for hyriv_id in reaches["reach_ids"])
        assert reaches["endorheic_reaches"] == []
        assert reaches["non_asia_region_reaches"] == []

    def test_reach_selection_is_a_downstream_sequence(self):
        """Consecutive ids must be linked by NEXT_DOWN, not merely listed.

        Checked against the source when it is present; without it, the record
        itself must at least state the rule and the stop reason.
        """
        selection = _provenance()["geography"]["reach_selection"]
        assert "downstream" in selection["selection_rule"].lower()
        assert selection["downstream_stop_reason"] in {
            "no_downstream_connection",
            "caller_stop_reach",
        }

    def test_construction_parameters_match_the_configuration(self):
        """The record and the config must not drift apart."""
        config = _config()["construction"]
        record = _provenance()["geography"]
        assert record["reach_selection"]["buffer_m"] == float(config["buffer_m"])
        assert record["reach_selection"]["cap_style"] == config["cap_style"]
        assert record["reach_selection"]["join_style"] == config["join_style"]
        assert record["terminus_rule"] == config["terminus_rule"]
        assert record["buffer"]["quad_segs"] == config["quad_segs"]
        assert _provenance()["conventions"]["crs_construction"] == config["study_crs"]
        assert _provenance()["conventions"]["crs_delivery"] == config["delivery_crs"]

    def test_buffer_sensitivity_is_recorded_for_every_configured_width(self):
        """AGENTS.md §9: a threshold that matters must be tested, not asserted."""
        widths = _config()["construction"]["buffer_sensitivity_m"]
        sensitivity = _provenance()["geography"]["buffer"]["sensitivity"]
        assert [entry["buffer_m"] for entry in sensitivity] == [float(w) for w in widths]
        for entry in sensitivity:
            assert entry["status"] in {"constructed", "no_fully_covered_corridor"}
            if entry["status"] == "constructed":
                assert entry["area_km2"] > 0
                assert entry["terminus_reach_id"]
            else:
                assert entry["reason"], "a refusal must say why, not just fail"

    def test_the_primary_buffer_is_one_of_the_tested_widths(self):
        record = _provenance()["geography"]
        tested = [entry["buffer_m"] for entry in record["buffer"]["sensitivity"]]
        assert float(record["reach_selection"]["buffer_m"]) in tested

    def test_governance_states_that_hydrorivers_is_not_an_inference_input(self):
        governance = _provenance()["governance"]
        assert "inference input" in governance["permitted_use"]
        assert "AGENTS.md" in governance["permitted_use"]
        assert "project-defined" in governance["not_a_competition_boundary"].lower()
        assert "does not supply" in governance["not_a_competition_boundary"].lower()

    def test_all_mandatory_attributions_are_carried(self):
        attributions = _provenance()["governance"]["attributions"]
        for required in REQUIRED_ATTRIBUTIONS:
            assert required in attributions
        assert any("HydroRIVERS" in line or "Lehner" in line for line in attributions)

    def test_limitations_are_recorded_not_omitted(self):
        """AGENTS.md §17: a limitation that affects the result must be stated."""
        limitations = _provenance()["limitations"]
        assert len(limitations) >= 4
        joined = " ".join(limitations).lower()
        assert "tributar" in joined, "main-stem-only exclusion must be disclosed"
        assert "coverage" in joined, "the instrument-bounded terminus must be disclosed"
        assert "460 m" in joined or "coarse" in joined


# ---------------------------------------------------------------------------
# 3. Hash pin and loading through the production interface
# ---------------------------------------------------------------------------


class TestHashPin:
    def test_recorded_digest_is_the_digest_of_the_geojson(self):
        record = _provenance()
        digest = _sha256(_geojson_path())
        assert record["geometry"]["sha256"] == digest
        assert record["aoi_id"].endswith(digest[:8])

    def test_recorded_byte_length_matches_the_file(self):
        assert _provenance()["geometry"]["bytes"] == _geojson_path().stat().st_size

    def test_the_loader_accepts_the_artifact_with_the_pinned_digest(self):
        digest = _provenance()["geometry"]["sha256"]
        aoi = AreaOfInterest.from_geojson_file(
            "derived-corridor", _geojson_path(), expected_sha256=digest
        )
        assert aoi.source_sha256 == digest
        assert aoi.crs == DELIVERY_CRS
        assert len(aoi.geometry.exterior) == _provenance()["geometry"]["vertex_count"]

    def test_a_swapped_geometry_with_the_old_digest_is_refused(self, tmp_path):
        """The failure this prevents: a different extent under the same name."""
        digest = _provenance()["geometry"]["sha256"]
        swapped = tmp_path / "swapped.geojson"
        swapped.write_text(
            '{"type": "Polygon", "coordinates": [[[85.0,28.0],[85.1,28.0],'
            '[85.1,28.1],[85.0,28.1],[85.0,28.0]]]}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="was pinned"):
            AreaOfInterest.from_geojson_file("swapped", swapped, expected_sha256=digest)

    def test_the_old_artifact_is_untouched(self):
        """Replacing the study extent must not rewrite the evidence it replaced."""
        assert (_root() / OLD_AOI).is_file()
        assert _sha256(_root() / OLD_AOI) == OLD_AOI_SHA256
        recorded = _provenance()["comparison_to_replaced_artifact"]
        assert recorded["sha256"] == OLD_AOI_SHA256

    def test_the_two_artifacts_are_different_files_with_different_bytes(self):
        assert _geojson_path().resolve() != (_root() / OLD_AOI).resolve()
        assert _sha256(_geojson_path()) != OLD_AOI_SHA256


class TestUpstreamTerminal:
    def test_the_reference_point_lies_on_the_corridor_boundary(self):
        """The upstream end is a flat cap through the documented reference.

        The reference projects onto the main stem; the cap passes through that
        projection perpendicular to the river, so the documented place is on the
        boundary and nothing upstream of it is inside the AOI. Allowing a round
        cap here would have added half a disc (~1.57 km^2) of unclaimed upstream
        area at the border.
        """
        reference = _provenance()["geography"]["reference"]
        point = Point(reference["lon"], reference["lat"])
        polygon = _polygon_wgs84()
        distance_m = polygon.exterior.distance(point) * 111_320.0
        assert distance_m < 1.0, f"reference is {distance_m:.2f} m from the AOI boundary"
        assert not polygon.contains(point), "a boundary point is not strictly contained"

    def test_the_reported_offset_from_the_river_is_plausible(self):
        """The reference must sit on the reach it was matched to.

        A large offset would mean the reference was matched to the wrong reach and
        the corridor starts somewhere the evidence does not support. The bound is
        the published site uncertainty, not a tolerance in the match: at this
        location the published coordinate falls on the mapped stem to within a
        centimetre, which is recorded rather than relied on.
        """
        reference = _provenance()["geography"]["reference"]
        selection = _provenance()["geography"]["reach_selection"]
        offset_m = float(selection["reference_offset_m"])
        assert 0.0 <= offset_m < reference["uncertainty_m"]


class TestReplacementComparison:
    def test_the_derived_corridor_overlaps_the_artifact_it_replaces(self):
        """The replacement must describe the same reach, not merely rename it.

        A low overlap would mean the hydrographic source put the corridor
        somewhere else entirely; the comparison is the evidence that the derived
        extent is a documented version of the same corridor.
        """
        comparison = _provenance()["comparison_to_replaced_artifact"]
        assert comparison["path"] == OLD_AOI.as_posix()
        # The replaced artifact's area is re-derived from its own bytes, so the
        # comparison is against the file rather than against a remembered number.
        assert comparison["area_km2"] == pytest.approx(132.9284, rel=1e-6)
        assert comparison["jaccard"] > 0.7
        assert comparison["intersection_km2"] > 0.9 * min(
            comparison["area_km2"], _provenance()["geometry"]["area_km2"]
        )


# ---------------------------------------------------------------------------
# 4. Authorisation: geometry work does not open the acquisition gate
# ---------------------------------------------------------------------------


class TestAuthorisationInvariants:
    def test_the_record_does_not_authorise_a_download(self):
        authorisation = _provenance()["authorisation"]
        assert authorisation["authorises_download"] is False
        assert authorisation["provenance_status"] == "unverified"
        assert authorisation["adoption"] == "not_ratified"
        assert authorisation["exploratory_exception"] == "absent"

    def test_the_configured_status_is_unverified(self):
        assert _config()["provenance_status"] == "unverified"

    def test_loading_the_artifact_does_not_authorise_a_download(self):
        aoi = AreaOfInterest.from_geojson_file("derived-corridor", _geojson_path())
        assert aoi.provenance_status is AoiProvenanceStatus.UNVERIFIED
        assert aoi.authorises_download is False

    def test_verified_status_is_what_opens_the_gate_not_valid_geometry(self):
        """States the mechanism, so nobody "fixes" the status to unblock a run.

        The same bytes, loaded with a stronger status, do authorise a download.
        That is exactly why the recorded status is a provenance decision and is
        not set by the constructor of this artifact.
        """
        aoi = AreaOfInterest.from_geojson_file(
            "derived-corridor",
            _geojson_path(),
            provenance_status=AoiProvenanceStatus.VERIFIED,
        )
        assert aoi.authorises_download is True
        assert _config()["provenance_status"] == "unverified"

    def test_acquisition_adopts_the_approved_aoi_but_preserves_historical_replacement(self):
        """M6.14 approves the new geometry only; M6.10 remains historical."""
        aoi = load_config("data")["aoi"]
        assert aoi["geojson_path"] == "configs/aoi/trishuli-corridor-hydrorivers-v10-m615.geojson"
        assert aoi["provenance_status"] == "unverified"
        assert _config()["replaces"] == "configs/aoi/trishuli-corridor-UNVERIFIED.geojson"

    def test_acquisition_download_is_still_disabled_in_configuration(self):
        from floodmap.acquisition.config import load_acquisition_config

        config, _ = load_acquisition_config()
        assert config.download.enabled is False
        # The exception exists as an object but is switched off and names no
        # product: an unset exception must not read as a granted one.
        assert config.download.exploratory_authorisation.enabled is False
        assert config.download.exploratory_authorisation.product_ids == ()
        assert config.aoi is not None
        assert config.aoi.authorises_download is False

    def test_the_builder_refuses_to_run_without_the_source(self, tmp_path):
        """A missing source is a blocker, never a silently synthetic corridor."""
        config = json.loads(json.dumps(_config()))
        config["construction"]["source"]["gdb_path"] = str(tmp_path / "absent.gdb")
        with pytest.raises(FileNotFoundError, match="HydroRIVERS source not found"):
            _load_builder().build(config)


# ---------------------------------------------------------------------------
# 5. Raster coverage (needs the delivered products)
# ---------------------------------------------------------------------------


class TestCoverage:
    @staticmethod
    def _footprints() -> list[shapely.Geometry]:
        import rasterio

        geometry = []
        for entry in _provenance()["coverage"]["footprints"]:
            with rasterio.open(_root() / entry["path"]) as dataset:
                geometry.append(
                    reproject_geometry(
                        box(*dataset.bounds),
                        src_crs=dataset.crs.to_string(),
                        dst_crs=DELIVERY_CRS,
                    )
                )
        return geometry

    def test_the_recorded_footprint_digests_match_the_products(self):
        products = _provenance()["coverage"]["footprints"]
        assert len(products) == 2
        for entry in products:
            path = _root() / entry["path"]
            if not path.is_file():
                pytest.skip(f"product raster not present (git-ignored): {entry['path']}")
            assert _sha256(path) == entry["sha256"]

    def test_the_corridor_lies_inside_both_product_footprints(self):
        """The AOI is only usable where both scenes observe it.

        Coverage is asserted as area, not by eye: a small uncovered sliver at the
        downstream terminus would make part of the AOI unmeasured.
        """
        entries = _provenance()["coverage"]["footprints"]
        if not all((_root() / entry["path"]).is_file() for entry in entries):
            pytest.skip("delivered RTC products not present (git-ignored)")
        observed = self._footprints()[0]
        for footprint in self._footprints()[1:]:
            observed = observed.intersection(footprint)
        polygon = _polygon_wgs84()
        covered = polygon.intersection(observed).area / polygon.area
        assert covered == pytest.approx(1.0, abs=1e-9)
        assert _provenance()["coverage"]["corridor_coverage_fraction"] == 1.0

    def test_the_terminus_is_set_by_coverage_and_says_so(self):
        coverage = _provenance()["coverage"]
        assert "product footprints" in coverage["definition"]
        assert coverage["observed_intersection_bounds_4326"]
        selection = _provenance()["geography"]["reach_selection"]
        assert selection["terminus_is_river_end"] is False


# ---------------------------------------------------------------------------
# 6. Reproducibility from the source (needs the geodatabase)
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_rebuilding_reproduces_the_committed_bytes(self):
        """The strongest claim available: the artifact is a function of the source.

        A recorded hash only proves the file has not changed since it was hashed.
        Re-running the builder proves the file follows from the documented inputs,
        which is what makes the extent auditable rather than merely pinned.
        """
        source = _root() / _config()["construction"]["source"]["gdb_path"]
        if not source.exists():
            pytest.skip(f"HydroRIVERS source not present (git-ignored): {source}")
        builder = _load_builder()
        config = _config()
        document, provenance = builder.build(config)
        rebuilt = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        assert hashlib.sha256(rebuilt).hexdigest() == _provenance()["geometry"]["sha256"]
        # The construction inputs, not just the outputs, must agree.
        recorded = _provenance()
        assert provenance["geography"]["reach_selection"] == recorded["geography"]["reach_selection"]
        assert provenance["coverage"]["corridor_coverage_fraction"] == 1.0


# ---------------------------------------------------------------------------
# 7. Module behaviour, on synthetic networks
#
# These do not touch the Trishuli data: they pin the rules that the corridor's
# provenance claims to have followed.
# ---------------------------------------------------------------------------


class TestNetworkSelection:
    def test_upstream_branch_prefers_the_larger_upstream_area(self):
        chosen = choose_upstream_branch([_reach(1, upland=10.0), _reach(2, upland=5.0)])
        assert chosen.hyriv_id == 1

    def test_upstream_branch_breaks_ties_by_strahler_order_then_id(self):
        """Deterministic tie-breaks are what make the choice reproducible."""
        equal_area = [
            _reach(7, upland=5.0, ord_stra=3),
            _reach(9, upland=5.0, ord_stra=4),
        ]
        assert choose_upstream_branch(equal_area).hyriv_id == 9
        all_equal = [_reach(9, upland=5.0, ord_stra=4), _reach(7, upland=5.0, ord_stra=4)]
        assert choose_upstream_branch(all_equal).hyriv_id == 7
        # Order of the input must not matter.
        assert choose_upstream_branch(list(reversed(all_equal))).hyriv_id == 7

    def test_empty_candidate_set_is_refused(self):
        with pytest.raises(ValueError, match="no upstream candidates"):
            choose_upstream_branch([])

    def test_downstream_path_follows_the_trunk_and_ignores_tributaries(self):
        """A Y-shape: the walk must take the merged channel, never a branch."""
        network = HydroRiversNetwork(
            [
                _reach(40000001, 40000003),  # headwater A
                _reach(40000002, 40000003),  # headwater B
                _reach(40000003, 40000004),  # trunk below the confluence
                _reach(40000004, 0),  # trunk end
            ]
        )
        path, reason = network.downstream_path(40000003)
        assert path == [40000003, 40000004]
        assert reason == "no_downstream_connection"

    def test_downstream_path_reports_a_missing_reach_rather_than_stopping(self):
        """An absent reach is not the river's end, and must not be reported as one."""
        network = HydroRiversNetwork([_reach(40000001, 40000002)])
        path, reason = network.downstream_path(40000001)
        assert path == [40000001]
        assert reason == "reach_40000002_absent_from_loaded_set"

    def test_downstream_path_honours_a_caller_stop_reach(self):
        network = HydroRiversNetwork(
            [
                _reach(40000001, 40000002),
                _reach(40000002, 40000003),
                _reach(40000003, 0),
            ]
        )
        path, reason = network.downstream_path(40000001, stop_at=[40000002])
        assert path == [40000001, 40000002]
        assert reason == "caller_stop_reach"

    def test_a_cycle_is_refused_rather_than_truncated(self):
        network = HydroRiversNetwork(
            [_reach(40000001, 40000002), _reach(40000002, 40000001)]
        )
        with pytest.raises(ValueError, match="cycle"):
            network.downstream_path(40000001)

    def test_duplicate_reach_ids_are_refused(self):
        with pytest.raises(ValueError, match="duplicate HYRIV_ID"):
            HydroRiversNetwork([_reach(40000001), _reach(40000001)])

    def test_unknown_seed_is_refused(self):
        network = HydroRiversNetwork([_reach(40000001)])
        with pytest.raises(KeyError, match="not in the loaded network"):
            network.downstream_path(49999999)

    def test_summarised_path_records_area_gain_and_region(self):
        network = HydroRiversNetwork(
            [_reach(40000001, 40000002, upland=10.0, length=2.0),
             _reach(40000002, 0, upland=30.0, length=3.0)]
        )
        record = network.summarise_path([40000001, 40000002]).to_record()
        assert record["reach_count"] == 2
        assert record["length_km"] == 5.0
        assert record["upstream_area_start_skm"] == 10.0
        assert record["upstream_area_end_skm"] == 30.0
        assert record["non_asia_region_reaches"] == []


class TestCorridorConstruction:
    def test_discontinuous_reaches_are_refused(self):
        """A path that does not merge into one line is not a river course."""
        geometries = {
            1: LineString([(85.0, 28.0), (85.1, 28.0)]),
            2: LineString([(85.5, 28.5), (85.6, 28.5)]),
        }
        with pytest.raises(ValueError, match="disconnected parts"):
            build_corridor_line(geometries, [1, 2])

    def test_missing_geometry_is_refused(self):
        with pytest.raises(ValueError, match="no geometry supplied"):
            build_corridor_line({1: LineString([(85.0, 28.0), (85.1, 28.0)])}, [1, 2])

    def test_repeated_vertices_are_dropped(self):
        """A duplicated coordinate carries no shape and breaks flat-capped buffers."""
        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
        cleaned = drop_repeated_vertices(line)
        assert list(cleaned.coords) == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        assert cleaned.length == line.length

    def test_buffer_of_a_line_ending_in_a_repeat_stays_a_single_polygon(self):
        """Regression: a trailing duplicate plus a flat cap produced a MultiPolygon.

        That failure had nothing to do with the corridor and would have been read
        as "the buffer does not fit", sending a future reader after the wrong
        problem.
        """
        line = LineString([(0.0, 0.0), (500.0, 0.0), (500.0, 0.0)])
        polygon = buffer_corridor(line, 100.0, cap_style="flat")
        assert polygon.geom_type == "Polygon"
        assert polygon.is_valid

    def test_buffer_requires_a_positive_distance(self):
        with pytest.raises(ValueError, match="must be positive"):
            buffer_corridor(LineString([(0, 0), (100, 0)]), 0.0)

    def test_flat_caps_end_at_the_terminal_cross_section(self):
        """Round caps would extend one radius past each chosen endpoint."""
        line = LineString([(0.0, 0.0), (1_000.0, 0.0)])
        flat = buffer_corridor(line, 100.0, cap_style="flat")
        round_capped = buffer_corridor(line, 100.0, cap_style="round")
        assert flat.bounds == pytest.approx((0.0, -100.0, 1_000.0, 100.0))
        assert round_capped.bounds[2] == pytest.approx(1_100.0, abs=1e-6)
        assert round_capped.area > flat.area

    def test_trim_keeps_the_downstream_part(self):
        """The cut is at the projected point, not at a nearby vertex."""
        line = LineString([(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)])
        trimmed = trim_line_to_point(line, Point(250.0, 40.0))
        assert trimmed.coords[0] == pytest.approx((250.0, 0.0))
        assert trimmed.coords[-1] == pytest.approx((500.0, 500.0))
        assert trimmed.length == pytest.approx(750.0)

    def test_trim_refuses_a_cut_at_the_upstream_end(self):
        """A no-op trim would let a recorded endpoint differ from the geometry."""
        line = LineString([(0.0, 0.0), (100.0, 0.0)])
        with pytest.raises(ValueError, match="projects to the upstream end"):
            trim_line_to_point(line, Point(0.0, 5.0))

    def test_trim_refuses_a_cut_at_the_last_vertex(self):
        line = LineString([(0.0, 0.0), (100.0, 0.0)])
        with pytest.raises(ValueError, match="nothing would remain"):
            trim_line_to_point(line, Point(100.0, 0.0))

    def test_reprojection_round_trips_within_a_metre(self):
        polygon = box(85.0, 28.0, 85.1, 28.1)
        utm = reproject_geometry(polygon)
        back = reproject_geometry(utm, src_crs=STUDY_AREA_CRS, dst_crs=DELIVERY_CRS)
        assert back.bounds == pytest.approx(polygon.bounds, abs=1e-6)

    def test_selection_requires_a_footprint(self):
        """Without coverage the walk would run the length of the river system."""
        with pytest.raises(ValueError, match="coverage footprint is required"):
            select_coverage_bounded_corridor(
                HydroRiversNetwork([_reach(40000001)]),
                {40000001: LineString([(85.0, 28.0), (85.05, 28.01)])},
                (85.0, 28.0),
                None,
            )

    def test_selection_refuses_a_footprint_too_small_for_the_buffer(self):
        geometry = {40000001: LineString([(85.0, 28.0), (85.05, 28.01)])}
        network = HydroRiversNetwork([_reach(40000001)])
        tiny = box(84.999, 27.999, 85.001, 28.001)
        with pytest.raises(ValueError, match="no reach of the main stem"):
            select_coverage_bounded_corridor(
                network, geometry, (85.005, 28.001), tiny, buffer_m=1_000.0
            )

    def test_a_reference_off_the_network_is_refused(self):
        network = HydroRiversNetwork([_reach(40000001)])
        geometry = {40000001: LineString([(85.0, 28.0), (85.05, 28.01)])}
        with pytest.raises(ValueError, match="no HydroRIVERS reach within"):
            select_coverage_bounded_corridor(
                network, geometry, (86.5, 28.5), box(84.0, 27.0, 87.0, 29.0)
            )

    def test_selection_is_coverage_bounded_and_reports_the_terminus(self):
        """The walk stops at the last *fully covered* reach, not at the river's end.

        Two reaches run east; the footprint ends inside the second. The terminus
        must therefore be the first reach, and the second must not appear in the
        corridor at all — a partially covered reach would put unobserved area
        inside the AOI.
        """
        geometry = {
            40000001: LineString([(85.00, 28.00), (85.03, 28.00)]),
            40000002: LineString([(85.03, 28.00), (85.10, 28.00)]),
        }
        network = HydroRiversNetwork(
            [_reach(40000001, 40000002, length=2.95), _reach(40000002, 0, length=6.9)]
        )
        footprint = box(84.98, 27.99, 85.04, 28.01)
        selection = select_coverage_bounded_corridor(
            network, geometry, (85.015, 28.0), footprint, buffer_m=200.0
        )
        assert selection.terminus_reach_id == 40000001
        assert selection.path.reach_ids == (40000001,)
        assert selection.centreline_length_m < 10_000.0
        assert selection.coverage_fraction == pytest.approx(1.0, abs=1e-9)
        # The corridor starts at the reference, not at the upstream reach end.
        assert selection.upstream_trim_m > 0.0
        assert selection.centreline_length_m == pytest.approx(
            selection.path.length_km * 1_000.0 - selection.upstream_trim_m, rel=0.02
        )
        assert selection.polygon.is_valid
