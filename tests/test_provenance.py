"""Provenance schema tests.

Verify that the provenance record is explicit, serializable and round-trips,
and that its two scientific guards actually hold.

These test software behaviour. Nothing here asserts that any real artifact was
produced correctly — no artifact exists yet.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from floodmap.utils.provenance import (
    OSM_ATTRIBUTION,
    REQUIRED_ATTRIBUTIONS,
    SENTINEL_ATTRIBUTION,
    WORLDDEM_ATTRIBUTION,
    ArtifactProvenance,
    ArtifactType,
    BeforeAfterPair,
    ProductionInput,
    SceneReference,
    Unknown,
    UnknownReason,
    ValidationOnlySource,
)


def _minimal() -> ArtifactProvenance:
    return ArtifactProvenance(artifact_type=ArtifactType.SEGMENTATION_PREDICTION)


def _populated() -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_type=ArtifactType.SEGMENTATION_PREDICTION,
        artifact_id="pred-0001",
        aoi="TODO: AOI representation not yet fixed",
        aoi_crs="EPSG:4326",
        event_date="2026-08-15",
        sentinel1=BeforeAfterPair(
            before=SceneReference(
                scene_id="S1_BEFORE",
                acquired_at="2026-08-01T12:00:00+00:00",
                relative_orbit=12,
                orbit_direction="DESCENDING",
                source=ProductionInput.SENTINEL1,
            ),
            after=SceneReference(
                scene_id="S1_AFTER",
                acquired_at="2026-08-20T12:00:00+00:00",
                relative_orbit=12,
                orbit_direction="DESCENDING",
                source=ProductionInput.SENTINEL1,
            ),
            same_relative_orbit=True,
        ),
        dem_version="TODO: product variant unverified",
        osm_snapshot_date="2026-07-01",
        production_inputs=[
            ProductionInput.SENTINEL1,
            ProductionInput.COPERNICUS_DEM,
            ProductionInput.OSM_PRE_EVENT,
        ],
        model_version="unset",
        preprocessing_version="unset",
        config_version="0.1.0",
        code_version="909d77f",
        random_seed=0,
        limitations=["no post-event optical scene available"],
    )


class TestConstruction:
    def test_artifact_type_is_required(self):
        with pytest.raises(ValidationError):
            ArtifactProvenance()

    def test_minimal_record_is_valid(self):
        record = _minimal()
        assert record.artifact_type is ArtifactType.SEGMENTATION_PREDICTION

    def test_generated_at_is_populated_automatically(self):
        assert _minimal().generated_at

    def test_unknown_fields_default_to_none_meaning_not_recorded(self):
        record = _minimal()
        assert record.aoi is None
        assert record.event_date is None

    def test_unexpected_field_is_rejected(self):
        """extra='forbid' keeps records from silently absorbing typos."""
        with pytest.raises(ValidationError):
            ArtifactProvenance(artifact_type=ArtifactType.EVALUATION_RESULT, scene_ids=["x"])


class TestUnknownVersusAbsent:
    """'Not recorded' and 'known to be unavailable' must stay distinguishable.

    Collapsing both into None would let a gap in the record read as a measured
    absence.
    """

    def test_unknown_requires_a_reason(self):
        with pytest.raises(ValidationError):
            Unknown()

    def test_unknown_carries_reason_and_note(self):
        unknown = Unknown(reason=UnknownReason.NOT_AVAILABLE_FROM_SOURCE, note="cloud cover")
        assert unknown.unknown is True
        assert unknown.reason is UnknownReason.NOT_AVAILABLE_FROM_SOURCE
        assert unknown.note == "cloud cover"

    def test_unknown_is_distinguishable_from_none_after_serialization(self):
        explicit = ArtifactProvenance(
            artifact_type=ArtifactType.PREPROCESSED_RASTER,
            event_date=Unknown(reason=UnknownReason.NOT_YET_VERIFIED),
        )
        absent = ArtifactProvenance(artifact_type=ArtifactType.PREPROCESSED_RASTER)

        assert explicit.to_dict()["event_date"] != absent.to_dict()["event_date"]
        assert absent.to_dict()["event_date"] is None
        assert explicit.to_dict()["event_date"]["unknown"] is True

    def test_unknown_survives_round_trip(self):
        record = ArtifactProvenance(
            artifact_type=ArtifactType.PREPROCESSED_RASTER,
            event_date=Unknown(reason=UnknownReason.ACQUISITION_FAILED, note="no scene"),
        )
        restored = ArtifactProvenance.from_yaml(record.to_yaml())
        assert isinstance(restored.event_date, Unknown)
        assert restored.event_date.reason is UnknownReason.ACQUISITION_FAILED


class TestSerialization:
    def test_to_dict_is_json_serializable(self):
        json.dumps(_populated().to_dict())

    def test_json_round_trip_is_lossless(self):
        record = _populated()
        assert ArtifactProvenance.from_json(record.to_json()) == record

    def test_yaml_round_trip_is_lossless(self):
        record = _populated()
        assert ArtifactProvenance.from_yaml(record.to_yaml()) == record

    def test_dict_round_trip_is_lossless(self):
        record = _populated()
        assert ArtifactProvenance.from_dict(record.to_dict()) == record

    def test_yaml_output_is_plain_mapping(self):
        """Serialized form must be readable, not Python-tagged objects."""
        text = _populated().to_yaml()
        assert "!!python" not in text
        assert isinstance(yaml.safe_load(text), dict)

    def test_enums_serialize_as_strings(self):
        data = _populated().to_dict()
        assert data["artifact_type"] == "segmentation_prediction"
        assert data["production_inputs"][0] == "sentinel-1"

    def test_architecture_document_fields_are_present(self):
        """architecture.md §12 names the expected provenance fields."""
        data = _populated().to_dict()
        for field in (
            "artifact_type",
            "aoi",
            "event_date",
            "sentinel1",
            "sentinel2",
            "dem_version",
            "osm_snapshot_date",
            "model_version",
            "preprocessing_version",
            "config_version",
            "generated_at",
        ):
            assert field in data, f"provenance must record {field}"


class TestProductionInputBoundary:
    """AGENTS.md §3 / architecture.md §18 encoded in the type system."""

    def test_production_input_enum_matches_permitted_sources(self):
        assert {s.value for s in ProductionInput} == {
            "sentinel-1",
            "sentinel-2",
            "copernicus-dem",
            "osm-pre-event",
            "permitted-training-dataset",
        }

    def test_validation_only_enum_lists_forbidden_sources(self):
        assert {s.value for s in ValidationOnlySource} == {
            "emsr927",
            "copernicus-ems",
            "unosat",
            "published-damage-map",
            "osm-post-event",
        }

    def test_the_two_enums_are_disjoint(self):
        """A source cannot be both permitted and forbidden."""
        assert not {s.value for s in ProductionInput} & {s.value for s in ValidationOnlySource}

    @pytest.mark.parametrize("forbidden", [s.value for s in ValidationOnlySource])
    def test_validation_only_source_rejected_as_production_input(self, forbidden: str):
        """The realistic defect path: a raw string bypassing the enum."""
        with pytest.raises(ValidationError):
            ArtifactProvenance(
                artifact_type=ArtifactType.SEGMENTATION_PREDICTION,
                production_inputs=[forbidden],
            )

    def test_permitted_sources_are_accepted(self):
        record = ArtifactProvenance(
            artifact_type=ArtifactType.SEGMENTATION_PREDICTION,
            production_inputs=list(ProductionInput),
        )
        assert len(record.production_inputs) == len(ProductionInput)


class TestSameOrbitTrackGuard:
    """AGENTS.md §4 — a same-track claim must be supported by recorded orbits."""

    def test_same_track_claim_requires_known_orbits(self):
        with pytest.raises(ValidationError):
            BeforeAfterPair(
                before=SceneReference(scene_id="a"),
                after=SceneReference(scene_id="b"),
                same_relative_orbit=True,
            )

    def test_same_track_claim_cannot_contradict_recorded_orbits(self):
        with pytest.raises(ValidationError):
            BeforeAfterPair(
                before=SceneReference(scene_id="a", relative_orbit=12),
                after=SceneReference(scene_id="b", relative_orbit=41),
                same_relative_orbit=True,
            )

    def test_same_track_claim_accepted_when_orbits_match(self):
        pair = BeforeAfterPair(
            before=SceneReference(scene_id="a", relative_orbit=12),
            after=SceneReference(scene_id="b", relative_orbit=12),
            same_relative_orbit=True,
        )
        assert pair.same_relative_orbit is True

    def test_differing_orbits_may_be_recorded_as_not_same_track(self):
        """Cross-track is recordable — it just must not be claimed as same-track."""
        pair = BeforeAfterPair(
            before=SceneReference(scene_id="a", relative_orbit=12),
            after=SceneReference(scene_id="b", relative_orbit=41),
            same_relative_orbit=False,
        )
        assert pair.same_relative_orbit is False

    def test_unknown_orbit_relationship_stays_none(self):
        pair = BeforeAfterPair(
            before=SceneReference(scene_id="a"),
            after=SceneReference(scene_id="b"),
        )
        assert pair.same_relative_orbit is None


class TestAutomaticAttribution:
    """AGENTS.md §18 — attribution must be automatic, not a caller's duty.

    The design intent is that it should be *impossible* to produce an artifact
    without the required attributions, including by accident. These tests probe
    the ways a caller could realistically drop them.
    """

    def test_attached_without_being_asked(self):
        assert _minimal().attribution == list(REQUIRED_ATTRIBUTIONS)

    def test_attached_to_a_fully_populated_record(self):
        for required in REQUIRED_ATTRIBUTIONS:
            assert required in _populated().attribution

    def test_survives_an_empty_list(self):
        """An explicit ``[]`` must not defeat the requirement."""
        record = ArtifactProvenance(artifact_type=ArtifactType.SITUATION_REPORT, attribution=[])
        assert record.attribution == list(REQUIRED_ATTRIBUTIONS)

    def test_partial_list_is_completed(self):
        """Passing one required string must not drop the other two."""
        record = ArtifactProvenance(
            artifact_type=ArtifactType.SITUATION_REPORT,
            attribution=[OSM_ATTRIBUTION],
        )
        assert record.attribution == list(REQUIRED_ATTRIBUTIONS)

    def test_extra_citations_are_appended_not_substituted(self):
        """Dataset citations coexist with the required strings (AGENTS.md §18)."""
        citation = "Bountos et al., 2024 (NeurIPS 2024)"
        record = ArtifactProvenance(
            artifact_type=ArtifactType.SEGMENTATION_PREDICTION,
            attribution=[citation],
        )
        assert record.attribution == list(REQUIRED_ATTRIBUTIONS) + [citation]

    def test_order_is_deterministic(self):
        """Required strings lead, in a fixed order, so rendered output is diffable."""
        record = ArtifactProvenance(
            artifact_type=ArtifactType.SITUATION_REPORT,
            attribution=["extra", OSM_ATTRIBUTION, SENTINEL_ATTRIBUTION],
        )
        assert record.attribution[:3] == [
            SENTINEL_ATTRIBUTION,
            WORLDDEM_ATTRIBUTION,
            OSM_ATTRIBUTION,
        ]

    def test_duplicates_are_collapsed(self):
        record = ArtifactProvenance(
            artifact_type=ArtifactType.SITUATION_REPORT,
            attribution=[SENTINEL_ATTRIBUTION, SENTINEL_ATTRIBUTION, "x", "x"],
        )
        assert record.attribution == list(REQUIRED_ATTRIBUTIONS) + ["x"]

    def test_present_in_every_serialization_format(self):
        """Attribution is worthless if it is lost on the way to disk."""
        record = _populated()
        assert record.to_dict()["attribution"] == list(REQUIRED_ATTRIBUTIONS)
        assert SENTINEL_ATTRIBUTION in json.loads(record.to_json())["attribution"]
        assert SENTINEL_ATTRIBUTION in yaml.safe_load(record.to_yaml())["attribution"]

    def test_round_trips(self):
        record = _populated()
        assert ArtifactProvenance.from_json(record.to_json()).attribution == record.attribution
        assert ArtifactProvenance.from_yaml(record.to_yaml()).attribution == record.attribution

    def test_every_artifact_type_carries_attribution(self):
        """No artifact kind is exempt — reports, rasters and manifests alike."""
        for artifact_type in ArtifactType:
            record = ArtifactProvenance(artifact_type=artifact_type)
            assert record.attribution == list(REQUIRED_ATTRIBUTIONS), artifact_type
