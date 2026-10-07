"""Configuration loading tests.

Verify that the four configuration files exist, parse, and expose the sections
later milestones will read.

Deliberately NOT tested: whether the configuration *values* are scientifically
correct. Values that are still ``null``/TODO are intentionally not asserted,
because pinning them would mean inventing parameters this milestone leaves open.
Values marked VERIFIED(spec) in ``configs/data.yaml`` **are** asserted, since
they are fixed by the official Track B specification and a silent change to any
of them would alter what the project is required to do.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS
from floodmap.utils.config import (
    EXPECTED_CONFIGS,
    config_path,
    configs_dir,
    load_all_configs,
    load_config,
    missing_configs,
    project_root,
)
from tests.conftest import REPO_ROOT


class TestConfigDiscovery:
    def test_project_root_resolves_to_repository(self):
        assert project_root() == REPO_ROOT

    def test_configs_dir_exists(self):
        assert configs_dir().is_dir()

    def test_no_expected_config_missing(self):
        assert missing_configs() == []


class TestConfigLoading:
    @pytest.mark.parametrize("name", EXPECTED_CONFIGS)
    def test_config_parses_to_mapping(self, name: str):
        config = load_config(name)
        assert isinstance(config, dict) and config

    @pytest.mark.parametrize("name", EXPECTED_CONFIGS)
    def test_config_declares_version(self, name: str):
        assert "version" in load_config(name)

    def test_load_all_configs_returns_every_config(self):
        configs = load_all_configs()
        assert set(configs) == set(EXPECTED_CONFIGS)

    def test_suffix_is_optional(self):
        assert load_config("data") == load_config("data.yaml")

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("does-not-exist")


class TestConfigStructure:
    """Top-level sections each milestone will depend on."""

    def test_data_config_sections(self):
        config = load_config("data")
        for key in ("aoi", "event", "production", "validation_only", "paths"):
            assert key in config, f"configs/data.yaml must define {key}"

    def test_preprocessing_config_sections(self):
        config = load_config("preprocessing")
        for key in (
            "target_grid",
            "alignment_checks",
            "sentinel1",
            "sentinel2",
            "dem",
            "osm",
            "qa",
        ):
            assert key in config, f"configs/preprocessing.yaml must define {key}"

    def test_features_config_sections(self):
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

    def test_segmentation_config_sections(self):
        config = load_config("segmentation")
        for key in ("classes", "model", "features", "training", "inference", "tracking"):
            assert key in config, f"configs/segmentation.yaml must define {key}"

    def test_evaluation_config_sections(self):
        config = load_config("evaluation")
        for key in (
            "splits",
            "leakage_controls",
            "metrics",
            "threshold_selection",
            "error_analysis",
        ):
            assert key in config, f"configs/evaluation.yaml must define {key}"


class TestConfigScientificGuards:
    """Config flags that encode a rule from AGENTS.md rather than a preference.

    These are the settings a future edit could relax without obviously breaking
    anything, so they are pinned here.
    """

    def test_sentinel1_requires_same_relative_orbit(self):
        """AGENTS.md §4 — different tracks must not be compared pixel-by-pixel."""
        s1 = load_config("data")["production"]["sentinel1"]
        assert s1["require_same_relative_orbit"] is True
        assert s1["on_no_same_track_pair"] == "fail"

    def test_post_event_osm_edits_are_disallowed(self):
        """AGENTS.md §3 — post-event OSM is validation-only."""
        assert load_config("data")["production"]["osm"]["allow_post_event_edits"] is False

    def test_validation_only_block_is_disabled_by_default(self):
        assert load_config("data")["validation_only"]["enabled"] is False

    def test_training_datasets_are_exactly_the_permitted_list(self):
        """VERIFIED(spec) — the permitted list is closed.

        This replaces an earlier guard that asserted the list was empty while
        the specification was unavailable. The list is now closed rather than
        unknown, so the guard asserts exact equality: a *new* entry appearing
        here is as much a governance violation as a forbidden one, because
        training on an unlisted dataset breaks AGENTS.md §3.
        """
        td = load_config("data")["production"]["training_datasets"]
        assert td["allowed_training_datasets"] == ["kuro_siwo", "sen1floods11"]
        assert [d["id"] for d in td["datasets"]] == ["kuro_siwo", "sen1floods11"]

    def test_training_datasets_carry_license_and_citation(self):
        """AGENTS.md §18 — datasets must be cited per license and paper.

        Licence is asserted as a *structure*, not a single string. Both
        permitted datasets have a licence problem that a lone identifier would
        erase: Kuro Siwo's LICENSE file says MIT while its README says CC BY,
        and Sen1Floods11 has no licence file at all. The audit recorded each
        declaration separately plus a conservative reading, so these tests pin
        that the discrepancy survives rather than being tidied into a claim.
        """
        for dataset in load_config("data")["production"]["training_datasets"]["datasets"]:
            assert dataset["citation"], f"{dataset['id']} has no citation"
            declared = dataset["license_declared"]
            assert isinstance(declared, dict) and declared, f"{dataset['id']} has no declarations"
            assert dataset["license_conservative"], f"{dataset['id']} has no conservative reading"
            assert dataset["license_status"], f"{dataset['id']} has no licence status"

    def test_license_discrepancies_are_preserved_not_collapsed(self):
        """A single `license:` key would silently assert something unverified."""
        datasets = {
            d["id"]: d for d in load_config("data")["production"]["training_datasets"]["datasets"]
        }

        kuro = datasets["kuro_siwo"]
        assert kuro["license_status"].startswith("DISCREPANCY")
        assert kuro["license_declared"]["repository_license_file"] == "MIT"
        assert "CC BY" in kuro["license_declared"]["repository_readme"]
        # Conservative reading must be the more restrictive one for the data.
        assert "CC BY" in kuro["license_conservative"]

        sen1 = datasets["sen1floods11"]
        assert sen1["license_declared"]["repository_license_file"] == "ABSENT"
        assert sen1["license_status"].startswith("UNVERIFIED")
        assert sen1["redistribution_permitted"] is False

    def test_no_training_dataset_claims_a_debris_class(self):
        """Verified against both corpora's own class definitions.

        Neither Kuro Siwo nor Sen1Floods11 labels debris, sediment or mud. This
        is pinned because flipping either flag to true would license a product
        claim the training labels cannot support (PRD.md FR-06).
        """
        for dataset in load_config("data")["production"]["training_datasets"]["datasets"]:
            assert (
                dataset["has_debris_or_sediment_class"] is False
            ), f"{dataset['id']}: a debris class claim requires label evidence"

    def test_label_class_maps_are_recorded_for_both_datasets(self):
        datasets = {
            d["id"]: d for d in load_config("data")["production"]["training_datasets"]["datasets"]
        }
        kuro = datasets["kuro_siwo"]["label_classes"]
        assert kuro[0] == "No water"
        assert kuro[1] == "Permanent Waters"
        assert kuro[2] == "Floods"
        assert datasets["kuro_siwo"]["separates_permanent_from_flood_water"] is True

        sen1 = datasets["sen1floods11"]["label_classes"]
        assert sen1[0] == "Not Water"
        assert sen1[1] == "Water"
        assert datasets["sen1floods11"]["separates_permanent_from_flood_water"] is False

    def test_backscatter_representations_differ_between_corpora(self):
        """The harmonisation requirement must stay visible in configuration.

        Kuro Siwo is linear power, Sen1Floods11 is decibels. Combining them
        without converting one would train across two unit systems, so the
        mismatch is pinned rather than left to be rediscovered.
        """
        datasets = {
            d["id"]: d for d in load_config("data")["production"]["training_datasets"]["datasets"]
        }
        assert datasets["kuro_siwo"]["backscatter_representation"] == "linear"
        assert datasets["sen1floods11"]["backscatter_representation"] == "decibel"

    def test_upstream_test_activations_are_marked_off_limits(self):
        """Kuro Siwo ships official splits; reusing its test events would leak."""
        kuro = next(
            d
            for d in load_config("data")["production"]["training_datasets"]["datasets"]
            if d["id"] == "kuro_siwo"
        )
        assert kuro["upstream_test_activations_are_offlimits"] is True
        splits = kuro["official_split_activation_ids"]
        assert splits["test"] and splits["val"]
        # The Nepal-labelled activation sits in the upstream test split.
        assert 1111007 in splits["test"]
        assert not set(splits["test"]) & set(splits["val"]), "upstream splits overlap"

    def test_detail_entries_match_the_allowed_list(self):
        """The two representations of the permitted list cannot drift apart."""
        td = load_config("data")["production"]["training_datasets"]
        assert {d["id"] for d in td["datasets"]} == set(td["allowed_training_datasets"])

    def test_event_date_is_the_specified_event(self):
        """VERIFIED(spec) — Trishuli flood event date."""
        assert load_config("data")["event"]["date"] == "2026-08-26"

    def test_dem_source_is_the_specified_product(self):
        """VERIFIED(spec) — Copernicus WorldDEM-30."""
        assert load_config("data")["production"]["dem"]["product"] == "Copernicus WorldDEM-30"

    def test_osm_snapshot_is_the_specified_pre_event_date(self):
        """VERIFIED(spec) — ohsome historical snapshot, 2026-07-27."""
        osm = load_config("data")["production"]["osm"]
        assert osm["snapshot_date"] == "2026-07-27"
        assert osm["snapshot_source"] == "ohsome API"

    def test_osm_snapshot_predates_the_event(self):
        """AGENTS.md §3 hard rule, checked as a date comparison rather than by eye.

        This is the guard that actually matters: the two dates are configured
        independently, so a future edit to either could silently invert them.
        """
        config = load_config("data")
        snapshot = date.fromisoformat(config["production"]["osm"]["snapshot_date"])
        event = date.fromisoformat(config["event"]["date"])
        assert snapshot < event, (
            f"OSM snapshot {snapshot} must predate the event {event}; a snapshot at "
            "or after the event imports post-event knowledge into a production input."
        )

    def test_emsr927_is_marked_prohibited_as_production_input(self):
        """VERIFIED(spec) — EMSR927 is validation-only."""
        sources = load_config("data")["validation_only"]["sources"]
        emsr = next(s for s in sources if s["name"] == "EMSR927")
        assert emsr["prohibited_as_production_input"] is True

    def test_random_pixel_splits_are_forbidden(self):
        """Spatially correlated pixels across splits inflate scores."""
        assert load_config("evaluation")["splits"]["forbid_random_pixel_split"] is True

    def test_himalaya_test_split_is_not_tunable(self):
        """AGENTS.md §5 — do not tune on the final unseen-Himalaya evaluation set."""
        test_split = load_config("evaluation")["splits"]["unseen_himalaya_test"]
        assert test_split["may_be_used_for_tuning"] is False
        assert test_split["may_be_used_for_threshold_selection"] is False
        assert test_split["may_be_used_for_checkpoint_selection"] is False

    def test_threshold_is_selected_on_validation(self):
        assert load_config("evaluation")["threshold_selection"]["selection_split"] == "validation"

    def test_required_metrics_are_present(self):
        """AGENTS.md §6 minimum metric set."""
        required = load_config("evaluation")["metrics"]["required"]
        assert {"iou", "dice_f1", "precision", "recall"}.issubset(set(required))

    def test_no_metric_targets_are_fabricated(self):
        """A target chosen before a baseline exists would be arbitrary."""
        assert load_config("evaluation")["metrics"]["targets"] is None

    def test_no_model_architecture_is_preselected(self):
        """AGENTS.md §5 — model choice must follow measured performance."""
        model = load_config("segmentation")["model"]
        assert model["architecture"] is None
        assert model["candidates"] == []

    def test_normalisation_statistics_come_from_training_only(self):
        normalisation = load_config("segmentation")["features"]["normalisation"]
        assert normalisation["statistics_source"] == "training_split_only"

    def test_optical_invalid_pixels_are_masked_not_interpolated(self):
        """An interpolated cloud gap would be presented as an observation."""
        assert load_config("preprocessing")["sentinel2"]["invalid_pixel_policy"] == "mask"

    def test_alignment_is_verified_not_assumed(self):
        """AGENTS.md §7 — verify CRS, transform, resolution, extent, pixel alignment."""
        checks = load_config("preprocessing")["alignment_checks"]
        for key in (
            "verify_crs",
            "verify_transform",
            "verify_resolution",
            "verify_extent",
            "verify_pixel_alignment",
            "verify_acquisition_geometry",
        ):
            assert checks[key] is True, f"alignment check {key} must be enabled"
        assert checks["on_failure"] == "fail"


class TestAuditedTrainingDatasetFacts:
    """Pin the dataset-audit findings that govern the M4 architecture.

    These are not preferences. Each one was read from a primary source
    (repository file, LICENSE file or paper) and each one, if silently flipped,
    would license a scientific claim the data does not support. The audit and
    its source URLs are in ``docs/dataset-registry.md`` §1.5.
    """

    @staticmethod
    def _datasets():
        return {
            d["id"]: d for d in load_config("data")["production"]["training_datasets"]["datasets"]
        }

    def test_neither_corpus_separates_debris_from_water(self):
        """The decision to narrow the product claim rests on this."""
        for dataset_id, dataset in self._datasets().items():
            assert (
                dataset["has_debris_or_sediment_class"] is False
            ), f"{dataset_id}: a debris class claim requires label evidence"

    def test_kuro_siwo_separates_permanent_water_from_flood_water(self):
        """The main reason Kuro Siwo is usable for a river valley.

        Without this distinction the Trishuli river itself is reported as flood
        on every run (docs/scientific-assumptions.md §7).
        """
        assert self._datasets()["kuro_siwo"]["separates_permanent_from_flood_water"] is True

    def test_sen1floods11_does_not_separate_permanent_water(self):
        """Why a naive union of the two corpora is forbidden."""
        assert self._datasets()["sen1floods11"]["separates_permanent_from_flood_water"] is False

    def test_label_class_maps_are_recorded_verbatim(self):
        kuro = self._datasets()["kuro_siwo"]["label_classes"]
        assert kuro[0] == "No water"
        assert kuro[1] == "Permanent Waters"
        assert kuro[2] == "Floods"
        sen1 = self._datasets()["sen1floods11"]["label_classes"]
        assert sen1[-1] == "No Data / Not Valid"
        assert sen1[0] == "Not Water"
        assert sen1[1] == "Water"

    def test_the_two_corpora_disagree_on_backscatter_representation(self):
        """The reason M3's representation gate is per corpus, not global.

        Kuro Siwo is linear sigma-nought, Sen1Floods11 is decibels. Reading
        either under the other's setting would silently corrupt every SAR
        feature while producing a plausible-looking raster.
        """
        datasets = self._datasets()
        assert datasets["kuro_siwo"]["backscatter_representation"] == "linear"
        assert datasets["sen1floods11"]["backscatter_representation"] == "decibel"

    def test_kuro_siwo_representation_is_verified_by_numeric_evidence(self):
        """Resolved from four independent lines of evidence.

        No sentence in the paper states the scale, so the status string must
        keep saying that even though the conclusion is now firm. The published
        channel statistics are pinned because they are the adapter's
        cross-check: a delivered raster whose distribution does not match them
        is not the product we think it is.
        """
        kuro = self._datasets()["kuro_siwo"]
        status = kuro["backscatter_representation_status"]
        assert status.startswith("VERIFIED")
        assert "not stated in prose" in status
        stats = kuro["published_channel_statistics"]
        assert stats["data_mean"] == {"vv": 0.0953, "vh": 0.0264}
        assert stats["data_std"] == {"vv": 0.0427, "vh": 0.0215}
        assert stats["clamp_max"] == 0.15
        # A linear-power mean is order 0.1; a decibel mean would be order -10.
        assert 0.0 < stats["data_mean"]["vv"] < 1.0

    def test_kuro_siwo_validity_is_a_separate_raster_not_a_label_value(self):
        """Regression guard for a real error this repository made and fixed.

        An earlier revision recorded "3: Invalid pixels" as a stored class with
        ignore_index semantics. It is not: the mask carries {0,1,2} and validity
        is a separate binary raster. An adapter built on the wrong reading would
        filter label==3, match nothing, and silently train on invalid pixels.
        """
        kuro = self._datasets()["kuro_siwo"]
        assert set(kuro["label_classes"]) == {0, 1, 2}
        assert 3 not in kuro["label_classes"]
        assert kuro["label_values_are_exhaustive"] is True
        assert kuro["validity_representation"] == "separate_binary_raster"
        assert "0=invalid" in kuro["validity_semantics"]

    def test_kuro_siwo_flood_definition_is_recorded_as_unpublished(self):
        """The process is documented; the decision rules are not released.

        Pinned so a later reader does not mistake "we read the supplement" for
        "the class boundary is defined". Our Floods semantics inherit an
        undocumented expert judgement, and no external water reference was used.
        """
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["flood_definition"].startswith("UNPUBLISHED")
        known = kuro["flood_definition_known"]
        assert "NONE" in known["permanent_water_reference"]
        assert known["includes_wet_soil_or_submerged_vegetation"] == "UNSTATED"
        assert "NOT MENTIONED" in known["sediment_laden_or_muddy_water"]
        assert "NONE REPORTED" in kuro["stated_label_noise"]

    def test_kuro_siwo_split_verification_is_recorded(self):
        """Splits were checked, not trusted: disjointness and coverage."""
        v = self._datasets()["kuro_siwo"]["split_verification"]
        assert v["pairwise_disjoint"] is True
        assert v["all_split_ids_present_in_catalogue"] is True
        assert v["train_count"] + v["val_count"] + v["test_count"] == v["union_count"]
        # One catalogue activation belongs to no split and must not be used blind.
        assert v["activations_in_no_split"] == [1111012]
        assert v["union_count"] < v["catalogue_activation_id_count"]

    def test_only_sen1floods11_provides_optical_data(self):
        """Why the required corpus cannot supervise optical channels."""
        datasets = self._datasets()
        assert datasets["kuro_siwo"]["optical"] is False
        assert datasets["sen1floods11"]["optical"] is True

    def test_sen1floods11_optical_is_toa_not_surface_reflectance(self):
        """L1C vs our planned L2A is a silent index-level domain shift."""
        assert self._datasets()["sen1floods11"]["optical_processing_level"] == "L1C"

    def test_neither_corpus_bundles_the_production_dem(self):
        """Kuro Siwo ships SRTM; production uses WorldDEM-30."""
        datasets = self._datasets()
        assert datasets["kuro_siwo"]["bundled_dem"] == "SRTM 1Sec"
        assert datasets["sen1floods11"]["bundled_dem"] is None
        production_dem = load_config("data")["production"]["dem"]["product"]
        assert datasets["kuro_siwo"]["bundled_dem"] != production_dem

    def test_upstream_test_activations_are_recorded_and_off_limits(self):
        """Training on an upstream test event voids comparison with published work."""
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["upstream_test_activations_are_offlimits"] is True
        test_ids = kuro["official_split_activation_ids"]["test"]
        val_ids = kuro["official_split_activation_ids"]["val"]
        assert test_ids and val_ids
        assert not set(test_ids) & set(val_ids), "upstream splits must be disjoint"
        # The Nepal-labelled activation sits in the upstream test split.
        assert 1111007 in test_ids

    def test_neither_corpus_may_be_redistributed_from_this_repository(self):
        """Licences are unresolved; committing a sample would be a legal defect."""
        datasets = self._datasets()
        assert datasets["sen1floods11"]["redistribution_permitted"] is False
        assert datasets["kuro_siwo"]["redistribution_permitted"] is None

    def test_sen1floods11_has_no_pre_event_image(self):
        """It therefore cannot supervise any change feature."""
        assert "no pre-event image" in self._datasets()["sen1floods11"]["temporal_structure"]

    def test_kuro_siwo_provides_pre_event_imagery(self):
        """Its triplet is what lets change features be supervised at all."""
        assert "pre-event" in self._datasets()["kuro_siwo"]["temporal_structure"]


class TestValidationOnlyAttributionIsNotGlobal:
    """The EMS attribution must not be attached to production artifacts.

    ``REQUIRED_ATTRIBUTIONS`` is attached to *every* ``ArtifactProvenance``.
    Adding the Copernicus EMS string there would stamp an EMS credit onto
    acquisition manifests, preprocessed rasters and feature stacks that EMS
    never contributed to — which would imply exactly the input relationship
    AGENTS.md §3 forbids. It belongs only on the comparison artifact.
    """

    EMS_ATTRIBUTION = "European Union, Copernicus Emergency Management Service data"

    def test_ems_attribution_is_not_in_the_global_required_set(self):
        for required in REQUIRED_ATTRIBUTIONS:
            assert "Emergency Management" not in required

    def test_ems_attribution_is_not_in_the_config_required_set(self):
        for required in load_config("data")["attribution"]["required"]:
            assert "Emergency Management" not in required


class TestConfigFilesAreValidYaml:
    @pytest.mark.parametrize("name", EXPECTED_CONFIGS)
    def test_file_is_parseable_yaml(self, name: str):
        with config_path(name).open("r", encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), dict)


class TestMandatoryAttribution:
    """AGENTS.md §18 — the three required attribution strings.

    The strings exist in two places by design: ``provenance.py`` enforces them
    on artifacts, ``configs/data.yaml`` exposes them to config consumers. Two
    copies of a legally significant string is a drift hazard, so these tests
    assert byte-equality rather than merely asserting each copy is non-empty.
    """

    def test_config_declares_the_required_attributions(self):
        assert load_config("data")["attribution"]["required"] == list(REQUIRED_ATTRIBUTIONS)

    def test_there_are_exactly_three(self):
        assert len(REQUIRED_ATTRIBUTIONS) == 3

    def test_sentinel_attribution_is_exact(self):
        assert REQUIRED_ATTRIBUTIONS[0] == "Contains modified Copernicus Sentinel data 2026."

    def test_osm_attribution_is_exact(self):
        assert REQUIRED_ATTRIBUTIONS[2] == "© OpenStreetMap contributors."

    def test_worlddem_attribution_retains_both_copyright_marks(self):
        """Regression guard for the known upstream wording discrepancy.

        A restatement of the specification omitted the "©" before "DLR e.V.".
        AGENTS.md §18 and README.md §12 both include it, and dropping a
        copyright mark from a required attribution is a legal defect, so this
        pins the form with both marks present.
        """
        worlddem = REQUIRED_ATTRIBUTIONS[1]
        assert "© DLR e.V. 2010-2014" in worlddem
        assert "© Airbus Defence and Space GmbH 2014-2018" in worlddem
        assert worlddem.endswith("all rights reserved.")

    def test_attribution_matches_agents_md_source_of_truth(self):
        """AGENTS.md §2 is the declared requirement source; stay consistent with it.

        Compared on collapsed whitespace because AGENTS.md hard-wraps the
        blockquote, and with "--" normalised to "-" because markdown prose uses
        the double hyphen for an en-dash.
        """
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        normalised = " ".join(agents.replace("--", "-").replace(">", " ").split())
        for required in REQUIRED_ATTRIBUTIONS:
            assert " ".join(required.split()) in normalised, f"not found in AGENTS.md: {required}"
