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
        """Resolved from four lines of evidence, then confirmed on real data.

        No sentence in the paper states the scale, so the status string must keep
        saying that even though the conclusion is now firm. The reference channel
        statistics are pinned because they are the published values -- but see
        `test_reference_statistics_are_not_a_per_raster_gate` for why they are a
        tolerance-banded reference rather than an acceptance test.
        """
        kuro = self._datasets()["kuro_siwo"]
        status = kuro["backscatter_representation_status"]
        assert status.startswith("VERIFIED")
        assert kuro["backscatter_representation"] == "linear"
        stats = kuro["reference_statistics"]
        assert stats["data_mean"] == {"vv": 0.0953, "vh": 0.0264}
        assert stats["data_std"] == {"vv": 0.0427, "vh": 0.0215}
        assert stats["clamp_max"] == 0.15
        # A linear-power mean is order 0.1; a decibel mean would be order -10.
        assert 0.0 < stats["data_mean"]["vv"] < 1.0

    def test_reference_statistics_are_not_a_per_raster_gate(self):
        """Corrected 2026-10-08 against the delivered dataset.

        An earlier revision claimed "a delivered raster whose distribution does
        not match these is not what we think it is". Measuring 150 delivered MS1
        tiles from 3 of 27 train activations gave clipped VV mean 0.1180 against
        a published 0.0953 -- 24% high, from a correct product read correctly.
        A per-raster equality gate would therefore reject valid data.

        The published values stay recorded as a reference for a tolerance-banded
        check over the full training split, and are explicitly NOT usable as our
        fitted normalisation parameters because their pre/post, split and
        aggregation bases are all unstated upstream.
        """
        stats = self._datasets()["kuro_siwo"]["reference_statistics"]
        assert stats["cross_check_is_per_raster_equality_gate"] is False
        assert stats["cross_check_policy"] == "tolerance_banded_over_full_training_split"
        assert stats["usable_as_fitted_normalisation_parameters"] is False
        for basis in ("basis_pre_or_post", "basis_split", "basis_aggregation"):
            assert stats[basis] == "UNSTATED"

    def test_fitted_statistics_come_only_from_the_adapter_training_split(self):
        """The hard leakage invariant, stated at the corpus level."""
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["fitted_statistics_source"] == "adapter_training_split_only"

    def test_delivered_sar_is_not_pre_clipped(self):
        """Verified on delivered data: observed VV max is 901.92, not 0.15.

        The upstream 0.15 clamp is a loader-time operation, not a property of
        the stored raster. An adapter that assumed the clip was already applied
        would train on a distribution thousands of times wider than intended,
        so the clip must be an explicit declared pipeline step.
        """
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["delivered_is_pre_clipped"] is False
        assert kuro["clip_0_15_is_explicit_adapter_operation"] is True
        assert kuro["observed_max_linear"]["vv"] > 0.15

    def test_kuro_siwo_label_nodata_is_stored_but_is_not_a_semantic_class(self):
        """Corrected 2026-10-08. Guards BOTH directions of a real error.

        History: an early revision recorded "3: Invalid pixels" as a stored
        class with ignore_index semantics. Commit 93ab592 removed it, correctly
        reasoning that 3 is not a semantic class -- but then over-corrected by
        declaring the stored value set exhaustive at {0,1,2}.

        The delivered raster disproves that: MK0_MLU stores {0,1,2,3}, value 3
        is 6.48% of label pixels, and info.json declares MK0_MLU.nodata = 3.

        So both readings were wrong in opposite directions. Treating 3 as a
        class trains a phantom category; treating 3 as absent feeds an
        undeclared value into the label tensor. This test pins the narrow truth
        in between: semantic {0,1,2}, stored {0,1,2,3}, nodata 3.
        """
        kuro = self._datasets()["kuro_siwo"]
        # 3 is NOT a semantic class -- the invariant 93ab592 got right.
        assert set(kuro["label_classes"]) == {0, 1, 2}
        assert 3 not in kuro["label_classes"]
        assert kuro["label_semantic_values"] == [0, 1, 2]
        assert kuro["label_nodata_is_a_semantic_class"] is False
        # ... but it IS a stored value, which 93ab592 denied.
        assert kuro["label_stored_values"] == [0, 1, 2, 3]
        assert kuro["label_nodata_value"] == 3
        assert kuro["label_values_are_exhaustive"] is False
        assert 3 in kuro["label_stored_values"]
        assert 3 not in kuro["label_semantic_values"]

    def test_kuro_siwo_validity_mechanisms_are_redundant_not_exclusive(self):
        """Corrected 2026-10-08 against the delivered dataset.

        The contract previously recorded validity as a separate binary raster
        "NOT a label value". Measurement over 120 tile pairs found zero
        disagreement between MLU==3, MNA==0 and SAR==0.0 -- validity is encoded
        three times, redundantly, not one way exclusively.

        The redundancy is kept because it yields three independent integrity
        checks on a delivered tile.
        """
        kuro = self._datasets()["kuro_siwo"]
        mechanisms = kuro["validity_mechanisms"]
        assert set(mechanisms) == {
            "separate_binary_raster",
            "label_nodata_sentinel",
            "sar_nodata_zero",
        }
        assert kuro["validity_mechanisms_are_redundant"] is True
        assert kuro["validity_canonical_source"] == "MK0_MNA == 1"
        assert "0=invalid" in kuro["validity_semantics"]

    def test_kuro_siwo_validity_integrity_assertions_are_declared(self):
        """The three-way agreement must be asserted by the adapter, not assumed."""
        kuro = self._datasets()["kuro_siwo"]
        assertions = " ; ".join(kuro["validity_integrity_assertions"])
        assert "MK0_MLU == 3" in assertions and "MK0_MNA == 0" in assertions
        assert "SAR == 0.0" in assertions
        assert kuro["validity_integrity_verified_on_delivered_data"] is True

    def test_only_the_labelled_partition_is_valid_adapter_input(self):
        """Partition 00 carries no label raster at all.

        Verified on delivered data: partition 01 holds 10 tif + info.json
        including MK0_MLU; partition 00 holds 7 tif + info.json with no label,
        aoiid None, and an "_NA_" filename component. An adapter that globbed
        every info.json would ingest label-free tiles as supervised samples.
        """
        contract = self._datasets()["kuro_siwo"]["partition_contract"]
        assert contract["required_for_supervised_training"] == "LABELLED_01"
        assert contract["rejected_for_supervised_training"] == "UNLABELLED_00"
        assert contract["must_not_glob_all_info_json"] is True
        assert contract["partition_01_discriminators"]["aoiid"] == 1
        assert contract["partition_00_discriminators"]["aoiid"] is None

    def test_catalogue_rows_are_not_an_on_disk_inventory(self):
        """catalogue.gkpg is source-level and lists non-exported grids.

        Verified on all five delivered activations: 3 rows per grid_id, and
        (rows where exported=1) / 3 equals the on-disk grid count exactly. The
        catalogue enumerates more grids than exist, so `exported == 1` is a
        required filter rather than an optional one.
        """
        contract = self._datasets()["kuro_siwo"]["catalogue_contract"]
        assert contract["required_state"] == "exported == 1"
        assert contract["rows_per_grid"] == 3
        assert contract["granularity"].startswith("source_level")
        assert contract["exported_1_iff_on_disk"] is True

    def test_temporal_roles_come_from_metadata_not_filenames(self):
        """info.json declares master/crank; the adapter must read them."""
        kuro = self._datasets()["kuro_siwo"]
        assert "info.json" in kuro["temporal_role_source"]
        assert "master" in kuro["temporal_role_source"]
        assert "crank" in kuro["temporal_role_source"]
        roles = kuro["temporal_roles"]
        assert "MS1" in roles["post"]
        assert "SL1" in roles["pre_rank_1"]
        assert "SL2" in roles["pre_rank_2"]

    def test_sl1_is_the_canonical_pre_event_and_sl2_is_recorded_as_dropped(self):
        """Dropping real data is allowed; dropping it silently is not."""
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["baseline_pre_selection"] == "SL1"
        assert kuro["baseline_dropped_epoch"] == "SL2"
        assert kuro["baseline_dropped_must_be_recorded"] is True

    def test_native_pixel_spacing_is_not_declared_as_true_ground_metres(self):
        """Corrected 2026-10-08. The corpus is not a 10 m ground product.

        Every inspected raster carries a 10.0 x -10.0 EPSG:3857 transform --
        10 PROJECTED units. Web Mercator scales by 1/cos(lat), so measured true
        ground spacing ranges 8.736 m to 9.798 m across the five delivered
        activations and nothing in the corpus is 10 m except at the equator.

        A bare `resolution_m: 10` would have told the adapter a Kuro Siwo pixel
        and a 10 m UTM inference pixel were the same size.
        """
        kuro = self._datasets()["kuro_siwo"]
        assert "resolution_m" not in kuro, "ambiguous key must stay removed"
        assert kuro["source_crs"] == "EPSG:3857"
        assert kuro["source_pixel_spacing_projected_units"] == 10
        assert kuro["source_pixel_spacing_is_true_ground_metres"] is False
        spacing = kuro["true_ground_spacing_m"]
        assert spacing["status"].startswith("PER_ACTIVATION")
        low, high = spacing["measured_range_m"]
        assert low < 10.0 and high < 10.0, "no activation is 10 m true ground"
        for actid, metres in spacing["measured"].items():
            assert 8.0 < metres < 10.0, f"{actid} spacing {metres} out of range"

    def test_bundled_slope_and_dem_are_excluded_from_the_baseline(self):
        """MK0_SLOPE is not degree-valued and the bundled DEM breaks parity.

        Measured slope is radian-like in bulk (p50 0.124) yet 0.001% of pixels
        exceed pi/2 and one tile reached 4344.96, so it is neither degrees nor
        clean radians. M3's dem_slope_degrees declares range (0, 90). The DEM is
        SRTM 1Sec against a production WorldDEM-30. The baseline is SAR-only, so
        both are excluded rather than reconciled.
        """
        kuro = self._datasets()["kuro_siwo"]
        assert kuro["bundled_slope_present"] is True
        assert kuro["bundled_slope_excluded"] is True
        assert kuro["bundled_slope_observed_max"] > 90.0
        assert kuro["bundled_dem_excluded_from_baseline"] is True
        assert "slope_horn_degrees" in kuro["slope_recompute_policy"]

    def test_local_availability_is_recorded_without_altering_the_split(self):
        """Data scarcity is a readiness blocker, never a reason to bend splits.

        3 of 27 train activations and 1 of 7 validation activations are present.
        19 validation tiles from a single activation cannot support threshold or
        checkpoint selection, so that is recorded as insufficient rather than
        quietly accepted.
        """
        kuro = self._datasets()["kuro_siwo"]
        avail = kuro["local_availability"]
        assert avail["official_split_modified"] is False
        assert avail["train_activations_present"] < avail["train_activations_total"]
        assert avail["val_activations_present"] < avail["val_activations_total"]
        assert avail["sufficient_for_production_training_run"] is False
        assert avail["sufficient_for_threshold_selection"] is False
        assert avail["activation_1111012_present"] is False
        assert "BLOCKER" in avail["status"]
        # The upstream split itself must be untouched by any of this.
        verification = kuro["split_verification"]
        assert verification["train_count"] == 27
        assert verification["val_count"] == 7
        assert verification["test_count"] == 10

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
