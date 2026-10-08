"""Contract-freeze guards for decisions C1-C14.

These tests pin the **contract**, not a future implementation. There is no M4
adapter, no model and no training loop, and nothing here assumes one: every
assertion is against a machine-readable decision in ``configs/`` or against the
typed schema that validates it.

Why a dedicated file. The decisions below were made once, with reasoning
recorded in ``docs/m4-architecture-decision.md`` §10. A future agent must be
able to change one of them deliberately and see exactly what breaks, rather than
discover six months later that an adapter quietly reinterpreted the contract.
Each test therefore states *why* the value matters, not only what it is.

Two kinds of test appear here and both are necessary:

* positive — the frozen value is present and has the frozen semantics;
* negative — a plausible mistake is *rejected*. A contract that cannot reject a
  misspelled key is decoration, which is precisely the defect measured in
  ``configs/preprocessing.yaml`` before 2026-10-08.
"""

from __future__ import annotations

import pytest
import yaml

from floodmap.preprocessing.config import PreprocessingConfig
from floodmap.preprocessing.errors import PreprocessingConfigurationError
from floodmap.utils.config import load_config
from floodmap.utils.contract import (
    ContractConfigurationError,
    M4Contract,
    load_dataset_capabilities,
    load_m4_contract,
)
from tests.conftest import REPO_ROOT

#: Case-insensitive markers for validation-only sources (AGENTS.md §3).
#:
#: Duplicated from tests/test_data_boundary.py on purpose. That guard scans the
#: ``production:`` block; the frozen contract is a new TOP-LEVEL block, which
#: that scan does not reach. Re-importing would couple two independent guards,
#: and a shared helper that silently stopped covering one block is the failure
#: being avoided.
FORBIDDEN_MARKERS = (
    "emsr927",
    "emsr 927",
    "copernicus ems",
    "copernicus_ems",
    "unosat",
    "unitar",
)


def _datasets():
    """Training-dataset detail entries, keyed by id."""
    entries = load_config("data")["production"]["training_datasets"]["datasets"]
    return {entry["id"]: entry for entry in entries}


def _contract_block():
    return load_config("data")["m4_contract"]


# ---------------------------------------------------------------------------
# The block exists, validates, and cannot absorb a typo
# ---------------------------------------------------------------------------


class TestContractBlockIsValidated:
    def test_contract_block_loads_and_validates(self):
        """The freeze is only real if it parses under its own schema."""
        contract = load_m4_contract()
        assert contract.version
        assert contract.frozen_at

    def test_unknown_contract_key_is_rejected(self):
        """A misspelled key must fail, not be ignored.

        This is the whole reason the schema exists. ``configs/data.yaml`` is
        otherwise read as a plain mapping, so ``clip_0_15_enabled`` instead of
        ``clip.enabled`` would leave the contract looking complete and the
        decision unenforced.
        """
        block = dict(_contract_block())
        block["clip_0_15_enabled"] = True
        with pytest.raises(ContractConfigurationError):
            M4Contract.from_mapping(block)

    def test_missing_contract_block_is_an_explicit_failure(self):
        """An absent contract must say so rather than default to something."""
        with pytest.raises(ContractConfigurationError, match="not frozen"):
            load_m4_contract({"production": {}})

    def test_contract_block_names_no_validation_only_source(self):
        """The production-boundary guard does not reach a top-level block.

        tests/test_data_boundary.py scans ``production:``. The frozen contract
        lives at the top level, so it needs its own scan or the boundary would
        have a hole exactly where the newest configuration is.
        """
        dumped = yaml.safe_dump(_contract_block()).lower()
        found = [marker for marker in FORBIDDEN_MARKERS if marker in dumped]
        assert not found, f"validation-only source named in m4_contract: {found}"

    def test_contract_block_is_top_level_not_a_sixth_production_source(self):
        """`production:` is a closed set of five permitted data sources."""
        config = load_config("data")
        assert "m4_contract" in config
        assert "m4_contract" not in config["production"]
        assert set(config["production"]) == {
            "sentinel1",
            "sentinel2",
            "dem",
            "osm",
            "training_datasets",
        }


# ---------------------------------------------------------------------------
# C1 — canonical projected CRS
# ---------------------------------------------------------------------------


class TestC1CanonicalCrs:
    def test_canonical_crs_is_a_metre_based_target_not_the_source(self):
        """The target grid must be metre-based, and must not be the source CRS.

        Inheriting EPSG:3857 would preserve the defect recorded against it: it
        is metre-LABELLED but not metre-TRUE, which is why the corpus's 10
        "metre" pixels measure 8.736-9.798 m of ground.
        """
        crs = load_m4_contract().canonical_crs
        assert crs.units == "metre"
        assert crs.is_target_not_source is True
        assert crs.forbid_geographic_crs_for_canonical_grid is True
        assert "UTM" in crs.family

    def test_no_single_global_epsg_is_pretended_to_exist(self):
        """A per-sample rule, because the corpus is global.

        Kuro Siwo spans roughly -18 to +60 degrees latitude across all
        longitudes. Pinning one projected EPSG would be metre-accurate for a
        minority of the corpus and wrong everywhere else.
        """
        crs = load_m4_contract().canonical_crs
        assert crs.single_global_epsg_chosen is False
        assert crs.single_global_epsg_rationale
        assert "centroid" in crs.zone_rule
        for dataset in ("kuro_siwo", "sen1floods11", "production"):
            assert dataset in crs.zone_sources

    def test_production_epsg_is_derived_not_pinned(self):
        """`aoi` is deliberately null, so a pinned EPSG would be invented.

        The expected zone is recorded so the decision is reproducible, with its
        status marked DERIVED so nobody mistakes it for a frozen value. The
        only candidate geometry on disk has no provenance.
        """
        crs = load_m4_contract().canonical_crs
        assert crs.production_expected_epsg_status.startswith("DERIVED")
        assert load_config("data")["aoi"]["bbox"] is None
        assert load_config("data")["aoi"]["geojson_path"] is None

    def test_source_crs_is_retained_independently_for_both_corpora(self):
        """A target-CRS decision must not overwrite recorded source metadata."""
        datasets = _datasets()
        assert datasets["kuro_siwo"]["source_crs"] == "EPSG:3857"
        assert datasets["sen1floods11"]["source_crs"] == "EPSG:4326"


# ---------------------------------------------------------------------------
# C2 — canonical GSD
# ---------------------------------------------------------------------------


class TestC2CanonicalGsd:
    def test_target_resolution_is_explicit_and_machine_readable(self):
        gsd = load_m4_contract().canonical_gsd
        assert gsd.value == 10.0
        assert gsd.units == "metre"
        assert load_config("preprocessing")["target_grid"]["resolution_m"] == 10

    def test_target_resolution_is_not_claimed_as_any_source_native_gsd(self):
        """The failure this prevents is the one already made and corrected.

        A bare `resolution_m: 10` previously told the adapter that a Kuro Siwo
        pixel and a 10 m UTM pixel were the same size. They are not, and the
        error grows with latitude.
        """
        gsd = load_m4_contract().canonical_gsd
        assert gsd.is_native_resolution_of_any_dataset is False
        preprocessing = load_config("preprocessing")["target_grid"]
        assert preprocessing["resolution_is_native_for_any_source"] is False

    def test_relationship_to_both_source_spacings_is_recorded(self):
        gsd = load_m4_contract().canonical_gsd
        assert "projected" in gsd.relationship_to_kuro_source.lower()
        assert "degree" in gsd.relationship_to_sen1floods11_source.lower()

    def test_the_resolution_harmonisation_cost_is_disclosed(self):
        """Unmeasured activations reach ~60N, where 10 units is ~5 m of ground.

        Coarsening those to 10 m is roughly a 2x loss. Recorded as a cost of
        the decision rather than left to be discovered during training.
        """
        assert load_m4_contract().canonical_gsd.disclosed_cost

    def test_neither_corpus_declares_a_bare_resolution_m(self):
        """The ambiguous key must stay removed from BOTH datasets.

        Sen1Floods11 carried `resolution_m: 10` under EPSG:4326, which is a
        unit error rather than an approximation: a geographic CRS has no metre
        spacing at all.
        """
        for dataset_id, dataset in _datasets().items():
            assert "resolution_m" not in dataset, (
                f"{dataset_id}: ambiguous `resolution_m` must stay removed in "
                "favour of an explicit source spacing with its units"
            )

    def test_source_pixel_spacing_is_never_asserted_as_true_ground_metres(self):
        """Neither corpus is a 10 m true-ground product."""
        datasets = _datasets()
        kuro = datasets["kuro_siwo"]
        assert kuro["source_pixel_spacing_projected_units"] == 10
        assert kuro["source_pixel_spacing_is_true_ground_metres"] is False
        low, high = kuro["true_ground_spacing_m"]["measured_range_m"]
        assert low < 10.0 and high < 10.0

        sen1 = datasets["sen1floods11"]
        assert sen1["source_pixel_spacing_degrees"] == pytest.approx(0.000090)
        assert sen1["source_pixel_spacing_is_true_ground_metres"] is False
        assert sen1["true_ground_spacing_m"]["status"].startswith("PER_CHIP")


# ---------------------------------------------------------------------------
# C3 — resampling
# ---------------------------------------------------------------------------


class TestC3Resampling:
    def test_every_raster_kind_has_its_own_method(self):
        methods = load_m4_contract().resampling.methods
        for kind in (
            "continuous_sar",
            "continuous_optical",
            "categorical_label",
            "binary_validity_mask",
            "dem",
        ):
            assert methods[kind], f"{kind} has no resampling method"

    def test_categorical_data_is_never_interpolated(self):
        """Bilinear on a class label invents classes that were never present.

        A 0/2 boundary interpolates to 1, which in this label scheme is
        "Permanent Waters" — a class the source never asserted at that pixel.
        """
        methods = load_m4_contract().resampling.methods
        assert methods["categorical_label"] == "nearest"
        assert methods["binary_validity_mask"] == "nearest"

    def test_continuous_sar_is_not_resampled_by_nearest_neighbour(self):
        """Nearest on continuous backscatter discards the averaging.

        Under C2 every Kuro activation is coarsened to 10 m, and a coarsened
        radiometric pixel is only meaningful if it averages what it covers.
        """
        assert load_m4_contract().resampling.methods["continuous_sar"] != "nearest"

    def test_a_generic_method_for_all_types_is_forbidden(self):
        resampling = load_m4_contract().resampling
        assert resampling.forbid_generic_method is True
        assert resampling.forbid_default is True
        # No generic fallback key may exist in the shipped config either: a
        # fallback is what lets a new raster kind inherit someone else's method.
        shipped = load_config("preprocessing")["target_grid"]["resampling"]
        assert "continuous" not in shipped
        assert "categorical" not in shipped

    def test_sar_is_resampled_in_linear_power_before_db_conversion(self):
        """Averaging decibels averages logarithms and is biased low."""
        resampling = load_m4_contract().resampling
        assert resampling.sar_resample_domain == "linear_power"
        assert resampling.sar_resample_before_db_conversion is True

    def test_unresolved_resampling_kind_fails_rather_than_defaulting(self):
        config = PreprocessingConfig.model_validate({})
        with pytest.raises(PreprocessingConfigurationError, match="unresolved"):
            config.require_resampling("categorical_label")

    def test_unknown_resampling_kind_is_rejected(self):
        config = PreprocessingConfig.model_validate({})
        with pytest.raises(PreprocessingConfigurationError, match="unknown resampling kind"):
            config.require_resampling("continuous")


# ---------------------------------------------------------------------------
# C4 — speckle
# ---------------------------------------------------------------------------


class TestC4Speckle:
    def test_speckle_decision_is_explicit_on_every_path(self):
        speckle = load_m4_contract().speckle
        assert speckle.applied_by_our_pipeline is False
        for path in ("kuro_siwo", "sen1floods11", "production_sentinel1"):
            assert speckle.source_state[path]

    def test_no_unsupported_parity_claim_is_made(self):
        """Parity is impossible, not merely unimplemented.

        Kuro Siwo is delivered already Lee Sigma filtered and cannot be
        un-filtered, so no setting of our own filter makes the two paths
        identical in texture statistics. Saying otherwise would overstate how
        well any reported number transfers.
        """
        speckle = load_m4_contract().speckle
        assert speckle.parity_established is False
        assert speckle.parity_possible is False
        assert speckle.parity_impossible_reason

    def test_the_residual_mismatch_is_disclosed_and_must_be_ablated(self):
        """A CNN reads texture, and speckle is texture.

        Training on filtered imagery and serving unfiltered imagery is the kind
        of domain shift that produces good validation numbers and poor field
        performance, so it is disclosed and must be quantified.
        """
        speckle = load_m4_contract().speckle
        assert speckle.mismatch_disclosed is True
        assert speckle.ablation_required_before_transfer_claim is True

    def test_speckle_configuration_is_not_silently_discarded(self):
        """The measured defect, now fixed.

        Before 2026-10-08 `speckle_filter` sat under an unmodelled
        `sentinel1.steps` block on a model with `extra="allow"`, so it parsed
        into `__pydantic_extra__` and was thrown away. A speckle decision could
        be written down and have no effect at all.
        """
        config = PreprocessingConfig.model_validate(load_config("preprocessing"))
        assert "steps" in type(config.sentinel1).model_fields
        assert "speckle" in type(config.sentinel1).model_fields
        assert not (config.sentinel1.model_extra or {})
        assert config.sentinel1.steps.speckle_filter is False
        assert config.require_speckle_policy().applied_by_our_pipeline is False
        assert load_m4_contract().speckle.config_can_be_silently_discarded is False

    def test_a_misspelled_speckle_key_is_rejected(self):
        """`speckle_fliter` used to be accepted without complaint."""
        with pytest.raises(PreprocessingConfigurationError):
            PreprocessingConfig.from_mapping({"sentinel1": {"steps": {"speckle_fliter": True}}})

    def test_contradictory_speckle_declarations_are_rejected(self):
        """Two places can express one decision, so they can disagree.

        The dangerous direction is a policy block saying "no filter" beside a
        step flag saying "filter", because the step flag is what a pipeline
        would act on. Fail rather than pick a winner.
        """
        with pytest.raises(PreprocessingConfigurationError, match="contradicts"):
            PreprocessingConfig.from_mapping(
                {
                    "sentinel1": {
                        "steps": {"speckle_filter": True},
                        "speckle": {"applied_by_our_pipeline": False},
                    }
                }
            )

    def test_an_enabled_filter_must_name_its_family_and_site(self):
        """An unreproducible filter is not a decision."""
        with pytest.raises(PreprocessingConfigurationError):
            PreprocessingConfig.from_mapping(
                {"sentinel1": {"speckle": {"applied_by_our_pipeline": True}}}
            )

    def test_an_unstated_speckle_decision_fails_rather_than_defaulting(self):
        """`None` means nobody chose, which is not the same as a chosen false."""
        config = PreprocessingConfig.model_validate({})
        with pytest.raises(PreprocessingConfigurationError, match="C4"):
            config.require_speckle_policy()


# ---------------------------------------------------------------------------
# C5 — statistics artifact
# ---------------------------------------------------------------------------


class TestC5Statistics:
    def test_schema_is_versioned_and_distinguishes_three_kinds(self):
        statistics = load_m4_contract().statistics
        assert statistics.schema_id == "floodmap.statistics/1"
        assert statistics.schema_version
        assert set(statistics.kinds) == {"reference", "fitted", "validation"}

    def test_provenance_can_answer_every_required_question(self):
        """A statistics file whose basis is unknown cannot be audited."""
        fields = load_m4_contract().statistics.provenance_required_fields
        for question in (
            "dataset_id",
            "split",
            "activations",
            "channels",
            "representation",
            "preprocessing_state",
            "clip_state",
            "schema_version",
        ):
            assert question in fields

    def test_reference_statistics_are_not_fitted_parameters(self):
        """Their pre/post, split and aggregation bases are unstated upstream.

        Adopting them as our normalisation parameters would normalise by an
        unknown recipe; using them as a per-raster equality gate would reject
        valid data, since a correct product gave a clipped VV mean 24% above
        the published value on a small, skewed subset.
        """
        reference = load_m4_contract().statistics.reference
        assert reference.usable_as_fitted_normalisation_parameters is False
        assert reference.cross_check_is_per_raster_equality_gate is False
        assert reference.cross_check_policy == "tolerance_banded_over_full_training_split"

    def test_no_fitted_statistics_are_invented(self):
        """The schema is frozen; the values do not exist and are not faked."""
        statistics = load_m4_contract().statistics
        assert statistics.fitted.exists is False
        assert statistics.fitted.path is None
        assert statistics.fitted.fitted_from_validation_or_test is False
        normalisation = load_config("segmentation")["features"]["normalisation"]
        assert normalisation["statistics_path"] is None
        assert normalisation["fitted_statistics_exist"] is False

    def test_normalisation_method_is_explicitly_outside_the_contract(self):
        """Said plainly rather than left as an unexplained null."""
        statistics = load_m4_contract().statistics
        assert statistics.normalisation_method_is_part_of_this_contract is False
        assert statistics.normalisation_method_status.startswith("DEFERRED")
        assert load_config("segmentation")["features"]["normalisation"]["method"] is None
        assert load_config("features")["normalisation"]["method"] is None

    def test_fitted_statistics_come_only_from_the_training_split(self):
        statistics = load_m4_contract().statistics
        assert "train" in statistics.fitted.source_split
        assert (
            load_config("segmentation")["features"]["normalisation"]["statistics_source"]
            == "training_split_only"
        )

    def test_validation_statistics_never_normalise(self):
        validation = load_m4_contract().statistics.validation
        assert validation.may_be_used_for_normalisation is False

    def test_values_file_stays_flat_for_the_existing_reader(self):
        """Forced, not chosen.

        ``floodmap.features.pipeline`` coerces ``float()`` over every value of
        every top-level key in the statistics file, so an embedded provenance
        block would raise rather than be ignored. Hence a sidecar pair.
        """
        statistics = load_m4_contract().statistics
        assert statistics.layout == "sidecar_pair"
        assert statistics.values_file != statistics.provenance_file


# ---------------------------------------------------------------------------
# C6 — threshold / checkpoint selection
# ---------------------------------------------------------------------------


class TestC6ThresholdSelection:
    def test_primary_criterion_is_explicit_and_class_aware(self):
        """Macro averaging would let the two easy classes mask flood failure.

        "No water" dominates by area and "Permanent Waters" is a largely
        static, high-contrast target, so a macro average can rise while flood
        IoU falls.
        """
        selection = load_m4_contract().threshold_selection
        assert selection.primary_criterion == "iou_flood_class"
        assert selection.primary_criterion_class_id == 2
        assert selection.primary_criterion_class_name == "Floods"
        evaluation = load_config("evaluation")["threshold_selection"]
        assert evaluation["selection_criterion"] == "iou_flood_class"
        assert evaluation["selection_criterion_class_id"] == 2

    def test_the_other_metrics_are_still_reported(self):
        """Not selecting on a metric is not the same as not reporting it."""
        reported = load_m4_contract().threshold_selection.reported_but_not_selected_on
        for metric in ("macro_iou", "confusion_matrix", "per_class_iou"):
            assert metric in reported
        metrics = load_config("evaluation")["metrics"]
        assert metrics["per_class"] is True
        assert metrics["confusion_matrix"] is True

    def test_three_split_roles_are_distinguished(self):
        """Training, validation and external validation are different things."""
        selection = load_m4_contract().threshold_selection
        for role in ("training", "validation", "external_validation"):
            assert role in selection.split_roles
        splits = load_config("evaluation")["splits"]
        for role in ("train", "validation", "external_validation", "unseen_himalaya_test"):
            assert role in splits

    def test_selection_happens_on_validation_only(self):
        selection = load_m4_contract().threshold_selection
        assert selection.selection_split_role == "validation"
        assert selection.forbid_selection_on_training is True
        assert selection.forbid_selection_on_test is True
        assert selection.forbid_selection_on_external_validation is True
        assert load_config("evaluation")["threshold_selection"]["selection_split"] == "validation"

    def test_external_validation_is_never_tuned_against(self):
        """It measures transfer, and it could not score this criterion anyway.

        Sen1Floods11 is binary, so it cannot evaluate the flood-vs-
        permanent-water distinction the frozen criterion depends on.
        """
        external = load_config("evaluation")["splits"]["external_validation"]
        assert external["may_be_used_for_tuning"] is False
        assert external["may_be_used_for_threshold_selection"] is False
        assert external["may_be_used_for_checkpoint_selection"] is False
        assert external["scores_flood_vs_permanent_water"] is False

    def test_the_unseen_himalaya_split_is_still_not_tunable(self):
        test_split = load_config("evaluation")["splits"]["unseen_himalaya_test"]
        assert test_split["may_be_used_for_tuning"] is False
        assert test_split["may_be_used_for_threshold_selection"] is False
        assert test_split["may_be_used_for_checkpoint_selection"] is False

    def test_the_19_tile_local_subset_cannot_produce_a_reportable_threshold(self):
        """The criterion is frozen; the value is blocked on data.

        One validation activation cannot separate event-specific behaviour from
        generalisable behaviour. The error being prevented is selecting on 19
        tiles and reporting the number as if it came from the official
        7-activation validation split.
        """
        selection = load_m4_contract().threshold_selection
        assert selection.validation_activations_available_locally == 1
        assert selection.min_validation_activations_for_a_reportable_threshold == 2
        assert selection.value_status.startswith("DEFERRED")
        assert selection.development_only_until_satisfied is True
        assert load_config("evaluation")["threshold_selection"]["search_space"] is None
        assert load_config("segmentation")["inference"]["probability_threshold"] is None

    def test_checkpoint_selection_uses_the_same_criterion_and_split(self):
        checkpoint = load_config("segmentation")["training"]["checkpoint"]
        assert checkpoint["monitor"] == "val_iou_flood_class"
        assert checkpoint["mode"] == "max"
        assert checkpoint["selection_split"] == "validation"
        assert checkpoint["forbid_selection_on_external_validation"] is True
        assert checkpoint["forbid_selection_on_test"] is True
        assert checkpoint["blocked_on_validation_data"] is True

    def test_insufficient_data_cannot_be_declared_sufficient(self):
        """The honest-accounting rule is enforced, not merely written down."""
        block = dict(_contract_block())
        selection = dict(block["threshold_selection"])
        selection["development_only_until_satisfied"] = False
        block["threshold_selection"] = selection
        with pytest.raises(ContractConfigurationError):
            M4Contract.from_mapping(block)


# ---------------------------------------------------------------------------
# C7 — spatial buffer / leakage
# ---------------------------------------------------------------------------


class TestC7SpatialIsolation:
    def test_zero_buffer_is_a_decision_with_a_named_mechanism(self):
        """Not an unset placeholder, and not an invented distance.

        Whole activations are assigned to one split, so no cross-split tile
        adjacency exists for a buffer to act on. A non-zero number here would
        have nothing to do.
        """
        isolation = load_m4_contract().spatial_isolation
        assert isolation.spatial_buffer_m == 0
        assert isolation.additional_buffer_required is False
        assert isolation.buffer_is_a_decision_not_an_unset_value is True
        assert isolation.primary_mechanism == "activation_level_partition"
        assert isolation.mechanism_rationale

        controls = load_config("evaluation")["leakage_controls"]
        assert controls["spatial_buffer_m"] == 0
        assert controls["spatial_isolation_mechanism"] == "activation_level_partition"
        assert controls["spatial_buffer_is_a_decision_not_an_unset_value"] is True

    def test_activation_disjointness_is_asserted(self):
        isolation = load_m4_contract().spatial_isolation
        assert isolation.assert_disjoint_activation_ids is True
        controls = load_config("evaluation")["leakage_controls"]
        assert controls["assert_disjoint_activation_ids"] is True
        assert controls["assert_disjoint_scene_ids"] is True

    def test_sub_activation_splitting_is_forbidden(self):
        """The condition under which a zero buffer stops being defensible.

        A tile-level split inside one activation WOULD create cross-split
        adjacency, at which point a justified buffer becomes mandatory.
        """
        isolation = load_m4_contract().spatial_isolation
        assert isolation.forbid_sub_activation_split is True
        assert isolation.forbid_sub_activation_split_reason
        controls = load_config("evaluation")["leakage_controls"]
        assert controls["forbid_sub_activation_split"] is True
        assert load_config("evaluation")["splits"]["forbid_random_pixel_split"] is True

    def test_zero_buffer_without_the_mechanism_is_rejected(self):
        """The buffer and the mechanism are checked together, not separately."""
        block = dict(_contract_block())
        isolation = dict(block["spatial_isolation"])
        isolation["forbid_sub_activation_split"] = False
        block["spatial_isolation"] = isolation
        with pytest.raises(ContractConfigurationError):
            M4Contract.from_mapping(block)

    def test_the_geographic_gap_is_disclosed_as_pending_not_closed(self):
        """Activation-level splitting isolates events, not geography.

        Two activations over the same basin at different dates could sit in
        different splits. Verifying needs all 45 activation geometries; 5 are
        available locally.
        """
        isolation = load_m4_contract().spatial_isolation
        assert isolation.assert_activation_geometry_disjointness.startswith("PENDING")
        assert isolation.assert_activation_geometry_disjointness_note
        controls = load_config("evaluation")["leakage_controls"]
        assert controls["assert_activation_geometry_disjointness"].startswith("PENDING")


# ---------------------------------------------------------------------------
# C8 — dataset capability model
# ---------------------------------------------------------------------------


class TestC8CapabilityModel:
    def test_both_datasets_declare_a_structured_capability_block(self):
        capabilities = load_dataset_capabilities()
        assert set(capabilities) == {"kuro_siwo", "sen1floods11"}

    def test_the_model_is_not_a_single_boolean_or_a_validity_form_literal(self):
        """Explicitly guards against the stale design.

        An earlier Sen1Floods11 audit proposed one `validity_form` Literal.
        Kuro Siwo encodes validity three redundant ways and Sen1Floods11 one
        way, so a one-of enum cannot represent both.
        """
        contract = load_m4_contract()
        assert contract.capability_model_is_a_single_boolean is False
        assert contract.capability_model_uses_single_validity_form_literal is False
        for dataset in _datasets().values():
            assert isinstance(dataset["validity_mechanisms"], list)

    def test_the_two_corpora_genuinely_differ(self):
        """The point of the model. Equal capabilities would make it pointless."""
        capabilities = load_dataset_capabilities()
        kuro = capabilities["kuro_siwo"]
        sen1 = capabilities["sen1floods11"]
        assert kuro.imagery.pre_event != sen1.imagery.pre_event
        assert kuro.imagery.temporal_pair != sen1.imagery.temporal_pair
        assert kuro.imagery.change_features != sen1.imagery.change_features
        assert kuro.imagery.optical != sen1.imagery.optical
        assert kuro.labels.permanent_water != sen1.labels.permanent_water
        assert (
            kuro.labels.separates_flood_from_permanent_water
            != sen1.labels.separates_flood_from_permanent_water
        )
        assert kuro.labels.multiclass_segmentation != sen1.labels.multiclass_segmentation
        assert kuro.sar_representation.native != sen1.sar_representation.native
        assert load_m4_contract().capability_keys_that_differ

    def test_kuro_siwo_validity_has_three_redundant_mechanisms(self):
        """The redundancy is an asset: three independent integrity checks."""
        kuro = _datasets()["kuro_siwo"]
        assert set(kuro["validity_mechanisms"]) == {
            "separate_binary_raster",
            "label_nodata_sentinel",
            "sar_nodata_zero",
        }
        assert kuro["validity_mechanisms_are_redundant"] is True
        assert kuro["validity_canonical_source"] == "MK0_MNA == 1"

    def test_sen1floods11_validity_has_exactly_one_mechanism(self):
        """It has no separate validity raster and no SAR nodata convention.

        The adapter gets one validity source and must not invent the others.
        """
        sen1 = _datasets()["sen1floods11"]
        assert sen1["validity_mechanisms"] == ["label_nodata_sentinel"]
        assert sen1["validity_mechanisms_are_redundant"] is False
        assert sen1["validity_integrity_assertions"] == []

    def test_a_capability_block_may_be_absent_not_merely_false(self):
        """Kuro Siwo has no optical branch at all.

        `optical_representation` is absent for it and present for
        Sen1Floods11, which is why this is a structured model rather than a
        flat set of booleans.
        """
        capabilities = load_dataset_capabilities()
        assert capabilities["kuro_siwo"].optical_representation is None
        assert capabilities["sen1floods11"].optical_representation is not None

    def test_change_features_cannot_be_declared_without_a_temporal_pair(self):
        """The guard against faking a pre-event image.

        A synthesised counterpart produces a change feature that is identically
        zero or pure noise, and nothing downstream can detect it.
        """
        block = load_config("data")
        datasets = block["production"]["training_datasets"]["datasets"]
        patched = []
        for entry in datasets:
            entry = dict(entry)
            if entry["id"] == "sen1floods11":
                capabilities = dict(entry["capabilities"])
                imagery = dict(capabilities["imagery"])
                imagery["change_features"] = True
                capabilities["imagery"] = imagery
                entry["capabilities"] = capabilities
            patched.append(entry)
        config = {"production": {"training_datasets": {"datasets": patched}}}
        with pytest.raises(ContractConfigurationError, match="change_features"):
            load_dataset_capabilities(config)

    def test_sar_representation_and_sign_must_agree(self):
        """Decibels are signed; linear power is not.

        Getting the pair wrong is how a dB raster gets read as linear, which
        yields a plausible-looking, physically meaningless product.
        """
        capabilities = load_dataset_capabilities()
        kuro = capabilities["kuro_siwo"].sar_representation
        sen1 = capabilities["sen1floods11"].sar_representation
        assert kuro.native == "linear_sigma0"
        assert kuro.native_is_decibel is False
        assert kuro.non_negative is True
        assert sen1.native == "decibel"
        assert sen1.native_is_decibel is True
        assert sen1.non_negative is False


# ---------------------------------------------------------------------------
# C9 — output nodata
# ---------------------------------------------------------------------------


class TestC9OutputNodata:
    def test_output_nodata_is_distinct_from_every_semantic_class(self):
        """A class ID reused as nodata makes every downstream count ambiguous.

        A pixel labelled 0 could mean "we observed dry ground" or "we had
        nothing to say", and no consumer could tell them apart.
        """
        nodata = load_m4_contract().output_nodata
        assert nodata.class_mask_nodata not in nodata.semantic_class_ids
        assert nodata.class_mask_nodata == 255
        assert nodata.forbid_semantic_class_id_as_nodata is True
        output = load_config("segmentation")["output"]
        assert output["class_mask_nodata"] == 255
        assert (
            output["class_mask_nodata"]
            not in load_config("segmentation")["classes"]["semantic_values"]
        )

    def test_output_nodata_is_distinct_from_the_input_label_sentinel(self):
        """Keeps a raster's provenance readable from its values.

        3 is Kuro Siwo's in-band INPUT sentinel. It is not reused for output
        even though it is not a semantic class.
        """
        nodata = load_m4_contract().output_nodata
        assert nodata.input_label_sentinel == 3
        assert nodata.class_mask_nodata != nodata.input_label_sentinel
        assert nodata.distinct_from_input_label_sentinel is True

    def test_continuous_outputs_have_their_own_sentinel(self):
        """-9999.0 is outside the range of every continuous output."""
        nodata = load_m4_contract().output_nodata
        assert nodata.probability_nodata == -9999.0
        assert nodata.continuous_nodata == -9999.0
        assert load_config("segmentation")["output"]["probability_nodata"] == -9999.0

    def test_the_ignore_index_is_the_input_sentinel_not_a_semantic_class(self):
        """The narrow truth between two errors this repository already made."""
        nodata = load_m4_contract().output_nodata
        assert nodata.loss_and_metrics_ignore_index == 3
        assert nodata.loss_and_metrics_ignore_index not in nodata.semantic_class_ids
        assert nodata.loss_and_metrics_ignore_index_is_an_input_concern is True
        classes = load_config("segmentation")["classes"]
        assert classes["ignore_index"] == 3
        assert classes["ignore_index_is_a_semantic_class"] is False

    def test_a_semantic_class_id_as_output_nodata_is_rejected(self):
        block = dict(_contract_block())
        nodata = dict(block["output_nodata"])
        nodata["class_mask_nodata"] = 2
        block["output_nodata"] = nodata
        with pytest.raises(ContractConfigurationError, match="collides"):
            M4Contract.from_mapping(block)

    def test_reusing_the_input_sentinel_as_output_nodata_is_rejected(self):
        block = dict(_contract_block())
        nodata = dict(block["output_nodata"])
        nodata["class_mask_nodata"] = 3
        block["output_nodata"] = nodata
        with pytest.raises(ContractConfigurationError):
            M4Contract.from_mapping(block)


# ---------------------------------------------------------------------------
# C10 — clip site
# ---------------------------------------------------------------------------


class TestC10ClipSite:
    def test_clip_is_explicit_configurable_and_sited_once(self):
        clip = load_m4_contract().clip
        assert clip.enabled is True
        assert clip.configurable is True
        assert clip.max_linear == 0.15
        assert clip.site == "dataset_adapter"
        assert "before linear-to-dB" in clip.site_detail

    def test_clip_is_applied_in_the_linear_domain_before_db_conversion(self):
        """0.15 is the exact published bound; its dB form is rounded.

        Clipping linear-then-converting applies the exact bound. Clipping in dB
        would make the training distribution depend on how many decimals were
        written down.
        """
        clip = load_m4_contract().clip
        assert clip.domain == "linear_sigma0"
        assert clip.applied_before_db_conversion is True
        assert clip.equivalent_db == pytest.approx(-8.2391, abs=1e-4)
        assert clip.equivalent_db_is_recorded_not_applied is True

    def test_raw_source_files_are_never_modified(self):
        """The clip is an adapter/preprocessing transformation."""
        clip = load_m4_contract().clip
        assert clip.raw_source_mutated is False
        assert clip.delivered_is_pre_clipped is False
        assert _datasets()["kuro_siwo"]["delivered_is_pre_clipped"] is False

    def test_the_delivery_is_known_not_to_be_pre_clipped(self):
        """Observed VV max 901.92, far above the 0.15 ceiling.

        An adapter that assumed the clip was already applied would train on a
        distribution thousands of times wider than intended.
        """
        kuro = _datasets()["kuro_siwo"]
        assert kuro["observed_max_linear"]["vv"] > 0.15
        assert kuro["clip_0_15_is_explicit_adapter_operation"] is True

    def test_three_representations_are_named_separately(self):
        """Source, preprocessing and model must not be conflated.

        Conflating them is how a clipped training distribution ends up served
        against an unclipped inference one.
        """
        representations = load_m4_contract().clip.representations
        assert "unclipped" in representations["source"]
        assert "0.15" in representations["preprocessing"]
        assert "decibel" in representations["model"]

    def test_the_clip_applies_to_both_paths_or_neither(self):
        clip = load_m4_contract().clip
        assert clip.both_paths_or_neither is True
        assert set(clip.applies_to_paths) == {"training_adapter", "production_preprocessing"}
        assert clip.production_path_status.startswith("PENDING")

    def test_the_clip_is_not_the_m3_feature_level_clamp(self):
        """M3's clamp stays off; this is a representation parameter."""
        clip = load_m4_contract().clip
        assert clip.is_m3_feature_level_clamp is False
        assert clip.m3_clip_to_valid_range_remains_disabled is True
        assert load_config("features")["numerics"]["clip_to_valid_range"] is False


# ---------------------------------------------------------------------------
# C11 — licensing and redistribution
# ---------------------------------------------------------------------------


class TestC11Redistribution:
    def test_no_dataset_is_redistributed_from_this_repository(self):
        policy = load_m4_contract().redistribution_policy
        assert policy.raw_data_stays_external is True
        assert policy.derived_dataset_copies_committed is False
        assert policy.sample_tiles_committed is False
        assert policy.samples_dir_may_hold_training_corpus_samples is False
        assert policy.docs_may_imply_redistribution is False

    def test_no_licence_is_manufactured_and_both_remain_unresolved(self):
        """Stating "unresolved" is the honest answer, not a gap to fill."""
        policy = load_m4_contract().redistribution_policy
        assert policy.license_manufactured is False
        assert "unresolved" in policy.upstream_license_status["kuro_siwo"].lower()
        assert "UNVERIFIED" in policy.upstream_license_status["sen1floods11"]

    def test_our_policy_is_kept_separate_from_the_unresolved_upstream_fact(self):
        """ "We do not know" and "we have decided not to" are different.

        Kuro Siwo's `redistribution_permitted` stays null because the upstream
        right is genuinely unknown; `repository_redistribution_allowed` records
        our decision, which does not wait for it.
        """
        datasets = _datasets()
        assert datasets["kuro_siwo"]["redistribution_permitted"] is None
        assert datasets["kuro_siwo"]["repository_redistribution_allowed"] is False
        assert datasets["sen1floods11"]["redistribution_permitted"] is False
        assert datasets["sen1floods11"]["repository_redistribution_allowed"] is False

    def test_both_corpora_are_declared_external_to_the_repository(self):
        for dataset_id, dataset in _datasets().items():
            assert dataset["external_to_repository"] is True, dataset_id
            assert dataset["path"] is None, dataset_id

    def test_external_dataset_paths_are_not_repository_paths(self):
        """Config may DESCRIBE an external path; it must not be inside the repo."""
        from pathlib import Path

        local = _datasets()["kuro_siwo"]["local_availability"]["dataset_path"]
        assert not Path(local).is_relative_to(
            REPO_ROOT
        ), f"external dataset path {local!r} is inside the repository"
        policy = load_m4_contract().redistribution_policy
        assert policy.metadata_and_config_may_describe_external_paths is True

    def test_no_training_corpus_raster_is_committed(self):
        """The guard that would actually catch an accident.

        Checks the working tree rather than a flag, because a policy that says
        "no samples committed" while a sample sits in `data/samples/` is worse
        than no policy.
        """
        samples = REPO_ROOT / "data" / "samples"
        if not samples.is_dir():
            return
        offenders = [
            path.name for path in samples.rglob("*") if path.is_file() and path.name != ".gitkeep"
        ]
        assert not offenders, f"training-corpus samples committed: {offenders}"


# ---------------------------------------------------------------------------
# C12 — local subset sufficiency
# ---------------------------------------------------------------------------


class TestC12LocalSubset:
    def test_the_local_corpus_is_recorded_as_incomplete(self):
        subset = load_m4_contract().local_subset
        assert subset.activations_present < subset.activations_catalogued
        assert subset.train_activations_present < subset.train_activations_total
        assert subset.val_activations_present < subset.val_activations_total
        assert "INCOMPLETE" in subset.status

    def test_development_is_permitted_but_claims_are_not(self):
        """Both halves, because the failure is treating one as the other."""
        subset = load_m4_contract().local_subset
        assert subset.development_use_permitted is True
        assert subset.production_training_run_permitted is False
        assert subset.threshold_selection_permitted is False
        assert subset.scientific_model_selection_permitted is False

    def test_the_official_split_is_not_redefined_by_what_is_on_disk(self):
        subset = load_m4_contract().local_subset
        assert subset.official_split_modified is False
        assert subset.official_split_redefined_by_local_availability is False
        assert subset.must_resolve_split_by_official_activation_id is True
        # The upstream counts themselves must be untouched.
        verification = _datasets()["kuro_siwo"]["split_verification"]
        assert verification["train_count"] == 27
        assert verification["val_count"] == 7
        assert verification["test_count"] == 10
        assert verification["catalogue_activation_id_count"] == 45

    def test_an_absent_activation_is_missing_not_unsplit(self):
        """A download gap must never masquerade as a split decision."""
        subset = load_m4_contract().local_subset
        assert subset.absent_activation_is_missing_not_unsplit is True
        assert subset.must_report_missing_activations is True

    def test_train_skew_is_disclosed(self):
        """1111004 alone is 71% of available train tiles, at latitude 29.1."""
        assert "1111004" in load_m4_contract().local_subset.train_tile_skew

    def test_incompleteness_cannot_be_turned_into_permission(self):
        block = dict(_contract_block())
        subset = dict(block["local_subset"])
        subset["production_training_run_permitted"] = True
        block["local_subset"] = subset
        with pytest.raises(ContractConfigurationError, match="incomplete"):
            M4Contract.from_mapping(block)


# ---------------------------------------------------------------------------
# C13 / C14 — adapter contract invariants
# ---------------------------------------------------------------------------


class TestC13KuroAdapterInvariants:
    def test_all_eighteen_invariants_are_recorded(self):
        invariants = _datasets()["kuro_siwo"]["adapter_invariants"]
        assert len(invariants) == 18
        assert load_m4_contract().adapter_invariant_counts.kuro_siwo == 18
        # Numbered, so an omission is visible rather than silent.
        for index, text in enumerate(invariants, start=1):
            assert text.startswith(f"{index}. "), f"invariant {index} is out of order"

    @pytest.mark.parametrize(
        "needle",
        [
            "partition == 01",
            "catalogue.exported == 1",
            "MS1",
            "SL1",
            "SL2",
            "{0,1,2,3}",
            "{0,1,2}",
            "nodata",
            "MK0_MNA == 1",
            "MLU==3",
            "EPSG:3857",
            "transform",
            "10 projected units",
            "MK0_SLOPE",
            "MK0_DEM",
            "linear sigma0",
            "clip",
            "split identity",
        ],
    )
    def test_each_required_subject_is_covered(self, needle: str):
        """Subject coverage, not prose matching.

        Asserted per subject so that deleting any one invariant fails with the
        name of the thing that stopped being enforced.
        """
        invariants = " ;; ".join(_datasets()["kuro_siwo"]["adapter_invariants"])
        assert needle in invariants

    def test_the_invariants_agree_with_the_dataset_facts_they_constrain(self):
        """An invariant that drifts from its facts protects nothing."""
        kuro = _datasets()["kuro_siwo"]
        assert kuro["label_stored_values"] == [0, 1, 2, 3]
        assert kuro["label_semantic_values"] == [0, 1, 2]
        assert kuro["label_nodata_value"] == 3
        assert kuro["label_nodata_is_a_semantic_class"] is False
        assert kuro["validity_canonical_source"] == "MK0_MNA == 1"
        assert kuro["partition_contract"]["required_for_supervised_training"] == "LABELLED_01"
        assert kuro["catalogue_contract"]["required_state"] == "exported == 1"
        assert kuro["baseline_pre_selection"] == "SL1"
        assert kuro["baseline_dropped_epoch"] == "SL2"
        assert kuro["baseline_dropped_must_be_recorded"] is True


class TestC14Sen1FloodsAdapterInvariants:
    def test_all_fifteen_invariants_are_recorded(self):
        invariants = _datasets()["sen1floods11"]["adapter_invariants"]
        assert len(invariants) == 15
        assert load_m4_contract().adapter_invariant_counts.sen1floods11 == 15
        for index, text in enumerate(invariants, start=1):
            assert text.startswith(f"{index}. "), f"invariant {index} is out of order"

    @pytest.mark.parametrize(
        "needle",
        [
            "decibel",
            "VV and VH",
            "10000",
            "binary",
            "-1 is nodata",
            "permanent-water vs flood",
            "post-event only",
            "synthesised",
            "change features",
            "DEM",
            "EPSG:4326",
            "transform",
            "0.000090 degrees",
            "Kuro Siwo semantics",
            "licence",
        ],
    )
    def test_each_required_subject_is_covered(self, needle: str):
        invariants = " ;; ".join(_datasets()["sen1floods11"]["adapter_invariants"])
        assert needle in invariants

    def test_sen1floods11_labels_are_binary_with_a_negative_nodata(self):
        sen1 = _datasets()["sen1floods11"]
        assert sen1["label_is_binary"] is True
        assert sen1["label_semantic_values"] == [0, 1]
        assert sen1["label_stored_values"] == [-1, 0, 1]
        assert sen1["label_nodata_value"] == -1
        assert sen1["label_nodata_is_a_semantic_class"] is False
        assert sen1["label_classes"][-1] == "No Data / Not Valid"

    def test_sen1floods11_has_no_pre_event_or_change_capability(self):
        """It is one post-event acquisition, so neither can be supervised."""
        capabilities = load_dataset_capabilities()["sen1floods11"]
        assert capabilities.imagery.pre_event is False
        assert capabilities.imagery.temporal_pair is False
        assert capabilities.imagery.change_features is False
        assert "no pre-event image" in _datasets()["sen1floods11"]["temporal_structure"]

    def test_sen1floods11_does_not_separate_permanent_water_from_flood(self):
        """The JRC chips are a chip SELECTION, not a label class."""
        sen1 = _datasets()["sen1floods11"]
        assert sen1["separates_permanent_from_flood_water"] is False
        assert sen1["permanent_water_chips_are_a_label_class"] is False
        assert sen1["permanent_water_chips"] == 814
        assert sen1["hand_labelled_flood_chips"] == 446

    def test_forbidden_capability_upgrades_are_machine_readable(self):
        """So the prohibition is testable, not a comment to be remembered."""
        forbidden = " ;; ".join(_datasets()["sen1floods11"]["forbidden_capability_upgrades"])
        for subject in (
            "pre_event",
            "temporal pairs",
            "change detection",
            "permanent-water",
            "BOA",
            "terrain",
            "Kuro-style",
        ):
            assert subject in forbidden

    def test_sen1floods11_is_external_validation_by_default(self):
        """Permission to train on it is not a plan to train on it."""
        sen1 = _datasets()["sen1floods11"]
        assert sen1["role"] == "external_validation_input"
        assert sen1["auxiliary_training_enabled"] is False
        assert sen1["auxiliary_training_permitted_by_specification"] is True


# ---------------------------------------------------------------------------
# Baseline scope — what the freeze does and does not cover
# ---------------------------------------------------------------------------


class TestBaselineScope:
    def test_the_baseline_is_sar_only(self):
        """No DEM and no bundled slope as model input channels."""
        baseline = load_m4_contract().baseline
        assert baseline.modality == "SAR only"
        assert baseline.dem_input is False
        assert baseline.bundled_slope_input is False
        assert baseline.optical_input is False

    def test_bundled_terrain_products_are_excluded_with_reasons(self):
        """Slope is not degree-valued; the DEM breaks train/inference parity.

        MK0_SLOPE's observed maximum is 4344.96, which is impossible as degrees
        and as radians. MK0_DEM is SRTM 1Sec against a production WorldDEM-30.
        """
        kuro = _datasets()["kuro_siwo"]
        assert kuro["bundled_slope_excluded"] is True
        assert kuro["bundled_slope_observed_max"] > 90.0
        assert kuro["bundled_dem_excluded_from_baseline"] is True
        capabilities = load_dataset_capabilities()["kuro_siwo"]
        assert capabilities.terrain.dem_available_to_canonical_sample is False
        assert capabilities.terrain.slope_trusted is False

    def test_no_corpus_union_and_no_debris_claim(self):
        baseline = load_m4_contract().baseline
        assert baseline.corpus_union_permitted is False
        assert baseline.debris_class_claimed is False
        for dataset in _datasets().values():
            assert dataset["has_debris_or_sediment_class"] is False

    def test_the_threshold_stays_outside_the_model(self):
        assert load_m4_contract().baseline.threshold_inside_model is False

    def test_temporal_depth_is_one_pre_plus_one_post(self):
        baseline = load_m4_contract().baseline
        assert baseline.temporal_depth == "1 pre + 1 post"

    def test_the_label_scheme_is_kuro_siwo_native_three_class(self):
        baseline = load_m4_contract().baseline
        assert baseline.label_scheme == "kuro_siwo native 3-class"
        classes = load_config("segmentation")["classes"]
        assert classes["semantic_values"] == [0, 1, 2]
        assert classes["names"] == ["No water", "Permanent Waters", "Floods"]
        assert load_config("segmentation")["model"]["num_classes"] == 3

    def test_a_sar_only_baseline_with_a_dem_channel_is_rejected(self):
        block = dict(_contract_block())
        baseline = dict(block["baseline"])
        baseline["dem_input"] = True
        block["baseline"] = baseline
        with pytest.raises(ContractConfigurationError, match="SAR-only"):
            M4Contract.from_mapping(block)


# ---------------------------------------------------------------------------
# M4 has not been started
# ---------------------------------------------------------------------------


class TestM4IsNotStarted:
    """The freeze encodes decisions; it must not quietly become M4.

    These guards fail if a contract commit starts carrying an implementation.
    """

    def test_the_segmentation_package_holds_no_implementation(self):
        package = REPO_ROOT / "src" / "floodmap" / "segmentation"
        modules = [path.name for path in package.glob("*.py")]
        assert modules == ["__init__.py"], f"M4 implementation must not exist yet, found {modules}"

    def test_no_model_architecture_is_selected(self):
        model = load_config("segmentation")["model"]
        assert model["architecture"] is None
        assert model["encoder"] is None
        assert model["candidates"] == []

    def test_no_training_hyperparameters_are_invented(self):
        training = load_config("segmentation")["training"]
        for key in ("seed", "epochs", "batch_size", "optimizer", "learning_rate", "loss"):
            assert training[key] is None, f"training.{key} must not be guessed"

    def test_the_contract_schema_is_not_an_adapter(self):
        """`utils/contract.py` validates configuration and nothing more."""
        source = (REPO_ROOT / "src" / "floodmap" / "utils" / "contract.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("import torch", "import rasterio", "def train", "DataLoader"):
            assert (
                forbidden not in source
            ), f"the contract schema must contain no implementation: found {forbidden!r}"
