"""Typed schema for the frozen M4 dataset contract.

``configs/data.yaml -> m4_contract`` records the decisions C1-C14 that M4 needs
in order to begin: the canonical grid, resampling policy, speckle policy,
statistics artifact schema, selection criterion, leakage policy, output nodata
semantics, clip site, redistribution policy and local-subset policy. Per-dataset
capabilities (C8) and adapter invariants (C13, C14) live beside the dataset
facts they constrain, under
``production.training_datasets.datasets[*]``, and are validated here too.

Why this module exists at all
-----------------------------
``configs/data.yaml`` is otherwise loaded as a plain mapping, which means a
misspelled key is simply a key nobody reads. For a block whose entire purpose is
to be the single authoritative record of a frozen decision, that failure mode is
unacceptable: ``clip_0_15_enabled`` instead of ``enabled`` would leave the
contract looking complete and the decision unenforced. Every model below uses
``extra="forbid"``.

Why it lives in ``utils`` and not ``segmentation``
--------------------------------------------------
This is a *configuration schema*, not M4. It contains no model, no adapter, no
training loop and no inference path. Placing it under
``src/floodmap/segmentation/`` would misrepresent a contract as an
implementation, and the M4 freeze gate explicitly requires that no segmentation
implementation exists yet.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import load_config

__all__ = [
    "AdapterInvariantCounts",
    "BaselineScope",
    "CanonicalCrsConfig",
    "CanonicalGsdConfig",
    "ClipConfig",
    "ContractConfigurationError",
    "DatasetCapabilities",
    "LocalSubsetPolicy",
    "M4Contract",
    "OutputNodataConfig",
    "RedistributionPolicy",
    "ResamplingPolicy",
    "SpatialIsolationPolicy",
    "SpeckleDecision",
    "StatisticsSchema",
    "ThresholdSelectionPolicy",
    "load_dataset_capabilities",
    "load_m4_contract",
]


class ContractConfigurationError(ValueError):
    """Raised when the frozen contract block is malformed or incomplete.

    A distinct type rather than a bare ``ValueError`` so a caller can tell a
    broken contract from an ordinary validation failure: the first means the
    freeze is not trustworthy, the second means one value is wrong.
    """


class _Strict(BaseModel):
    """Base for every contract model: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# C1 / C2 / C3 — canonical grid
# ---------------------------------------------------------------------------


class CanonicalCrsConfig(_Strict):
    """C1. A rule and a family, not one global EPSG code.

    ``single_global_epsg_chosen`` is ``False`` deliberately and is asserted
    rather than merely documented: the training corpus spans roughly -18 to +60
    degrees latitude across all longitudes, and no single projected CRS holds
    metre fidelity over that extent.
    """

    is_target_not_source: bool
    family: str
    units: str
    epsg_pattern: str
    single_global_epsg_chosen: bool
    single_global_epsg_rationale: str
    zone_rule: str
    zone_sources: Mapping[str, str]
    production_expected_epsg: Optional[str] = None
    production_expected_epsg_status: Optional[str] = None
    forbid_geographic_crs_for_canonical_grid: bool

    @model_validator(mode="after")
    def _target_crs_must_be_metre_based(self) -> "CanonicalCrsConfig":
        if self.units != "metre":
            raise ValueError(
                f"canonical_crs.units must be 'metre', got {self.units!r}; "
                "areas, lengths and the slope kernel are meaningless otherwise"
            )
        if not self.is_target_not_source:
            raise ValueError(
                "canonical_crs must be declared as the TARGET grid CRS; source "
                "CRS is recorded per dataset and must not be overwritten by it"
            )
        return self


class CanonicalGsdConfig(_Strict):
    """C2. One target resolution, explicitly native to nothing."""

    value: float
    units: str
    measured_in: str
    is_native_resolution_of_any_dataset: bool
    rationale: str
    relationship_to_kuro_source: str
    relationship_to_sen1floods11_source: str
    disclosed_cost: str

    @model_validator(mode="after")
    def _must_be_positive_metres(self) -> "CanonicalGsdConfig":
        if self.units != "metre":
            raise ValueError(f"canonical_gsd.units must be 'metre', got {self.units!r}")
        if self.value <= 0:
            raise ValueError("canonical_gsd.value must be positive")
        return self


class ResamplingPolicy(_Strict):
    """C3. One method per raster kind, with no generic fallback."""

    forbid_generic_method: bool
    forbid_default: bool
    methods: Mapping[str, str]
    sar_resample_domain: str
    sar_resample_before_db_conversion: bool
    sar_resample_domain_rationale: str
    categorical_rationale: str
    dem_status: str

    #: Kinds that must each carry their own method.
    REQUIRED_KINDS: ClassVar[Tuple[str, ...]] = (
        "continuous_sar",
        "continuous_optical",
        "categorical_label",
        "binary_validity_mask",
        "dem",
    )

    @model_validator(mode="after")
    def _every_kind_is_covered_and_typed(self) -> "ResamplingPolicy":
        missing = [kind for kind in self.REQUIRED_KINDS if kind not in self.methods]
        if missing:
            raise ValueError(
                f"resampling.methods is missing {', '.join(missing)}; every raster "
                "kind must declare its own method"
            )
        # The whole point of C3: categorical data must not be interpolated.
        # Bilinear on a class label produces values that were never in the
        # source -- a 0/2 boundary becomes 1, which is "Permanent Waters".
        for kind in ("categorical_label", "binary_validity_mask"):
            if self.methods[kind] != "nearest":
                raise ValueError(
                    f"resampling.methods.{kind} must be 'nearest', got "
                    f"{self.methods[kind]!r}; interpolating categorical data "
                    "invents classes that were never in the source"
                )
        if self.methods["continuous_sar"] == "nearest":
            raise ValueError(
                "resampling.methods.continuous_sar must not be 'nearest'; that "
                "discards the averaging that makes a coarsened pixel meaningful"
            )
        if self.sar_resample_domain != "linear_power":
            raise ValueError(
                "resampling.sar_resample_domain must be 'linear_power'; averaging "
                "decibels averages logarithms and is biased low"
            )
        return self


# ---------------------------------------------------------------------------
# C4 — speckle
# ---------------------------------------------------------------------------


class SpeckleDecision(_Strict):
    """C4. An explicit decision, including the explicit absence of parity."""

    applied_by_our_pipeline: bool
    filter_family: Optional[str] = None
    parameters: Optional[Mapping[str, Any]] = None
    pipeline_site: Optional[str] = None
    parity_established: bool
    parity_possible: bool
    parity_impossible_reason: Optional[str] = None
    source_state: Mapping[str, str]
    mismatch_disclosed: bool
    ablation_required_before_transfer_claim: bool
    config_can_be_silently_discarded: bool
    config_schema_owner: str

    @model_validator(mode="after")
    def _unestablished_parity_must_be_disclosed(self) -> "SpeckleDecision":
        """No silent mismatch, and no unsupported parity claim.

        The two failure directions are symmetric and both matter. Claiming
        parity that does not exist overstates how well a reported number
        transfers; leaving a known mismatch undisclosed hides it entirely.
        """
        if not self.parity_established and not self.mismatch_disclosed:
            raise ValueError(
                "speckle parity is not established, so mismatch_disclosed must "
                "be true; an undisclosed train/serve texture mismatch is exactly "
                "the failure this block exists to prevent"
            )
        if self.parity_established and not self.parity_possible:
            raise ValueError(
                "speckle.parity_established cannot be true while " "parity_possible is false"
            )
        if self.applied_by_our_pipeline and (
            self.filter_family is None or self.pipeline_site is None
        ):
            raise ValueError(
                "an enabled speckle filter must name its family and its site in "
                "the pipeline, otherwise it is unreproducible"
            )
        if self.config_can_be_silently_discarded:
            raise ValueError(
                "speckle configuration must not be silently discardable; this is "
                "the defect fixed in src/floodmap/preprocessing/config.py"
            )
        return self


# ---------------------------------------------------------------------------
# C5 — statistics
# ---------------------------------------------------------------------------


class _FittedStatistics(_Strict):
    exists: bool
    source_split: str
    fitted_from_validation_or_test: bool
    path: Optional[str] = None

    @model_validator(mode="after")
    def _no_leakage_and_no_phantom_artifact(self) -> "_FittedStatistics":
        if self.fitted_from_validation_or_test:
            raise ValueError(
                "fitted statistics must come from the training split only; "
                "fitting on validation or test data is test-time leakage"
            )
        if not self.exists and self.path is not None:
            raise ValueError(
                "statistics.fitted.exists is false but a path is set; a declared "
                "artifact that does not exist is worse than none"
            )
        return self


class _ReferenceStatistics(_Strict):
    dataset_id: str
    usable_as_fitted_normalisation_parameters: bool
    cross_check_policy: str
    cross_check_is_per_raster_equality_gate: bool

    @model_validator(mode="after")
    def _reference_values_are_not_our_parameters(self) -> "_ReferenceStatistics":
        """Published statistics are a cross-check, not a normalisation source.

        Their pre/post, split and aggregation bases are all unstated upstream,
        so adopting them as fitted parameters would normalise our data by an
        unknown recipe. The per-raster equality gate is separately forbidden:
        measurement on a correct product gave a clipped VV mean 24% above the
        published value, so an equality gate would reject valid data.
        """
        if self.usable_as_fitted_normalisation_parameters:
            raise ValueError(
                "reference statistics must not be declared usable as fitted "
                "normalisation parameters; their pre/post, split and aggregation "
                "bases are unstated upstream"
            )
        if self.cross_check_is_per_raster_equality_gate:
            raise ValueError(
                "reference statistics must not be a per-raster equality gate; a "
                "correct product read correctly does not reproduce them on a "
                "small, geographically skewed subset"
            )
        return self


class _ValidationStatistics(_Strict):
    purpose: str
    may_be_used_for_normalisation: bool

    @model_validator(mode="after")
    def _validation_statistics_never_normalise(self) -> "_ValidationStatistics":
        if self.may_be_used_for_normalisation:
            raise ValueError(
                "validation statistics are for drift reporting only; normalising "
                "with them would leak validation data into the model"
            )
        return self


class StatisticsSchema(_Strict):
    """C5. A versioned schema and three distinct kinds. No invented values."""

    schema_id: str
    schema_version: str
    kinds: Tuple[str, ...]
    layout: str
    values_file: str
    values_shape: str
    values_shape_rationale: str
    provenance_file: str
    provenance_required_fields: Tuple[str, ...]
    fitted: _FittedStatistics
    reference: _ReferenceStatistics
    validation: _ValidationStatistics
    normalisation_method_is_part_of_this_contract: bool
    normalisation_method_status: str

    #: Provenance fields that must be answerable for any statistics artifact.
    REQUIRED_PROVENANCE: ClassVar[Tuple[str, ...]] = (
        "kind",
        "dataset_id",
        "split",
        "activations",
        "channels",
        "representation",
        "preprocessing_state",
        "clip_state",
        "schema_version",
    )

    @model_validator(mode="after")
    def _kinds_and_provenance_are_sufficient(self) -> "StatisticsSchema":
        if set(self.kinds) != {"reference", "fitted", "validation"}:
            raise ValueError(
                "statistics.kinds must distinguish exactly reference, fitted and "
                f"validation, got {list(self.kinds)}"
            )
        missing = [
            field
            for field in self.REQUIRED_PROVENANCE
            if field not in self.provenance_required_fields
        ]
        if missing:
            raise ValueError(
                "statistics.provenance_required_fields cannot answer "
                f"{', '.join(missing)}; a statistics artifact whose dataset, "
                "split, representation or clip state is unknown is unusable"
            )
        return self


# ---------------------------------------------------------------------------
# C6 / C7 — selection and leakage
# ---------------------------------------------------------------------------


class ThresholdSelectionPolicy(_Strict):
    """C6. Criterion frozen; value blocked on data, and said so."""

    primary_criterion: str
    primary_criterion_class_id: int
    primary_criterion_class_name: str
    primary_criterion_rationale: str
    reported_but_not_selected_on: Tuple[str, ...]
    selection_split_role: str
    forbid_selection_on_training: bool
    forbid_selection_on_test: bool
    forbid_selection_on_external_validation: bool
    split_roles: Mapping[str, str]
    min_validation_activations_for_a_reportable_threshold: int
    min_validation_activations_rationale: str
    validation_activations_available_locally: int
    value_status: str
    development_only_until_satisfied: bool

    @model_validator(mode="after")
    def _selection_is_on_validation_and_data_sufficiency_is_honest(
        self,
    ) -> "ThresholdSelectionPolicy":
        if self.selection_split_role != "validation":
            raise ValueError(
                "threshold selection must happen on the validation role only, got "
                f"{self.selection_split_role!r}"
            )
        for role in ("training", "validation", "external_validation"):
            if role not in self.split_roles:
                raise ValueError(f"threshold_selection.split_roles must distinguish {role}")
        if not (self.forbid_selection_on_test and self.forbid_selection_on_external_validation):
            raise ValueError(
                "selection on the test split or on the external-validation corpus "
                "must be forbidden; either would convert a reporting set into a "
                "tuning set"
            )
        # The honest-accounting rule: if the available data cannot meet the
        # stated floor, the contract must say the value is deferred rather than
        # quietly accept a threshold tuned on one event.
        insufficient = (
            self.validation_activations_available_locally
            < self.min_validation_activations_for_a_reportable_threshold
        )
        if insufficient and not self.development_only_until_satisfied:
            raise ValueError(
                "fewer validation activations are available than the stated "
                "minimum, so development_only_until_satisfied must be true; a "
                "threshold tuned on one activation is a property of that event"
            )
        if insufficient and not self.value_status.startswith("DEFERRED"):
            raise ValueError(
                "fewer validation activations are available than the stated "
                "minimum, so value_status must be DEFERRED(...)"
            )
        return self


class SpatialIsolationPolicy(_Strict):
    """C7. A zero buffer, justified by the partition unit rather than guessed."""

    additional_buffer_required: bool
    spatial_buffer_m: float
    buffer_is_a_decision_not_an_unset_value: bool
    primary_mechanism: str
    mechanism_rationale: str
    assert_disjoint_activation_ids: bool
    forbid_sub_activation_split: bool
    forbid_sub_activation_split_reason: str
    assert_activation_geometry_disjointness: str
    assert_activation_geometry_disjointness_note: str

    @model_validator(mode="after")
    def _zero_buffer_needs_a_mechanism(self) -> "SpatialIsolationPolicy":
        """A zero buffer is only defensible while splits are whole activations.

        If sub-activation splitting were ever permitted, tiles from one event
        would straddle splits and a justified buffer distance would become
        mandatory. The two settings are therefore checked together rather than
        left to be noticed later.
        """
        if self.spatial_buffer_m < 0:
            raise ValueError("spatial_buffer_m cannot be negative")
        if self.spatial_buffer_m == 0:
            if not self.assert_disjoint_activation_ids:
                raise ValueError(
                    "a zero spatial buffer requires activation-id disjointness to "
                    "be asserted; otherwise nothing isolates the splits"
                )
            if not self.forbid_sub_activation_split:
                raise ValueError(
                    "a zero spatial buffer requires sub-activation splitting to be "
                    "forbidden; a tile-level split inside one activation would "
                    "create the cross-split adjacency a buffer exists to prevent"
                )
            if not self.buffer_is_a_decision_not_an_unset_value:
                raise ValueError(
                    "a zero spatial buffer must be flagged as a decision, so it "
                    "cannot be mistaken for an unset placeholder"
                )
        return self


# ---------------------------------------------------------------------------
# C9 / C10 — nodata and clipping
# ---------------------------------------------------------------------------


class OutputNodataConfig(_Strict):
    """C9. Output nodata is never a semantic class ID."""

    semantic_class_ids: Tuple[int, ...]
    class_mask_dtype: str
    class_mask_nodata: int
    probability_dtype: str
    probability_nodata: float
    continuous_nodata: float
    forbid_semantic_class_id_as_nodata: bool
    distinct_from_input_label_sentinel: bool
    input_label_sentinel: int
    loss_and_metrics_ignore_index: int
    loss_and_metrics_ignore_index_is_an_input_concern: bool
    rationale: str
    propagated_to_m2_m3_configs: bool
    propagated_to_m2_m3_configs_status: str

    @model_validator(mode="after")
    def _nodata_cannot_collide(self) -> "OutputNodataConfig":
        if self.class_mask_nodata in self.semantic_class_ids:
            raise ValueError(
                f"class_mask_nodata={self.class_mask_nodata} collides with a "
                f"semantic class ID {list(self.semantic_class_ids)}; a pixel would "
                "then be ambiguous between an observation and an absence"
            )
        if self.class_mask_nodata == self.input_label_sentinel:
            raise ValueError(
                "class_mask_nodata must differ from the upstream input label "
                "sentinel, so a raster's provenance stays readable from its values"
            )
        if self.loss_and_metrics_ignore_index in self.semantic_class_ids:
            raise ValueError(
                "loss_and_metrics_ignore_index must not be a semantic class ID; "
                "ignoring a real class would silently drop it from training"
            )
        if self.class_mask_dtype == "uint8" and not 0 <= self.class_mask_nodata <= 255:
            raise ValueError(
                f"class_mask_nodata={self.class_mask_nodata} is outside the uint8 range"
            )
        if not self.forbid_semantic_class_id_as_nodata:
            raise ValueError(
                "forbid_semantic_class_id_as_nodata must be true; this is the "
                "invariant the rest of this block implements"
            )
        return self


class ClipConfig(_Strict):
    """C10. One site, one domain, and the raw files left alone."""

    enabled: bool
    domain: str
    max_linear: float
    equivalent_db: float
    equivalent_db_is_recorded_not_applied: bool
    site: str
    site_detail: str
    applied_before_db_conversion: bool
    raw_source_mutated: bool
    delivered_is_pre_clipped: bool
    configurable: bool
    applies_to_paths: Tuple[str, ...]
    both_paths_or_neither: bool
    production_path_site: str
    production_path_status: str
    is_m3_feature_level_clamp: bool
    m3_clip_to_valid_range_remains_disabled: bool
    representations: Mapping[str, str]

    @model_validator(mode="after")
    def _clip_site_is_unambiguous(self) -> "ClipConfig":
        if self.raw_source_mutated:
            raise ValueError(
                "the clip must never modify raw delivered files; it is an "
                "adapter/preprocessing transformation"
            )
        if self.enabled and self.domain != "linear_sigma0":
            raise ValueError(
                "the clip must be applied in the linear sigma0 domain: 0.15 is the "
                "exact published bound, while its decibel form is a rounded "
                f"transform of it (got domain={self.domain!r})"
            )
        if self.enabled and not self.applied_before_db_conversion:
            raise ValueError(
                "the clip must be applied before the linear-to-dB conversion so "
                "the exact published bound is the one that takes effect"
            )
        # Source, preprocessing and model representations must each be named,
        # because conflating them is how a clipped training distribution ends up
        # served against an unclipped inference one.
        for stage in ("source", "preprocessing", "model"):
            if stage not in self.representations:
                raise ValueError(f"clip.representations must name the {stage} representation")
        if self.enabled and self.delivered_is_pre_clipped:
            raise ValueError(
                "delivered_is_pre_clipped is true while the clip is enabled; the "
                "measured delivery is NOT pre-clipped (observed VV max 901.92)"
            )
        if self.is_m3_feature_level_clamp:
            raise ValueError(
                "the clip is a representation parameter at the adapter boundary, "
                "not M3's feature-level clamp"
            )
        return self


# ---------------------------------------------------------------------------
# C11 / C12 — policy
# ---------------------------------------------------------------------------


class RedistributionPolicy(_Strict):
    """C11. Our policy, kept separate from the unresolved upstream licences."""

    raw_data_stays_external: bool
    derived_dataset_copies_committed: bool
    sample_tiles_committed: bool
    samples_dir_may_hold_training_corpus_samples: bool
    metadata_and_config_may_describe_external_paths: bool
    docs_may_imply_redistribution: bool
    upstream_license_status: Mapping[str, str]
    license_manufactured: bool
    resolution_required_before: Tuple[str, ...]

    @model_validator(mode="after")
    def _nothing_is_redistributed_and_no_licence_is_invented(
        self,
    ) -> "RedistributionPolicy":
        if self.license_manufactured:
            raise ValueError("a licence must never be manufactured")
        if not self.raw_data_stays_external:
            raise ValueError("raw training-corpus data must stay external to this repository")
        for flag in (
            "derived_dataset_copies_committed",
            "sample_tiles_committed",
            "samples_dir_may_hold_training_corpus_samples",
            "docs_may_imply_redistribution",
        ):
            if getattr(self, flag):
                raise ValueError(
                    f"redistribution_policy.{flag} must be false while the upstream "
                    "licences are unresolved"
                )
        return self


class LocalSubsetPolicy(_Strict):
    """C12. Development permitted, claims blocked, official split untouched."""

    official_split_modified: bool
    official_split_redefined_by_local_availability: bool
    absent_activation_is_missing_not_unsplit: bool
    development_use_permitted: bool
    production_training_run_permitted: bool
    threshold_selection_permitted: bool
    scientific_model_selection_permitted: bool
    must_report_missing_activations: bool
    must_resolve_split_by_official_activation_id: bool
    activations_present: int
    activations_catalogued: int
    train_activations_present: int
    train_activations_total: int
    val_activations_present: int
    val_activations_total: int
    test_activations_present: int
    test_activations_total: int
    train_tile_skew: str
    status: str

    @model_validator(mode="after")
    def _incompleteness_does_not_become_permission(self) -> "LocalSubsetPolicy":
        if self.official_split_modified or self.official_split_redefined_by_local_availability:
            raise ValueError(
                "the official split must never be modified or redefined by what "
                "happens to be downloaded locally"
            )
        if not self.absent_activation_is_missing_not_unsplit:
            raise ValueError("a locally absent activation is MISSING, never 'not in the split'")
        incomplete = self.activations_present < self.activations_catalogued
        if incomplete:
            for flag in (
                "production_training_run_permitted",
                "threshold_selection_permitted",
                "scientific_model_selection_permitted",
            ):
                if getattr(self, flag):
                    raise ValueError(
                        f"the local corpus is incomplete ({self.activations_present} of "
                        f"{self.activations_catalogued} activations), so {flag} must be "
                        "false; development use is permitted, scientific claims are not"
                    )
            if not self.must_report_missing_activations:
                raise ValueError(
                    "an incomplete corpus must report its missing activations rather "
                    "than present the subset as the whole"
                )
        return self


# ---------------------------------------------------------------------------
# C8 — per-dataset capabilities
# ---------------------------------------------------------------------------


class _ImageryCapabilities(_Strict):
    sar: bool
    optical: bool
    pre_event: bool
    post_event: bool
    temporal_pair: bool
    change_features: bool

    @model_validator(mode="after")
    def _change_requires_a_pair(self) -> "_ImageryCapabilities":
        """Change features cannot exist without two epochs.

        This is the guard against the tempting shortcut of letting one code
        path serve both corpora by faking a pre-event image. A synthesised
        counterpart produces a change feature that is identically zero or pure
        noise, and nothing downstream can detect it.
        """
        if self.change_features and not self.temporal_pair:
            raise ValueError(
                "change_features requires temporal_pair; a change feature without "
                "two epochs would be computed against a synthesised image"
            )
        if self.temporal_pair and not (self.pre_event and self.post_event):
            raise ValueError("temporal_pair requires both pre_event and post_event imagery")
        return self


class _TerrainCapabilities(_Strict):
    dem_bundled: bool
    dem_available_to_canonical_sample: bool
    slope_bundled: bool
    slope_trusted: bool

    @model_validator(mode="after")
    def _cannot_expose_what_is_not_bundled(self) -> "_TerrainCapabilities":
        if self.dem_available_to_canonical_sample and not self.dem_bundled:
            raise ValueError("a DEM cannot be exposed to the canonical sample unless it is bundled")
        if self.slope_trusted and not self.slope_bundled:
            raise ValueError("a slope product cannot be trusted unless it is bundled")
        return self


class _LabelNodata(_Strict):
    present: bool
    value: Optional[int] = None
    form: Optional[str] = None

    @model_validator(mode="after")
    def _present_nodata_is_specified(self) -> "_LabelNodata":
        if self.present and (self.value is None or self.form is None):
            raise ValueError("a present label nodata sentinel must declare its value and its form")
        return self


class _LabelCapabilities(_Strict):
    flood: bool
    permanent_water: bool
    separates_flood_from_permanent_water: bool
    multiclass_segmentation: bool
    binary_segmentation: bool
    nodata: _LabelNodata

    @model_validator(mode="after")
    def _separation_requires_both_classes(self) -> "_LabelCapabilities":
        if self.separates_flood_from_permanent_water and not (self.flood and self.permanent_water):
            raise ValueError(
                "a corpus cannot separate flood from permanent water unless it " "labels both"
            )
        if self.multiclass_segmentation and not self.separates_flood_from_permanent_water:
            raise ValueError(
                "multiclass segmentation here means the flood/permanent-water "
                "distinction; declaring it without that separation would promise a "
                "class the labels do not contain"
            )
        return self


class _SarRepresentation(_Strict):
    native: str
    native_is_decibel: bool
    nodata_value: Optional[float] = None
    non_negative: bool
    dtype: str
    source_speckle_filtered: bool
    source_clipped: bool

    @model_validator(mode="after")
    def _representation_and_sign_agree(self) -> "_SarRepresentation":
        """Linear power is non-negative; decibels are signed.

        Getting this pair wrong is how a dB raster gets read as linear (or the
        reverse), which yields a plausible-looking and physically meaningless
        product. Checked rather than trusted.
        """
        if self.native_is_decibel and self.non_negative:
            raise ValueError("decibel backscatter is signed; non_negative cannot be true")
        if not self.native_is_decibel and not self.non_negative:
            raise ValueError("linear sigma0 is a power ratio and cannot be negative")
        if self.native_is_decibel != (self.native == "decibel"):
            raise ValueError(
                f"native={self.native!r} contradicts native_is_decibel=" f"{self.native_is_decibel}"
            )
        return self


class _OpticalRepresentation(_Strict):
    processing_level: str
    quantity: str
    scale_factor: float
    dtype: str
    band_count: int
    quality_mask_included: bool
    boa_surface_reflectance: bool


class _SplitCapabilities(_Strict):
    type: str
    authoritative_unit: str
    official_split_is_authoritative: bool


class DatasetCapabilities(_Strict):
    """C8. What a corpus can and cannot do.

    ``optical_representation`` is optional because Kuro Siwo has no optical
    branch at all. That asymmetry is the reason this is a structured model
    rather than a flat set of booleans: the contract must allow a capability
    block to be *absent*, not merely false.
    """

    imagery: _ImageryCapabilities
    terrain: _TerrainCapabilities
    labels: _LabelCapabilities
    sar_representation: _SarRepresentation
    optical_representation: Optional[_OpticalRepresentation] = None
    splits: _SplitCapabilities

    @model_validator(mode="after")
    def _optical_block_matches_the_optical_flag(self) -> "DatasetCapabilities":
        if self.imagery.optical and self.optical_representation is None:
            raise ValueError(
                "a corpus declaring optical imagery must describe its optical " "representation"
            )
        if not self.imagery.optical and self.optical_representation is not None:
            raise ValueError(
                "a corpus with no optical imagery must not describe an optical " "representation"
            )
        return self


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


class AdapterInvariantCounts(_Strict):
    kuro_siwo: int
    sen1floods11: int


class BaselineScope(_Strict):
    modality: str
    dem_input: bool
    bundled_slope_input: bool
    optical_input: bool
    temporal_depth: str
    label_scheme: str
    corpus_union_permitted: bool
    debris_class_claimed: bool
    threshold_inside_model: bool

    @model_validator(mode="after")
    def _sar_only_means_sar_only(self) -> "BaselineScope":
        if self.modality == "SAR only" and (
            self.dem_input or self.bundled_slope_input or self.optical_input
        ):
            raise ValueError(
                "a SAR-only baseline cannot take DEM, bundled slope or optical "
                "model input channels"
            )
        if self.debris_class_claimed:
            raise ValueError(
                "no permitted corpus labels debris; claiming the class would be "
                "unsupported by any label evidence"
            )
        if self.threshold_inside_model:
            raise ValueError(
                "the decision threshold stays outside the model so it can be "
                "selected on validation and reported with its procedure"
            )
        return self


class M4Contract(_Strict):
    """The frozen M4 dataset contract: decisions C1-C14."""

    version: str
    frozen_at: str
    sample_contract_reference: str
    canonical_crs: CanonicalCrsConfig
    canonical_gsd: CanonicalGsdConfig
    resampling: ResamplingPolicy
    speckle: SpeckleDecision
    statistics: StatisticsSchema
    threshold_selection: ThresholdSelectionPolicy
    spatial_isolation: SpatialIsolationPolicy
    output_nodata: OutputNodataConfig
    clip: ClipConfig
    redistribution_policy: RedistributionPolicy
    local_subset: LocalSubsetPolicy
    capabilities_location: str
    capability_model_is_a_single_boolean: bool
    capability_model_uses_single_validity_form_literal: bool
    capability_keys_that_differ: Tuple[str, ...]
    adapter_invariants_location: str
    adapter_invariant_counts: AdapterInvariantCounts
    baseline: BaselineScope

    @model_validator(mode="after")
    def _capability_model_is_structured(self) -> "M4Contract":
        """C8 must not regress to a boolean or to a single validity enum.

        The single ``validity_form`` Literal proposed by an earlier audit is
        named explicitly so it cannot be reintroduced by accident: Kuro Siwo
        encodes validity three ways and Sen1Floods11 one way, so a one-of enum
        cannot represent both.
        """
        if self.capability_model_is_a_single_boolean:
            raise ValueError(
                "the capability model must not collapse to a single boolean; the "
                "two corpora differ on several independent capabilities"
            )
        if self.capability_model_uses_single_validity_form_literal:
            raise ValueError(
                "validity must be a LIST of mechanisms, not a single "
                "`validity_form` Literal: Kuro Siwo encodes validity three ways "
                "and Sen1Floods11 one way"
            )
        if not self.capability_keys_that_differ:
            raise ValueError(
                "capability_keys_that_differ must name the capabilities on which "
                "the corpora disagree; an empty list would imply they are "
                "interchangeable"
            )
        return self

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "M4Contract":
        try:
            return cls.model_validate(mapping)
        except Exception as exc:
            raise ContractConfigurationError(f"invalid m4_contract configuration: {exc}") from exc


def _dataset_entries(data_config: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    try:
        return data_config["production"]["training_datasets"]["datasets"]
    except (KeyError, TypeError) as exc:
        raise ContractConfigurationError(
            "configs/data.yaml must define production.training_datasets.datasets"
        ) from exc


def load_m4_contract(
    data_config: Optional[Mapping[str, Any]] = None,
) -> M4Contract:
    """Load and validate the frozen contract block from ``configs/data.yaml``."""
    config = data_config if data_config is not None else load_config("data")
    block = config.get("m4_contract")
    if block is None:
        raise ContractConfigurationError(
            "configs/data.yaml does not define an `m4_contract` block; the M4 "
            "dataset contract is not frozen"
        )
    return M4Contract.from_mapping(block)


def load_dataset_capabilities(
    data_config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, DatasetCapabilities]:
    """Validate every dataset's capability block, keyed by dataset id (C8)."""
    config = data_config if data_config is not None else load_config("data")
    capabilities = {}
    for entry in _dataset_entries(config):
        dataset_id = entry.get("id")
        block = entry.get("capabilities")
        if block is None:
            raise ContractConfigurationError(
                f"dataset {dataset_id!r} declares no `capabilities` block; an "
                "absent block is indistinguishable from an unchecked one"
            )
        try:
            capabilities[dataset_id] = DatasetCapabilities.model_validate(block)
        except Exception as exc:
            raise ContractConfigurationError(
                f"invalid capabilities for dataset {dataset_id!r}: {exc}"
            ) from exc
    return capabilities
