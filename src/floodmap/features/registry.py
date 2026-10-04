"""Machine-readable feature contract.

Why a registry
--------------
``architecture.md`` §5 requires feature construction to be reproducible and
configurable, and ``AGENTS.md`` §22 requires the system to answer "How do you
know this?". A segmentation model that receives an anonymous N-channel stack
cannot answer that question for any of its channels. The registry therefore
carries, for every generated feature: what physical quantity it represents,
which permitted source it came from, the exact inputs it consumed, its
mathematical definition, units, dtype, any scientifically justified range, its
nodata policy, its version, and the rationale for computing it at all.

Why templates rather than a fixed feature list
----------------------------------------------
The project has deliberately not established which Sentinel-1 polarisations or
Sentinel-2 bands the delivered products contain
(``configs/preprocessing.yaml`` leaves ``required_polarizations`` and
``required_bands`` null, and ``docs/data-contract.md`` §1.1/§2.3 mark them
TODO(verify)). Hard-coding ``B03`` as "green" or assuming VV+VH exists would be
exactly the invention this repository forbids.

So the catalogue below is written against *roles* — ``vv``, ``green``, ``nir``
— and :func:`build_registry` binds roles to the band descriptions that M2
actually wrote. A role with no binding yields an explicit error naming the
missing binding; it never yields a guessed band.

Why the backscatter representation is a gate, not a note
--------------------------------------------------------
M2 can emit Sentinel-1 either as a decibel conversion (``linear_to_db``) or in
an unspecified source-calibrated representation. The two demand different
arithmetic for the same physical quantity: in decibels the log-ratio *is* the
plain difference, while in linear power it is ``10*log10(post/pre)``. Applying
a logarithm to decibel data produces a plausible-looking raster that is
physically meaningless, and nothing downstream would reveal it. The
representation is therefore a required, explicit input, and each affected
template resolves to a different transform depending on it.

Nothing here classifies a pixel. A feature is evidence for the segmentation
model in M4, not a flood decision.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..utils.provenance import ProductionInput
from .errors import (
    FeatureConfigurationError,
    FeatureRegistryError,
    UnsupportedFeatureError,
)

__all__ = [
    "ArtifactRole",
    "BackscatterRepresentation",
    "FeatureDefinition",
    "FeatureFamily",
    "FeatureInput",
    "FeatureRegistry",
    "FeatureTemplate",
    "TemplateExpansion",
    "TransformKind",
    "FEATURE_TEMPLATES",
    "template_ids",
]


class FeatureFamily(str, Enum):
    """Scientific grouping of a feature, used for configuration and reporting."""

    SENTINEL1_BACKSCATTER = "sentinel1_backscatter"
    SENTINEL1_CHANGE = "sentinel1_change"
    SENTINEL1_POLARIMETRIC = "sentinel1_polarimetric"
    SENTINEL2_REFLECTANCE = "sentinel2_reflectance"
    SENTINEL2_INDEX = "sentinel2_index"
    SENTINEL2_CHANGE = "sentinel2_change"
    TERRAIN = "terrain"


class ArtifactRole(str, Enum):
    """Which M2 analysis-ready artifact an input band is read from."""

    SENTINEL1_BEFORE = "sentinel1_before"
    SENTINEL1_AFTER = "sentinel1_after"
    SENTINEL2_BEFORE = "sentinel2_before"
    SENTINEL2_AFTER = "sentinel2_after"
    DEM = "dem"


class BackscatterRepresentation(str, Enum):
    """How M2 expressed Sentinel-1 backscatter in the artifact it wrote.

    ``LINEAR`` means linear power (sigma-nought as delivered/calibrated);
    ``DECIBEL`` means ``10*log10`` has already been applied. There is no
    default: an unrecorded representation is a configuration error.
    """

    LINEAR = "linear"
    DECIBEL = "decibel"


class TransformKind(str, Enum):
    """The exact arithmetic applied, implemented in :mod:`.numerics`."""

    PASSTHROUGH = "passthrough"
    DIFFERENCE = "difference"
    LOG_RATIO_DB = "log_ratio_db"
    CROSS_POL_DIFFERENCE_DB = "cross_polarisation_difference_db"
    NORMALIZED_DIFFERENCE = "normalized_difference"
    DERIVED_DIFFERENCE = "derived_difference"
    SLOPE_HORN_DEGREES = "slope_horn_degrees"


class TemplateExpansion(str, Enum):
    """How one catalogue entry becomes concrete feature definitions."""

    #: Once per configured Sentinel-1 polarisation role.
    PER_POLARISATION = "per_polarisation"
    #: Once per configured Sentinel-2 spectral band role.
    PER_SPECTRAL_BAND = "per_spectral_band"
    #: Exactly once, against the specific roles the template names.
    FIXED = "fixed"


class FeatureInput(BaseModel):
    """One band of one M2 artifact consumed by a feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: ArtifactRole
    band_role: str = Field(
        description="Logical role, e.g. 'vv' or 'green'. Bound to a real band description by config."
    )
    band_name: Optional[str] = Field(
        default=None,
        description="The band description in the M2 artifact that the role resolved to.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class FeatureDefinition(BaseModel):
    """Complete, serializable contract for one generated feature.

    Every field is required reading for a downstream consumer. In particular
    ``definition`` is the arithmetic actually performed, not a description of
    it, and ``rationale`` states why the feature is scientifically worth
    computing — ``AGENTS.md`` forbids adding a feature merely because it is
    common in flood-mapping tutorials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    template_id: str
    family: FeatureFamily
    source: ProductionInput
    inputs: Tuple[FeatureInput, ...]
    derived_from: Tuple[str, ...] = Field(
        default=(),
        description="Names of other registry features this one is computed from.",
    )
    transform: TransformKind
    definition: str = Field(description="The exact transformation applied.")
    units: str = Field(description="Physical units, or 'dimensionless'.")
    dtype: str
    valid_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Scientifically justified range, where one exists. Recorded and "
            "reported, never enforced by clipping."
        ),
    )
    valid_range_condition: Optional[str] = Field(
        default=None,
        description="The assumption under which valid_range holds, when it is conditional.",
    )
    nodata_policy: str
    rationale: str
    limitations: Tuple[str, ...] = ()
    feature_version: str
    citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def required_artifacts(self) -> Tuple[ArtifactRole, ...]:
        seen: list[ArtifactRole] = []
        for item in self.inputs:
            if item.artifact not in seen:
                seen.append(item.artifact)
        return tuple(seen)


class FeatureTemplate(BaseModel):
    """A catalogue entry: a feature definition awaiting its role bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    expansion: TemplateExpansion
    family: FeatureFamily
    source: ProductionInput
    #: ``(artifact, role-or-placeholder)`` pairs. ``"{role}"`` is substituted
    #: during per-polarisation / per-band expansion.
    inputs: Tuple[Tuple[ArtifactRole, str], ...] = ()
    #: Template IDs this feature is derived from, with the same placeholder rule.
    derived_from: Tuple[str, ...] = ()
    #: ``name_pattern`` and ``definition`` may contain ``{role}``.
    name_pattern: str
    transform: TransformKind
    #: Transform used instead of ``transform`` when Sentinel-1 is in linear power.
    linear_transform: Optional[TransformKind] = None
    definition: str
    linear_definition: Optional[str] = None
    units: str
    linear_units: Optional[str] = None
    dtype: str = "float32"
    valid_range: Optional[Tuple[float, float]] = None
    valid_range_condition: Optional[str] = None
    nodata_policy: str
    rationale: str
    limitations: Tuple[str, ...] = ()
    feature_version: str
    citation: Optional[str] = None


# =============================================================================
# Nodata policy strings
#
# Factored out so the same policy reads identically on every feature it applies
# to; divergent wording for identical behaviour would be a documentation defect.
# =============================================================================

_POLICY_ALL_INPUTS = (
    "Invalid where any contributing input pixel is invalid in its M2 valid mask "
    "or non-finite. Invalid pixels are written as the configured feature nodata "
    "value and marked 0 in the feature valid mask."
)

_POLICY_POSITIVE_DOMAIN = (
    _POLICY_ALL_INPUTS + " Additionally invalid where an input is not strictly positive, since the "
    "logarithm is undefined there. No epsilon is substituted."
)

_POLICY_NONZERO_SUM = (
    _POLICY_ALL_INPUTS + " Additionally invalid where the two band values sum to zero, since the "
    "normalised difference is undefined there."
)

_POLICY_SLOPE = (
    _POLICY_ALL_INPUTS + " The one-pixel grid border is invalid because no 3x3 window exists, and "
    "an interior pixel is invalid if any of its nine window cells is invalid."
)


# =============================================================================
# Closed feature catalogue
#
# Each entry states the physical quantity, the sensor, the inputs, the
# assumptions, the invalid-data behaviour and the downstream purpose, as the
# milestone contract requires before a feature may exist.
# =============================================================================

FEATURE_TEMPLATES: Tuple[FeatureTemplate, ...] = (
    # ---------------------------------------------------------------- S1 levels
    FeatureTemplate(
        template_id="s1_backscatter_pre",
        expansion=TemplateExpansion.PER_POLARISATION,
        family=FeatureFamily.SENTINEL1_BACKSCATTER,
        source=ProductionInput.SENTINEL1,
        inputs=((ArtifactRole.SENTINEL1_BEFORE, "{role}"),),
        name_pattern="s1_{role}_pre",
        transform=TransformKind.PASSTHROUGH,
        definition="Pre-event Sentinel-1 {role} backscatter, taken unchanged from the M2 artifact.",
        units="dB",
        linear_units="linear power (sigma-nought)",
        linear_definition=(
            "Pre-event Sentinel-1 {role} backscatter in linear power, taken "
            "unchanged from the M2 artifact."
        ),
        linear_transform=TransformKind.PASSTHROUGH,
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The absolute pre-event backscatter level is retained alongside any "
            "change feature. A change value alone is ambiguous: a given decrease "
            "means something different over an already-dark permanent water body "
            "than over a bright vegetated slope, and the model cannot make that "
            "distinction without the pre-event level."
        ),
        limitations=(
            "Low backscatter is ambiguous between smooth open water, some dry "
            "smooth surfaces and radar shadow (docs/scientific-assumptions.md §7).",
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s1_backscatter_post",
        expansion=TemplateExpansion.PER_POLARISATION,
        family=FeatureFamily.SENTINEL1_BACKSCATTER,
        source=ProductionInput.SENTINEL1,
        inputs=((ArtifactRole.SENTINEL1_AFTER, "{role}"),),
        name_pattern="s1_{role}_post",
        transform=TransformKind.PASSTHROUGH,
        definition="Post-event Sentinel-1 {role} backscatter, taken unchanged from the M2 artifact.",
        units="dB",
        linear_units="linear power (sigma-nought)",
        linear_definition=(
            "Post-event Sentinel-1 {role} backscatter in linear power, taken "
            "unchanged from the M2 artifact."
        ),
        linear_transform=TransformKind.PASSTHROUGH,
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The post-event level is the observation the prediction is actually "
            "about. Smooth open water is a specular reflector and returns little "
            "energy to the sensor, so a low post-event value is the primary SAR "
            "evidence for inundation — evidence, not a decision."
        ),
        limitations=(
            "Surface conditions at the acquisition timestamp only; not peak "
            "extent (docs/scientific-assumptions.md §5).",
        ),
        feature_version="1.0.0",
    ),
    # ---------------------------------------------------------------- S1 change
    FeatureTemplate(
        template_id="s1_change_db",
        expansion=TemplateExpansion.PER_POLARISATION,
        family=FeatureFamily.SENTINEL1_CHANGE,
        source=ProductionInput.SENTINEL1,
        inputs=(
            (ArtifactRole.SENTINEL1_AFTER, "{role}"),
            (ArtifactRole.SENTINEL1_BEFORE, "{role}"),
        ),
        name_pattern="s1_{role}_change_db",
        transform=TransformKind.DIFFERENCE,
        definition=(
            "post_dB - pre_dB for Sentinel-1 {role}. Because the inputs are "
            "already logarithmic, this difference is identically "
            "10*log10(post_linear / pre_linear)."
        ),
        linear_transform=TransformKind.LOG_RATIO_DB,
        linear_definition=(
            "10 * log10(post_linear / pre_linear) for Sentinel-1 {role}, the "
            "log-ratio change in decibels."
        ),
        units="dB",
        linear_units="dB",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The log-ratio is the standard SAR change quantity because speckle is "
            "multiplicative: a ratio turns it into an additive term with "
            "approximately terrain-independent statistics, whereas a linear "
            "difference scales with the local backscatter level. The decibel form "
            "is symmetric about zero, so a decrease and an equivalent increase "
            "have comparable magnitude. A pronounced decrease is consistent with "
            "a surface becoming smoother and more specular, which inundation can "
            "cause."
        ),
        limitations=(
            "Only interpretable between comparable viewing geometries; M2 enforces "
            "the same-relative-orbit requirement (AGENTS.md §4).",
            "Backscatter decrease is not specific to flooding, and partially "
            "submerged vegetation can increase backscatter through double-bounce "
            "scattering rather than decrease it.",
        ),
        feature_version="1.0.0",
    ),
    # ---------------------------------------------------------- S1 polarimetric
    FeatureTemplate(
        template_id="s1_co_cross_ratio_db_pre",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL1_POLARIMETRIC,
        source=ProductionInput.SENTINEL1,
        inputs=(
            (ArtifactRole.SENTINEL1_BEFORE, "vv"),
            (ArtifactRole.SENTINEL1_BEFORE, "vh"),
        ),
        name_pattern="s1_vv_vh_ratio_db_pre",
        transform=TransformKind.CROSS_POL_DIFFERENCE_DB,
        definition="pre VV_dB - pre VH_dB, i.e. 10*log10(VV_linear / VH_linear).",
        linear_transform=TransformKind.LOG_RATIO_DB,
        linear_definition="10 * log10(pre VV_linear / pre VH_linear).",
        units="dB",
        linear_units="dB",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The co- to cross-polarised ratio responds to the dominant scattering "
            "mechanism rather than to brightness alone. Surface (single-bounce) "
            "scattering from smooth surfaces depolarises weakly and gives a high "
            "ratio, while volume scattering from vegetation depolarises strongly "
            "and gives a low one. This discriminates dark-but-rough from "
            "dark-and-smooth surfaces, which absolute backscatter cannot."
        ),
        limitations=(
            "Requires both a co- and a cross-polarised channel in the delivered "
            "product; the pipeline does not assume dual polarisation exists.",
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s1_co_cross_ratio_db_post",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL1_POLARIMETRIC,
        source=ProductionInput.SENTINEL1,
        inputs=(
            (ArtifactRole.SENTINEL1_AFTER, "vv"),
            (ArtifactRole.SENTINEL1_AFTER, "vh"),
        ),
        name_pattern="s1_vv_vh_ratio_db_post",
        transform=TransformKind.CROSS_POL_DIFFERENCE_DB,
        definition="post VV_dB - post VH_dB, i.e. 10*log10(VV_linear / VH_linear).",
        linear_transform=TransformKind.LOG_RATIO_DB,
        linear_definition="10 * log10(post VV_linear / post VH_linear).",
        units="dB",
        linear_units="dB",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The post-event scattering-mechanism indicator, paired with its "
            "pre-event counterpart so the model sees the polarimetric state on "
            "both dates rather than only the later one."
        ),
        limitations=(
            "Requires both a co- and a cross-polarised channel in the delivered "
            "product; the pipeline does not assume dual polarisation exists.",
        ),
        feature_version="1.0.0",
    ),
    # --------------------------------------------------------- S2 reflectance
    FeatureTemplate(
        template_id="s2_reflectance_pre",
        expansion=TemplateExpansion.PER_SPECTRAL_BAND,
        family=FeatureFamily.SENTINEL2_REFLECTANCE,
        source=ProductionInput.SENTINEL2,
        inputs=((ArtifactRole.SENTINEL2_BEFORE, "{role}"),),
        name_pattern="s2_{role}_pre",
        transform=TransformKind.PASSTHROUGH,
        definition="Pre-event Sentinel-2 {role} reflectance, taken unchanged from the M2 artifact.",
        units="reflectance (dimensionless)",
        valid_range=(0.0, 1.0),
        valid_range_condition=(
            "Holds for physically valid surface reflectance. Atmospheric "
            "correction can return values slightly outside it over dark or "
            "shadowed surfaces, so the range is reported, not enforced."
        ),
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "Retained so the spectral state before the event is available "
            "alongside any index or change feature, and so a change value can be "
            "interpreted relative to its starting point."
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s2_reflectance_post",
        expansion=TemplateExpansion.PER_SPECTRAL_BAND,
        family=FeatureFamily.SENTINEL2_REFLECTANCE,
        source=ProductionInput.SENTINEL2,
        inputs=((ArtifactRole.SENTINEL2_AFTER, "{role}"),),
        name_pattern="s2_{role}_post",
        transform=TransformKind.PASSTHROUGH,
        definition="Post-event Sentinel-2 {role} reflectance, taken unchanged from the M2 artifact.",
        units="reflectance (dimensionless)",
        valid_range=(0.0, 1.0),
        valid_range_condition=(
            "Holds for physically valid surface reflectance. Atmospheric "
            "correction can return values slightly outside it over dark or "
            "shadowed surfaces, so the range is reported, not enforced."
        ),
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The post-event spectral observation, available only where cloud and "
            "quality masking left a valid pixel."
        ),
        limitations=(
            "Optical observation is frequently unavailable in monsoon conditions; "
            "an invalid pixel is 'not observed', never 'observed, not flooded' "
            "(docs/scientific-assumptions.md §8).",
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s2_reflectance_change",
        expansion=TemplateExpansion.PER_SPECTRAL_BAND,
        family=FeatureFamily.SENTINEL2_CHANGE,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_AFTER, "{role}"),
            (ArtifactRole.SENTINEL2_BEFORE, "{role}"),
        ),
        name_pattern="s2_{role}_change",
        transform=TransformKind.DIFFERENCE,
        definition="post {role} reflectance - pre {role} reflectance.",
        units="reflectance difference (dimensionless)",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Follows from both inputs lying in [0, 1].",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "A per-band reflectance change isolates surface change from the "
            "static spectral background, which varies strongly with land cover. "
            "Water bodies absorb near-infrared strongly, so a NIR decrease "
            "accompanying a shortwave-infrared decrease is evidence of new "
            "standing water; sediment-laden water instead tends to brighten in "
            "the visible bands."
        ),
        limitations=(
            "Requires a valid pixel on both dates, so cloud on either date "
            "removes the change observation entirely.",
        ),
        feature_version="1.0.0",
    ),
    # ------------------------------------------------------------- S2 indices
    FeatureTemplate(
        template_id="s2_ndwi_pre",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_BEFORE, "green"),
            (ArtifactRole.SENTINEL2_BEFORE, "nir"),
        ),
        name_pattern="s2_ndwi_pre",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(pre green - pre nir) / (pre green + pre nir).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "Normalised Difference Water Index. Water absorbs near-infrared "
            "strongly while still reflecting green, so open water sits at the "
            "high end of this index and most land cover below it. Computed on "
            "the pre-event date to establish the permanent water baseline: "
            "without it the river itself would be indistinguishable from new "
            "flooding (docs/scientific-assumptions.md §7)."
        ),
        limitations=(
            "Built-up surfaces are a known source of high NDWI values, which is "
            "why MNDWI exists.",
            "An index value is evidence, not a water classification; no threshold "
            "is applied in this milestone.",
        ),
        feature_version="1.0.0",
        citation=(
            "McFeeters, S.K. (1996). The use of the Normalized Difference Water "
            "Index (NDWI) in the delineation of open water features. "
            "International Journal of Remote Sensing 17(7), 1425-1432."
        ),
    ),
    FeatureTemplate(
        template_id="s2_ndwi_post",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_AFTER, "green"),
            (ArtifactRole.SENTINEL2_AFTER, "nir"),
        ),
        name_pattern="s2_ndwi_post",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(post green - post nir) / (post green + post nir).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "The post-event water-sensitive index, the optical counterpart to the "
            "post-event SAR backscatter level."
        ),
        limitations=(
            "Built-up surfaces are a known source of high NDWI values.",
            "An index value is evidence, not a water classification.",
        ),
        feature_version="1.0.0",
        citation=(
            "McFeeters, S.K. (1996). The use of the Normalized Difference Water "
            "Index (NDWI) in the delineation of open water features. "
            "International Journal of Remote Sensing 17(7), 1425-1432."
        ),
    ),
    FeatureTemplate(
        template_id="s2_ndwi_change",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_CHANGE,
        source=ProductionInput.SENTINEL2,
        derived_from=("s2_ndwi_post", "s2_ndwi_pre"),
        name_pattern="s2_ndwi_change",
        transform=TransformKind.DERIVED_DIFFERENCE,
        definition="s2_ndwi_post - s2_ndwi_pre.",
        units="dimensionless",
        valid_range=(-2.0, 2.0),
        valid_range_condition="Follows from both indices lying in [-1, 1].",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "The increase in water-sensitive index between the two dates. This is "
            "the quantity that separates new inundation from permanent water: a "
            "river channel has a high index on both dates and therefore a change "
            "near zero, whereas a newly inundated field shows a large positive "
            "change."
        ),
        limitations=(
            "Requires a cloud-free pixel on both dates.",
            "A positive change is evidence of increased water-like spectral "
            "response, not a flood classification.",
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s2_mndwi_pre",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_BEFORE, "green"),
            (ArtifactRole.SENTINEL2_BEFORE, "swir16"),
        ),
        name_pattern="s2_mndwi_pre",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(pre green - pre swir16) / (pre green + pre swir16).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "Modified NDWI. Substituting shortwave infrared for near-infrared "
            "suppresses the built-up false positives that affect NDWI, because "
            "built surfaces reflect SWIR strongly while water absorbs it almost "
            "completely. Retained on the pre-event date as the permanent-water "
            "baseline."
        ),
        limitations=(
            "The SWIR bands are delivered at coarser native resolution than the "
            "visible bands, so M2's documented resampling decision is embedded in "
            "this feature.",
        ),
        feature_version="1.0.0",
        citation=(
            "Xu, H. (2006). Modification of normalised difference water index "
            "(NDWI) to enhance open water features in remotely sensed imagery. "
            "International Journal of Remote Sensing 27(14), 3025-3033."
        ),
    ),
    FeatureTemplate(
        template_id="s2_mndwi_post",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_AFTER, "green"),
            (ArtifactRole.SENTINEL2_AFTER, "swir16"),
        ),
        name_pattern="s2_mndwi_post",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(post green - post swir16) / (post green + post swir16).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "The post-event SWIR-based water-sensitive index, less susceptible to "
            "built-up false positives than NDWI."
        ),
        limitations=(
            "The SWIR bands are delivered at coarser native resolution than the " "visible bands.",
        ),
        feature_version="1.0.0",
        citation=(
            "Xu, H. (2006). Modification of normalised difference water index "
            "(NDWI) to enhance open water features in remotely sensed imagery. "
            "International Journal of Remote Sensing 27(14), 3025-3033."
        ),
    ),
    FeatureTemplate(
        template_id="s2_mndwi_change",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_CHANGE,
        source=ProductionInput.SENTINEL2,
        derived_from=("s2_mndwi_post", "s2_mndwi_pre"),
        name_pattern="s2_mndwi_change",
        transform=TransformKind.DERIVED_DIFFERENCE,
        definition="s2_mndwi_post - s2_mndwi_pre.",
        units="dimensionless",
        valid_range=(-2.0, 2.0),
        valid_range_condition="Follows from both indices lying in [-1, 1].",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "Change in the SWIR-based water index, separating new inundation from "
            "permanent water with less built-up contamination than the NDWI "
            "change."
        ),
        limitations=("Requires a cloud-free pixel on both dates.",),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="s2_ndvi_pre",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_BEFORE, "nir"),
            (ArtifactRole.SENTINEL2_BEFORE, "red"),
        ),
        name_pattern="s2_ndvi_pre",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(pre nir - pre red) / (pre nir + pre red).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "Normalised Difference Vegetation Index, included for "
            "vegetation/water discrimination rather than as a water indicator. "
            "Dense vegetation and open water occupy opposite ends of this index, "
            "and the pre-event vegetation state is needed to interpret SAR change: "
            "flooding under vegetation can raise backscatter through double-bounce "
            "scattering, the opposite of the open-water response."
        ),
        limitations=(
            "Not a water index. Its role here is to contextualise the SAR and "
            "water-index features, not to detect flooding.",
        ),
        feature_version="1.0.0",
        citation=(
            "Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). "
            "Monitoring vegetation systems in the Great Plains with ERTS. "
            "NASA SP-351, Third ERTS Symposium, 309-317."
        ),
    ),
    FeatureTemplate(
        template_id="s2_ndvi_post",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_INDEX,
        source=ProductionInput.SENTINEL2,
        inputs=(
            (ArtifactRole.SENTINEL2_AFTER, "nir"),
            (ArtifactRole.SENTINEL2_AFTER, "red"),
        ),
        name_pattern="s2_ndvi_post",
        transform=TransformKind.NORMALIZED_DIFFERENCE,
        definition="(post nir - post red) / (post nir + post red).",
        units="dimensionless",
        valid_range=(-1.0, 1.0),
        valid_range_condition="Holds when both reflectances are non-negative.",
        nodata_policy=_POLICY_NONZERO_SUM,
        rationale=(
            "The post-event vegetation state. A drop relative to the pre-event "
            "value is consistent with vegetation being submerged or stripped by "
            "debris, which is contextual evidence for the segmentation model."
        ),
        limitations=(
            "Not a water index, and a vegetation decrease has many causes "
            "unrelated to flooding.",
        ),
        feature_version="1.0.0",
        citation=(
            "Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). "
            "Monitoring vegetation systems in the Great Plains with ERTS. "
            "NASA SP-351, Third ERTS Symposium, 309-317."
        ),
    ),
    FeatureTemplate(
        template_id="s2_ndvi_change",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.SENTINEL2_CHANGE,
        source=ProductionInput.SENTINEL2,
        derived_from=("s2_ndvi_post", "s2_ndvi_pre"),
        name_pattern="s2_ndvi_change",
        transform=TransformKind.DERIVED_DIFFERENCE,
        definition="s2_ndvi_post - s2_ndvi_pre.",
        units="dimensionless",
        valid_range=(-2.0, 2.0),
        valid_range_condition="Follows from both indices lying in [-1, 1].",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "Change in vegetation index, which helps separate a surface that was "
            "already bare from one where vegetation was removed or submerged."
        ),
        limitations=(
            "Requires a cloud-free pixel on both dates, and phenological change "
            "between the two dates is a confounder independent of the event.",
        ),
        feature_version="1.0.0",
    ),
    # ----------------------------------------------------------------- terrain
    FeatureTemplate(
        template_id="dem_elevation",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.TERRAIN,
        source=ProductionInput.COPERNICUS_DEM,
        inputs=((ArtifactRole.DEM, "elevation"),),
        name_pattern="dem_elevation",
        transform=TransformKind.PASSTHROUGH,
        definition="Elevation, taken unchanged from the M2 DEM artifact.",
        units="m",
        nodata_policy=_POLICY_ALL_INPUTS,
        rationale=(
            "Absolute elevation is weak evidence on its own but constrains where "
            "water can plausibly stand, and it lets the model learn the local "
            "valley-floor context of the Bhote Koshi-Trishuli corridor rather "
            "than treating every dark pixel identically."
        ),
        limitations=(
            "Pre-event terrain prior, not a current observation: an avalanche and "
            "debris flow changes real topography "
            "(docs/scientific-assumptions.md §10.1).",
            "Vertical datum is preserved from M2 where the source declared it and "
            "is otherwise explicitly unknown.",
        ),
        feature_version="1.0.0",
    ),
    FeatureTemplate(
        template_id="dem_slope_degrees",
        expansion=TemplateExpansion.FIXED,
        family=FeatureFamily.TERRAIN,
        source=ProductionInput.COPERNICUS_DEM,
        inputs=((ArtifactRole.DEM, "elevation"),),
        name_pattern="dem_slope_degrees",
        transform=TransformKind.SLOPE_HORN_DEGREES,
        definition=(
            "degrees(arctan(hypot(dz/dx, dz/dy))) with Horn's third-order "
            "finite-difference 3x3 kernel: "
            "dz/dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cellsize), "
            "dz/dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8 * cellsize)."
        ),
        units="degrees",
        valid_range=(0.0, 90.0),
        valid_range_condition="Guaranteed by the arctangent of a non-negative gradient magnitude.",
        nodata_policy=_POLICY_SLOPE,
        rationale=(
            "Slope is the single most useful terrain covariate here for two "
            "reasons. Hydrologically, standing water is implausible on a steep "
            "face, so slope constrains where an inundation prediction is "
            "physically reasonable. Radiometrically, SAR geometry distortion - "
            "shadow, layover, foreshortening - is a function of local slope and "
            "aspect relative to the look direction, and radar shadow in steep "
            "terrain is a systematic false-positive mechanism for water detection "
            "(docs/scientific-assumptions.md §7). Slope therefore also supports "
            "the error stratification that docs/data-contract.md §3.4 requires."
        ),
        limitations=(
            "Derived from a ~30 m DEM while the imagery is finer, so the slope "
            "surface is smoother than the optical/SAR detail and cannot resolve "
            "narrow channels or road cuttings "
            "(docs/scientific-assumptions.md §10.5).",
            "Assumes the analysis grid is projected with linear units matching "
            "the elevation units. A geographic CRS in degrees would make the "
            "gradient meaningless, so the caller must declare this explicitly.",
            "This is a terrain derivative only. No flow direction, accumulation, "
            "routing or downstream tracing is computed here.",
        ),
        feature_version="1.0.0",
        citation=(
            "Horn, B.K.P. (1981). Hill shading and the reflectance map. "
            "Proceedings of the IEEE 69(1), 14-47."
        ),
    ),
)


def template_ids() -> Tuple[str, ...]:
    return tuple(template.template_id for template in FEATURE_TEMPLATES)


_TEMPLATES_BY_ID: Mapping[str, FeatureTemplate] = {
    template.template_id: template for template in FEATURE_TEMPLATES
}

if len(_TEMPLATES_BY_ID) != len(FEATURE_TEMPLATES):  # pragma: no cover - import-time guard
    raise FeatureRegistryError("duplicate template_id in the feature catalogue")


class FeatureRegistry(BaseModel):
    """The resolved, ordered set of features one feature set will contain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_version: str
    feature_set_version: str
    config_version: str
    backscatter_representation: Optional[BackscatterRepresentation] = None
    band_role_bindings: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Resolved role -> M2 band description, per artifact role.",
    )
    features: Tuple[FeatureDefinition, ...]

    def names(self) -> Tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def get(self, name: str) -> FeatureDefinition:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise FeatureRegistryError(f"feature {name!r} is not in this registry")

    def required_artifacts(self) -> Tuple[ArtifactRole, ...]:
        seen: list[ArtifactRole] = []
        for feature in self.features:
            for role in feature.required_artifacts():
                if role not in seen:
                    seen.append(role)
        return tuple(seen)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def _substitute(text: str, role: Optional[str]) -> str:
    return text.replace("{role}", role) if role is not None else text


def _resolve_inputs(
    template: FeatureTemplate,
    role: Optional[str],
    bindings: Mapping[ArtifactRole, Mapping[str, str]],
) -> Tuple[FeatureInput, ...]:
    resolved: list[FeatureInput] = []
    for artifact, slot in template.inputs:
        band_role = _substitute(slot, role)
        artifact_bindings = bindings.get(artifact)
        if artifact_bindings is None:
            raise FeatureConfigurationError(
                f"{template.template_id}: no band-role bindings configured for "
                f"artifact {artifact.value!r}; M3 does not guess band names"
            )
        band_name = artifact_bindings.get(band_role)
        if band_name is None:
            raise FeatureConfigurationError(
                f"{template.template_id}: band role {band_role!r} is not bound to a "
                f"band description for artifact {artifact.value!r}. Bind it in "
                "configs/features.yaml -> band_roles once the delivered product's "
                "band names are verified; no band name is assumed."
            )
        resolved.append(FeatureInput(artifact=artifact, band_role=band_role, band_name=band_name))
    return tuple(resolved)


def _sentinel1_variant(
    template: FeatureTemplate,
    representation: Optional[BackscatterRepresentation],
) -> Tuple[TransformKind, str, str]:
    """Pick the transform, definition and units that match the representation."""
    if representation is None:
        raise FeatureConfigurationError(
            f"{template.template_id} consumes Sentinel-1, but "
            "sentinel1.backscatter_representation is unresolved. The correct "
            "arithmetic differs between linear power and decibels, so it cannot "
            "be defaulted."
        )
    if representation is BackscatterRepresentation.DECIBEL:
        return template.transform, template.definition, template.units
    if template.linear_transform is None:
        raise UnsupportedFeatureError(
            f"{template.template_id} is not defined for linear-power backscatter"
        )
    return (
        template.linear_transform,
        template.linear_definition or template.definition,
        template.linear_units or template.units,
    )


def build_registry(
    *,
    enabled_template_ids: Sequence[str],
    polarisation_roles: Sequence[str],
    spectral_band_roles: Sequence[str],
    band_role_bindings: Mapping[ArtifactRole, Mapping[str, str]],
    backscatter_representation: Optional[BackscatterRepresentation],
    registry_version: str,
    feature_set_version: str,
    config_version: str,
) -> FeatureRegistry:
    """Resolve enabled catalogue templates into concrete feature definitions.

    Ordering is the catalogue order, then the configured role order, so an
    identical configuration always produces an identical feature sequence. Band
    order in the written artifact is part of the contract and must be stable.

    Derived features are emitted after the features they are derived from, and
    a derived feature whose base features are not both enabled is an explicit
    configuration error rather than a silently dropped channel.
    """
    if not enabled_template_ids:
        raise FeatureConfigurationError(
            "no features are enabled; features.enabled_templates is empty"
        )
    unknown = [tid for tid in enabled_template_ids if tid not in _TEMPLATES_BY_ID]
    if unknown:
        raise FeatureRegistryError(
            f"unknown feature template(s) {unknown}; the catalogue is closed. "
            f"Available: {list(template_ids())}"
        )

    enabled = list(dict.fromkeys(enabled_template_ids))
    # Catalogue order, not caller order, so the band sequence is deterministic.
    ordered = [t for t in FEATURE_TEMPLATES if t.template_id in set(enabled)]

    base: list[FeatureDefinition] = []
    derived: list[FeatureDefinition] = []

    for template in ordered:
        if template.expansion is TemplateExpansion.PER_POLARISATION:
            roles: Iterable[Optional[str]] = list(polarisation_roles)
            if not polarisation_roles:
                raise FeatureConfigurationError(
                    f"{template.template_id} expands per polarisation, but "
                    "sentinel1.polarisation_roles is unresolved"
                )
        elif template.expansion is TemplateExpansion.PER_SPECTRAL_BAND:
            roles = list(spectral_band_roles)
            if not spectral_band_roles:
                raise FeatureConfigurationError(
                    f"{template.template_id} expands per spectral band, but "
                    "sentinel2.spectral_band_roles is unresolved"
                )
        else:
            roles = [None]

        for role in roles:
            transform = template.transform
            definition = template.definition
            units = template.units
            if template.source is ProductionInput.SENTINEL1:
                transform, definition, units = _sentinel1_variant(
                    template, backscatter_representation
                )
            definition = _substitute(definition, role)

            inputs = (
                ()
                if template.transform is TransformKind.DERIVED_DIFFERENCE
                else _resolve_inputs(template, role, band_role_bindings)
            )
            feature = FeatureDefinition(
                name=_substitute(template.name_pattern, role),
                template_id=template.template_id,
                family=template.family,
                source=template.source,
                inputs=inputs,
                derived_from=tuple(_substitute(item, role) for item in template.derived_from),
                transform=transform,
                definition=definition,
                units=units,
                dtype=template.dtype,
                valid_range=template.valid_range,
                valid_range_condition=template.valid_range_condition,
                nodata_policy=template.nodata_policy,
                rationale=template.rationale,
                limitations=template.limitations,
                feature_version=template.feature_version,
                citation=template.citation,
            )
            (derived if feature.derived_from else base).append(feature)

    features = tuple(base) + tuple(derived)
    names = [feature.name for feature in features]
    available = set(names)
    for feature in derived:
        missing = [name for name in feature.derived_from if name not in available]
        if missing:
            raise FeatureConfigurationError(
                f"{feature.name} is derived from {missing}, which are not enabled. "
                "Enable the base features or disable the derived feature; M3 does "
                "not silently drop a requested channel."
            )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FeatureRegistryError(f"duplicate feature names resolved: {duplicates}")

    return FeatureRegistry(
        registry_version=registry_version,
        feature_set_version=feature_set_version,
        config_version=config_version,
        backscatter_representation=backscatter_representation,
        band_role_bindings={
            artifact.value: dict(mapping) for artifact, mapping in band_role_bindings.items()
        },
        features=features,
    )


__all__ = list(__all__) + ["build_registry"]
