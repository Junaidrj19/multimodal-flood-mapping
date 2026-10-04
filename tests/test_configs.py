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
        """AGENTS.md §18 — datasets must be cited per license and paper."""
        for dataset in load_config("data")["production"]["training_datasets"]["datasets"]:
            assert dataset["license"], f"{dataset['id']} has no license"
            assert dataset["citation"], f"{dataset['id']} has no citation"

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
