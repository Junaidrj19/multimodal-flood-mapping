"""Configuration loading tests.

Verify that the four configuration files exist, parse, and expose the sections
later milestones will read.

Deliberately NOT tested: whether the configuration *values* are scientifically
correct. Most are intentionally ``null``/TODO at this stage, and asserting
specific values would mean inventing the parameters this milestone leaves open.
"""

from __future__ import annotations

import pytest
import yaml

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

    def test_training_datasets_are_not_invented(self):
        """The permitted list is UNKNOWN (blocking); it must stay empty until supplied."""
        assert load_config("data")["production"]["training_datasets"]["datasets"] == []

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
