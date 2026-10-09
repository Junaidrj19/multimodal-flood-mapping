"""Quarantine policy tests.

A quarantine is the one mechanism in this pipeline whose entire purpose is to
remove data, so the properties worth testing are the ones that stop it becoming
a silent filter:

* an exclusion is explained by a reason code from a closed set;
* asking for an excluded sample by name raises rather than returning less;
* the audit record exists even when nothing is excluded;
* a policy cannot be applied to the wrong corpus, or without a version;
* nothing about label or validity semantics changes.
"""

from __future__ import annotations

import pytest
import yaml

from floodmap.datasets.quarantine import (
    QuarantineError,
    QuarantinePolicy,
    load_quarantine,
    quarantine_path,
)

SAMPLE_B = "1111004:df3d7e17-0a15-5d76-aad9-4571285aae39"
SAMPLE_A = "1111004:304feeed-d5a7-5ad8-b08d-ea711c073f9d"


def _write_policy(tmp_path, body: dict):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _valid_body(**overrides) -> dict:
    body = {
        "version": "1.0.0",
        "dataset_id": "kuro_siwo",
        "reason_codes": {"broken": "the data is broken"},
        "samples": [{"sample_id": "a:1", "reason_code": "broken"}],
    }
    body.update(overrides)
    return body


class TestTheCommittedPolicy:
    """The real policy file, as reviewed."""

    def test_it_exists_and_loads(self):
        assert quarantine_path("kuro_siwo").is_file()
        policy = load_quarantine("kuro_siwo", required=True)
        assert policy.dataset_id == "kuro_siwo"
        assert policy.version
        assert policy.evidence == "docs/m5-source-data-quarantine.md"

    def test_it_excludes_sample_b_only(self):
        """Sample A is deliberately NOT quarantined: its consumed channels are clean."""
        policy = load_quarantine("kuro_siwo", required=True)
        assert policy.is_quarantined(SAMPLE_B)
        assert not policy.is_quarantined(SAMPLE_A)
        assert len(policy.entries) == 1

    def test_the_exclusion_carries_a_declared_reason_and_measurements(self):
        entry = load_quarantine("kuro_siwo", required=True).entry(SAMPLE_B)
        assert entry.reason_code == "nan_on_declared_valid_consumed_sar"
        assert "CONSUMED" in entry.reason
        assert entry.details["official_split"] == "train"
        assert entry.details["consumed_channel_nan_on_valid_pixels"] == 464
        assert entry.details["recovery_attempted"] is False
        assert entry.details["recovery_blocked_by"]
        assert entry.details["reinstate_if"]

    def test_the_excluded_sample_is_not_in_validation_or_test(self):
        """So no reported metric is affected by the exclusion either way."""
        entry = load_quarantine("kuro_siwo", required=True).entry(SAMPLE_B)
        assert entry.details["official_split"] == "train"


class TestPartitionAndAudit:
    def test_partition_preserves_order_and_separates_excluded(self, tmp_path):
        policy = load_quarantine("kuro_siwo", path=_write_policy(tmp_path, _valid_body()))
        result = policy.partition(["a:0", "a:1", "a:2"])
        assert result.eligible == ("a:0", "a:2")
        assert [e.sample_id for e in result.excluded] == ["a:1"]
        assert result.n_eligible == 2 and result.n_excluded == 1

    def test_audit_record_is_present_even_with_no_exclusions(self, tmp_path):
        policy = load_quarantine("kuro_siwo", path=_write_policy(tmp_path, _valid_body()))
        record = policy.partition(["a:0", "a:2"]).audit_record()
        assert record["policy_applied"] is True
        assert record["n_excluded"] == 0
        assert record["excluded_samples"] == []
        assert record["excluded_by_reason_code"] == {}

    def test_audit_record_counts_by_reason_code(self, tmp_path):
        body = _valid_body(
            reason_codes={"broken": "x", "other": "y"},
            samples=[
                {"sample_id": "a:1", "reason_code": "broken"},
                {"sample_id": "a:2", "reason_code": "broken"},
                {"sample_id": "a:3", "reason_code": "other"},
            ],
        )
        policy = load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))
        record = policy.partition(["a:1", "a:2", "a:3", "a:4"]).audit_record()
        assert record["excluded_by_reason_code"] == {"broken": 2, "other": 1}
        assert record["n_eligible"] == 1

    def test_an_empty_policy_reports_that_none_was_applied(self):
        record = QuarantinePolicy.empty("kuro_siwo").partition(["a:1"]).audit_record()
        assert record["policy_applied"] is False
        assert record["policy_version"] is None
        assert record["n_excluded"] == 0

    def test_requiring_a_quarantined_sample_raises(self, tmp_path):
        """Never silently return fewer samples than were asked for."""
        policy = load_quarantine("kuro_siwo", path=_write_policy(tmp_path, _valid_body()))
        policy.require_eligible(["a:0", "a:2"])  # no raise
        with pytest.raises(QuarantineError, match="explicitly requested"):
            policy.require_eligible(["a:0", "a:1"])

    def test_the_raise_names_the_sample_and_reason_code(self, tmp_path):
        policy = load_quarantine("kuro_siwo", path=_write_policy(tmp_path, _valid_body()))
        with pytest.raises(QuarantineError) as excinfo:
            policy.require_eligible(["a:1"])
        assert "a:1" in str(excinfo.value)
        assert "broken" in str(excinfo.value)


class TestPolicyValidation:
    def test_a_missing_file_is_empty_by_default_and_fatal_when_required(self, tmp_path):
        assert load_quarantine("nothing", path=tmp_path / "absent.yaml").entries == {}
        with pytest.raises(QuarantineError, match="not found"):
            load_quarantine("nothing", path=tmp_path / "absent.yaml", required=True)

    def test_a_policy_for_another_corpus_is_rejected(self, tmp_path):
        path = _write_policy(tmp_path, _valid_body(dataset_id="sen1floods11"))
        with pytest.raises(QuarantineError, match="another corpus"):
            load_quarantine("kuro_siwo", path=path)

    def test_an_unversioned_policy_is_rejected(self, tmp_path):
        body = _valid_body()
        del body["version"]
        with pytest.raises(QuarantineError, match="no version"):
            load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))

    def test_an_entry_without_a_reason_code_is_rejected(self, tmp_path):
        body = _valid_body(samples=[{"sample_id": "a:1"}])
        with pytest.raises(QuarantineError, match="no reason_code"):
            load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))

    def test_an_undeclared_reason_code_is_rejected(self, tmp_path):
        body = _valid_body(samples=[{"sample_id": "a:1", "reason_code": "improvised"}])
        with pytest.raises(QuarantineError, match="undeclared reason_code"):
            load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))

    def test_a_duplicate_entry_is_rejected(self, tmp_path):
        body = _valid_body(
            samples=[
                {"sample_id": "a:1", "reason_code": "broken"},
                {"sample_id": "a:1", "reason_code": "broken"},
            ]
        )
        with pytest.raises(QuarantineError, match="duplicate"):
            load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))

    def test_an_entry_without_a_sample_id_is_rejected(self, tmp_path):
        body = _valid_body(samples=[{"reason_code": "broken"}])
        with pytest.raises(QuarantineError, match="no sample_id"):
            load_quarantine("kuro_siwo", path=_write_policy(tmp_path, body))


class TestTheFrozenSemanticsAreUntouched:
    """Quarantine operates on whole samples, never on pixels."""

    def test_the_policy_declares_no_label_or_validity_override(self):
        raw = yaml.safe_load(quarantine_path("kuro_siwo").read_text(encoding="utf-8"))
        forbidden = {
            "ignore_index",
            "semantic_values",
            "label_mapping",
            "validity_mechanism",
            "validity_mechanisms",
            "mna",
            "nodata_value",
        }
        assert not (set(raw) & forbidden)
        for entry in raw["samples"]:
            assert not (set(entry) & forbidden)

    def test_frozen_label_and_validity_config_is_unchanged(self):
        from floodmap.utils.config import load_config

        classes = load_config("segmentation")["classes"]
        assert classes["semantic_values"] == [0, 1, 2]
        assert classes["ignore_index"] == 3
        assert classes["ignore_index_is_a_semantic_class"] is False
