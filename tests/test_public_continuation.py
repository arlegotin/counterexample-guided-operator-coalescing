"""Public evidence checks require no scientific dependencies or private inputs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_public_continuation", ROOT / "provenance/verify_public_continuation.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture
def scores():
    return json.loads((ROOT / "provenance/continuation-v2/recovered-scores.json").read_text())["original"]


def test_published_package_reconstructs_raw_reductions_and_selections():
    result = VERIFIER.verify_package(ROOT / "provenance/continuation-v2")
    assert result["native_artifacts"] == 8
    assert result["hashed_files"] == 92


def test_missing_candidate_is_rejected(scores):
    scores["score_matrices"]["student_only"].pop()
    with pytest.raises(ValueError, match="incomplete score matrix"):
        VERIFIER.verify_scores(scores)


def test_zero_denominator_is_rejected(scores):
    scores["raw"][0]["conditional_probes"][0]["valid_targets"] = 0
    with pytest.raises(ValueError, match="target denominator"):
        VERIFIER.verify_scores(scores)


def test_changed_selection_is_rejected(scores):
    scores["point_estimates"]["natural_kl"]["selected_pair"] = scores["pairs"][0]
    with pytest.raises(ValueError, match="selection mismatch"):
        VERIFIER.verify_scores(scores)


def test_changed_formula_is_rejected(scores):
    scores["score_matrices"]["active_1"][0][0] += .01
    with pytest.raises(ValueError, match="numeric reduction mismatch"):
        VERIFIER.verify_scores(scores)


def test_tampered_file_is_rejected(tmp_path):
    (tmp_path / "row.json").write_text('{}')
    (tmp_path / "SHA256.json").write_text(json.dumps({"files": {"row.json": "0" * 64}}))
    with pytest.raises(ValueError, match="file hash mismatch"):
        VERIFIER.verify_package(tmp_path)


def test_unlisted_nested_manifest_is_rejected(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/SHA256.json").write_text('{}')
    (tmp_path / "SHA256.json").write_text('{"files": {}}')
    with pytest.raises(ValueError, match="manifest coverage mismatch"):
        VERIFIER.verify_package(tmp_path)
