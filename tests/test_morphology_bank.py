import json

from oxygen_shape.experiments.morphology_bank import prepare_bank
from oxygen_shape.experiments.morphology_study import MorphologyStudyConfig


def test_bank_is_distinct_and_disjoint_before_any_pde_outcome(tmp_path):
    config = MorphologyStudyConfig()
    mapping = prepare_bank(tmp_path, config)
    manifest = json.loads((tmp_path / "bank_manifest.json").read_text())
    training = {row["mask_sha256"] for row in manifest["training"]}
    evaluation = {row["mask_sha256"] for row in manifest["evaluation"]}
    assert len(evaluation) == 45
    assert len(training) == 7
    assert training.isdisjoint(evaluation)
    assert mapping == prepare_bank(tmp_path, config)
