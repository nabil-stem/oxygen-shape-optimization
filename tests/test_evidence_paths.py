from pathlib import Path

import pytest

from oxygen_shape.experiments.evidence_paths import EvidencePaths


def test_relocated_windows_identity_and_cross_study_reference(tmp_path):
    main = tmp_path / "experiments/main"
    control = tmp_path / "experiments/control"
    paths = EvidencePaths(main, control)
    assert paths.resolve(r"C:\old\experiments\main\search\field.npz", control) == (
        main / "search/field.npz"
    )
    assert paths.resolve(r"..\main\search\field.npz", control) == main / "search/field.npz"
    assert paths.resolve("/old/experiments/main/search/field.npz", control) == (
        main / "search/field.npz"
    )
    assert paths.resolve(str(main), control) == main


@pytest.mark.parametrize(
    "recorded",
    [
        "../../secret.txt",
        r"C:\private\main\field.npz",
        r"C:\old\experiments\unknown\field.npz",
        r"C:\old\experiments\main\..\secret.txt",
    ],
)
def test_relocation_rejects_unknown_roots_and_escape(tmp_path, recorded):
    main = tmp_path / "experiments/main"
    with pytest.raises(ValueError):
        EvidencePaths(main).resolve(recorded, main)


def test_duplicate_study_names_rejected():
    with pytest.raises(ValueError):
        EvidencePaths(Path("a/study"), Path("b/study"))
