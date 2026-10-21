import json
from datetime import datetime, timedelta, timezone

import pytest

from oxygen_shape.experiments.long_study import StudyConfig, build_plan, controller
from oxygen_shape.experiments.study_worker import atomic_json


def test_study_plan_has_unique_jobs_and_independent_training_seed_ranges():
    config = StudyConfig(deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat())
    jobs = build_plan(config)
    assert len({job["id"] for job in jobs}) == len(jobs)
    assert sum(job["kind"] == "train" for job in jobs) == 5
    evaluations = [job for job in jobs if job["kind"] == "evaluate"]
    assert all(job["config"]["solve_budget"] == 41 for job in evaluations)
    assert all(job["config"]["refinement_factors"] == [2, 4] for job in evaluations)
    assert (
        len(
            [
                job
                for job in evaluations
                if job["condition"] == "primary" and job["policy_label"] == "uniform"
            ]
        )
        == 20
    )
    bad = config.model_copy(update={"training_seeds": [1042, 1043]})
    with pytest.raises(ValueError, match="overlap"):
        build_plan(bad)


@pytest.mark.parametrize("stop", [False, True])
def test_controller_stops_before_launching_when_budget_or_stop_request_applies(
    tmp_path, monkeypatch, stop
):
    job = {"id": "never_launch", "kind": "tests", "estimate_seconds": 60}
    atomic_json(tmp_path / "plan.json", {"jobs": [job]})
    atomic_json(
        tmp_path / "status.json",
        {
            "status": "prepared",
            "jobs": {},
            "deadline_utc": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            "updated_utc": "initial",
        },
    )
    if stop:
        (tmp_path / "STOP").touch()

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("No process should be launched after stop/deadline")

    monkeypatch.setattr("oxygen_shape.experiments.long_study.subprocess.Popen", forbidden_launch)
    controller(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == ("stopped_by_request" if stop else "budget_complete")
    assert status["idle_sleep_prevented"] is False
    assert status["active_progress"] is None
    assert (tmp_path / "report.md").is_file()


def test_atomic_progress_retries_transient_file_locks(tmp_path, monkeypatch):
    import os

    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}')
    original_replace = os.replace
    calls, delays = [], []

    def locked_replace(source, target):
        calls.append(source)
        if len(calls) <= 2:
            assert json.loads(destination.read_text()) == {"old": True}
            raise PermissionError("Windows sharing lock")
        original_replace(source, target)

    monkeypatch.setattr("oxygen_shape.experiments.study_worker.os.replace", locked_replace)
    monkeypatch.setattr("oxygen_shape.experiments.study_worker.sleep", delays.append)
    atomic_json(destination, {"new": 3})
    assert json.loads(destination.read_text()) == {"new": 3}
    assert len(calls) == 3
    assert delays == [0.02, 0.04]
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_progress_permanent_lock_keeps_old_file_and_raises(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}')
    delays = []

    def locked_replace(*args):
        raise PermissionError("Permanent lock")

    monkeypatch.setattr("oxygen_shape.experiments.study_worker.os.replace", locked_replace)
    monkeypatch.setattr("oxygen_shape.experiments.study_worker.sleep", delays.append)
    with pytest.raises(PermissionError, match="Permanent"):
        atomic_json(destination, {"new": 3})
    assert len(delays) == 6
    assert sum(delays) <= 1
    assert json.loads(destination.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))
