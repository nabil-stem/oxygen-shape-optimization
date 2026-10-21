"""Launch and supervise a deadline-bounded study in a hidden local process."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import Field

from oxygen_shape.experiments.config import RunConfig
from oxygen_shape.experiments.reporting import create_run, provenance
from oxygen_shape.experiments.study_report import write_study_report
from oxygen_shape.experiments.study_worker import atomic_json, execute_job
from oxygen_shape.optimization.config import RLConfig


class StudyConfig(RunConfig):
    reference_checkpoint: Path = Path("../models/disk_q_policy.npz")
    deadline_utc: str
    training_episodes: int = Field(default=40, ge=1)
    training_seeds: list[int] = Field(default_factory=lambda: [1042, 2042, 3042, 4042, 5042])
    primary_seeds: list[int] = Field(default_factory=lambda: list(range(200, 220)))
    transfer_seeds: list[int] = Field(default_factory=lambda: list(range(300, 305)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_plan(config: StudyConfig) -> list[dict]:
    primary = config.primary_seeds
    all_eval = primary + config.transfer_seeds
    train_rng = {
        10000 + seed + i for seed in config.training_seeds for i in range(config.training_episodes)
    }
    if len(set(config.training_seeds)) != len(config.training_seeds) or set(all_eval) & train_rng:
        raise ValueError("Training replicas and evaluation seeds must be distinct")
    if len(train_rng) != len(config.training_seeds) * config.training_episodes:
        raise ValueError("Training episode seed ranges overlap")
    base = RLConfig(
        seed=42,
        solve_budget=41,
        max_attempts_per_solve=100,
        training_episodes=config.training_episodes,
        evaluation_seeds=primary,
    )
    jobs = [{"id": "regression_gate", "kind": "tests", "estimate_seconds": 60}]

    def evaluation(
        condition,
        label,
        seed,
        *,
        shape="deep_cavity",
        Bi=1.0,
        checkpoint="checkpoints/reference/policy.npz",
        dependency=None,
        fallback=50,
    ):
        cfg = base.model_copy(
            update={
                "seed": seed,
                "initial_shape": shape,
                "model": base.model.model_copy(update={"Bi": Bi}),
                "policy_stall_limit": fallback,
            }
        )
        jobs.append(
            {
                "id": f"{condition}_{label}_{seed}",
                "kind": "evaluate",
                "condition": condition,
                "policy_label": label,
                "checkpoint": checkpoint,
                "depends_on": dependency,
                "transfer": condition.startswith("transfer"),
                "config": cfg.model_dump(mode="json"),
                "estimate_seconds": 55 if fallback == 50 else 120,
            }
        )

    for seed in primary[:10]:
        for label in ("uniform", "untrained", "reference_rl"):
            evaluation("primary", label, seed)
    for index, seed in enumerate(config.training_seeds):
        name = f"replica_{index + 1}"
        job_id = f"train_{name}"
        cfg = base.model_copy(update={"seed": seed})
        jobs.append(
            {
                "id": job_id,
                "kind": "train",
                "config": cfg.model_dump(mode="json"),
                "estimate_seconds": 500,
            }
        )
        for eval_seed in primary[:5]:
            evaluation(
                "primary",
                name,
                eval_seed,
                checkpoint=f"jobs/{job_id}/policy.npz",
                dependency=job_id,
            )
    for condition, shape, Bi in (
        ("transfer_rectangle", "rectangle", 1.0),
        ("transfer_sinusoidal", "sinusoidal", 1.0),
        ("transfer_Bi_0.1", "deep_cavity", 0.1),
        ("transfer_Bi_10", "deep_cavity", 10.0),
    ):
        for seed in config.transfer_seeds:
            for label in ("uniform", "untrained", "reference_rl"):
                evaluation(condition, label, seed, shape=shape, Bi=Bi)
    for seed in primary[10:]:
        for label in ("uniform", "untrained", "reference_rl"):
            evaluation("primary", label, seed)
    for index, _ in enumerate(config.training_seeds):
        name = f"replica_{index + 1}"
        for seed in primary[5:10]:
            evaluation(
                "primary",
                name,
                seed,
                checkpoint=f"jobs/train_{name}/policy.npz",
                dependency=f"train_{name}",
            )
    for seed in config.transfer_seeds:
        evaluation("fallback_ablation", "reference_without_fallback", seed, fallback=100000)
    return jobs


def prepare(config_path: Path) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = StudyConfig.model_validate(raw)
    deadline = datetime.fromisoformat(config.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.timestamp() <= time.time():
        raise ValueError("Deadline must be a future timezone-aware UTC time")
    root_output = config.output_root
    if not root_output.is_absolute():
        root_output = (config_path.parent / root_output).resolve()
    config = config.model_copy(update={"output_root": root_output})
    root = create_run(config, "two_hour_rl_study")
    snapshot = root / "source_snapshot"
    workspace = Path(__file__).resolve().parents[3]
    shutil.copytree(
        workspace / "tests", snapshot / "tests", ignore=shutil.ignore_patterns("__pycache__")
    )
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    for test in (snapshot / "tests").glob("*.py"):
        manifest[test.relative_to(snapshot).as_posix()] = hashlib.sha256(
            test.read_bytes()
        ).hexdigest()
    atomic_json(root / "source_manifest.json", manifest)
    checkpoint = (config_path.parent / config.reference_checkpoint).resolve()
    target = root / "checkpoints/reference"
    target.mkdir(parents=True)
    shutil.copyfile(checkpoint, target / "policy.npz")
    shutil.copyfile(checkpoint.with_name("policy_metadata.json"), target / "policy_metadata.json")
    jobs = build_plan(config)
    atomic_json(root / "plan.json", {"jobs": jobs})
    atomic_json(
        root / "environment.json",
        {
            **provenance(),
            "workspace": str(workspace),
            "reference_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "deadline_utc": config.deadline_utc,
        },
    )
    state = {
        "status": "prepared",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "deadline_utc": config.deadline_utc,
        "jobs": {},
        "active_job": None,
    }
    atomic_json(root / "status.json", state)
    write_study_report(root, state, jobs)
    return root


def worker_environment(root: Path) -> dict:
    return {
        **os.environ,
        "PYTHONPATH": str(root / "source_snapshot/src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def prevent_idle_sleep(enabled: bool) -> bool:
    if os.name != "nt":
        return False
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    return bool(function(0x80000000 | (0x00000001 if enabled else 0)))


def controller(root: Path) -> None:
    jobs = json.loads((root / "plan.json").read_text(encoding="utf-8"))["jobs"]
    state = json.loads((root / "status.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(state["deadline_utc"].replace("Z", "+00:00")).timestamp()
    state.update(
        status="running", controller_pid=os.getpid(), idle_sleep_prevented=prevent_idle_sleep(True)
    )
    durations: dict[str, list[float]] = {}

    def publish():
        state["updated_utc"] = utc_now()
        atomic_json(root / "status.json", state)
        write_study_report(root, state, jobs)

    publish()
    for job in jobs:
        if (root / "STOP").exists():
            state["status"] = "stopped_by_request"
            break
        dependency = job.get("depends_on")
        if dependency and state["jobs"].get(dependency) != "complete":
            state["jobs"][job["id"]] = "skipped_dependency"
            publish()
            continue
        estimate = job["estimate_seconds"]
        if durations.get(job["kind"]):
            ordered = sorted(durations[job["kind"]])
            estimate = max(estimate / 2, ordered[len(ordered) // 2])
        if deadline - time.time() < estimate * 1.2 + 30:
            state["status"] = "budget_complete"
            break
        path = root / "jobs" / job["id"]
        path.mkdir(parents=True)
        atomic_json(path / "job.json", job)
        started = time.monotonic()
        with (path / "run.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "oxygen_shape.experiments.long_study",
                    "worker",
                    str(root),
                    job["id"],
                ],
                env=worker_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            state.update(active_job=job["id"], worker_pid=process.pid, active_progress=None)
            publish()
            last_publish = time.monotonic()
            while process.poll() is None:
                if time.time() >= deadline or (root / "STOP").exists():
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    state["jobs"][job["id"]] = "interrupted"
                    state["status"] = (
                        "budget_complete" if time.time() >= deadline else "stopped_by_request"
                    )
                    break
                if time.monotonic() - last_publish >= 30:
                    progress_path = path / "progress.json"
                    if progress_path.exists():
                        state["active_progress"] = json.loads(
                            progress_path.read_text(encoding="utf-8")
                        )
                    publish()
                    last_publish = time.monotonic()
                time.sleep(1)
        duration = time.monotonic() - started
        state["active_job"] = None
        state["worker_pid"] = None
        state["active_progress"] = None
        if state["status"] != "running":
            break
        state["jobs"][job["id"]] = "complete" if process.returncode == 0 else "failed"
        if process.returncode and (job["kind"] == "tests" or not (path / "result.json").exists()):
            state["status"] = "halted_worker_or_regression_failure"
            publish()
            break
        if process.returncode == 2:
            state["status"] = "halted_numerical_failure"
            publish()
            break
        if process.returncode == 0:
            durations.setdefault(job["kind"], []).append(duration)
        publish()
    else:
        state["status"] = "all_planned_jobs_complete"
    prevent_idle_sleep(False)
    state.update(
        active_job=None,
        worker_pid=None,
        active_progress=None,
        idle_sleep_prevented=False,
        finished_utc=utc_now(),
    )
    publish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "controller", "worker"))
    parser.add_argument("path", type=Path)
    parser.add_argument("job_id", nargs="?")
    args = parser.parse_args()
    if args.mode == "launch":
        root = prepare(args.path.resolve())
        with (root / "controller.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "oxygen_shape.experiments.long_study",
                    "controller",
                    str(root),
                ],
                env=worker_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )
        print(json.dumps({"study_directory": str(root), "controller_pid": process.pid}))
    elif args.mode == "controller":
        try:
            controller(args.path.resolve())
        except BaseException as exc:
            state = json.loads((args.path / "status.json").read_text(encoding="utf-8"))
            state.update(status="halted_controller_error", reason=str(exc), updated_utc=utc_now())
            atomic_json(args.path / "status.json", state)
            jobs = json.loads((args.path / "plan.json").read_text(encoding="utf-8"))["jobs"]
            write_study_report(args.path, state, jobs)
            raise
        finally:
            prevent_idle_sleep(False)
    else:
        jobs = json.loads((args.path / "plan.json").read_text(encoding="utf-8"))["jobs"]
        job = next(job for job in jobs if job["id"] == args.job_id)
        raise SystemExit(execute_job(args.path.resolve(), job))


if __name__ == "__main__":
    main()
