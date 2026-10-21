"""Command-line entry points for the validated reference solver."""

from pathlib import Path
from typing import Annotated

import typer

from oxygen_shape.experiments.baselines import run_baselines
from oxygen_shape.experiments.config import BaselineConfig, SlabConfig, SolveConfig, load_config
from oxygen_shape.experiments.deletion_config import DeletionConfig
from oxygen_shape.experiments.deletion_hypothesis import run_deletion_study
from oxygen_shape.experiments.feasibility import run_feasibility_study
from oxygen_shape.experiments.feasibility_config import FeasibilityConfig
from oxygen_shape.experiments.feasible_pool_study import PoolStudyConfig, run_feasible_pool_study
from oxygen_shape.experiments.optimization import run_optimization
from oxygen_shape.experiments.pioneer_training import run_pioneer_training
from oxygen_shape.experiments.policy_evaluation import evaluate_policy
from oxygen_shape.experiments.prediction_config import PredictionConfig
from oxygen_shape.experiments.prediction_diagnostic import run_prediction_diagnostic
from oxygen_shape.experiments.revision_training import run_revision_study
from oxygen_shape.experiments.rl import run_rl
from oxygen_shape.experiments.rl_benchmark import benchmark_checkpoint
from oxygen_shape.experiments.runner import run_slab_validation, run_solve
from oxygen_shape.experiments.sensitivity_benchmark import run_sensitivity_benchmark
from oxygen_shape.optimization.config import RLConfig, SearchConfig
from oxygen_shape.optimization.guided_config import SensitivityBenchmarkConfig
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig
from oxygen_shape.optimization.revision_config import RevisionConfig

app = typer.Typer(no_args_is_help=True, help="Oxygen diffusion reference experiments (2D).")
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


@app.command("benchmark-valid-pools")
def benchmark_valid_pools(config: ConfigOption) -> None:
    """Compare fixed pools with bounded refill to distinct admissible alternatives."""
    try:
        directory, summary = run_feasible_pool_study(
            load_config(config, PoolStudyConfig), progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Valid-pool study failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Status={summary['status']}; output={directory}")


@app.command("revise-rl")
def revise_rl(
    config: ConfigOption,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    prepare_only: Annotated[bool, typer.Option("--prepare-only")] = False,
) -> None:
    """Ablate rewards, flux priors and geometry screening, then test fresh starts."""
    try:
        directory, summary = run_revision_study(
            load_config(config, RevisionConfig),
            resume=resume,
            prepare_only=prepare_only,
            progress=typer.echo,
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"RL revision study failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Status={summary['status']}; output={directory}")
    if summary["status"] not in ("ready", "complete"):
        raise typer.Exit(1)


@app.command("train-rl-candidates")
def train_rl_candidates(
    config: ConfigOption,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    prepare_only: Annotated[bool, typer.Option("--prepare-only")] = False,
) -> None:
    """Prepare strong split-safe starts, train replicas, and benchmark frozen policies."""
    try:
        directory, summary = run_pioneer_training(
            load_config(config, PioneerRLConfig),
            resume=resume,
            prepare_only=prepare_only,
            progress=typer.echo,
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Candidate RL study failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Status={summary['status']}; output={directory}")
    if summary["status"] not in ("ready", "complete"):
        raise typer.Exit(1)


@app.command("benchmark-feasibility")
def benchmark_feasibility(config: ConfigOption) -> None:
    """Audit geometry-only selection and compare frozen equal-budget searches."""
    try:
        directory, summary = run_feasibility_study(
            load_config(config, FeasibilityConfig), progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Feasibility benchmark failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Status={summary['status']}; PDE solves={summary['number_of_pde_solves']}")
    typer.echo(f"Report: {directory / 'report.md'}")
    if summary["status"] != "complete":
        raise typer.Exit(1)


@app.command("diagnose-prediction")
def diagnose_prediction(config: ConfigOption) -> None:
    """Test fixed-geometry response predictions on disjoint held-out edits."""
    try:
        directory, summary = run_prediction_diagnostic(
            load_config(config, PredictionConfig), progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Prediction diagnostic failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Status={summary['status']}; solves={summary['number_of_pde_solves']}")
    typer.echo(f"Report: {directory / 'report.md'}")
    if summary["status"] != "complete":
        raise typer.Exit(1)


@app.command("benchmark-sensitivity")
def benchmark_sensitivity(config: ConfigOption) -> None:
    """Compare uniform, flux and empirical sensitivity with all probe costs charged."""
    try:
        directory, summary = run_sensitivity_benchmark(
            load_config(config, SensitivityBenchmarkConfig),
            progress=typer.echo,
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Sensitivity benchmark failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Status={summary['status']}; search solves={summary['number_of_pde_solves']}; "
        f"included probes={summary['probe_pde_solves_included']}"
    )
    typer.echo(f"Report: {directory / 'report.md'}")
    if summary["status"] != "complete":
        raise typer.Exit(1)


@app.command("test-deletions")
def test_deletions(config: ConfigOption) -> None:
    """Measure global effects of low-flux/random cuts and fixed-area paired edits."""
    try:
        directory, summary = run_deletion_study(
            load_config(config, DeletionConfig), progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Deletion study failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Status={summary['status']}; solves={summary['number_of_pde_solves']}; "
        f"low-flux deletion gains/losses="
        f"{summary['low_flux_deletion_positive']}/{summary['low_flux_deletion_negative']}"
    )
    typer.echo(f"Report: {directory / 'report.md'}")
    if summary["status"] != "complete":
        raise typer.Exit(1)


@app.command("validate-1d")
def validate_1d(config: ConfigOption) -> None:
    """Validate the finite-volume solver against the exact Robin slab."""
    try:
        directory, summary = run_slab_validation(load_config(config, SlabConfig))
    except (ValueError, ArithmeticError, OSError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Slab validation passed: max |u-u_exact|={summary['max_concentration_error']:.3e}, "
        f"max |q-q_exact|={summary['max_flux_density_error']:.3e}"
    )
    typer.echo(f"Output: {directory}")


@app.command("solve")
def solve_rectangle(config: ConfigOption) -> None:
    """Solve a rectangle and save concentrations, fluxes, and provenance."""
    try:
        directory, summary = run_solve(load_config(config, SolveConfig))
    except (ValueError, ArithmeticError, OSError) as exc:
        typer.echo(f"Solve failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    eta = summary["final_eta"]
    eta_text = "undefined" if eta is None else f"{eta:.6g}"
    typer.echo(
        f"Q={summary['final_Q']:.8g}, eta={eta_text}, "
        f"conservation={summary['conservation_error']:.3e}"
    )
    typer.echo(f"Output: {directory}")


@app.command("compare-shapes")
def compare_shapes(config: ConfigOption) -> None:
    """Run matched-resource baseline shapes, permeability sweep, and grid refinement."""
    try:
        directory, summary = run_baselines(load_config(config, BaselineConfig), progress=typer.echo)
    except (ValueError, ArithmeticError, OSError) as exc:
        typer.echo(f"Comparison failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Baseline sweep passed: {summary['number_of_pde_solves']} solves; "
        f"max conservation={summary['max_conservation_error']:.3e}; "
        f"refinement flags={summary['refinement_flagged_cases']}"
    )
    typer.echo(f"Report: {directory / 'report.md'}")


@app.command("optimize")
def optimize_disks(config: ConfigOption) -> None:
    """Run constrained, area-balanced random disk search and validate on finer grids."""
    try:
        directory, summary = run_optimization(
            load_config(config, SearchConfig), progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Optimization failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Q: {summary['initial_Q']:.7g} -> {summary['final_Q']:.7g}; "
        f"accepted={summary['accepted_moves']}; solves={summary['number_of_pde_solves']}"
    )
    typer.echo(f"Output: {directory}")


@app.command("train-rl")
def train_rl(config: ConfigOption) -> None:
    """Train a tabular Q-policy and benchmark a frozen checkpoint on held-out seeds."""
    try:
        directory, summary = run_rl(load_config(config, RLConfig), progress=typer.echo)
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"RL experiment failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Training solves={summary['training_pde_solves']}; "
        f"evaluation solves={summary['evaluation_pde_solves']}"
    )
    typer.echo(f"Policy: {directory / 'policy.npz'}")
    typer.echo(f"Report: {directory / 'report.md'}")


@app.command("evaluate-rl")
def evaluate_rl(
    config: ConfigOption,
    policy: Annotated[Path, typer.Option("--policy", exists=True, dir_okay=False)],
) -> None:
    """Use a frozen trained policy for a new seed without retraining."""
    try:
        directory, summary = evaluate_policy(load_config(config, RLConfig), policy)
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Policy evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Frozen policy Q: {summary['initial_Q']:.7g} -> {summary['final_Q']:.7g}")
    typer.echo(f"Output: {directory}")


@app.command("benchmark-rl")
def benchmark_rl(
    config: ConfigOption,
    training_run: Annotated[Path, typer.Option("--training-run", exists=True, file_okay=False)],
) -> None:
    """Benchmark an existing training run without repeating its training solves."""
    try:
        directory, _ = benchmark_checkpoint(
            load_config(config, RLConfig), training_run, progress=typer.echo
        )
    except (ValueError, ArithmeticError, OSError, RuntimeError) as exc:
        typer.echo(f"Checkpoint benchmark failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Report: {directory / 'report.md'}")


if __name__ == "__main__":
    app()
