"""Frozen multi-start curriculum, controls, and held-out RL study settings."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import StrictConfig
from oxygen_shape.experiments.prediction_config import FrozenReference, PredictionConfig
from oxygen_shape.optimization.guided_config import GuidedConfig


class TrainingReference(StrictConfig):
    family: Literal["deep_cavity", "rectangle"]
    reference: FrozenReference


class PioneerRLConfig(GuidedConfig):
    training_references: list[TrainingReference] = Field(default_factory=list)
    replica_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: list(range(50000, 50005))
    )
    validation_generation_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [61000]
    )
    test_generation_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [62000, 62001]
    )
    validation_rollout_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [71000]
    )
    test_rollout_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [72000, 72001]
    )
    episodes: int = Field(default=24, ge=1, strict=True)
    solve_budget: int = Field(default=17, ge=2, strict=True)
    raw_draw_budget: int = Field(default=6000, ge=1, strict=True)
    pool_size: int = Field(default=8, ge=2, le=64, strict=True)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    # Episodes terminate within raw_draw_budget decisions, so gamma=1 is finite.
    discount: float = Field(default=0.95, ge=0, le=1)
    candidate_screening: Literal["cheap", "full"] = "cheap"
    refill_quota: Literal[0, 2, 4] = 0
    refill_draw_limit: int = Field(default=256, ge=4, strict=True)
    flux_prior_scale: float = Field(default=0, ge=0)
    excluded_geometry_hashes: list[str] = Field(default_factory=list)
    epsilon_start: float = Field(default=1, ge=0, le=1)
    epsilon_end: float = Field(default=0.15, ge=0, le=1)
    reward_scale: float = Field(default=100, gt=0)
    invalid_reward: float = Field(default=-0.01, le=0)
    families: list[Literal["deep_cavity", "rectangle"]] = Field(
        default_factory=lambda: ["deep_cavity", "rectangle"]
    )

    @model_validator(mode="after")
    def check_protocol(self) -> "PioneerRLConfig":
        if self.refill_quota and (
            self.candidate_screening != "full" or self.refill_draw_limit < self.pool_size
        ):
            raise ValueError("Refill requires full screening and a cap at least the base pool")
        if self.model.Bi <= 0:
            raise ValueError("RL flux features require positive permeability")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in self.excluded_geometry_hashes
        ):
            raise ValueError("Excluded geometries must be SHA256 mask hashes")
        streams = [
            self.replica_seeds,
            self.validation_generation_seeds,
            self.test_generation_seeds,
            self.validation_rollout_seeds,
            self.test_rollout_seeds,
        ]
        if any(not s or len(set(s)) != len(s) for s in streams):
            raise ValueError("Use nonempty distinct seed lists")
        if any(set(a) & set(b) for i, a in enumerate(streams) for b in streams[i + 1 :]):
            raise ValueError("Training, generation and evaluation streams must be disjoint")
        training = {seed * 1000 + e for seed in self.replica_seeds for e in range(self.episodes)}
        if (
            training & set(sum(streams, []))
            or len(training) != len(self.replica_seeds) * self.episodes
        ):
            raise ValueError("Episode seed streams overlap")
        if self.episodes >= 1000 or self.epsilon_start < self.epsilon_end:
            raise ValueError("Use fewer than 1000 episodes and decaying exploration")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("Use distinct geometry families")
        if self.n != 100 or self.refinement_factors != [2, 4]:
            raise ValueError("This validated RL curriculum uses n=100 and refinements 200/400")
        if any(r.family not in self.families for r in self.training_references):
            raise ValueError("Reference family is outside the curriculum")
        if {r.reference.expected_seed for r in self.training_references} & set(sum(streams, [])):
            raise ValueError("New seed streams overlap development-reference seeds")
        return self

    def reference_config(self) -> PredictionConfig:
        fields = {
            key: value
            for key, value in self.model_dump().items()
            if key in PredictionConfig.model_fields
        }
        return PredictionConfig(**fields, shapes=self.families)

    def resolve_paths(self, base: Path) -> "PioneerRLConfig":
        resolved = super().resolve_paths(base)
        refs = [
            r.model_copy(
                update={
                    "reference": r.reference.model_copy(
                        update={
                            "run_directory": r.reference.run_directory
                            if r.reference.run_directory.is_absolute()
                            else (base / r.reference.run_directory).resolve()
                        }
                    )
                }
            )
            for r in self.training_references
        ]
        return resolved.model_copy(update={"training_references": refs})
