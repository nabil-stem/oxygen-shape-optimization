"""Reproducible random-search and reinforcement-learning configurations."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import ExperimentConfig


class SearchConfig(ExperimentConfig):
    n: int = Field(default=100, ge=20, strict=True)
    initial_shape: Literal["deep_cavity", "rectangle", "sinusoidal"] = "deep_cavity"
    target_area: float = Field(default=0.24, gt=0, lt=0.7)
    perimeter_limit: float = Field(default=4.0, gt=0)
    minimum_feature_width: float = Field(default=0.04, gt=0)
    protected_length: float = Field(default=0.12, gt=0, lt=0.5)
    radii: tuple[float, float] = (0.05, 0.08)
    solve_budget: int = Field(default=81, ge=2, strict=True)
    max_attempts_per_solve: int = Field(default=30, ge=1, strict=True)
    min_improvement: float = Field(default=1e-7, ge=0)
    policy_stall_limit: int = Field(default=50, ge=1, strict=True)
    refinement_factors: list[Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=lambda: [2, 4]
    )

    @model_validator(mode="after")
    def validate_search(self) -> "SearchConfig":
        if self.n % 20:
            raise ValueError("n must be a multiple of 20 to preserve the fixed source")
        if abs(self.protected_length * self.n - round(self.protected_length * self.n)) > 1e-10:
            raise ValueError("Protected collar length must align with the search cell edges")
        if not 3 / self.n <= self.radii[0] <= self.radii[1] <= 0.25:
            raise ValueError("Require 3h <= radius_min <= radius_max <= 0.25")
        if self.minimum_feature_width < 2 / self.n:
            raise ValueError("Minimum feature width must span at least two grid cells")
        if not self.refinement_factors or self.refinement_factors != sorted(
            set(self.refinement_factors)
        ):
            raise ValueError("Use distinct increasing refinement factors >=2")
        return self


class RLConfig(SearchConfig):
    training_episodes: int = Field(default=40, ge=1, strict=True)
    training_solve_budget: int = Field(default=25, ge=2, strict=True)
    training_attempts_per_solve: int = Field(default=30, ge=1, strict=True)
    evaluation_seeds: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4]
    )
    training_seed_offset: int = Field(default=10000, ge=0, strict=True)
    learning_rate: float = Field(default=0.15, gt=0, le=1)
    discount: float = Field(default=0.9, ge=0, lt=1)
    epsilon_start: float = Field(default=1.0, ge=0, le=1)
    epsilon_end: float = Field(default=0.15, ge=0, le=1)
    evaluation_epsilon: float = Field(default=0.1, ge=0, le=1)
    invalid_reward: float = Field(default=-0.01, le=0)
    reward_scale: float = Field(default=100.0, gt=0)

    @model_validator(mode="after")
    def validate_training(self) -> "RLConfig":
        seeds = self.evaluation_seeds
        training = {
            self.training_seed_offset + self.seed + i for i in range(self.training_episodes)
        }
        if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & training:
            raise ValueError(
                "Evaluation seeds must be nonempty, unique, and disjoint from training"
            )
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("Exploration must decay from epsilon_start to epsilon_end")
        return self
