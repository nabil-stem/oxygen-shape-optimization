"""Prespecified geometry audit and frozen-input equal-budget search."""

from typing import Annotated

from pydantic import Field, model_validator

from oxygen_shape.experiments.prediction_config import PredictionConfig
from oxygen_shape.optimization.perimeter_pool import PoolMethod


class FeasibilityConfig(PredictionConfig):
    seed: int = Field(default=8600, ge=0, strict=True)
    pools_per_replicate: int = Field(default=64, ge=1, strict=True)
    pool_size: int = Field(default=8, ge=2, le=64, strict=True)
    epsilon_uniform: float = Field(default=0.25, gt=0, le=1)
    search_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: list(range(8800, 8805))
    )
    search_solve_budget: int = Field(default=17, ge=2, strict=True)
    raw_draw_budget: int = Field(default=6000, ge=1, strict=True)
    methods: list[PoolMethod] = Field(default_factory=lambda: ["uniform", "perimeter_pool"])
    audit_response_solves: bool = False

    @model_validator(mode="after")
    def validate_feasibility(self) -> "FeasibilityConfig":
        if not self.frozen_references:
            raise ValueError("The feasibility study requires pinned frozen inputs")
        if (
            not self.methods
            or len(set(self.methods)) != len(self.methods)
            or not {"uniform", "perimeter_pool"}.issubset(self.methods)
        ):
            raise ValueError("Use distinct methods including uniform and perimeter_pool controls")
        if self.audit_response_solves and "screened_uniform" not in self.methods:
            raise ValueError("Response ablation requires screened_uniform")
        if not self.search_seeds or len(set(self.search_seeds)) != len(self.search_seeds):
            raise ValueError("Search seeds must be nonempty and unique")
        if set(self.search_seeds) & set(range(self.seed, self.seed + self.replicates)):
            raise ValueError("Audit and search seeds must be disjoint")
        return self
