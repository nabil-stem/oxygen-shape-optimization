"""Common solve-budget protocol for uniform, flux and empirical sensitivity."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.optimization.config import SearchConfig


class GuidedConfig(SearchConfig):
    batch_size: int = Field(default=8, ge=2, strict=True)
    probe_count: int = Field(default=4, ge=1, strict=True)
    epsilon_uniform: float = Field(default=0.25, gt=0, le=1)
    kernel_bandwidth: float = Field(default=0.15, gt=0)

    @model_validator(mode="after")
    def check_guidance(self) -> "GuidedConfig":
        if self.probe_count >= self.batch_size:
            raise ValueError("Probe count must leave room for guided candidates")
        if self.model.Bi <= 0:
            raise ValueError("Shape search requires positive permeability")
        return self


class SensitivityBenchmarkConfig(GuidedConfig):
    evaluation_seeds: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [8000, 8001, 8002, 8003, 8004]
    )
    shapes: list[Literal["deep_cavity", "rectangle"]] = Field(
        default_factory=lambda: ["deep_cavity", "rectangle"]
    )
    target_relative_gain: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def check_benchmark(self) -> "SensitivityBenchmarkConfig":
        if not self.evaluation_seeds or len(self.evaluation_seeds) != len(
            set(self.evaluation_seeds)
        ):
            raise ValueError("Evaluation seeds must be nonempty and unique")
        if not self.shapes or len(self.shapes) != len(set(self.shapes)):
            raise ValueError("Shape families must be nonempty and unique")
        return self
