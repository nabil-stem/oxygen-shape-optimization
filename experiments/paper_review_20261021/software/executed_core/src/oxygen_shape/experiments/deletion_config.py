"""Prespecified, constrained experiment for the low-flux deletion hypothesis."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import RunConfig
from oxygen_shape.optimization.config import SearchConfig


class DeletionConfig(RunConfig):
    n: int = Field(default=100, ge=20, strict=True)
    shapes: list[Literal["deep_cavity", "rectangle", "sinusoidal"]] = Field(
        default_factory=lambda: ["deep_cavity", "rectangle"]
    )
    permeabilities: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=lambda: [0.1, 1.0, 10.0]
    )
    target_area: float = Field(default=0.24, gt=0, lt=0.7)
    perimeter_limit: float = Field(default=4.0, gt=0)
    minimum_feature_width: float = Field(default=0.04, gt=0)
    protected_length: float = Field(default=0.12, gt=0, lt=0.5)
    radii: tuple[float, float] = (0.05, 0.08)
    refinement_factors: list[Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=lambda: [2, 4]
    )
    samples_per_group: int = Field(default=8, ge=1, strict=True)
    max_attempts_per_group: int = Field(default=2000, ge=1, strict=True)
    low_flux_fraction: float = Field(default=0.2, gt=0, lt=1)
    max_removed_area_fraction: float = Field(default=0.05, gt=0, lt=0.5)
    effect_tolerance: float = Field(default=1e-7, gt=0)

    @model_validator(mode="after")
    def check_matrix(self) -> "DeletionConfig":
        # Reuse the optimizer's dimensional, source-alignment and grid checks.
        SearchConfig.model_validate(
            self.model_dump(
                include={
                    "n",
                    "target_area",
                    "perimeter_limit",
                    "minimum_feature_width",
                    "protected_length",
                    "radii",
                    "refinement_factors",
                }
            )
        )
        if not self.shapes or len(set(self.shapes)) != len(self.shapes):
            raise ValueError("shapes must be nonempty and unique")
        if not self.permeabilities or self.permeabilities != sorted(set(self.permeabilities)):
            raise ValueError("permeabilities must be nonempty, distinct and increasing")
        if self.max_attempts_per_group < self.samples_per_group:
            raise ValueError("Attempt budget must cover the sample quota")
        return self
