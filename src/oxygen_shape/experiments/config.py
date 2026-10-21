"""Strict YAML configuration with paths relative to the configuration file."""

from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from oxygen_shape.physics.model import DiffusionModel, SolverSettings

ConfigType = TypeVar("ConfigType", bound="RunConfig")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RectangleConfig(StrictConfig):
    nx: int = Field(default=80, ge=1, strict=True)
    ny: int = Field(default=60, ge=1, strict=True)
    h: float = Field(default=0.0125, gt=0)
    boundary_mode: Literal["chamber", "slab"] = "chamber"


class RunConfig(StrictConfig):
    seed: int = Field(default=42, ge=0, strict=True)
    output_root: Path = Path("../experiments")
    solver: SolverSettings = Field(default_factory=SolverSettings)

    def resolve_paths(self: ConfigType, base: Path) -> ConfigType:
        root = self.output_root
        if not root.is_absolute():
            root = (base / root).resolve()
        return self.model_copy(update={"output_root": root})


class ExperimentConfig(RunConfig):
    model: DiffusionModel = Field(default_factory=DiffusionModel)


class SolveConfig(ExperimentConfig):
    geometry: RectangleConfig = Field(default_factory=RectangleConfig)


class SlabConfig(ExperimentConfig):
    resolutions: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=lambda: [8, 16, 32, 64, 128]
    )
    length: float = Field(default=1.0, gt=0)
    transverse_cells: int = Field(default=3, ge=1, strict=True)
    validation_tolerance: float = Field(default=1e-9, gt=0)

    @model_validator(mode="after")
    def valid_resolutions(self) -> "SlabConfig":
        if (
            len(self.resolutions) < 2
            or len(set(self.resolutions)) != len(self.resolutions)
            or any(isinstance(n, bool) or n < 1 for n in self.resolutions)
            or self.resolutions != sorted(self.resolutions)
        ):
            raise ValueError(
                "resolutions must contain at least two distinct increasing positive integers"
            )
        return self


class BaselineConfig(RunConfig):
    resolutions: list[Annotated[int, Field(strict=True, ge=20)]] = Field(
        default_factory=lambda: [100, 200, 400]
    )
    permeabilities: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0]
    )
    target_area: float = Field(default=0.24, gt=0, lt=0.7)
    perimeter_budget: float = Field(default=18.0, gt=0)
    target_perimeter: float = Field(default=3.0, gt=0)
    hilbert_order: int = Field(default=2, ge=1, le=4, strict=True)
    refinement_warning_rtol: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_matrix(self) -> "BaselineConfig":
        if (
            len(self.resolutions) < 2
            or self.resolutions != sorted(set(self.resolutions))
            or any(n % 20 for n in self.resolutions)
            or any(
                b != 2 * a for a, b in zip(self.resolutions[:-1], self.resolutions[1:], strict=True)
            )
        ):
            raise ValueError(
                "Use at least two increasing nested resolutions, multiples of 20, doubling"
            )
        if self.permeabilities != sorted(set(self.permeabilities)) or not self.permeabilities:
            raise ValueError("permeabilities must be distinct, positive, and increasing")
        return self


def load_config(path: Path, schema: type[ConfigType]) -> ConfigType:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = schema.model_validate(data)
    return config.resolve_paths(path.parent)
