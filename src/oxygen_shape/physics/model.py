"""Validated dimensionless model parameters."""

from pydantic import BaseModel, ConfigDict, Field


class DiffusionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    Bi: float = Field(default=1.0, ge=0)


class SolverSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    conservation_rtol: float = Field(default=1e-9, gt=0)
    conservation_atol: float = Field(default=1e-12, gt=0)
    residual_tolerance: float = Field(default=1e-10, gt=0)
    bounds_tolerance: float = Field(default=1e-10, gt=0)
