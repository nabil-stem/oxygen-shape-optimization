"""Portable protocol for interior square islands, distinct from the v0.1.0 study."""

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import RunConfig


class HolesConfig(RunConfig):
    n: int = Field(default=100, ge=40, strict=True)
    families: list[str] = ["maze_chamber", "rectangle", "hexagon", "acinar_sacs", "maze"]
    target_area: float = Field(default=0.24, gt=0, lt=0.7)
    widths: list[float] = [0.04, 0.08]
    position_stride: float = Field(default=0.08, gt=0)
    gap: float = Field(default=0.04, gt=0)
    minimum_feature_width: float = Field(default=0.04, gt=0)
    protected_length: float = Field(default=0.12, gt=0)
    permeabilities: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
    validation_Bi: list[float] = [0.01, 1.0, 100.0]
    refinement_factors: list[int] = [2, 4]
    fem_max_areas: list[float] = [8e-5, 2e-5]
    search_seeds: list[int] = [420000, 420001, 420002, 420003, 420004]
    search_budget: int = Field(default=129, ge=1, strict=True)
    search_raw_budget: int = Field(default=50000, ge=1, strict=True)
    search_Bi: float = Field(default=1.0, gt=0)
    exploration: float = Field(default=0.25, gt=0, le=1)
    min_improvement: float = Field(default=1e-10, ge=0)

    @model_validator(mode="after")
    def check_protocol(self) -> "HolesConfig":
        if self.n % 20:
            raise ValueError("Use a grid multiple of 20")
        if (
            not self.families
            or len(set(self.families)) != len(self.families)
            or any(
                f not in ("maze_chamber", "rectangle", "acinar_sacs", "maze", "hexagon")
                for f in self.families
            )
        ):
            raise ValueError("Declare distinct supported geometry families")
        for values in (self.widths, self.permeabilities, self.validation_Bi, self.fem_max_areas):
            if not values or any(v <= 0 for v in values) or len(set(values)) != len(values):
                raise ValueError("Widths, Bi values and FEM areas must be distinct and positive")
        for value in [*self.widths, self.position_stride, 0.08]:
            if abs(value * self.n - round(value * self.n)) > 1e-10:
                raise ValueError("Widths, position stride and 0.08 search squares must fit grid")
        if any(w < self.minimum_feature_width for w in self.widths):
            raise ValueError("Hole width cannot be below the feature cutoff")
        if not set(self.validation_Bi).issubset(self.permeabilities):
            raise ValueError("Validate Bi values from the declared sweep")
        if self.search_Bi not in self.permeabilities:
            raise ValueError("Search Bi must be in the declared sweep")
        if (
            not self.refinement_factors
            or any(
                isinstance(f, bool) or not isinstance(f, int) or f < 2
                for f in self.refinement_factors
            )
            or len(set(self.refinement_factors)) != len(self.refinement_factors)
        ):
            raise ValueError("Declare distinct integer refinement factors >=2")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.search_seeds):
            raise ValueError("Search seeds must be nonnegative integers")
        if len(set(self.search_seeds)) != len(self.search_seeds):
            raise ValueError("Duplicate search seed")
        return self
