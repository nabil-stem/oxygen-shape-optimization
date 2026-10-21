"""Fixed geometry, disjoint calibration/test streams and radius diagnostics."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import ExperimentConfig, StrictConfig
from oxygen_shape.optimization.guided_config import GuidedConfig


class FrozenReference(StrictConfig):
    run_directory: Path
    mask_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_seed: int = Field(ge=0, strict=True)
    expected_method: str = Field(default="uniform", min_length=1)
    minimum_perimeter_fraction: float = Field(default=0.99, gt=0, le=1)


class PredictionConfig(ExperimentConfig):
    seed: int = Field(default=8200, ge=0, strict=True)
    replicates: int = Field(default=5, ge=1, le=100, strict=True)
    n: int = Field(default=100, ge=20, strict=True)
    shapes: list[Literal["deep_cavity", "rectangle"]] = Field(
        default_factory=lambda: ["deep_cavity", "rectangle"]
    )
    target_area: float = Field(default=0.24, gt=0, lt=0.7)
    perimeter_limit: float = Field(default=4, gt=0)
    minimum_feature_width: float = Field(default=0.04, gt=0)
    protected_length: float = Field(default=0.12, gt=0, lt=0.5)
    radii: tuple[float, float] = (0.05, 0.08)
    refinement_factors: list[Annotated[int, Field(ge=2, strict=True)]] = Field(
        default_factory=lambda: [2, 4]
    )
    calibration_sizes: list[Annotated[int, Field(ge=2, strict=True)]] = Field(
        default_factory=lambda: [4, 8, 16, 24]
    )
    heldout_samples: int = Field(default=48, ge=4, strict=True)
    max_attempts_per_pool: int = Field(default=6000, ge=1, strict=True)
    radius_families: int = Field(default=8, ge=1, strict=True)
    matched_radii: list[float] = Field(default_factory=lambda: [0.05, 0.065, 0.08])
    spatial_bandwidth: float = Field(default=0.15, gt=0)
    radius_bandwidth: float = Field(default=0.01, gt=0)
    top_fraction: float = Field(default=0.25, gt=0, lt=1)
    effect_tolerance: float = Field(default=1e-7, gt=0)
    frozen_references: dict[Literal["deep_cavity", "rectangle"], FrozenReference] = Field(
        default_factory=dict
    )
    include_initial_controls: bool = False

    @model_validator(mode="after")
    def validate_diagnostic(self) -> "PredictionConfig":
        self.search_config()
        if not self.shapes or len(set(self.shapes)) != len(self.shapes):
            raise ValueError("Use nonempty unique shape families")
        if self.frozen_references and set(self.frozen_references) != set(self.shapes):
            raise ValueError("Provide a frozen reference for every configured shape")
        if self.include_initial_controls and not self.frozen_references:
            raise ValueError("Initial controls require frozen references")
        if not self.calibration_sizes or self.calibration_sizes != sorted(
            set(self.calibration_sizes)
        ):
            raise ValueError("Calibration prefixes must be distinct and increasing")
        if self.max_attempts_per_pool < max(self.calibration_sizes[-1], self.heldout_samples):
            raise ValueError("Attempt budget must cover each pool quota")
        if (
            len(self.matched_radii) < 2
            or self.matched_radii != sorted(set(self.matched_radii))
            or any(not self.radii[0] <= radius <= self.radii[1] for radius in self.matched_radii)
        ):
            raise ValueError(
                "Matched radii must be increasing, distinct and inside the proposal range"
            )
        return self

    @property
    def case_specs(self) -> dict[str, tuple[str, str]]:
        cases = {}
        for shape in self.shapes:
            if not self.frozen_references:
                cases[shape] = (shape, "initial")
            else:
                if self.include_initial_controls:
                    cases[f"{shape}__initial"] = (shape, "initial")
                cases[f"{shape}__optimized"] = (shape, "optimized")
        return cases

    @property
    def case_labels(self) -> list[str]:
        return list(self.case_specs)

    def resolve_paths(self, base: Path) -> "PredictionConfig":
        resolved = super().resolve_paths(base)
        references = {
            shape: reference.model_copy(
                update={
                    "run_directory": (
                        reference.run_directory
                        if reference.run_directory.is_absolute()
                        else (base / reference.run_directory).resolve()
                    )
                }
            )
            for shape, reference in self.frozen_references.items()
        }
        return resolved.model_copy(update={"frozen_references": references})

    def search_config(self, shape: str = "deep_cavity") -> GuidedConfig:
        fields = {
            key: value
            for key, value in self.model_dump().items()
            if key in GuidedConfig.model_fields
        }
        return GuidedConfig(**fields, initial_shape=shape)
