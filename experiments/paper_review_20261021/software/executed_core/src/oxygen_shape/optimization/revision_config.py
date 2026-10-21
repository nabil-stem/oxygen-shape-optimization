"""Prespecified reward, prior and geometric-screening comparisons."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from oxygen_shape.experiments.config import StrictConfig
from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


class RevisionArm(StrictConfig):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    discount: float = Field(ge=0, le=1)
    invalid_reward: float = Field(le=0, allow_inf_nan=False)
    flux_prior_scale: float = Field(default=0, ge=0, allow_inf_nan=False)
    candidate_screening: Literal["cheap", "full"] = "cheap"


def default_arms() -> list[RevisionArm]:
    return [
        RevisionArm(name="g095_penalty", discount=0.95, invalid_reward=-0.01),
        RevisionArm(name="g095_zero", discount=0.95, invalid_reward=0),
        RevisionArm(name="g1_penalty", discount=1, invalid_reward=-0.01),
        RevisionArm(name="g1_zero", discount=1, invalid_reward=0),
        RevisionArm(name="g1_zero_prior", discount=1, invalid_reward=0, flux_prior_scale=1),
        RevisionArm(name="g1_zero_full", discount=1, invalid_reward=0, candidate_screening="full"),
        RevisionArm(
            name="g1_zero_prior_full",
            discount=1,
            invalid_reward=0,
            flux_prior_scale=1,
            candidate_screening="full",
        ),
    ]


class RevisionConfig(PioneerRLConfig):
    seed: int = Field(default=149000, ge=0, strict=True)
    replica_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: list(range(150000, 150005))
    )
    validation_generation_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [161000]
    )
    test_generation_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [162000, 162001]
    )
    validation_rollout_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [171000, 171001]
    )
    test_rollout_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [172000, 172001]
    )
    episodes: int = Field(default=12, ge=1, lt=1000, strict=True)
    arms: list[RevisionArm] = Field(default_factory=default_arms)
    test_core_arms: list[str] = Field(
        default_factory=lambda: ["g095_penalty", "g1_zero_prior_full"]
    )
    excluded_geometry_hashes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_revision(self) -> "RevisionConfig":
        names = [a.name for a in self.arms]
        if not names or len(set(names)) != len(names):
            raise ValueError("Revision arms must have distinct names")
        if len(set(self.test_core_arms)) != len(self.test_core_arms) or not set(
            self.test_core_arms
        ) <= set(names):
            raise ValueError("Test core arms must be distinct configured arms")
        if self.candidate_screening != "cheap":
            raise ValueError("Bank generation uses the original cheap-screen controls")
        if any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in self.excluded_geometry_hashes
        ):
            raise ValueError("Excluded geometry hashes must be lowercase SHA-256")
        return self

    def for_arm(self, arm: RevisionArm) -> PioneerRLConfig:
        return self.model_copy(
            update={
                "discount": arm.discount,
                "invalid_reward": arm.invalid_reward,
                "candidate_screening": arm.candidate_screening,
            }
        )
