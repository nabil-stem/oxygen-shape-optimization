"""An explicit continuation protocol with a fixed exploration schedule."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from oxygen_shape.optimization.pioneer_config import PioneerRLConfig


class DurationConfig(PioneerRLConfig):
    reference_run: Path
    reference_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    episodes: int = Field(default=96, ge=1, lt=1000, strict=True)
    reused_episodes: int = Field(default=24, ge=1, strict=True)
    exploration_episodes: int = Field(default=24, ge=1, strict=True)
    checkpoint_episodes: list[Annotated[int, Field(ge=1, strict=True)]] = Field(
        default_factory=lambda: [12, 24, 48, 96]
    )
    development_rollout_seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(
        default_factory=lambda: [263000, 263001]
    )

    @model_validator(mode="after")
    def duration_protocol(self):
        if (
            self.checkpoint_episodes != sorted(set(self.checkpoint_episodes))
            or not self.checkpoint_episodes
            or self.checkpoint_episodes[-1] != self.episodes
            or self.reused_episodes not in self.checkpoint_episodes
            or self.episodes <= self.reused_episodes
            or self.exploration_episodes != self.reused_episodes
        ):
            raise ValueError("Use sorted checkpoints including the reused prefix and final episode")
        seeds = self.development_rollout_seeds
        old_streams = sum(
            (
                self.replica_seeds,
                self.validation_generation_seeds,
                self.test_generation_seeds,
                self.validation_rollout_seeds,
                self.test_rollout_seeds,
            ),
            [],
        )
        training = {r * 1000 + e for r in self.replica_seeds for e in range(self.episodes)}
        if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & (set(old_streams) | training):
            raise ValueError("Development seeds must be fresh, unique and separate from training")
        return self

    def resolve_paths(self, base: Path):
        config = super().resolve_paths(base)
        reference = self.reference_run
        if not reference.is_absolute():
            reference = (base / reference).resolve()
        return config.model_copy(update={"reference_run": reference})

    def exploration(self, episode: int) -> float:
        denominator = max(self.exploration_episodes - 1, 1)
        step = min(episode, denominator)
        # Match the reference operation order as well as its mathematical formula.
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * step / denominator
