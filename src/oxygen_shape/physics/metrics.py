"""Physical transfer, membrane utilization, and raster resource measures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FluxMetrics:
    physical_flux_Q: float
    source_influx: float
    ideal_flux_Q: float
    efficiency_eta: float | None
    area: float
    robin_perimeter: float
    source_length: float
    conservation_error: float
    residual_inf: float
