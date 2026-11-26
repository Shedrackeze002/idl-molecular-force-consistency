from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    record_id: str
    molecule: str
    kind: Literal["path", "minimum", "transition_state"]
    source_path: str
    connection_id: str | None
    branch: str | None
    frame_index: int | None
    species: tuple[str, ...]
    positions: np.ndarray
    energy: float
    forces: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class PreparationPolicy:
    duplicate_tolerance: float = 1e-5
    endpoint_tolerance: float = 0.05
    reference_tolerance: float = 1e-5
    graph_factor: float = 1.2
    graph_min_distance: float = 0.4
    search_limit: int = 100000
    seed: int = 20260929
    nonstationary_force: float = 1e-3
    exact_energy_tolerance: float = 1e-7
    exact_force_tolerance: float = 1e-6
