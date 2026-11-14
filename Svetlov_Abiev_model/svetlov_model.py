"""Compatibility imports. Prefer ``droplet_formation`` in new code."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from droplet_formation import (  # noqa: E402
    AnalyticFields,
    Channel,
    ChebyshevFields,
    Forces,
    ModelParameterError,
    critical_diameter,
    equilibrium_diameter,
    flow_regime,
    forces_on_droplet,
    mlh,
    real_diameter,
    solve_analytical,
    solve_chebyshev,
)

__all__ = [
    "AnalyticFields",
    "Channel",
    "ChebyshevFields",
    "Forces",
    "ModelParameterError",
    "critical_diameter",
    "equilibrium_diameter",
    "flow_regime",
    "forces_on_droplet",
    "mlh",
    "real_diameter",
    "solve_analytical",
    "solve_chebyshev",
]
