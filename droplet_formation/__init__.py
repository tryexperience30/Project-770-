"""Microfluidic droplet-formation hydrodynamics.

Closed-form and Chebyshev solutions of axisymmetric Stokes flow in a
cylindrical channel, plus the equatorial force balance used to estimate
droplet size and the drip–slug transition.
"""

from droplet_formation.analytical import AnalyticFields, solve_analytical
from droplet_formation.collocation import ChebyshevFields, solve_chebyshev
from droplet_formation.exceptions import ModelParameterError
from droplet_formation.forces import (
    Forces,
    critical_diameter,
    equilibrium_diameter,
    flow_regime,
    forces_on_droplet,
    real_diameter,
)
from droplet_formation.geometry import Channel
from droplet_formation.units import mlh

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

__version__ = "1.0.0"
