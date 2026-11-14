"""Channel geometry and default physical properties."""

from __future__ import annotations

from dataclasses import dataclass

from droplet_formation.exceptions import ModelParameterError
from droplet_formation.validation import require_positive


@dataclass(frozen=True)
class Channel:
    """Cylindrical microchannel and two-phase fluid properties.

    Defaults match Svetlov & Abiev (2021), water–cyclohexane micromixer:
    ``dc = 300 μm``, ``μ1 = 2.24 mPa·s``, ``μ2 = 0.97 mPa·s``,
    ``σ = 5.932 mN/m``.
    """

    dc: float = 300e-6
    mu1: float = 2.24e-3
    mu2: float = 0.97e-3
    sigma: float = 5.932e-3

    def __post_init__(self) -> None:
        require_positive("channel diameter dc", self.dc)
        require_positive("continuous-phase viscosity mu1", self.mu1)
        require_positive("dispersed-phase viscosity mu2", self.mu2)
        if self.sigma < 0.0:
            raise ModelParameterError("Interfacial tension sigma cannot be negative.")

    @property
    def R(self) -> float:
        """Channel radius, m."""
        return 0.5 * self.dc
