"""Equatorial force balance and droplet-size estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from droplet_formation.analytical import AnalyticFields, solve_analytical
from droplet_formation.geometry import Channel
from droplet_formation.validation import require_positive


@dataclass(frozen=True)
class Forces:
    """Holding, shear, and pressure forces on the droplet, N."""

    Fr: float
    Fs: float
    Fdp: float

    @property
    def Ftot(self) -> float:
        return self.Fdp + self.Fs - self.Fr


def forces_on_droplet(fields: AnalyticFields, sigma: float) -> Forces:
    """Equatorial force estimates from the closed-form Stokes field.

    ``Fr = π dd σ``, ``Fs = τ(Rd) Ad``, ``FΔp = (∂p2/∂x) Vd``.
    """
    if sigma < 0.0:
        raise ValueError("Interfacial tension sigma cannot be negative.")
    Rd = fields.Rd
    dd = 2.0 * Rd
    Ad = np.pi * Rd**2
    Vd = 4.0 / 3.0 * np.pi * Rd**3
    Fr = np.pi * dd * sigma
    Fs = fields.shear_interface() * Ad
    Fdp = fields.P2 * Vd
    return Forces(float(Fr), float(Fs), float(Fdp))


def critical_diameter(dc: float, Q1: float, Q2: float) -> float:
    """Spherical-drop limit after neck collapse, ``d_crit``.

    ``d_crit = dc (1 − 6 Q2 / (π Q1))^{1/3}``.
    """
    dc = require_positive("dc", dc)
    Q1 = require_positive("Q1", Q1)
    Q2 = require_positive("Q2", Q2)
    arg = 1.0 - 6.0 * Q2 / (np.pi * Q1)
    if arg <= 0.0:
        return 0.0
    return float(dc * arg ** (1.0 / 3.0))


def equilibrium_diameter(
    ch: Channel,
    Q1: float,
    Q2: float,
    n_scan: int = 80,
) -> float | None:
    """Smallest ``dd`` in (0, dc) at which ``Ftot`` crosses through zero."""
    Q1 = require_positive("Q1", Q1)
    Q2 = require_positive("Q2", Q2)
    if n_scan < 8:
        raise ValueError("n_scan must be at least 8.")

    dds = np.linspace(0.15 * ch.dc, 0.98 * ch.dc, n_scan)
    prev: float | None = None
    d_prev: float | None = None
    for dd in dds:
        Rd = 0.5 * dd
        try:
            fld = solve_analytical(ch.R, Rd, ch.mu1, ch.mu2, Q1, Q2)
            F = forces_on_droplet(fld, ch.sigma).Ftot
        except np.linalg.LinAlgError:
            continue
        if prev is not None and d_prev is not None and prev < 0.0 <= F:
            return float(d_prev + (dd - d_prev) * (-prev) / (F - prev))
        prev, d_prev = F, float(dd)
    return None


def real_diameter(deq: float, dc: float, Q1: float, Q2: float) -> float:
    """Detached diameter from growth volume plus neck-collapse influx."""
    deq = require_positive("deq", deq)
    dc = require_positive("dc", dc)
    Q1 = require_positive("Q1", Q1)
    Q2 = require_positive("Q2", Q2)
    Vgrow = np.pi * deq**3 / 6.0
    tcol = dc**3 / Q1
    volume = Vgrow + Q2 * tcol
    return float((6.0 * volume / np.pi) ** (1.0 / 3.0))


def flow_regime(deq: float, dcrit: float) -> str:
    """Classify slug vs drip from the equilibrium / critical comparison."""
    if deq > dcrit:
        return "slug"
    return "drip"
