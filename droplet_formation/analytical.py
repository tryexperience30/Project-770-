"""Closed-form axisymmetric Stokes field in the film and droplet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from droplet_formation.validation import require_positive, require_strict_radius_pair

Array = NDArray[np.floating]


def _film_flow_integrals(
    R: float, Rd: float, mu1: float
) -> tuple[float, float, float]:
    """Radial integrals of the film profile against P1, c11, and c21."""
    I_P1 = (R**4 - Rd**4) / (16.0 * mu1)
    I_c11 = 0.5 * (R**2 * np.log(R) - Rd**2 * np.log(Rd)) - 0.25 * (R**2 - Rd**2)
    I_c21 = 0.5 * (R**2 - Rd**2)
    return I_P1, I_c11, I_c21


@dataclass
class AnalyticFields:
    """Velocity field from the 6 × 6 Stokes system.

    Unknowns: ``P1 = ∂p1/∂x``, ``P2 = ∂p2/∂x``, interface speed ``Utp``,
    and integration constants ``c11``, ``c21``, ``c22`` (``c12 = 0``).
    """

    P1: float
    P2: float
    Utp: float
    c11: float
    c21: float
    c22: float
    R: float
    Rd: float
    mu1: float
    mu2: float

    def u1(self, r: Array | float) -> Array:
        r = np.asarray(r, dtype=float)
        return (r**2) * self.P1 / (4.0 * self.mu1) + self.c11 * np.log(r) + self.c21

    def u2(self, r: Array | float) -> Array:
        r = np.asarray(r, dtype=float)
        return (r**2) * self.P2 / (4.0 * self.mu2) + self.c22

    def u(self, r: Array | float) -> Array:
        r = np.asarray(r, dtype=float)
        out = np.empty_like(r, dtype=float)
        inside = r <= self.Rd
        out[inside] = self.u2(r[inside])
        out[~inside] = self.u1(r[~inside])
        return out

    def du1_dr(self, r: Array | float) -> Array:
        r = np.asarray(r, dtype=float)
        return r * self.P1 / (2.0 * self.mu1) + self.c11 / r

    def du2_dr(self, r: Array | float) -> Array:
        r = np.asarray(r, dtype=float)
        return r * self.P2 / (2.0 * self.mu2)

    def shear_interface(self) -> float:
        """Interfacial shear τ(Rd) = μ1 u1'(Rd) = μ2 u2'(Rd), Pa."""
        return float(self.mu1 * self.du1_dr(self.Rd))

    def reconstruct_Q(self) -> tuple[float, float]:
        """Reconstruct (Q1, Q2) by integrating the closed-form profiles."""
        R, Rd, mu1, mu2 = self.R, self.Rd, self.mu1, self.mu2
        I_P1, I_c11, I_c21 = _film_flow_integrals(R, Rd, mu1)
        Q1 = 2.0 * np.pi * (self.P1 * I_P1 + self.c11 * I_c11 + self.c21 * I_c21)
        Q2 = np.pi * Rd**2 * self.Utp - np.pi * self.P2 * Rd**4 / (8.0 * mu2)
        return float(Q1), float(Q2)


def solve_analytical(
    R: float,
    Rd: float,
    mu1: float,
    mu2: float,
    Q1: float,
    Q2: float,
) -> AnalyticFields:
    """Solve the Stokes BVP with flow-rate constraints for both phases.

    The linear system is assembled from the wall, interface, and centreline
    conditions plus the two volumetric-flow integrals, rather than from the
    typeset closed forms (21) and (25) in the source paper.
    """
    R, Rd = require_strict_radius_pair(R, Rd)
    mu1 = require_positive("mu1", mu1)
    mu2 = require_positive("mu2", mu2)
    Q1 = require_positive("Q1", Q1)
    Q2 = require_positive("Q2", Q2)

    A = np.zeros((6, 6))
    b = np.zeros(6)
    # unknowns: P1, P2, Utp, c11, c21, c22

    A[0] = [R**2 / (4.0 * mu1), 0.0, 0.0, np.log(R), 1.0, 0.0]
    b[0] = 0.0

    A[1] = [Rd**2 / (4.0 * mu1), 0.0, -1.0, np.log(Rd), 1.0, 0.0]
    b[1] = 0.0

    A[2] = [0.0, Rd**2 / (4.0 * mu2), -1.0, 0.0, 0.0, 1.0]
    b[2] = 0.0

    A[3] = [Rd / 2.0, -Rd / 2.0, 0.0, mu1 / Rd, 0.0, 0.0]
    b[3] = 0.0

    A[4] = [0.0, -np.pi * Rd**4 / (8.0 * mu2), np.pi * Rd**2, 0.0, 0.0, 0.0]
    b[4] = Q2

    I_P1, I_c11, I_c21 = _film_flow_integrals(R, Rd, mu1)
    A[5] = [
        2.0 * np.pi * I_P1,
        0.0,
        0.0,
        2.0 * np.pi * I_c11,
        2.0 * np.pi * I_c21,
        0.0,
    ]
    b[5] = Q1

    try:
        P1, P2, Utp, c11, c21, c22 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Analytical Stokes system is singular for the given inputs."
        ) from exc

    return AnalyticFields(
        float(P1),
        float(P2),
        float(Utp),
        float(c11),
        float(c21),
        float(c22),
        R,
        Rd,
        mu1,
        mu2,
    )


def radial_profile(fields: AnalyticFields, n: int = 400) -> tuple[Array, Array]:
    """Return radius in metres and axial velocity across the full channel."""
    r = np.linspace(1e-9, fields.R, n)
    return r, fields.u(r)
