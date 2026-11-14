"""
Svetlov & Abiev (2021) droplet-formation model.

Implements Chemical Engineering Science 235 (2021) 116493, eqs. (8)–(43):
  - axisymmetric Stokes flow in the film (phase 1) and droplet (phase 2)
  - two different pressure gradients fixed by Q1 and Q2
  - interface speed U_tp from shear continuity
  - Chebyshev collocation of the same two-phase Stokes problem
  - force balance FΔp + Fs − Fr = 0 and critical / real diameters

This file is a rewrite. It does not modify the original Project/ folder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MLH_TO_M3S = 1.0e-6 / 3600.0  # 1 ml/h → m³/s


# ---------------------------------------------------------------------------
# Paper default properties (Figs. 3–4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    dc: float = 300e-6          # channel diameter, m
    mu1: float = 2.24e-3        # continuous-phase viscosity, Pa·s
    mu2: float = 0.97e-3        # dispersed-phase viscosity, Pa·s
    sigma: float = 5.932e-3     # interfacial tension, N/m (Fig. 7)

    @property
    def R(self) -> float:
        return 0.5 * self.dc


def mlh(q_mlh: float) -> float:
    """Convert millilitres per hour to m³/s."""
    return q_mlh * MLH_TO_M3S


# ---------------------------------------------------------------------------
# Analytical solution of eqs. (10)–(27)
# ---------------------------------------------------------------------------
@dataclass
class AnalyticFields:
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

    def u1(self, r):
        r = np.asarray(r, dtype=float)
        return (r**2) * self.P1 / (4.0 * self.mu1) + self.c11 * np.log(r) + self.c21

    def u2(self, r):
        r = np.asarray(r, dtype=float)
        return (r**2) * self.P2 / (4.0 * self.mu2) + self.c22

    def u(self, r):
        r = np.asarray(r, dtype=float)
        out = np.empty_like(r, dtype=float)
        inside = r <= self.Rd
        out[inside] = self.u2(r[inside])
        out[~inside] = self.u1(r[~inside])
        return out

    def du1_dr(self, r):
        r = np.asarray(r, dtype=float)
        return r * self.P1 / (2.0 * self.mu1) + self.c11 / r

    def du2_dr(self, r):
        r = np.asarray(r, dtype=float)
        return r * self.P2 / (2.0 * self.mu2)

    def shear_interface(self) -> float:
        """τ(Rd) = μ1 u1'(Rd) = μ2 u2'(Rd), Pa."""
        return self.mu1 * self.du1_dr(self.Rd)

    def reconstruct_Q(self):
        """Return (Q1, Q2) by integrating the closed-form profiles."""
        R, Rd, mu1, mu2 = self.R, self.Rd, self.mu1, self.mu2
        I_P1 = (R**4 - Rd**4) / (16.0 * mu1)
        I_c11 = 0.5 * (R**2 * np.log(R) - Rd**2 * np.log(Rd)) - 0.25 * (R**2 - Rd**2)
        I_c21 = 0.5 * (R**2 - Rd**2)
        Q1 = 2.0 * np.pi * (self.P1 * I_P1 + self.c11 * I_c11 + self.c21 * I_c21)
        Q2 = np.pi * Rd**2 * self.Utp - np.pi * self.P2 * Rd**4 / (8.0 * mu2)
        return Q1, Q2


def solve_analytical(R: float, Rd: float, mu1: float, mu2: float,
                     Q1: float, Q2: float) -> AnalyticFields:
    """
    Solve the 6 × 6 linear system for (P1, P2, U_tp, c11, c21, c22).

    The paper typesetting of (21) and (25) is easy to misread, so the
    constants are obtained from the BCs and the two flow-rate integrals
    rather than from the printed closed forms.
    """
    if not (0.0 < Rd < R):
        raise ValueError("Need 0 < Rd < R")
    if Q1 <= 0.0 or Q2 <= 0.0:
        raise ValueError("Flow rates must be positive")

    A = np.zeros((6, 6))
    b = np.zeros(6)
    # unknowns: P1, P2, Utp, c11, c21, c22

    # (11) u1(R) = 0
    A[0] = [R**2 / (4.0 * mu1), 0.0, 0.0, np.log(R), 1.0, 0.0]
    b[0] = 0.0

    # (13) u1(Rd) = Utp
    A[1] = [Rd**2 / (4.0 * mu1), 0.0, -1.0, np.log(Rd), 1.0, 0.0]
    b[1] = 0.0

    # (13) u2(Rd) = Utp   (c12 = 0 already built in)
    A[2] = [0.0, Rd**2 / (4.0 * mu2), -1.0, 0.0, 0.0, 1.0]
    b[2] = 0.0

    # (15) μ1 u1'(Rd) = μ2 u2'(Rd)
    A[3] = [Rd / 2.0, -Rd / 2.0, 0.0, mu1 / Rd, 0.0, 0.0]
    b[3] = 0.0

    # (22)–(23) dispersed-phase flow rate
    A[4] = [0.0, -np.pi * Rd**4 / (8.0 * mu2), np.pi * Rd**2, 0.0, 0.0, 0.0]
    b[4] = Q2

    # (20) continuous-phase flow rate
    I_P1 = (R**4 - Rd**4) / (16.0 * mu1)
    I_c11 = 0.5 * (R**2 * np.log(R) - Rd**2 * np.log(Rd)) - 0.25 * (R**2 - Rd**2)
    I_c21 = 0.5 * (R**2 - Rd**2)
    A[5] = [2.0 * np.pi * I_P1, 0.0, 0.0, 2.0 * np.pi * I_c11, 2.0 * np.pi * I_c21, 0.0]
    b[5] = Q1

    P1, P2, Utp, c11, c21, c22 = np.linalg.solve(A, b)
    return AnalyticFields(P1, P2, Utp, c11, c21, c22, R, Rd, mu1, mu2)


# ---------------------------------------------------------------------------
# Chebyshev collocation of the same Stokes problem
# ---------------------------------------------------------------------------
def cheb_points(N: int) -> np.ndarray:
    return np.cos(np.pi * np.arange(N + 1) / N)


def cheb_D(N: int) -> np.ndarray:
    """Trefethen Chebyshev differentiation matrix on Gauss–Lobatto nodes."""
    x = cheb_points(N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                D[i, j] = (2.0 * N**2 + 1.0) / 6.0
            elif i == N and j == N:
                D[i, j] = -(2.0 * N**2 + 1.0) / 6.0
            elif i == j:
                D[i, j] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
            else:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j) / (x[i] - x[j]))
    return D


def clenshaw_curtis_weights(N: int) -> np.ndarray:
    """Weights for ∫_{-1}^{1} f(x) dx on Chebyshev–Gauss–Lobatto nodes."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1.0)
        w[N] = w[0]
        for k in range(1, N // 2):
            v = v - 2.0 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1.0)
        v = v - np.cos(N * theta[ii]) / (N**2 - 1.0)
    else:
        w[0] = 1.0 / N**2
        w[N] = w[0]
        for k in range(1, (N + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1.0)
    w[ii] = 2.0 * v / N
    return w


@dataclass
class ChebyshevFields:
    u1: np.ndarray
    u2: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    P1: float
    P2: float
    Utp: float

    def u_at(self, r: float) -> float:
        if r <= self.r2[0] + 1e-30:
            return float(np.interp(r, self.r2[::-1], self.u2[::-1]))
        return float(np.interp(r, self.r1[::-1], self.u1[::-1]))


def solve_chebyshev(R: float, Rd: float, mu1: float, mu2: float,
                    Q1: float, Q2: float, N: int = 50) -> ChebyshevFields:
    """
    Spectral solution of

        (1/r) d/dr ( r du_i/dr ) = (1/μ_i) d p_i / dx

    with BCs (11)–(15) and the two flow-rate constraints (20), (22).
    P1 and P2 are unknowns, so each phase has its own driving term.
    """
    if not (0.0 < Rd < R):
        raise ValueError("Need 0 < Rd < R")

    x = cheb_points(N)
    D = cheb_D(N)
    w = clenshaw_curtis_weights(N)

    # film: x=1 → r=R,  x=-1 → r=Rd
    r1 = R + 0.5 * (R - Rd) * (x - 1.0)
    dr1_dx = 0.5 * (R - Rd)
    D1_c = D / dr1_dx
    D2_c = D1_c @ D1_c
    R1 = np.diag(r1) @ D2_c + D1_c          # r u'' + u'

    # droplet: x=1 → r=0,  x=-1 → r=Rd
    r2 = 0.5 * Rd * (1.0 - x)
    dr2_dx = -0.5 * Rd
    D1_d = D / dr2_dx
    D2_d = D1_d @ D1_d
    R2 = np.diag(r2) @ D2_d + D1_d

    n1 = N + 1
    nunk = 2 * n1 + 2                       # u1, u2, P1, P2
    A = np.zeros((nunk, nunk))
    b = np.zeros(nunk)
    iP1, iP2 = 2 * n1, 2 * n1 + 1

    # Stokes in the film (all nodes; BCs overwrite four rows later)
    A[0:n1, 0:n1] = R1
    A[0:n1, iP1] = -r1 / mu1

    # Stokes in the droplet
    A[n1:2 * n1, n1:2 * n1] = R2
    A[n1:2 * n1, iP2] = -r2 / mu2

    # (11) u1(R) = 0   at i=0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    # (13) u1(Rd) = u2(Rd)   film i=N, drop i=N
    A[N, :] = 0.0
    A[N, N] = 1.0
    A[N, n1 + N] = -1.0
    b[N] = 0.0

    # (12) du2/dr = 0 at r=0   drop i=0
    A[n1, :] = 0.0
    A[n1, n1:2 * n1] = D1_d[0, :]
    b[n1] = 0.0

    # (15) μ1 u1'(Rd) = μ2 u2'(Rd)
    A[n1 + N, :] = 0.0
    A[n1 + N, 0:n1] = mu1 * D1_c[N, :]
    A[n1 + N, n1:2 * n1] = -mu2 * D1_d[N, :]
    b[n1 + N] = 0.0

    # (20) Q1 = 2π ∫_{Rd}^{R} u1 r dr = 2π ∫_{-1}^{1} u1 r1 (dr1/dx) dx
    A[iP1, :] = 0.0
    A[iP1, 0:n1] = 2.0 * np.pi * w * r1 * dr1_dx
    b[iP1] = Q1

    # (22) Q2 = 2π ∫_0^{Rd} u2 r dr = 2π ∫_{-1}^{1} u2 r2 (Rd/2) dx
    A[iP2, :] = 0.0
    A[iP2, n1:2 * n1] = 2.0 * np.pi * w * r2 * (0.5 * Rd)
    b[iP2] = Q2

    sol = np.linalg.solve(A, b)
    u1 = sol[0:n1]
    u2 = sol[n1:2 * n1]
    P1 = float(sol[iP1])
    P2 = float(sol[iP2])
    return ChebyshevFields(u1, u2, r1, r2, P1, P2, float(u1[N]))


# ---------------------------------------------------------------------------
# Force balance, eqs. (30)–(43)
# ---------------------------------------------------------------------------
@dataclass
class Forces:
    Fr: float
    Fs: float
    Fdp: float

    @property
    def Ftot(self) -> float:
        return self.Fdp + self.Fs - self.Fr


def forces_on_droplet(fields: AnalyticFields, sigma: float) -> Forces:
    """
    Fr = π dd σ                         (31)
    Fs = τ(Rd) · Ad ,  Ad = π Rd²       (33)
    FΔp = (∂p2/∂x) Vd , Vd = 4/3 π Rd³  (36)
    """
    Rd = fields.Rd
    dd = 2.0 * Rd
    Ad = np.pi * Rd**2
    Vd = 4.0 / 3.0 * np.pi * Rd**3
    Fr = np.pi * dd * sigma
    Fs = fields.shear_interface() * Ad
    Fdp = fields.P2 * Vd
    return Forces(Fr, Fs, Fdp)


def critical_diameter(dc: float, Q1: float, Q2: float) -> float:
    """d_crit = dc (1 − 6 Q2 / (π Q1))^{1/3}, eq. (43)."""
    arg = 1.0 - 6.0 * Q2 / (np.pi * Q1)
    if arg <= 0.0:
        return 0.0
    return dc * arg ** (1.0 / 3.0)


def equilibrium_diameter(ch: Channel, Q1: float, Q2: float,
                         n_scan: int = 80) -> float | None:
    """
    Smallest dd in (0, dc) at which Ftot changes from negative to
    non-negative (detaching forces overtake interfacial tension).
    """
    dds = np.linspace(0.15 * ch.dc, 0.98 * ch.dc, n_scan)
    prev = None
    d_prev = None
    for dd in dds:
        Rd = 0.5 * dd
        try:
            fld = solve_analytical(ch.R, Rd, ch.mu1, ch.mu2, Q1, Q2)
            F = forces_on_droplet(fld, ch.sigma).Ftot
        except np.linalg.LinAlgError:
            continue
        if prev is not None and prev < 0.0 <= F:
            # linear interpolate the zero
            return float(d_prev + (dd - d_prev) * (-prev) / (F - prev))
        prev, d_prev = F, dd
    return None


def real_diameter(deq: float, dc: float, Q1: float, Q2: float) -> float:
    """d_real from V_grow + Q2 t_col, t_col = dc³ / Q1 (Cubaud, eq. 40)."""
    Vgrow = np.pi * deq**3 / 6.0
    tcol = dc**3 / Q1
    V = Vgrow + Q2 * tcol
    return float((6.0 * V / np.pi) ** (1.0 / 3.0))


def profile_arrays(fields: AnalyticFields, n: int = 400):
    """Return (r_um, u) across the full radius for plotting."""
    r = np.linspace(1e-9, fields.R, n)
    r[r < fields.Rd] = np.clip(r[r < fields.Rd], 0.0, fields.Rd)
    return r * 1e6, fields.u(r)
