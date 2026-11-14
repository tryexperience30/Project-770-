"""Chebyshev–Gauss–Lobatto collocation of the two-phase Stokes problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from droplet_formation.validation import (
    require_collocation_order,
    require_positive,
    require_strict_radius_pair,
)

Array = NDArray[np.floating]


def chebyshev_nodes(N: int) -> Array:
    """Chebyshev–Gauss–Lobatto nodes on [-1, 1], index 0 at x = 1."""
    N = require_collocation_order(N)
    return np.cos(np.pi * np.arange(N + 1) / N)


def chebyshev_differentiation_matrix(N: int) -> Array:
    """Trefethen differentiation matrix on Gauss–Lobatto nodes."""
    N = require_collocation_order(N)
    x = chebyshev_nodes(N)
    D = np.ones((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
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


def clenshaw_curtis_weights(N: int) -> Array:
    """Weights for ∫_{-1}^{1} f(x) dx on Chebyshev–Gauss–Lobatto nodes."""
    N = require_collocation_order(N)
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    interior = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1.0)
        w[N] = w[0]
        for k in range(1, N // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k**2 - 1.0)
        v = v - np.cos(N * theta[interior]) / (N**2 - 1.0)
    else:
        w[0] = 1.0 / N**2
        w[N] = w[0]
        for k in range(1, (N + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k**2 - 1.0)
    w[interior] = 2.0 * v / N
    return w


@dataclass
class ChebyshevFields:
    """Spectral solution on film nodes ``r1`` and droplet nodes ``r2``."""

    u1: Array
    u2: Array
    r1: Array
    r2: Array
    P1: float
    P2: float
    Utp: float

    def u_at(self, r: float) -> float:
        radius = float(r)
        if radius <= float(self.r2[0]) + 1e-30:
            return float(np.interp(radius, self.r2[::-1], self.u2[::-1]))
        return float(np.interp(radius, self.r1[::-1], self.u1[::-1]))


def solve_chebyshev(
    R: float,
    Rd: float,
    mu1: float,
    mu2: float,
    Q1: float,
    Q2: float,
    N: int = 50,
) -> ChebyshevFields:
    """Spectral solution of the same BVP as :func:`solve_analytical`.

    Unknowns are ``u1`` on the film, ``u2`` in the droplet, and both
    pressure gradients. Flow-rate integrals use Clenshaw–Curtis quadrature.
    """
    R, Rd = require_strict_radius_pair(R, Rd)
    mu1 = require_positive("mu1", mu1)
    mu2 = require_positive("mu2", mu2)
    Q1 = require_positive("Q1", Q1)
    Q2 = require_positive("Q2", Q2)
    N = require_collocation_order(N)

    x = chebyshev_nodes(N)
    D = chebyshev_differentiation_matrix(N)
    w = clenshaw_curtis_weights(N)

    r1 = R + 0.5 * (R - Rd) * (x - 1.0)
    dr1_dx = 0.5 * (R - Rd)
    D1_c = D / dr1_dx
    D2_c = D1_c @ D1_c
    R1 = np.diag(r1) @ D2_c + D1_c

    r2 = 0.5 * Rd * (1.0 - x)
    dr2_dx = -0.5 * Rd
    D1_d = D / dr2_dx
    D2_d = D1_d @ D1_d
    R2 = np.diag(r2) @ D2_d + D1_d

    n1 = N + 1
    nunk = 2 * n1 + 2
    A = np.zeros((nunk, nunk))
    b = np.zeros(nunk)
    iP1, iP2 = 2 * n1, 2 * n1 + 1

    A[0:n1, 0:n1] = R1
    A[0:n1, iP1] = -r1 / mu1

    A[n1:2 * n1, n1:2 * n1] = R2
    A[n1:2 * n1, iP2] = -r2 / mu2

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    A[N, :] = 0.0
    A[N, N] = 1.0
    A[N, n1 + N] = -1.0
    b[N] = 0.0

    A[n1, :] = 0.0
    A[n1, n1:2 * n1] = D1_d[0, :]
    b[n1] = 0.0

    A[n1 + N, :] = 0.0
    A[n1 + N, 0:n1] = mu1 * D1_c[N, :]
    A[n1 + N, n1:2 * n1] = -mu2 * D1_d[N, :]
    b[n1 + N] = 0.0

    A[iP1, :] = 0.0
    A[iP1, 0:n1] = 2.0 * np.pi * w * r1 * dr1_dx
    b[iP1] = Q1

    A[iP2, :] = 0.0
    A[iP2, n1:2 * n1] = 2.0 * np.pi * w * r2 * (0.5 * Rd)
    b[iP2] = Q2

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Chebyshev Stokes system is singular for the given inputs."
        ) from exc

    u1 = sol[0:n1]
    u2 = sol[n1:2 * n1]
    return ChebyshevFields(
        u1,
        u2,
        r1,
        r2,
        float(sol[iP1]),
        float(sol[iP2]),
        float(u1[N]),
    )
