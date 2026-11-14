"""Input validation and edge cases."""

from __future__ import annotations

import pytest

from droplet_formation import (
    Channel,
    ModelParameterError,
    critical_diameter,
    mlh,
    solve_analytical,
    solve_chebyshev,
)


def test_rejects_non_positive_flow_rate() -> None:
    with pytest.raises(ModelParameterError):
        mlh(0.0)
    with pytest.raises(ModelParameterError):
        mlh(-1.0)


def test_rejects_invalid_geometry() -> None:
    with pytest.raises(ModelParameterError):
        Channel(dc=0.0)
    with pytest.raises(ModelParameterError):
        Channel(mu1=-1.0)
    ch = Channel()
    with pytest.raises(ModelParameterError):
        solve_analytical(ch.R, ch.R, ch.mu1, ch.mu2, mlh(1), mlh(1))
    with pytest.raises(ModelParameterError):
        solve_analytical(ch.R, 0.0, ch.mu1, ch.mu2, mlh(1), mlh(1))


def test_rejects_invalid_collocation_order() -> None:
    ch = Channel()
    with pytest.raises(ModelParameterError):
        solve_chebyshev(ch.R, 0.5 * ch.R, ch.mu1, ch.mu2, mlh(1), mlh(1), N=2)


def test_critical_diameter_vanishes_when_argument_negative() -> None:
    dc = 300e-6
    # 6 Q2 / (π Q1) > 1 → no spherical drop remains after collapse.
    assert critical_diameter(dc, Q1=1e-12, Q2=1.0) == 0.0
