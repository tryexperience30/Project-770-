"""Shared input checks. Scientific symbols keep their conventional names."""

from droplet_formation.exceptions import ModelParameterError


def require_positive(name: str, value: float) -> float:
    number = float(value)
    if not (number > 0.0):
        raise ModelParameterError(f"{name} must be positive, got {value!r}.")
    return number


def require_strict_radius_pair(R: float, Rd: float) -> tuple[float, float]:
    """Require a droplet radius strictly inside the channel, 0 < Rd < R."""
    R = require_positive("channel radius R", R)
    Rd = float(Rd)
    if not (0.0 < Rd < R):
        raise ModelParameterError(
            f"Droplet radius Rd must satisfy 0 < Rd < R; got Rd={Rd}, R={R}."
        )
    return R, Rd


def require_collocation_order(N: int) -> int:
    order = int(N)
    if order < 4:
        raise ModelParameterError(
            f"Chebyshev order N must be an integer >= 4, got {N!r}."
        )
    return order
