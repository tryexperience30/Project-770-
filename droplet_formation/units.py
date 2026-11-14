"""Unit conversions used by the solver."""

from droplet_formation.exceptions import ModelParameterError

MLH_TO_M3S = 1.0e-6 / 3600.0  # 1 ml/h → m³/s


def mlh(flow_rate_mlh: float) -> float:
    """Convert millilitres per hour to cubic metres per second."""
    value = float(flow_rate_mlh)
    if value <= 0.0:
        raise ModelParameterError("Flow rate in ml/h must be positive.")
    return value * MLH_TO_M3S
