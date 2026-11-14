"""Domain errors for the microfluidic droplet-formation solver."""


class ModelParameterError(ValueError):
    """Raised when a physical or numerical input is outside the valid range."""
