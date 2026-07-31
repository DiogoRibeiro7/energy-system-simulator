class EnergySystemError(Exception):
    """Base exception for the package."""


class ConfigurationError(EnergySystemError):
    """Raised when configuration values are invalid."""


class DataValidationError(EnergySystemError):
    """Raised when input data violate the documented contract."""


class OptimisationError(EnergySystemError):
    """Raised when the dispatch optimisation does not return a usable solution."""
