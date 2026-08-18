"""Project-specific exceptions."""


class ScaffoldCoderError(Exception):
    """Base exception."""


class UnsupportedSyntaxError(ScaffoldCoderError):
    """Raised when source code uses syntax outside the enabled v0 grammar."""


class IRValidationError(ScaffoldCoderError):
    """Raised when the typed program tree violates an invariant."""


class RuntimeInvariantError(ScaffoldCoderError):
    """Raised when a symbolic runtime transition would violate the spec."""


class BudgetExceededError(ScaffoldCoderError):
    """Raised when an oracle/runtime trace exceeds a configured hard limit."""

