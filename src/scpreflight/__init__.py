"""scpreflight: find out what is really inside a single-cell dataset before you analyze it."""

__version__ = "0.1.0"

from .core import check  # noqa: E402
from .model import Finding, MatrixState, Report  # noqa: E402

__all__ = ["check", "Report", "MatrixState", "Finding", "__version__"]
