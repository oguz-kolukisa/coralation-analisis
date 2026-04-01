"""Single source of truth for the project version."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("coralation-analisis")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for editable installs
