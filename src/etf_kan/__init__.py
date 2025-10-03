from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("etf-kan")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
