from importlib import metadata

from . import models

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = ["models"]
