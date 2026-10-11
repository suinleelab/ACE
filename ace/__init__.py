"""ACE package initialization."""

from ._constants import REGISTRY_KEYS

import logging

from rich.console import Console
from rich.logging import RichHandler

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "ace-vi"
try:
    __version__ = importlib_metadata.version(package_name)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.1.0"

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("ace: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False
