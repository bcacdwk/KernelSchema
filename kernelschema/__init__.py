"""CPU-only public workflow entry points. Backends are loaded on explicit use."""
from .api import design, implement, optimize
from .contracts import ProtocolError, UnsupportedError

__version__ = "0.1.0"
__all__ = ["design", "implement", "optimize", "ProtocolError", "UnsupportedError"]
