"""Experimental, finite event graphs; independent of the authoring checker."""

from .graph import build_phase
from .model import GraphError

__all__ = ["build_phase", "GraphError"]
