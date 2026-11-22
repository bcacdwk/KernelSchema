#!/usr/bin/env python3
"""Review implementation evidence and keep, repair, or revise its Schema."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from schema_optimize.tools.runner import main
    raise SystemExit(main())
