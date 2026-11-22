#!/usr/bin/env python3
"""Write a task Schema using the standalone Design CLI."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from schema_design.tools.runner import main
    raise SystemExit(main())
