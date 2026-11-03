#!/usr/bin/env python3
"""Run Schema Extract through the tool layer."""

import sys

sys.dont_write_bytecode = True

from tools.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
