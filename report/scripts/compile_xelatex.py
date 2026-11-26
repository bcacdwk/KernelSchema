#!/usr/bin/env python3
"""latexmk XeLaTeX command: refresh highlighting before every engine pass."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
subprocess.run([sys.executable, str(ROOT / 'scripts/highlight_yaml.py')], check=True)
engine = shutil.which('xelatex')
if not engine:
    raise SystemExit('XeLaTeX not found. Install TeX Live/MacTeX and add its bin directory to PATH.')
os.execv(engine, [engine, *sys.argv[1:]])
