#!/usr/bin/env python3
"""Keep the distribution PDF synchronized after a successful latexmk build."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'build/main.pdf'
target = ROOT / '基于 Schema 的 Kernel 表示与生成.pdf'
if not source.is_file():
    raise SystemExit('build/main.pdf missing: nothing to publish')
if not target.exists() or target.read_bytes() != source.read_bytes():
    shutil.copy2(source, target)
print('Distribution PDF: ' + str(target))
