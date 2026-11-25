#!/usr/bin/env python3
"""Build the report with XeLaTeX or Tectonic in an isolated staging tree."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "build")
    parser.add_argument("--engine", default="auto", help="auto, xelatex, tectonic, or an absolute executable path")
    parser.add_argument("--passes", type=int, choices=(2, 3), default=3, help="Compilation passes for stable contents/bookmarks (default: 3)")
    parser.add_argument("--draft", action="store_true", help="Permit visible placeholders for absent diagrams; never writes diagrams")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate body.tex and refresh original image copies before building")
    parser.add_argument("--only-cached", action="store_true", help="Use only cached tectonic packages")
    parser.add_argument("--bundle", help="Optional local Tectonic bundle or an approved mirror URL")
    args = parser.parse_args()
    engine = (shutil.which("xelatex") or shutil.which("tectonic")) if args.engine == "auto" else shutil.which(args.engine)
    if not engine:
        raise SystemExit("No TeX engine found; pass --engine /absolute/path/to/xelatex or install tectonic on PATH")
    # Preserve the executable basename: xelatex may be a symlink to xetex,
    # and TeX selects its format from argv[0].
    engine = str(Path(engine).absolute())
    engine_kind = "tectonic" if "tectonic" in Path(engine).name.lower() else "xelatex"
    if engine_kind == "xelatex" and (args.bundle or args.only_cached):
        raise SystemExit("--bundle and --only-cached apply only to tectonic")
    if args.bundle and Path(args.bundle).exists():
        args.bundle = str(Path(args.bundle).resolve())
    if args.regenerate:
        subprocess.run([sys.executable, str(ROOT / "scripts/convert.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/highlight_yaml.py")], check=True)
    missing = [f"diagram-{i:02d}.tex" for i in range(1, 14) if not (ROOT / "diagrams" / f"diagram-{i:02d}.tex").exists()]
    if missing and not args.draft:
        raise SystemExit("Missing TikZ files: " + ", ".join(missing) + ". Use --draft only for layout checks.")
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Snapshot only for compilation: no generated intermediates in the source tree.
    with tempfile.TemporaryDirectory(prefix="kernel-report-tex-") as temp:
        stage = Path(temp)
        for name in ("main.tex", "body.tex", "body-highlighted.tex", "yaml-style.tex"):
            shutil.copy2(ROOT / name, stage / name)
        for name in ("fonts", "assets", "diagrams"):
            if (ROOT / name).is_dir():
                (stage / name).symlink_to(ROOT / name, target_is_directory=True)
        if args.draft:
            main_tex = stage / "main.tex"
            main_tex.write_text("\\def\\AllowMissingDiagrams{1}\n" + main_tex.read_text())
        if engine_kind == "tectonic":
            cmd = [engine, "--keep-logs", "--keep-intermediates", "--reruns", str(args.passes - 1), "--outdir", str(args.outdir)]
            if args.only_cached:
                cmd.append("--only-cached")
            if args.bundle:
                cmd += ["--bundle", args.bundle]
            subprocess.run(cmd + ["main.tex"], cwd=stage, check=True)
        else:
            cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"-output-directory={args.outdir}", "main.tex"]
            for number in range(1, args.passes + 1):
                print(f"XeLaTeX pass {number}/{args.passes}", flush=True)
                subprocess.run(cmd, cwd=stage, check=True)
    print(f"PDF: {args.outdir / 'main.pdf'}")
    if args.draft and missing:
        print(f"DRAFT: {len(missing)} diagram placeholders; not a complete final report")


if __name__ == "__main__":
    main()
