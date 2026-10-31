"""Publication and invalidation of generated pipeline review artifacts."""
from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

from .render import RenderError, dot_text, gallery, graphviz, viewer

REPORT_SCHEMA = "pipeline-graph-trial-1"
SUFFIXES = (".json", ".dot", ".svg", ".view.html")


class GalleryIdentity(HTMLParser):
    """Read stable ownership metadata independently of gallery presentation."""
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("name") == "kernelschema-gallery-source":
            self.sources.append(attributes.get("content"))


class GraphOutput:
    def __init__(self, directory: Path):
        self.directory = directory

    def discard(self, node: str) -> None:
        for suffix in SUFFIXES:
            (self.directory / (node + suffix)).unlink(missing_ok=True)

    def reports(self, task_directory: Path) -> dict[str, dict]:
        reports = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError, RecursionError):
                continue
            if (isinstance(report, dict) and report.get("schema") == REPORT_SCHEMA
                    and isinstance(report.get("source"), str)
                    and Path(report["source"]).parent.resolve() == task_directory
                    and Path(report["source"]).stem == path.stem):
                reports[path.stem] = report
        return reports

    def owns_gallery(self, task_directory: Path) -> bool:
        try:
            identity = GalleryIdentity()
            identity.feed((self.directory / "index.html").read_text(encoding="utf-8"))
            return identity.sources == [str(task_directory)]
        except (OSError, UnicodeError, ValueError):
            return False

    def begin(self, source: Path, *, all_nodes: bool) -> None:
        """Invalidate the requested results before checking any input.

        A whole-task refresh also retires reports for removed nodes. Recognize
        those by their report format and source directory, not by file suffix.
        """
        task_directory = self.task_directory = source.parent.resolve()
        reports = self.reports(task_directory)
        nodes = {source.stem}
        if all_nodes:
            if task_directory.is_dir():
                nodes.update(p.stem for p in task_directory.iterdir() if p.suffix in {".yaml", ".yml"})
            nodes.update(reports)
        conflicts = [self.directory / (node + suffix) for node in sorted(nodes - reports.keys())
                     for suffix in SUFFIXES
                     if (self.directory / (node + suffix)).exists() or (self.directory / (node + suffix)).is_symlink()]
        index = self.directory / "index.html"
        if (index.exists() or index.is_symlink()) and not self.owns_gallery(task_directory):
            conflicts.append(index)
        if conflicts:
            raise OSError("output conflicts with files not owned by this task: "
                          + ", ".join(str(path) for path in conflicts) + "; choose a separate --out directory")
        index.unlink(missing_ok=True)
        for node in sorted(nodes):
            self.discard(node)

    def write(self, node: str, title: str, graphs: list, samples: dict, report: dict, format: str) -> None:
        """Publish complete files; a rendering failure retains checked DOT/JSON."""
        self.directory.mkdir(parents=True, exist_ok=True)
        render_error = None
        try:
            with TemporaryDirectory(prefix=".pipeline-", dir=self.directory) as temporary:
                base = Path(temporary) / node
                base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                base.with_suffix(".dot").write_text(dot_text(title, graphs, samples), encoding="utf-8")
                suffixes = [".json", ".dot"]
                if format == "svg":
                    try:
                        graphviz(base.with_suffix(".dot"), base.with_suffix(".svg"))
                        base.with_suffix(".view.html").write_text(viewer(title, node + ".svg", report), encoding="utf-8")
                        suffixes.extend((".svg", ".view.html"))
                    except RenderError as error:
                        render_error = error
                for suffix in suffixes:
                    base.with_suffix(suffix).replace(self.directory / (node + suffix))
        except OSError:
            self.discard(node)
            raise
        if render_error is not None:
            raise render_error

    def write_gallery(self, title: str, entries: list[dict]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".pipeline-", dir=self.directory) as temporary:
            index = Path(temporary) / "index.html"
            index.write_text(gallery(title, entries, str(self.task_directory)), encoding="utf-8")
            index.replace(self.directory / "index.html")
