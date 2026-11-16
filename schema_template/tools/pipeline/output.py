"""Atomic, sibling PNG output with provenance stored inside the image."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import zlib

from .render import RenderError, dot_text, graphviz

SIGNATURE = b"\x89PNG\r\n\x1a\n"
METADATA_KEY = b"KernelSchema"


def chunks(data: bytes):
    if not data.startswith(SIGNATURE):
        raise ValueError("not a PNG image")
    offset = len(SIGNATURE)
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:end - 4]
        if zlib.crc32(kind + payload) != struct.unpack_from(">I", data, end - 4)[0]:
            raise ValueError("invalid PNG checksum")
        yield kind, payload, data[offset:end]
        offset = end
        if kind == b"IEND" and offset == len(data):
            return
    raise ValueError("incomplete PNG image")


def metadata(path: Path) -> dict:
    try:
        entries = [json.loads(payload.split(b"\0", 1)[1]) for kind, payload, _ in chunks(path.read_bytes())
                   if kind == b"tEXt" and payload.startswith(METADATA_KEY + b"\0")]
        return entries[0] if len(entries) == 1 and isinstance(entries[0], dict) else {}
    except (OSError, ValueError, UnicodeError, RecursionError):
        return {}


def owner(path: Path) -> str | None:
    information = metadata(path)
    source = information.get("source")
    if (information.get("kind") == "pipeline_graph" and isinstance(source, str)
            and Path(source).name == source and Path(source).suffix in {".yaml", ".yml"}
            and Path(source).stem == path.stem):
        return source
    return None


def stamp(data: bytes, information: dict) -> bytes:
    payload = METADATA_KEY + b"\0" + json.dumps(information, sort_keys=True, ensure_ascii=True).encode("ascii")
    record = struct.pack(">I", len(payload)) + b"tEXt" + payload + struct.pack(">I", zlib.crc32(b"tEXt" + payload))
    result = [SIGNATURE]
    for kind, content, raw in chunks(data):
        if kind == b"IEND":
            result.append(record)
        if kind != b"tEXt" or not content.startswith(METADATA_KEY + b"\0"):
            result.append(raw)
    return b"".join(result)


class GraphOutput:
    def preflight(self, sources) -> None:
        """Check requested destinations, without deleting or scanning old PNGs."""
        destinations = {}
        conflicts = []
        for source in sorted({Path(source) for source in sources}):
            destination = source.with_suffix(".png")
            previous = destinations.get(destination)
            if previous is not None and previous != source:
                raise RenderError(f"{destination}: multiple YAML sources have the same PNG destination; PNG not updated")
            destinations[destination] = source
            if destination.is_symlink() or destination.exists() and owner(destination) != source.name:
                conflicts.append(destination)
        if conflicts:
            raise RenderError("PNG not updated: destinations are not owned by these YAML sources: "
                              + ", ".join(map(str, conflicts)))

    def write(self, source: Path, graphs: list, samples: dict, notes: list[str], *, source_sha256: str,
              source_manifest: dict[str, str] | None = None, effective_identity: str | None = None,
              origins: dict | None = None) -> Path:
        """Render and atomically replace only after ownership and source checks.

        All requested analyses must complete before callers publish any PNGs.
        Cyclic but complete samples may be published with a red diagnosis.
        Failures leave the previous destination in place, without a state file.
        """
        source = Path(source)
        try:
            self.preflight([source])
            if not graphs:
                raise RenderError("no completed step graphs were supplied")
            if {graph.step for graph in graphs} != samples.keys():
                raise RenderError("sample configurations must match the completed step graphs")
            manifest = dict(source_manifest) if source_manifest is not None else {source.name: source_sha256}
            if manifest.get(source.name) != source_sha256:
                raise RenderError(f"{source}: source manifest does not match the checked source")

            def unchanged():
                for name, digest in manifest.items():
                    if Path(name).name != name or Path(name).suffix not in {".yaml", ".yml"}:
                        raise RenderError(f"{source}: invalid source manifest filename {name!r}")
                    dependency = source.parent / name
                    try:
                        current = hashlib.sha256(dependency.read_bytes()).hexdigest()
                    except OSError as error:
                        raise RenderError(f"{dependency}: source changed after checking; rerun checking and rendering") from error
                    if current != digest:
                        raise RenderError(f"{dependency}: source changed after checking; rerun checking and rendering")

            unchanged()
            title = source.parent.name + " / " + source.stem
            image = graphviz(dot_text(title, graphs, samples, notes))
            information = {
                "kind": "pipeline_graph", "source": source.name, "source_sha256": source_sha256,
                "source_manifest": manifest, "effective_identity": effective_identity, "origins": origins or {},
                "samples": samples, "notes": notes, "acyclic": all(not graph.cycle for graph in graphs),
                "scope": "given sample; one representative worker per step; local event relations",
                "steps": [{"step": graph.step, "actions": len(graph.operations),
                           "events": 2 * len(graph.operations), "boundary_skips": len(graph.boundaries)}
                          for graph in graphs],
            }
            try:
                image = stamp(image, information)
            except ValueError as error:
                raise RenderError(f"Graphviz returned an invalid PNG: {error}") from error
            destination = source.with_suffix(".png")
            with TemporaryDirectory(prefix=".pipeline-", dir=source.parent) as temporary:
                staged = Path(temporary) / destination.name
                staged.write_bytes(image)
                unchanged()
                self.preflight([source])
                staged.replace(destination)
            return destination
        except (RenderError, OSError) as error:
            if "PNG not updated" in str(error):
                raise
            raise RenderError(f"{error}; PNG not updated, previous output retained") from error
