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
    def begin(self, source: Path, *, all_nodes: bool) -> None:
        sources = {source}
        if all_nodes:
            sources.update(path for path in source.parent.iterdir() if path.suffix in {".yaml", ".yml"})
            for path in source.parent.glob("*.png"):
                name = owner(path)
                if name is not None:
                    sources.add(source.parent / name)
        conflicts = [path.with_suffix(".png") for path in sorted(sources)
                     if (path.with_suffix(".png").exists() or path.with_suffix(".png").is_symlink())
                     and owner(path.with_suffix(".png")) != path.name]
        if conflicts:
            raise OSError("output conflicts with PNG files not generated for these YAML files: "
                          + ", ".join(str(path) for path in conflicts))
        for path in sorted(sources):
            path.with_suffix(".png").unlink(missing_ok=True)

    def write(self, source: Path, graphs: list, samples: dict, notes: list[str], *, source_sha256: str) -> Path:
        def unchanged():
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
                raise RenderError(f"{source}: source changed after checking; rerun checking and rendering")

        unchanged()
        title = source.parent.name + " / " + source.stem
        image = graphviz(dot_text(title, graphs, samples, notes))
        information = {"kind": "pipeline_graph", "source": source.name, "source_sha256": source_sha256,
                       "samples": samples, "notes": notes, "acyclic": all(not graph.cycle for graph in graphs)}
        try:
            image = stamp(image, information)
        except ValueError as error:
            raise RenderError(f"Graphviz returned an invalid PNG: {error}") from error
        destination = source.with_suffix(".png")
        with TemporaryDirectory(prefix=".pipeline-", dir=source.parent) as temporary:
            staged = Path(temporary) / destination.name
            staged.write_bytes(image)
            unchanged()
            staged.replace(destination)
        return destination
