"""Source-checkout resources; no cwd assumptions, downloads or repository scans."""
from pathlib import Path

from .contracts import ProtocolError, relative_path


def checkout_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "schema_template" / "tools" / "schema.py").is_file():
        raise ProtocolError("KernelSchema source resources missing; use the supported source checkout/editable installation")
    return root


def resource(path: str) -> Path:
    relative_path(path)
    result = checkout_root() / path
    if not result.is_file() or not result.resolve().is_relative_to(checkout_root()):
        raise ProtocolError(f"missing checkout resource: {path}")
    return result


def rule_files() -> tuple[Path, ...]:
    root = checkout_root()
    return tuple(sorted((root / "schema_docs").glob("[0-6].*.md"))) + tuple(
        resource(name) for name in ("DESIGN_SCHEMA.md", "OPTIMIZE_SCHEMA.md", "IMPLEMENT_SCHEMA.md", "schema_template/README.md")
    )


def freeze_rules(publication):
    """Freeze the actual role/docs/Template implementation bytes into Rules.

    Tools' transitive Python source files are explicit rule files. The tool
    version is its actual schema.py digest, never a floating Git/marketing tag.
    """
    from .contracts import PROTOCOL
    root = checkout_root()
    implementation = tuple(sorted(path for path in (root / "schema_template/tools").rglob("*.py") if "tests" not in path.relative_to(root).parts))
    refs = {str(path.relative_to(root)): publication.put_bytes("rules/" + str(path.relative_to(root)), path.read_bytes()) for path in (*rule_files(), *implementation)}
    source = refs["schema_template/tools/schema.py"]
    value = {"protocol": PROTOCOL, "kind": "rules", "files": sorted(refs.values(), key=lambda x: x["path"]),
             "tools": [{"name": "template", "version": source["sha256"], "source": source}]}
    return publication.put_json("rules/manifest.json", value)
