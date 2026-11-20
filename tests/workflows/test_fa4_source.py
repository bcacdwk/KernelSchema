"""Real codec/install/closure tests. No CUDA or original module imports."""
from copy import deepcopy
from pathlib import Path

import pytest

from kernelschema import artifacts
from kernelschema.artifacts import (Bundle, FileBytes, Publication, import_bundle, install_source,
    replace_source, source_files_manifest, source_manifest, verify_source_directory)
from kernelschema.contracts import ProtocolError, UnsupportedError, validate


def freeze(root, files):
    p = Publication(root)
    ref = p.put_source_files("source", {name: FileBytes(data, 0o644) for name, data in files.items()})
    p.commit([ref])
    return ref


def test_tree_roundtrip_import_modes_and_single_file_refusal(tmp_path):
    ref = freeze(tmp_path / "bundle", {"solution.py": b"# original CRLF\r\n", "pkg/__init__.py": b"", "pkg/op.py": b"x=1\n"})
    b = Bundle(tmp_path / "bundle")
    assert b.source_files(ref)["solution.py"].data == b"# original CRLF\r\n"
    with pytest.raises(UnsupportedError, match="single_file"):
        b.source(ref)
    import_bundle(b, tmp_path / "imported", [ref])
    imported = Bundle(tmp_path / "imported")
    assert imported.source_files(ref) == b.source_files(ref)
    install_source(imported, ref, tmp_path / "draft")
    verify_source_directory(b, ref, tmp_path / "draft")
    assert (tmp_path / "draft/pkg/op.py").stat().st_mode & 0o777 == 0o644
    with pytest.raises(ProtocolError, match="new directory"):
        install_source(b, ref, tmp_path / "draft")


def test_two_to_one_deletes_old_and_replay_is_idempotent(tmp_path):
    old = freeze(tmp_path, {"solution.py": b"old\n", "gone.py": b"remove me\n"})
    new = freeze(tmp_path, {"solution.py": b"new\n"})
    b = Bundle(tmp_path)
    install_source(b, old, tmp_path / "draft")
    replace_source(b, new, tmp_path / "draft", previous=old)
    replace_source(b, new, tmp_path / "draft", previous=old)
    assert sorted(p.name for p in (tmp_path / "draft").iterdir()) == ["solution.py"]
    assert (tmp_path / "draft/solution.py").read_bytes() == b"new\n"


def test_interrupted_exchange_recovers_without_overlay(tmp_path, monkeypatch):
    old = freeze(tmp_path, {"solution.py": b"old", "removed.py": b"old"})
    new = freeze(tmp_path, {"solution.py": b"new"})
    b = Bundle(tmp_path)
    install_source(b, old, tmp_path / "draft")
    sync = artifacts.fsync_directory
    def crash(path):
        if path == tmp_path and (tmp_path / "draft/solution.py").read_bytes() == b"new":
            raise RuntimeError("synthetic interruption after atomic exchange")
        return sync(path)
    monkeypatch.setattr(artifacts, "fsync_directory", crash)
    with pytest.raises(RuntimeError, match="synthetic"):
        replace_source(b, new, tmp_path / "draft", previous=old)
    verify_source_directory(b, new, tmp_path / "draft")
    monkeypatch.setattr(artifacts, "fsync_directory", sync)
    replace_source(b, new, tmp_path / "draft", previous=old)
    assert not list(tmp_path.glob(".ks-replace-*"))


@pytest.mark.parametrize("name", ["../out.py", "pkg/nested/a.py", "thing.so", "bin.sh", "pkg/data.json", "absolute/../x.py"])
def test_unsupported_tree_paths_reject(name):
    with pytest.raises(ProtocolError):
        source_files_manifest({"solution.py": FileBytes(b"", 0o644), name: FileBytes(b"", 0o644)})


def test_bytes_modes_extra_files_and_symlinks_are_checked(tmp_path):
    with pytest.raises(ProtocolError):
        source_files_manifest({"solution.py": FileBytes(b"\xff", 0o644)})
    with pytest.raises(UnsupportedError):
        source_files_manifest({"solution.py": FileBytes(b"", 0o755)})
    ref = freeze(tmp_path, {"solution.py": b"ok"})
    root = tmp_path / Path(ref["path"]).parent / "files"
    (root / "extra.py").write_bytes(b"not in snapshot")
    with pytest.raises(ProtocolError, match="extra"):
        Bundle(tmp_path).source_files(ref)
    (root / "extra.py").unlink()
    (root / "solution.py").chmod(0o755)
    with pytest.raises(ProtocolError, match="mode"):
        Bundle(tmp_path).source_files(ref)
    (root / "solution.py").unlink()
    (root / "solution.py").symlink_to("/etc/hosts")
    with pytest.raises(ProtocolError):
        Bundle(tmp_path).source_files(ref)


def test_finite_limits_and_legacy_format_remain_strict(tmp_path):
    with pytest.raises(ProtocolError, match="entrypoint"):
        source_files_manifest({"other.py": FileBytes(b"", 0o644)})
    with pytest.raises(UnsupportedError, match="32 files"):
        source_files_manifest({"solution.py": FileBytes(b"x" * (4 * 1024 * 1024 + 1), 0o644)})
    legacy = source_manifest(b"pass\n")
    wrong = deepcopy(legacy)
    wrong["files"][0]["name"] = "solution.py"
    with pytest.raises(UnsupportedError):
        validate(wrong)
    p = Publication(tmp_path)
    ref = p.put_source("old", b"pass\n")
    p.commit([ref])
    assert Bundle(tmp_path).source(ref) == b"pass\n"
