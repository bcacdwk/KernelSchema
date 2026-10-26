#!/usr/bin/env python3
"""Install the pinned official CUE binary locally, verifying its SHA-256."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import platform
import tarfile
import tempfile
import urllib.request

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEMPLATE_ROOT.parent
VERSION = "v0.17.1"
# Official cue-lang/cue v0.17.1 release asset digests. No moving download URLs.
SHA256 = {
    ("darwin", "amd64"): "80aa026c3f47400c7bfc228b3422fd56b7c6c5ea4d70686f8d8f01ced716d3de",
    ("darwin", "arm64"): "64921403f012a97f89494c03605db2fbf7d9daa77dc2631819ac4406cb2e8074",
    ("linux", "amd64"): "a39b0c97695069d95d276d99be0f5dbabb081d801bfdc9ba49b76efaf94e2369",
    ("linux", "arm64"): "0d729be30d52c952ca38fc9dcb692caa09d8463fa0b64df5781312779183fbcd",
}


def main() -> None:
    if ROOT.joinpath(".cue-version").read_text().strip() != VERSION:
        raise SystemExit("Update the pinned release digests together with .cue-version.")
    system = platform.system().lower()
    machine = {"x86_64": "amd64", "aarch64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )
    digest = SHA256.get((system, machine))
    if digest is None:
        raise SystemExit(f"Unsupported installer platform: {system}/{machine}; supply CUE={VERSION} binary.")
    asset = f"cue_{VERSION}_{system}_{machine}.tar.gz"
    url = f"https://github.com/cue-lang/cue/releases/download/{VERSION}/{asset}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != digest:
        raise SystemExit("CUE archive checksum mismatch; nothing installed.")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = next((m for m in bundle.getmembers() if m.name in ("cue", "./cue")), None)
        if member is None or not member.isfile():
            raise SystemExit("CUE executable missing from official archive.")
        binary = bundle.extractfile(member)
        assert binary is not None
        payload = binary.read()
    target = TEMPLATE_ROOT / ".tools" / "cue" / VERSION / "cue"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(payload)
    try:
        temporary.chmod(0o755)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(target)


if __name__ == "__main__":
    main()
