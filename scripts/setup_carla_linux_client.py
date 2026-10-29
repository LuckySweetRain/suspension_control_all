"""Extract only the matching Linux CARLA 0.9.11 API from its release archive.

No maps, engine binaries or Windows artifacts are installed into WSL. A local
archive can be supplied to avoid re-downloading the release stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "carsim_mpc_quick/orin"))
from runtime.paths import SETTINGS, resolve_resource

URL = "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.11.tar.gz"
NAME = "carla-0.9.11-py3.7-linux-x86_64.egg"


def extract(source, target):
    last = time.monotonic()
    with tarfile.open(fileobj=source, mode="r|gz", bufsize=1024 * 1024) as archive:
        for member in archive:
            if time.monotonic() - last > 20:
                print("Scanning release:", member.name, flush=True)
                last = time.monotonic()
            if Path(member.name).name != NAME:
                continue
            if not member.isfile() or not 1024 * 1024 <= member.size <= 200 * 1024 * 1024:
                raise ValueError("Unexpected CARLA API archive member")
            raw = archive.extractfile(member).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never extract arbitrary archive paths or execute its setup scripts.
            target.write_bytes(raw)
            return
    raise FileNotFoundError("Matching CARLA API missing from release")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    target = resolve_resource(SETTINGS["linux_carla_egg"])
    if target.name != NAME:
        raise ValueError("linux_carla_egg must name the matching Linux 0.9.11 Python 3.7 API")
    if not target.exists():
        with (
            args.archive.open("rb")
            if args.archive
            else urllib.request.urlopen(URL, timeout=60) as incoming
        ):
            extract(incoming, target)
    with zipfile.ZipFile(target) as api:
        if api.testzip() is not None:
            raise ValueError("Corrupt CARLA Python API")
    metadata = dict(
        source=str(args.archive) if args.archive else URL,
        file=str(target),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size,
    )
    target.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
