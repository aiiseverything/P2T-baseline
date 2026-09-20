#!/usr/bin/env python3
"""Pre-fetch the NVIDIA CUDA wheels from PyPI instead of pypi.nvidia.com.

The pinned environment resolves its `nvidia-*` packages through
``download.pytorch.org/whl/cu129``, whose index links to **pypi.nvidia.com**.
On this host that origin sustains roughly 0.4 MB/s while PyPI's own file host
does ~23 MB/s -- a ~50x difference across several GB of wheels.

This downloads the exact versions the lock pins, from files.pythonhosted.org,
into ``.wheels/`` so ``setup_env.sh`` can pass ``--find-links`` and have uv take
the local copies.  Wheels PyPI does not carry are reported and left to uv.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

# This host has no IPv6 route; urllib would otherwise prefer an AAAA record.
_getaddrinfo = socket.getaddrinfo


def _ipv4_only(*args, **kwargs):
    return [item for item in _getaddrinfo(*args, **kwargs) if item[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements/ssh-a6000-cu129.txt"
TARGET = ROOT / ".wheels"
WHEEL = re.compile(r'href="([^"]+\.whl[^"]*)"')
PLATFORM = ("x86_64", "manylinux")


def pinned_packages() -> dict[str, str]:
    lock = LOCK.read_text()
    pins = dict(re.findall(r"^(nvidia[a-z0-9-]+)==([^\s;]+)", lock, re.M))
    if not pins:
        raise SystemExit(f"no nvidia pins found in {LOCK}")
    return pins


def candidate_urls(name: str, version: str) -> list[str]:
    base = f"https://pypi.org/simple/{name}/"
    with urllib.request.urlopen(base, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    urls = [urllib.request.urljoin(base, href) for href in WHEEL.findall(html)]
    normalised = version.replace("-", "_")
    urls = [url for url in urls if normalised in Path(url).name]
    urls = [url for url in urls if all(tag in url for tag in PLATFORM)]
    urls = [url for url in urls if "aarch64" not in url and "win" not in url]
    return urls


def fetch(name: str, version: str) -> str:
    urls = candidate_urls(name, version)
    if not urls:
        return f"MISSING {name}=={version} is not on PyPI; uv will fetch it"
    url = urls[0]
    # Keep the wheel's own filename: `--find-links` only recognises files whose
    # names parse as wheel names, and a stripped name is silently ignored.
    destination = TARGET / urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
    if destination.is_file() and destination.stat().st_size:
        return f"cached  {destination.name} ({destination.stat().st_size / 2**20:.1f} MiB)"
    urllib.request.urlretrieve(url, destination)
    return f"fetched {destination.name} ({destination.stat().st_size / 2**20:.1f} MiB)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without downloading")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    TARGET.mkdir(exist_ok=True)
    pins = pinned_packages()
    print(f"{len(pins)} pinned nvidia packages -> {TARGET}")
    if args.check:
        for name, version in sorted(pins.items()):
            present = (TARGET / f"{name}-{version}").is_file()
            print(f"  {'ok     ' if present else 'MISSING'} {name}=={version}")
        return 0
    failures = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda item: fetch(*item), sorted(pins.items())):
            print("  " + result)
            failures += result.startswith("MISSING")
    (TARGET / "manifest.json").write_text(json.dumps({"pins": pins}, indent=2))
    print(f"done; {failures} package(s) left to uv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
