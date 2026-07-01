"""Download the cleaned LongMemEval-S dataset from HuggingFace.

The file is ~270 MB. Re-downloads are skipped if the destination already
exists. No HuggingFace credentials or extra dependencies are required;
the file is fetched over plain HTTPS.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
DEFAULT_DEST = Path("data/longmemeval_s_cleaned.json")


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")

    request = urllib.request.Request(url, headers={"User-Agent": "marvin-eval/0.1"})
    with (
        urllib.request.urlopen(request) as response,
        tmp.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)

    tmp.rename(dest)
    print(f"done: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination exists",
    )
    args = parser.parse_args(argv)
    try:
        download(args.url, args.dest, force=args.force)
    except Exception as exc:  # pragma: no cover - network
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
