"""Create a self-contained local handoff archive, excluding environments/secrets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORIES = ("data/raw/full_ten_stocks", "data/raw/full_ten_stocks_corrected",
                   "data/raw/primary_audit_2025_03_18")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/Financial_Time_Series_Project.zip")
    args = parser.parse_args()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {Path(name) for name in tracked if name and (ROOT / name).is_file()}
    for directory in RAW_DIRECTORIES:
        files.update(path.relative_to(ROOT) for path in (ROOT / directory).rglob("*") if path.is_file())
    metadata = ROOT / "data/raw/full_instrument_verification.json"
    if metadata.exists(): files.add(metadata.relative_to(ROOT))
    forbidden = {".env", "secrets.toml", ".DS_Store"}
    files = sorted(path for path in files if path.name not in forbidden
                   and not path.name.startswith(".env.") and "__pycache__" not in path.parts
                   and not any(part.startswith(".venv") for part in path.parts)
                   and (ROOT / path).resolve() != args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "financial-time-series-mining/"
    hashes = {}
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in files:
            payload = (ROOT / relative).read_bytes()
            hashes[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
            archive.writestr(prefix + relative.as_posix(), payload)
        archive.writestr(prefix + "ARCHIVE_CONTENTS.json", json.dumps(
            {"created_at_utc": datetime.now(timezone.utc).isoformat(), "files": hashes}, indent=2, sort_keys=True) + "\n")
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
    print(json.dumps({"archive": str(args.output), "files": len(files),
                      "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
