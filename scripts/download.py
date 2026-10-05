#!/usr/bin/env python3
"""Download every public input needed to reproduce the paper from the prediction stage.

Standard library only, so it runs before any environment is installed.

Groups:
  data         SynthRAD2023/2025 training archives (Zenodo). With ``--scope heldout``
               (default) only the held-out evaluation cases are extracted, straight from
               the remote zip with HTTP range requests (~7 GB instead of ~52 GB).
  transforms   IMPACT registration parameter files (Hugging Face datasets).
  elx          SynthRAD2025 organizer (ELX) registration parameter files, pinned to the
               SynthRAD2025/preprocessing commit, and the ELX transforms of the held-out
               cases computed with them by register_elx.py (Hugging Face, next to the
               checkpoints). Training cases: run register_elx.py.
  sim-cbct     The 103 simulated CBCT of the held-out SynthRAD2025 Task 2 cases (Hugging Face,
               next to the checkpoints).
  checkpoints  The 50 paper checkpoints listed in ``models/manifest.csv``.
  models       SAM 2.1 weights: official checkpoint (training loss) and the TorchScript
               encoder used by the calibrated d_SAM metric.

TotalSegmentator (Dice) and VGG16 are fetched automatically by KonfAI/torchvision on first use.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"

# Zenodo training archives: (year, task) -> (url, bytes, prefix inside the zip).
ZENODO = {
    (2023, 1): ("https://zenodo.org/api/records/7260705/files/Task1.zip/content", 14471900926, "Task1/"),
    (2023, 2): ("https://zenodo.org/api/records/7260705/files/Task2.zip/content", 10886603058, "Task2/"),
    (2025, 1): ("https://zenodo.org/api/records/15373853/files/synthRAD2025_Task1_Train.zip/content",
                14846718591, "synthRAD2025_Task1_Train/Task1/"),
    (2025, 2): ("https://zenodo.org/api/records/15373853/files/synthRAD2025_Task2_Train.zip/content",
                11978062547, "synthRAD2025_Task2_Train/Task2/"),
}
IMAGE_STEMS = ("ct", "mr", "cbct", "mask")

IMPACT_DATASETS = {
    2023: "VBoussot/synthrad2023-impact-registration",
    2025: "VBoussot/synthrad2025-impact-registration",
}
# Organizer deformable-registration parameters (stage2.py), GPL-3.0, fetched not vendored.
ELX_PARAMETERS = "https://raw.githubusercontent.com/SynthRAD2025/preprocessing/8a5b125e3a2a3e92287ed381519f5416bbb358b4/configs"
ELX_PARAMETER_FILES = [f"param_def_{modality}_{region}.txt" for modality in ("mr", "cbct") for region in ("AB", "HN", "TH")]
# The 50 paper checkpoints, stored under checkpoints/ with the layout of models/manifest.csv, the
# ELX transforms of the held-out cases, stored under transforms/ with the layout of data/transforms,
# and the simulated CBCT, stored under sim-cbct/{region}/{case}.mha.
CHECKPOINT_REPO = "VBoussot/RegistrationBias-sCT"
ELX_TRANSFORMS = "transforms/synthrad2025-elx-registration/"
SIM_CBCT = "sim-cbct/"

# (repo, path in repo, local path, sha256)
MODEL_FILES = [
    ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", None,
     "models/external/sam2.1_hiera_small.pt",
     "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"),
    ("VBoussot/ImpactSynth", "SAM2.1_Small.pt", "models/external/SAM2.1_Small.pt",
     "10c2fb000e57fa4adcb5f1c23aa7dfe6f622e1f649643ceb7e43e8188f024899"),
]


# --------------------------------------------------------------------------- HTTP

def open_url(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": "impactsynth-reproduce"}
    if url.startswith(HF) and os.environ.get("HF_TOKEN"):  # optional
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except urllib.error.URLError:
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(url)


def read_range(url: str, start: int, end: int) -> bytes:
    with open_url(url, start, end) as response:
        return response.read()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url: str, target: Path, expected_sha256: str | None = None) -> None:
    if target.is_file() and (expected_sha256 is None or sha256(target) == expected_sha256):
        print(f"ok    {target.relative_to(ROOT)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with open_url(url) as response, part.open("wb") as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    if expected_sha256 and sha256(part) != expected_sha256:
        part.unlink()
        raise SystemExit(f"SHA-256 mismatch for {url}")
    part.replace(target)
    print(f"got   {target.relative_to(ROOT)}")


# --------------------------------------------------------------------------- remote zip

def zip_index(url: str, size: int) -> list[dict]:
    """Parse the central directory of a remote (zip64) archive."""
    tail = read_range(url, max(0, size - 65558), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:
        z64_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        cd_size, cd_offset = struct.unpack("<QQ", read_range(url, z64_offset + 40, z64_offset + 55))
    directory = read_range(url, cd_offset, cd_offset + cd_size - 1)
    entries, pos = [], 0
    while directory[pos:pos + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", directory[pos + 10:pos + 12])
        crc, csize, usize = struct.unpack("<III", directory[pos + 16:pos + 28])
        n_name, n_extra, n_comment = struct.unpack("<HHH", directory[pos + 28:pos + 34])
        offset, = struct.unpack("<I", directory[pos + 42:pos + 46])
        name = directory[pos + 46:pos + 46 + n_name].decode("utf-8")
        extra = directory[pos + 46 + n_name:pos + 46 + n_name + n_extra]
        # zip64 extra field: 64-bit values replace, in order, the fields set to 0xFFFFFFFF.
        e = 0
        while e + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[e:e + 4])
            if tag == 1:
                values = list(struct.unpack(f"<{length // 8}Q", extra[e + 4:e + 4 + length]))
                if usize == 0xFFFFFFFF:
                    usize = values.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = values.pop(0)
                if offset == 0xFFFFFFFF:
                    offset = values.pop(0)
            e += 4 + length
        entries.append({"name": name, "method": method, "crc": crc, "csize": csize,
                        "usize": usize, "offset": offset})
        pos += 46 + n_name + n_extra + n_comment
    return entries


def extract_member(url: str, entry: dict, target: Path) -> None:
    if target.is_file() and target.stat().st_size == entry["usize"]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    end = entry["offset"] + 30 + 1024 + len(entry["name"]) + entry["csize"]
    with open_url(url, entry["offset"], end) as response, part.open("wb") as handle:
        header = response.read(30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {entry['name']}")
        n_name, n_extra = struct.unpack("<HH", header[26:30])
        response.read(n_name + n_extra)
        inflate = zlib.decompressobj(-15) if entry["method"] == 8 else None
        crc, remaining = 0, entry["csize"]
        while remaining:
            chunk = response.read(min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"truncated download for {entry['name']}")
            remaining -= len(chunk)
            data = inflate.decompress(chunk) if inflate else chunk
            crc = zlib.crc32(data, crc)
            handle.write(data)
        if inflate:
            tail = inflate.flush()
            crc = zlib.crc32(tail, crc)
            handle.write(tail)
    if crc != entry["crc"]:
        part.unlink()
        raise RuntimeError(f"CRC mismatch for {entry['name']}")
    part.replace(target)


def heldout_ids(task: int) -> set[str]:
    return set((ROOT / f"data/splits/task_{task}/Validation.txt").read_text().split())


def download_data(scope: str, max_cases: int | None, years, tasks, dry_run: bool) -> None:
    for (year, task), (url, size, prefix) in ZENODO.items():
        if year not in years or task not in tasks:
            continue
        wanted = heldout_ids(task) if scope == "heldout" else None
        selected, cases = [], {}
        for entry in zip_index(url, size):
            relative = entry["name"][len(prefix):]
            parts = relative.split("/")  # REGION/CASE/file
            if len(parts) != 3 or parts[2].split(".")[0] not in IMAGE_STEMS:
                continue
            region, case, _ = parts
            if wanted is None or case in wanted:
                cases.setdefault(region, set()).add(case)
                selected.append((entry, region, case, relative))
        if max_cases:
            cases = {region: set(sorted(ids)[:max_cases]) for region, ids in cases.items()}
        members = [(entry, ROOT / f"data/raw/synthrad{year}/Train/Task_{task}" / relative)
                   for entry, region, case, relative in selected if case in cases[region]]
        total = sum(entry["csize"] for entry, _ in members) / 1e9
        summary = ", ".join(f"{region} {len(ids)}" for region, ids in sorted(cases.items()))
        print(f"SynthRAD{year} Task {task}: {summary} cases, {len(members)} files, {total:.1f} GB")
        if dry_run:
            continue
        for index, (entry, target) in enumerate(members, 1):
            extract_member(url, entry, target)
            print(f"\r  {index}/{len(members)} {target.relative_to(ROOT)}", end="", flush=True)
        print()


# --------------------------------------------------------------------------- Hugging Face

def hf_files(repo: str, repo_type: str) -> list[dict]:
    """Recursive file listing, following the API's cursor pagination."""
    url = f"{HF}/api/{repo_type}s/{repo}/tree/main?recursive=true"
    files = []
    while url:
        with open_url(url) as response:
            files += [item for item in json.load(response) if item["type"] == "file"]
            link = response.headers.get("Link", "")
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return files


def hf_url(repo: str, repo_type: str, path: str) -> str:
    base = f"{HF}/datasets/{repo}" if repo_type == "dataset" else f"{HF}/{repo}"
    return f"{base}/resolve/main/{path}"


def download_transforms(scope: str, years, tasks, dry_run: bool) -> None:
    for year, repo in IMPACT_DATASETS.items():
        if year not in years:
            continue
        wanted = set().union(*(heldout_ids(task) for task in (1, 2))) if scope == "heldout" else None
        files = [item for item in hf_files(repo, "dataset")
                 if item["path"].endswith(".txt") and item["path"].count("/") == 2
                 and item["path"].split("/")[0] in {f"Task_{task}" for task in tasks}
                 and (wanted is None or Path(item["path"]).stem in wanted)]
        print(f"{repo}: {len(files)} transforms")
        if dry_run:
            continue
        for item in files:
            target = ROOT / "data/transforms" / repo.split("/")[1] / item["path"]
            if not target.is_file():
                fetch_file(hf_url(repo, "dataset", item["path"]), target)


def download_elx(tasks, dry_run: bool) -> None:
    print(f"SynthRAD2025 ELX parameters: {len(ELX_PARAMETER_FILES)} files")
    if not dry_run:
        for name in ELX_PARAMETER_FILES:
            fetch_file(f"{ELX_PARAMETERS}/{name}", ROOT / "data/transforms/synthrad2025-elx-parameters" / name)
    transforms = [item["path"] for item in hf_files(CHECKPOINT_REPO, "model")
                  if item["path"].startswith(ELX_TRANSFORMS)
                  and item["path"].split("/")[2] in {f"Task_{task}" for task in tasks}]
    print(f"{CHECKPOINT_REPO}: {len(transforms)} ELX transforms (held-out cases)")
    if dry_run:
        return
    for path in transforms:
        if not (ROOT / "data" / path).is_file():
            fetch_file(hf_url(CHECKPOINT_REPO, "model", path), ROOT / "data" / path)


def download_sim_cbct(dry_run: bool) -> None:
    files = [item["path"] for item in hf_files(CHECKPOINT_REPO, "model") if item["path"].startswith(SIM_CBCT)]
    print(f"{CHECKPOINT_REPO}: {len(files)} simulated CBCT")
    if dry_run:
        return
    for path in files:
        fetch_file(hf_url(CHECKPOINT_REPO, "model", path), ROOT / "data/raw/sim_cbct" / Path(path).name)


def download_checkpoints(pattern: str, dry_run: bool) -> None:
    rows = [row for row in csv.DictReader((ROOT / "models/manifest.csv").open())
            if fnmatch.fnmatch(row["repository_path"], pattern)]
    print(f"{CHECKPOINT_REPO}: {len(rows)} checkpoints, "
          f"{sum(int(row['bytes']) for row in rows) / 1e9:.1f} GB")
    if dry_run:
        return
    for row in rows:
        path = "checkpoints/" + row["repository_path"].removeprefix("models/checkpoints/")
        fetch_file(hf_url(CHECKPOINT_REPO, "model", path), ROOT / row["repository_path"], row["sha256"])


def download_models(dry_run: bool) -> None:
    for repo, path, local, digest in MODEL_FILES:
        url = repo if path is None else hf_url(repo, "model", path)
        print(f"model {local}")
        if not dry_run:
            fetch_file(url, ROOT / local, digest)


# --------------------------------------------------------------------------- main

GROUPS = ("data", "transforms", "elx", "sim-cbct", "checkpoints", "models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--scope", choices=("heldout", "all"), default="heldout",
                        help="heldout: evaluation cases only (prediction workflow); all: full training sets")
    parser.add_argument("--max-cases", type=int, help="limit cases per region (smoke test)")
    parser.add_argument("--year", type=int, nargs="+", choices=(2023, 2025), default=[2023, 2025],
                        help="data and transforms of these SynthRAD editions only")
    parser.add_argument("--task", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--checkpoints", default="*",
                        help="glob on models/manifest.csv paths, e.g. '*Task_2/IMPACT/MAE/*'")
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = parser.parse_args()

    missing = []
    for group in args.only:
        try:
            if group == "data":
                download_data(args.scope, args.max_cases, args.year, args.task, args.dry_run)
            elif group == "transforms":
                download_transforms(args.scope, args.year, args.task, args.dry_run)
            elif group == "elx":
                download_elx(args.task, args.dry_run)
            elif group == "sim-cbct":
                if 2025 in args.year and 2 in args.task:
                    download_sim_cbct(args.dry_run)
            elif group == "checkpoints":
                download_checkpoints(args.checkpoints, args.dry_run)
            else:
                download_models(args.dry_run)
        except (SystemExit, urllib.error.HTTPError) as error:
            missing.append(f"[{group}] {error}")
    if missing:
        print("\nNot available:\n" + "\n".join(missing), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
