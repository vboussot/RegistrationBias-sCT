"""Remote-zip extraction in scripts/download.py, served from an in-memory zip64 archive."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download", ROOT / "scripts" / "download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_remote_zip_member_extraction(tmp_path, monkeypatch) -> None:
    payload = bytes(range(256)) * 4000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Task1/AB/1ABA001/ct.mha", payload)
        with archive.open("Task1/AB/1ABA001/mask.mha", "w", force_zip64=True) as member:
            member.write(b"\x01" * 5000)
    blob = buffer.getvalue()

    def fake_open(url, start=None, end=None):
        return io.BytesIO(blob[start:end + 1] if start is not None else blob)

    monkeypatch.setattr(download, "open_url", fake_open)
    entries = {entry["name"]: entry for entry in download.zip_index("zip", len(blob))}
    assert set(entries) == {"Task1/AB/1ABA001/ct.mha", "Task1/AB/1ABA001/mask.mha"}

    target = tmp_path / "ct.mha"
    download.extract_member("zip", entries["Task1/AB/1ABA001/ct.mha"], target)
    assert target.read_bytes() == payload
    download.extract_member("zip", entries["Task1/AB/1ABA001/mask.mha"], tmp_path / "mask.mha")
    assert (tmp_path / "mask.mha").read_bytes() == b"\x01" * 5000
