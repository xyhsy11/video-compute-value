"""Bounded public HTTP acquisition. Never fetch an entire video archive."""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

from .common import Blocked, read_json, write_json


def request(url, headers=None, method="GET"):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method), timeout=30)


def fetch_json(url, ledger):
    with request(url) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    ledger.charge_transfer(len(content))
    if len(content) > 8 * 1024 * 1024:
        raise Blocked("BLOCKED_INPUT", "Metadata response exceeds 8 MiB")
    return json.loads(content)


def download_small(url, path, ledger, limit=16 * 1024 * 1024):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with request(url) as response:
        size = int(response.headers.get("Content-Length", 0))
        if size > limit:
            raise Blocked("BLOCKED_RESOURCE", "Annotation exceeds small-download limit")
        ledger.check(projected_bytes=limit)
        data = response.read(limit + 1)
    ledger.charge_transfer(len(data))
    if len(data) > limit:
        raise Blocked("BLOCKED_RESOURCE", "Oversized metadata transfer aborted")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def dataset_metadata(config, root, ledger):
    repo = config["annotation_repo"]
    info = fetch_json(f"https://huggingface.co/api/datasets/{repo}", ledger)
    revision = config.get("annotation_revision") or info["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Annotation revision must be an immutable SHA")
    annotation = Path(root, "data", "annotations", revision + ".parquet")
    if not annotation.exists():
        download_small(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{config['annotation_filename']}",
                       annotation, ledger)
    # The HF file listing is small; it contains metadata, not full archives.
    listing = fetch_json(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true", ledger)
    archives = [{"path": entry["path"], "size": entry["size"]} for entry in listing
                if entry.get("type") == "file" and re.fullmatch(r"videos_chunked_\d+\.zip", entry["path"])]
    if not archives:
        raise Blocked("BLOCKED_INPUT", "Official distribution archive layout changed; inspect before downloading")
    return annotation, {"repo": repo, "revision": revision, "archives": archives,
                        "acquisition_source": "Official Video-MME project link to LMMS-Lab/LMMS-Eval"}


class RangeReader(io.RawIOBase):
    """Seekable ZIP view with strict HTTP 206 and Content-Range validation."""
    def __init__(self, url, size, ledger):
        super().__init__()
        self.url, self.size, self.ledger, self.position = url, size, ledger, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 8 * 1024 * 1024:
            raise Blocked("BLOCKED_RESOURCE", "ZIP reader requested an unexpectedly large range")
        self.ledger.allow_video_transfer(size)
        self.ledger.check(projected_bytes=size)
        start, end = self.position, self.position + size - 1
        with request(self.url, {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise Blocked("BLOCKED_INPUT", "Server did not honor exact byte range; full archive transfer aborted")
            content = response.read(size + 1)
        self.ledger.charge_transfer(len(content), video=True)
        if len(content) != size:
            raise Blocked("BLOCKED_INPUT", "Truncated or oversized HTTP range response")
        self.position += len(content)
        return content


def archive_index(metadata, root, ledger):
    cache = Path(root, "data", "archive_index.json")
    if cache.exists():
        previous = read_json(cache)
        if previous["revision"] == metadata["revision"]:
            return previous["members"]
    members = {}
    for archive in metadata["archives"]:
        print(f"Inspecting ZIP directory: {archive['path']} ({archive['size']} bytes; ranges only)", flush=True)
        url = (f"https://huggingface.co/datasets/{metadata['repo']}/resolve/"
               f"{metadata['revision']}/{quote(archive['path'])}")
        with RangeReader(url, archive["size"], ledger) as reader, zipfile.ZipFile(reader) as zipped:
            for entry in zipped.infolist():
                if Path(entry.filename).suffix.lower() not in (".mp4", ".mkv", ".webm"):
                    continue
                source_id = Path(entry.filename).stem
                if source_id in members:
                    raise Blocked("INPUT_INTEGRITY_FAILED", f"Duplicate source in official archives: {source_id}")
                members[source_id] = {"archive": archive, "member": entry.filename,
                                      "compressed_size": entry.compress_size, "size": entry.file_size}
    write_json(cache, {"revision": metadata["revision"], "members": members})
    return members


def acquire_video(source_id, member, metadata, root, ledger):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Unsafe source identifier")
    target = Path(root, "data", "videos", source_id + Path(member["member"]).suffix)
    if target.exists():
        return target
    ledger.allow_video_transfer(member["compressed_size"] + 1024 * 1024)
    ledger.check(projected_bytes=member["size"])
    url = (f"https://huggingface.co/datasets/{metadata['repo']}/resolve/"
           f"{metadata['revision']}/{quote(member['archive']['path'])}")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Acquiring source {source_id}: {member['compressed_size']} compressed bytes", flush=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with RangeReader(url, member["archive"]["size"], ledger) as reader, zipfile.ZipFile(reader) as zipped:
            with zipped.open(member["member"]) as source, temporary.open("wb") as output:
                written = 0
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > member["size"]:
                        raise Blocked("INPUT_INTEGRITY_FAILED", "ZIP member expanded beyond declared size")
                    output.write(chunk)
        if temporary.stat().st_size != member["size"]:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Extracted video size mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
