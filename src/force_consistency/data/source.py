import hashlib
import os
import re
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

URL = "https://ndownloader.figshare.com/files/57341180"
SIZE = 53019632
SHA256 = "3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b"


class SourceError(ValueError):
    pass


@dataclass(frozen=True)
class SourceManifest:
    url: str
    size: int
    sha256: str
    members: tuple[str, ...]


def validate_members(archive: zipfile.ZipFile) -> None:
    names = set()
    for entry in archive.infolist():
        p = PurePosixPath(entry.filename)
        mode = entry.external_attr >> 16
        if (p.is_absolute() or ".." in p.parts or "\\" in entry.filename
                or re.match(r"^[A-Za-z]:", entry.filename)
                or stat.S_ISLNK(mode) or entry.filename in names):
            raise SourceError(f"Unsafe or duplicate archive member: {entry.filename}")
        names.add(entry.filename)


def verify_archive(path: Path) -> SourceManifest:
    if path.stat().st_size != SIZE:
        raise SourceError("Landscape17 archive size mismatch")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise SourceError("Landscape17 archive checksum mismatch")
    with zipfile.ZipFile(path) as z:
        validate_members(z)
        members = tuple(sorted(z.namelist()))
    return SourceManifest(URL, SIZE, SHA256, members)


def fetch_archive(destination: Path) -> SourceManifest:
    if destination.exists():
        return verify_archive(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(URL, timeout=60) as r:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
        result = verify_archive(temporary)
        temporary.replace(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)
