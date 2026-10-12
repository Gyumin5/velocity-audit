#!/usr/bin/env python3
"""Fetch a small subset of entries out of a very large remote zip by byte range.

Reviewer 1 (comment 3) asks for the residual comparison recomputed with matched
horizontal-speed definitions wherever the components are available.  On KITTI
raw the components live in the OXTS text files, which are a few megabytes inside
a 21~GB per-drive archive; we no longer hold a local copy of the release.  A zip
central directory lets us read just those entries: locate the directory, pick the
entries we want, and fetch the one contiguous byte range that covers them.

Nothing here is specific to KITTI; the caller supplies the url, a substring the
member names must contain, and an output directory.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import struct
import sys
import urllib.request
import zlib

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"


def get_range(url: str, start: int, end: int) -> bytes:
    """Inclusive byte range, as HTTP defines it."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def find_central_directory(url: str, size: int) -> tuple[int, int]:
    """Return (offset, size) of the central directory, zip64 included."""
    tail_len = min(size, 1 << 16)
    tail = get_range(url, size - tail_len, size - 1)
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise SystemExit("no end-of-central-directory record found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(EOCD64_LOC_SIG, 0, i)
    if j >= 0:
        eocd64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        blob = get_range(url, eocd64_off, eocd64_off + 55)
        if blob[:4] != EOCD64_SIG:
            raise SystemExit("zip64 locator does not point at a zip64 record")
        cd_size, cd_off = struct.unpack("<QQ", blob[40:56])
    return cd_off, cd_size


def zip64_extra(extra: bytes, want: int) -> list[int]:
    """Values from the zip64 extra field, in the order the spec stores them."""
    k = 0
    while k + 4 <= len(extra):
        tag, ln = struct.unpack("<HH", extra[k:k + 4])
        if tag == 0x0001:
            body = extra[k + 4:k + 4 + ln]
            return [struct.unpack("<Q", body[8 * m:8 * m + 8])[0]
                    for m in range(min(want, len(body) // 8))]
        k += 4 + ln
    return []


def parse_entries(cd: bytes, needle: str) -> list[dict]:
    out, k = [], 0
    while k + 46 <= len(cd) and cd[k:k + 4] == CEN_SIG:
        (method, csize, usize, nlen, elen, clen) = (
            struct.unpack("<H", cd[k + 10:k + 12])[0],
            struct.unpack("<I", cd[k + 20:k + 24])[0],
            struct.unpack("<I", cd[k + 24:k + 28])[0],
            struct.unpack("<H", cd[k + 28:k + 30])[0],
            struct.unpack("<H", cd[k + 30:k + 32])[0],
            struct.unpack("<H", cd[k + 32:k + 34])[0],
        )
        lho = struct.unpack("<I", cd[k + 42:k + 46])[0]
        name = cd[k + 46:k + 46 + nlen].decode("utf-8", "replace")
        extra = cd[k + 46 + nlen:k + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, lho):
            vals = zip64_extra(extra, 3)
            seq = iter(vals)
            if usize == 0xFFFFFFFF:
                usize = next(seq, usize)
            if csize == 0xFFFFFFFF:
                csize = next(seq, csize)
            if lho == 0xFFFFFFFF:
                lho = next(seq, lho)
        if needle in name and not name.endswith("/"):
            out.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "offset": lho})
        k += 46 + nlen + elen + clen
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--contains", required=True,
                    help="substring the member path must contain, e.g. /oxts/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-span-mb", type=float, default=512.0,
                    help="refuse to fetch a contiguous span larger than this")
    args = ap.parse_args()

    size = content_length(args.url)
    cd_off, cd_size = find_central_directory(args.url, size)
    cd = get_range(args.url, cd_off, cd_off + cd_size - 1)
    entries = parse_entries(cd, args.contains)
    if not entries:
        raise SystemExit(f"no entries matching {args.contains!r}")
    lo = min(e["offset"] for e in entries)
    hi = max(e["offset"] + e["csize"] for e in entries) + 4096
    span = hi - lo
    print(f"archive {size/1e9:.1f} GB, {len(entries)} matching entries, "
          f"contiguous span {span/1e6:.1f} MB")
    if span > args.max_span_mb * 1e6:
        raise SystemExit("matching entries are not contiguous enough; aborting")
    blob = get_range(args.url, lo, hi - 1)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for e in entries:
        p = e["offset"] - lo
        if blob[p:p + 4] != LOC_SIG:
            print(f"  skip {e['name']}: local header not where the directory says")
            continue
        nlen, elen = struct.unpack("<HH", blob[p + 26:p + 30])
        start = p + 30 + nlen + elen
        raw = blob[start:start + e["csize"]]
        data = raw if e["method"] == 0 else zlib.decompress(raw, -15)
        if len(data) != e["usize"]:
            print(f"  skip {e['name']}: size {len(data)} != {e['usize']}")
            continue
        dest = args.out / e["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    print(f"wrote {written} files under {args.out}")


if __name__ == "__main__":
    sys.exit(main())
