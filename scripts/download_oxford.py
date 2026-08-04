#!/usr/bin/env python3
"""mrgdatashare.robots.ox.ac.uk Oxford RobotCar downloader.

The official SDK (ori-mrg/robotcar-dataset-sdk) does NOT include a downloader,
so this script implements the Django login flow + scraping of /download/ to
fetch tar files for the categories we need (gps / ins / rtk only).

Credentials must be provided via environment variables to keep them out of
shell history and process listings:

    export OXFORD_USER='your-email'
    export OXFORD_PASS='your-password'
    python scripts/download_oxford.py --runs 2014-05-06-12-54-54 \\
        --categories ins rtk gps \\
        --out /mnt/Data/velref/oxford_robotcar

The script never prints, logs, or stores OXFORD_PASS. After completion you
should ``unset OXFORD_PASS``.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup


BASE = "https://mrgdatashare.robots.ox.ac.uk"
LOGIN_URL = f"{BASE}/accounts/login/"
DOWNLOAD_URL = f"{BASE}/download/"

DEFAULT_CATEGORIES = ("ins", "rtk", "gps")  # CSV-only metadata, no images/lidar


def login(session: requests.Session, username: str, password: str) -> None:
    r = session.get(LOGIN_URL, allow_redirects=True, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    csrf_input = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})
    if csrf_input is None:
        raise RuntimeError("login page missing CSRF token")
    csrf = csrf_input["value"]

    headers = {"Referer": LOGIN_URL}
    data = {
        "csrfmiddlewaretoken": csrf,
        "username": username,
        "password": password,
        "next": "/download/",
    }
    r2 = session.post(LOGIN_URL, data=data, headers=headers,
                      allow_redirects=True, timeout=30)
    r2.raise_for_status()
    if "/accounts/login" in r2.url or "Login" in r2.text[:500]:
        raise RuntimeError("login failed (still on login page); check credentials")


def list_downloads(session: requests.Session, run: str,
                   categories: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return [(href, label)] for tar files matching the run + categories.

    Parses the /download/ HTML table. Anchor tags whose href contains the
    run timestamp and whose label includes one of the requested categories
    are kept.
    """
    r = session.get(DOWNLOAD_URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    results: list[tuple[str, str]] = []
    cat_re = re.compile(r"\b(" + "|".join(re.escape(c) for c in categories) + r")\b",
                        flags=re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = (a.get_text() or "").strip()
        if run not in href and run not in label:
            continue
        if not (cat_re.search(href) or cat_re.search(label)):
            continue
        full = href if href.startswith("http") else BASE + href
        results.append((full, label or href))

    # Some sites encode a single ZIP per (run, sensor) using opaque IDs in href
    # but include the human-readable label in the row text. As a fallback we
    # also walk table rows and collect download links inside the row when the
    # row text contains run + a category keyword.
    if not results:
        for row in soup.find_all("tr"):
            row_text = row.get_text(" ", strip=True)
            if run not in row_text or not cat_re.search(row_text):
                continue
            for a in row.find_all("a", href=True):
                href = a["href"]
                if "logout" in href.lower() or href.startswith("#"):
                    continue
                full = href if href.startswith("http") else BASE + href
                results.append((full, row_text[:120]))

    # Deduplicate while preserving order.
    seen = set()
    deduped: list[tuple[str, str]] = []
    for h, l in results:
        if h in seen:
            continue
        seen.add(h)
        deduped.append((h, l))
    return deduped


def download_file(session: requests.Session, url: str, out_path: Path,
                  retries: int = 2) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  skip (exists): {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
        return
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                total = int(r.headers.get("content-length") or 0)
                seen_bytes = 0
                last_print = time.time()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            seen_bytes += len(chunk)
                            if time.time() - last_print > 2.0:
                                pct = (100.0 * seen_bytes / total) if total else 0.0
                                print(f"    {out_path.name}: {seen_bytes/1e6:.1f} MB"
                                      + (f" ({pct:.0f}%)" if total else ""), flush=True)
                                last_print = time.time()
                tmp.rename(out_path)
            print(f"  done: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  retry {attempt+1}: {e}", file=sys.stderr)
            time.sleep(3.0)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run IDs, e.g. 2014-05-06-12-54-54")
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    ap.add_argument("--out", type=Path, default=Path("/mnt/Data/velref/oxford_robotcar"))
    ap.add_argument("--list-only", action="store_true",
                    help="print discovered download URLs without fetching")
    args = ap.parse_args()

    user = os.environ.get("OXFORD_USER")
    pw = os.environ.get("OXFORD_PASS")
    if not user or not pw:
        print("ERROR: OXFORD_USER and OXFORD_PASS environment variables required",
              file=sys.stderr)
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": "velref-paper-downloader/0.1"})

    print(f"[1/3] login as {user} ...")
    login(session, user, pw)
    print("      logged in.")

    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        print(f"[2/3] discovering files for {run} (categories={args.categories}) ...")
        files = list_downloads(session, run, tuple(args.categories))
        if not files:
            print(f"      WARNING: no matching downloads for {run}")
            continue
        print(f"      found {len(files)} file(s):")
        for href, label in files:
            print(f"        - {label}")
            print(f"          {href}")
        if args.list_only:
            continue
        run_dir = args.out / run
        for href, label in files:
            # Best-effort filename: use last URL path component or label.
            name = href.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or label
            if not name or "/" in name:
                name = re.sub(r"\W+", "_", label)[:64] + ".bin"
            print(f"[3/3] fetching {name} ...")
            download_file(session, href, run_dir / name)

    print("all done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
