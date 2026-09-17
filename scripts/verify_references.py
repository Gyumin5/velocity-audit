#!/usr/bin/env python3
"""Check every ref.bib entry against CrossRef (and arXiv for preprints).

Reviewer 3 of the IEEE Access first round listed eight entries with a wrong
year, a wrong journal, a format problem, or an author list that does not match
the arXiv record.  A DOI that resolves is not enough: what has to agree is the
year, the container title, and the author list.  This checks those three.

Entries with a doi field are looked up by DOI.  Entries with an arXiv URL are
looked up through the arXiv API.  Everything else is searched by title and the
best match is reported for a human to accept or reject.

  python scripts/verify_references.py            # report
  python scripts/verify_references.py --json out.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "paper" / "ref.bib"
UA = "velocity-audit-refcheck/1.0 (mailto:ogm3614@snu.ac.kr)"


# The NeurIPS Datasets and Benchmarks Track prints no page numbers, so this
# entry has no page range to carry. Every other conference entry does.
NEURIPS_UNPAGINATED = {"wilson2023argoverse2"}

def norm_name(name: str) -> str:
    """Compare names on letters alone: the BibTeX source spells accents as
    macros (K{\\"u}mmerle) and CrossRef spells them as characters."""
    name = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-z]", "", name.lower())


# One registered record is itself wrong about a given name, and our entry is
# right, so the difference is recorded here rather than "corrected" into the
# bibliography. Keyed by (bib key, normalised family name).
RECORD_GIVEN_DEFECTS = {
    ("martinez2020pit30m", "barsan"):
        "CrossRef stores 'loan Andrei' with a lowercase L; the author is Ioan Andrei Barsan",
}


def norm_given(name: str) -> str:
    """Given names normalised to space-separated words. Unlike norm_name this keeps
    the word boundaries, because "Ioan A." and "Marcel J. E." collapse into single
    tokens without them and then read as disagreements with the spelled-out record."""
    name = unicodedata.normalize("NFKD", name)
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in re.split(r"[\s.\-]+", name)]
    return " ".join(w for w in words if w)


def people(author_field: str) -> list:
    """Every author in the entry as (family, given), in order.

    A resolving DOI says nothing about the author list, and reviewer 3 asked for
    the list to be right rather than only its first name. The given name is kept
    because the records carry it: nikolic2016imu named the right family with the
    wrong given name (Andreas for Amir) and a family-only check passed it."""
    out = []
    for part in re.split(r"\s+and\s+", author_field.strip()):
        part = part.replace("{", "").replace("}", "").strip()
        if not part:
            continue
        if "," in part:
            fam, _, given = part.partition(",")
        else:
            words = part.split()
            fam, given = words[-1], " ".join(words[:-1])
        out.append((norm_name(fam), norm_given(given)))
    return out


def families(author_field: str) -> list:
    return [f for f, _ in people(author_field)]


def given_agrees(ours: str, theirs: str) -> bool:
    """An initial agrees with the name it abbreviates; two spelled-out names do not
    agree unless they match. Either side may be initials, so compare at the shorter
    resolution rather than calling every abbreviation a disagreement."""
    if not ours or not theirs:
        return True
    a, b = ours.split(), theirs.split()
    if not a or not b:
        return True
    for x, y in zip(a, b):
        if len(x) == 1 or len(y) == 1:
            if x[0] != y[0]:
                return False
        elif x != y:
            return False
    return True


def first_family(author_field: str) -> str:
    """Family name of the first author, from either BibTeX spelling.

    A multi-word family name (Le Gentil) survives only when the entry writes
    "Family, Given", so the comma form decides where the family name ends."""
    first = re.split(r"\s+and\s+", author_field.strip())[0]
    first = first.replace("{", "").replace("}", "").strip()
    name = first.split(",")[0] if "," in first else first.split()[-1]
    return norm_name(name)


def get(url: str, tries: int = 3) -> str:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return ""


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        start = m.end()
        depth, i = 1, m.start(0) + text[m.start(0):].index("{")
        i += 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                depth, k = 1, j + 1
                while k < len(body) and depth:
                    depth += (body[k] == "{") - (body[k] == "}")
                    k += 1
                val = body[j + 1:k - 1]
            elif body[j] == '"':
                k = body.index('"', j + 1)
                val = body[j + 1:k]
            else:
                k = j
                while k < len(body) and body[k] not in ",\n":
                    k += 1
                val = body[j:k]
            fields[fm.group(1).lower()] = " ".join(val.split())
        entries.append({"type": m.group(1).lower(), "key": m.group(2).strip(), **fields})
    return entries


def clean(s: str) -> str:
    s = re.sub(r"[{}\\]", "", s or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def first_surname(author_field: str) -> str:
    a = re.split(r"\s+and\s+", author_field or "", maxsplit=1)[0]
    a = re.sub(r"[{}\\]", "", a)
    return clean(a.split(",")[0] if "," in a else a.split()[-1] if a.split() else "")


def norm_pages(s: str) -> str:
    """'1231--1237' and '1231-1237' are the same page range."""
    return re.sub(r"[^0-9]+", "-", s or "").strip("-")


def crossref_by_doi(doi: str) -> dict | None:
    try:
        return json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    except Exception:
        return None


def crossref_by_title(title: str) -> dict | None:
    try:
        q = urllib.parse.urlencode({"query.bibliographic": clean(title), "rows": 1})
        items = json.loads(get(f"https://api.crossref.org/works?{q}"))["message"]["items"]
        return items[0] if items else None
    except Exception:
        return None


def cr_year(msg: dict) -> int | None:
    for k in ("published-print", "published-online", "issued", "created"):
        p = msg.get(k, {}).get("date-parts", [[None]])[0]
        if p and p[0]:
            return int(p[0])
    return None


def arxiv(aid: str) -> dict | None:
    try:
        x = get(f"http://export.arxiv.org/api/query?id_list={aid}")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(x).find("a:entry", ns)
        if e is None:
            return None
        return {
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "published": e.findtext("a:published", "", ns),
        }
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    report, bad = [], 0
    for n, e in enumerate(entries, 1):
        row = {"n": n, "key": e["key"], "problems": [], "source": None}
        bib_year = re.search(r"\d{4}", e.get("year", ""))
        bib_year = int(bib_year.group()) if bib_year else None
        venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or ""

        url = e.get("howpublished", "") + " " + e.get("url", "")
        m = re.search(r"arxiv\.org/abs/([\d.]+)", url)
        if m:
            row["source"] = f"arXiv:{m.group(1)}"
            a = arxiv(m.group(1))
            if not a:
                row["problems"].append("arXiv record not retrievable")
            else:
                row["remote_title"] = a["title"]
                row["remote_authors"] = a["authors"]
                row["remote_year"] = int(a["published"][:4])
                row["compare"] = {
                    "year": (e.get("year", ""), a["published"][:4]),
                    "title": (e.get("title", ""), a["title"]),
                    "authors": (e.get("author", ""), "; ".join(a["authors"])),
                }
                if clean(a["title"]) != clean(e.get("title", "")):
                    row["problems"].append(f"title differs from arXiv: {a['title']!r}")
                if a["authors"] and first_surname(e.get("author", "")) != clean(a["authors"][0].split()[-1]):
                    row["problems"].append(
                        f"first author {e.get('author','')!r} vs arXiv {a['authors'][0]!r}")
                if bib_year and int(a["published"][:4]) != bib_year:
                    row["problems"].append(f"year {bib_year} vs arXiv {a['published'][:4]}")
        elif e.get("doi"):
            row["source"] = f"doi:{e['doi']}"
            msg = crossref_by_doi(e["doi"])
            if not msg:
                row["problems"].append("DOI does not resolve at CrossRef")
            else:
                ct = (msg.get("container-title") or [""])[0]
                y = cr_year(msg)
                rt = (msg.get("title") or [""])[0]
                row.update({"remote_title": rt, "remote_venue": ct, "remote_year": y})
                # Both sides of every compared field are kept so that the
                # comparison can be printed side by side for a human to read,
                # not only the fields that disagreed.
                row["compare"] = {
                    "year": (e.get("year", ""), str(y or "")),
                    "venue": (venue, ct),
                    "title": (e.get("title", ""), rt),
                    "authors": (e.get("author", ""),
                                "; ".join(f"{a.get('given','')} {a.get('family','')}".strip()
                                          for a in (msg.get("author") or []))),
                    "volume": (e.get("volume", ""), str(msg.get("volume") or "")),
                    "pages": (e.get("pages", ""), str(msg.get("page") or "")),
                }
                if y and bib_year and y != bib_year:
                    row["problems"].append(f"year {bib_year} vs CrossRef {y}")
                if ct and clean(ct) != clean(venue) and clean(venue) not in clean(ct):
                    row["problems"].append(f"venue {venue!r} vs CrossRef {ct!r}")
                if rt and clean(rt) != clean(e.get("title", "")):
                    row["problems"].append(f"title differs: CrossRef {rt!r}")
                # The author list is what reviewer 3 flagged on the arXiv entry,
                # and an entry can carry a resolving DOI while naming someone who
                # is not on the paper or dropping co-authors, so the whole list is
                # compared, in order.
                ra = [(norm_name(a.get("family", "")), norm_given(a.get("given", "")))
                      for a in (msg.get("author") or []) if a.get("family")]
                if ra and e.get("author"):
                    ours, theirs = people(e["author"]), ra
                    # "and others" means the tail is deliberately elided, so only
                    # the names the entry does print have to agree.
                    if ours and ours[-1][0] == "others":
                        ours, theirs = ours[:-1], theirs[:len(ours) - 1]
                    shown = [f"{g} {f}".strip() for f, g in ra]
                    if [f for f, _ in ours] != [f for f, _ in theirs]:
                        row["problems"].append(f"authors {e['author']!r} vs CrossRef {shown}")
                    else:
                        for (of, og), (tf, tg) in zip(ours, theirs):
                            if (e["key"], of) in RECORD_GIVEN_DEFECTS:
                                continue
                            if not given_agrees(og, tg):
                                row["problems"].append(
                                    f"given name {og!r} for {of!r} vs CrossRef {tg!r}")
                # Reviewer 3 asked for complete records, so the volume and page
                # range have to agree with the registered ones, not just exist.
                cv, cp = str(msg.get("volume") or ""), str(msg.get("page") or "")
                if cv and e.get("volume") and clean(cv) != clean(e["volume"]):
                    row["problems"].append(f"volume {e['volume']!r} vs CrossRef {cv!r}")
                if cp and e.get("pages") and norm_pages(cp) != norm_pages(e["pages"]):
                    row["problems"].append(f"pages {e['pages']!r} vs CrossRef {cp!r}")
        else:
            row["source"] = "title search"
            msg = crossref_by_title(e.get("title", ""))
            if msg:
                ct = (msg.get("container-title") or [""])[0]
                row.update({"remote_title": (msg.get("title") or [""])[0],
                            "remote_venue": ct, "remote_year": cr_year(msg),
                            "remote_doi": msg.get("DOI")})
                row["problems"].append("no DOI in ref.bib; nearest CrossRef match shown for review")
            else:
                row["problems"].append("no DOI and no CrossRef match found")

        # An @article printed without a volume or a page range is the
        # "citations appear incomplete" case, whatever CrossRef says.
        if e["type"] == "article":
            for f in ("volume", "pages"):
                if not e.get(f):
                    row["problems"].append(f"incomplete: no {f}")

        # A conference paper printed without a page range is the same case.
        # NEURIPS_UNPAGINATED lists the proceedings that carry no page numbers
        # at all, so that a missing range there is not reported as a defect.
        if e["type"] == "inproceedings" and not e.get("pages"):
            if e["key"] not in NEURIPS_UNPAGINATED:
                row["problems"].append("incomplete: no pages")

        if row["problems"]:
            bad += 1
        report.append(row)
        flag = "FAIL" if row["problems"] else "ok  "
        print(f"[{n:2d}] {flag} {e['key']}")
        for p in row["problems"]:
            print(f"        - {p}")
        time.sleep(0.2)

    print(f"\n{bad} of {len(entries)} entries need attention")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
