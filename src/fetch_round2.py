"""Round 2 data: a fresh test set plus a separate calibration pool.

Nothing here overlaps round 1: no shared images, patients or lesions.
- data2/test:  200 new cases (50 invasive, 50 in situ, 100 nevi) for the A/B test
- data2/calib: 80 cases (20 invasive, 20 in situ, 40 nevi). Opus runs on these
  cold, and its mistakes become the teaching examples. The test set is never
  used for teaching, so round B can't just memorize answers.
"""
import csv
import random
import subprocess
import sys
from pathlib import Path

import requests

API = "https://api.isic-archive.com/api/v2/images/"
ROOT = Path(__file__).resolve().parent.parent
D2 = ROOT / "data2"
SEED = 2026
COMMON = " AND diagnosis_confirm_type:histopathology AND image_type:dermoscopic"
BUCKETS = [  # (label, query, test_n, calib_n)
    ("melanoma", 'diagnosis_3:"Melanoma Invasive"', 50, 20),
    ("melanoma", 'diagnosis_3:"Melanoma in situ"', 50, 20),
    ("benign", "diagnosis_3:Nevus", 100, 40),
]
PAGES = 16  # 1,600 candidates per bucket


def round1_exclusions():
    ids, pats, lesions = set(), set(), set()
    with open(ROOT / "data" / "labels.csv") as f:
        for row in csv.DictReader(f):
            ids.add(row["isic_id"])
            clin = requests.get(API + row["isic_id"] + "/", timeout=60).json()["metadata"].get("clinical", {})
            if clin.get("patient_id"):
                pats.add(clin["patient_id"])
            if clin.get("lesion_id"):
                lesions.add(clin["lesion_id"])
    return ids, pats, lesions


def pool(query):
    out, url, params = [], API + "search/", {"query": query + COMMON, "limit": 100}
    for _ in range(PAGES):
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        out += d["results"]
        url, params = d.get("next"), None
        if not url:
            break
    return out


def main():
    rng = random.Random(SEED)
    ids, pats, lesions = round1_exclusions()
    print(f"excluding {len(ids)} images, {len(pats)} patients, {len(lesions)} lesions from round 1", file=sys.stderr)
    split = {"test": [], "calib": []}
    for label, query, n_test, n_cal in BUCKETS:
        cands = pool(query)
        rng.shuffle(cands)
        got = []
        for c in cands:
            clin = c["metadata"].get("clinical", {})
            pid, lid = clin.get("patient_id"), clin.get("lesion_id")
            if c["isic_id"] in ids or (pid and pid in pats) or (lid and lid in lesions):
                continue
            got.append((label, c))
            ids.add(c["isic_id"])
            pats.add(pid) if pid else None
            lesions.add(lid) if lid else None
            if len(got) == n_test + n_cal:
                break
        split["test"] += got[:n_test]
        split["calib"] += got[n_test:]
        print(f"{query}: pool {len(cands)}, picked {len(got)}", file=sys.stderr)

    for name, items in split.items():
        rng.shuffle(items)
        raw, inp = D2 / name / "raw", D2 / name / "model_input"
        raw.mkdir(parents=True, exist_ok=True)
        inp.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (label, c) in enumerate(items, 1):
            case = f"{name}_{i:03d}"
            src = raw / f"{case}.jpg"
            if not src.exists():
                r = requests.get(c["files"]["full"]["url"], timeout=120)
                r.raise_for_status()
                src.write_bytes(r.content)
            # same resize as round 1: long edge 1568px, JPEG q90
            subprocess.run(["sips", "-Z", "1568", "-s", "format", "jpeg", "-s", "formatOptions", "90",
                            str(src), "--out", str(inp / src.name)], check=True, capture_output=True)
            clin = c["metadata"].get("clinical", {})
            rows.append({"case": case, "label": label, "isic_id": c["isic_id"],
                         "diagnosis": clin.get("diagnosis_3"), "age": clin.get("age_approx"),
                         "sex": clin.get("sex"), "site": clin.get("anatom_site_1"),
                         "license": c.get("copyright_license"), "attribution": c.get("attribution")})
        with open(D2 / name / "labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: saved {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
