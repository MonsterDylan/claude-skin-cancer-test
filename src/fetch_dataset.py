"""Pull a balanced, biopsy-confirmed sample from the ISIC Archive.

100 melanomas (50 invasive, 50 in situ) + 100 benign nevi.
Every image is dermoscopic and confirmed by histopathology. The benign moles
were biopsied too, so they looked suspicious enough that a doctor cut them out.
That makes this the hard version of the test, the one dermatologists face.

Images are saved under anonymized names (case_001.jpg ...) so the filename
never leaks the answer. The answer key lives in data/labels.csv.
"""
import csv
import random
import sys
from pathlib import Path

import requests

API = "https://api.isic-archive.com/api/v2/images/search/"
ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
SEED = 42

BUCKETS = [
    ("melanoma", 'diagnosis_3:"Melanoma Invasive"', 50),
    ("melanoma", 'diagnosis_3:"Melanoma in situ"', 50),
    ("benign", "diagnosis_3:Nevus", 100),
]
COMMON = ' AND diagnosis_confirm_type:histopathology AND image_type:dermoscopic'
POOL_PAGES = 8  # 8 x 100 = up to 800 candidates per bucket to sample from


def pool(query):
    out, url, params = [], API, {"query": query + COMMON, "limit": 100}
    for _ in range(POOL_PAGES):
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
    picked, seen_patients, seen_lesions = [], set(), set()
    for label, query, n in BUCKETS:
        cands = pool(query)
        rng.shuffle(cands)
        got = 0
        for c in cands:
            clin = c["metadata"].get("clinical", {})
            pid, lid = clin.get("patient_id"), clin.get("lesion_id")
            if (pid and pid in seen_patients) or (lid and lid in seen_lesions):
                continue  # one image per patient/lesion, no near-duplicates
            picked.append((label, c))
            seen_patients.add(pid) if pid else None
            seen_lesions.add(lid) if lid else None
            got += 1
            if got == n:
                break
        print(f"{query}: pool {len(cands)}, picked {got}", file=sys.stderr)

    rng.shuffle(picked)  # mix the order so classes aren't in runs
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (label, c) in enumerate(picked, 1):
        case = f"case_{i:03d}"
        dest = IMG_DIR / f"{case}.jpg"
        if not dest.exists():
            img = requests.get(c["files"]["full"]["url"], timeout=120)
            img.raise_for_status()
            dest.write_bytes(img.content)
        clin = c["metadata"].get("clinical", {})
        rows.append({
            "case": case, "label": label, "isic_id": c["isic_id"],
            "diagnosis": clin.get("diagnosis_3"), "age": clin.get("age_approx"),
            "sex": clin.get("sex"), "site": clin.get("anatom_site_1"),
            "license": c.get("copyright_license"), "attribution": c.get("attribution"),
        })
    with open(ROOT / "data" / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
