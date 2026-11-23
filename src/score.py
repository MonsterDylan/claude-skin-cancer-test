"""Score every results/<model>.jsonl against the biopsy answer key.

Writes results/scorecard.json and prints a summary.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Published dermatologist benchmarks on the same task type (dermoscopic image,
# melanoma vs biopsied mole). Different image sets than ours, so a rough yardstick.
BENCHMARKS = [
    {"name": "157 dermatologists (Brinker et al., Eur J Cancer 2019)",
     "sensitivity": 0.741, "specificity": 0.600},
    {"name": "58 dermatologists (Haenssle et al., Ann Oncol 2018)",
     "sensitivity": 0.866, "specificity": 0.713},
]


def auc(pos, neg):
    """Probability a random melanoma scores higher than a random mole (ties = half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(rows, labels):
    ok = [r for r in rows if not r.get("error") and not r.get("refused")]
    tp = sum(1 for r in ok if labels[r["case"]]["label"] == "melanoma" and r["diagnosis"] == "melanoma")
    fn = sum(1 for r in ok if labels[r["case"]]["label"] == "melanoma" and r["diagnosis"] == "benign")
    tn = sum(1 for r in ok if labels[r["case"]]["label"] == "benign" and r["diagnosis"] == "benign")
    fp = sum(1 for r in ok if labels[r["case"]]["label"] == "benign" and r["diagnosis"] == "melanoma")
    pos = [r["melanoma_probability"] for r in ok if labels[r["case"]]["label"] == "melanoma"]
    neg = [r["melanoma_probability"] for r in ok if labels[r["case"]]["label"] == "benign"]

    def sub(diag):
        sel = [r for r in ok if labels[r["case"]]["diagnosis"] == diag]
        hit = sum(1 for r in sel if r["diagnosis"] == "melanoma")
        return {"n": len(sel), "caught": hit, "sensitivity": round(hit / len(sel), 3) if sel else None}

    u = [r.get("usage") or {} for r in rows]
    return {
        "model": rows[0]["model"],
        "answered": len(ok), "refused": sum(1 for r in rows if r.get("refused")),
        "errors": sum(1 for r in rows if r.get("error")),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "accuracy": round((tp + tn) / len(ok), 3),
        "sensitivity": round(tp / (tp + fn), 3),
        "specificity": round(tn / (tn + fp), 3),
        "auc": round(auc(pos, neg), 3),
        "invasive": sub("Melanoma Invasive"),
        "in_situ": sub("Melanoma in situ"),
        "input_tokens": sum(x.get("in", 0) for x in u),
        "output_tokens": sum(x.get("out", 0) for x in u),
        "mean_prob_melanoma": round(sum(pos) / len(pos), 1),
        "mean_prob_benign": round(sum(neg) / len(neg), 1),
    }


def main():
    with open(ROOT / "data" / "labels.csv") as f:
        labels = {r["case"]: r for r in csv.DictReader(f)}
    cards = {}
    for path in sorted((ROOT / "results").glob("*.jsonl")):
        latest = {}
        for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip()):
            if r["case"] not in latest or not r.get("error"):
                latest[r["case"]] = r  # keep the successful answer when a case was retried
        rows = list(latest.values())
        if rows:
            cards[path.stem] = score(rows, labels)
    out = {"n_images": len(labels), "models": cards, "benchmarks": BENCHMARKS}
    (ROOT / "results" / "scorecard.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
