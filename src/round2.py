"""Round 2: does teaching Opus with examples in the prompt make it better?

Steps (run in order):
  round2.py calib           Opus cold on the 80-case calibration pool
  round2.py pick            choose 20 teaching examples, weighted to its mistakes
  round2.py run zeroshot    Round A: Opus cold on the fresh 200
  round2.py run fewshot     Round B: Opus with the 20 examples in every prompt
  round2.py score           compare A vs B

Model, prompt, schema, effort and image size are identical to round 1.
The 20 examples sit in a cached prefix, so they're paid for once rather than 200 times.
"""
import base64
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_model import PROMPT, SCHEMA  # noqa: E402  same wording as round 1
from score import BENCHMARKS, auc  # noqa: E402

MODEL = "claude-opus-5-5"
D2 = ROOT / "data2"
OUT = ROOT / "results2"
OUT.mkdir(exist_ok=True)
client = anthropic.Anthropic(max_retries=6)


def b64(p):
    return base64.standard_b64encode(Path(p).read_bytes()).decode()


def labels(split):
    with open(D2 / split / "labels.csv") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def image_block(path):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(path)}}


def teaching_blocks():
    """The fixed prefix: intro, then 20 labeled example photos. Cached after the last one."""
    ex = json.loads((D2 / "examples.json").read_text())
    blocks = [{"type": "text", "text": (
        "Before the case you'll classify, here are 20 reference dermoscopic photos from "
        "other patients, each with its biopsy result. Several are cases that are easy to "
        "get wrong: melanomas that look harmless and moles that look alarming. Study them, "
        "then classify the final image.")}]
    for i, e in enumerate(ex, 1):
        result = ("MELANOMA (" + e["diagnosis"].replace("Melanoma ", "").lower() + ")"
                  if e["label"] == "melanoma" else "BENIGN mole (nevus)")
        blocks.append({"type": "text", "text": f"Reference {i}. Biopsy result: {result}."})
        blocks.append(image_block(D2 / "calib" / "model_input" / f"{e['case']}.jpg"))
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def ask(path, prefix=None):
    content = (prefix or []) + [
        {"type": "text", "text": "Now the case to classify:"} if prefix else None,
        image_block(path),
        {"type": "text", "text": PROMPT},
    ]
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [c for c in content if c]}],
    )
    u = resp.usage
    usage = {"in": u.input_tokens, "out": u.output_tokens,
             "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0}
    if resp.stop_reason == "refusal":
        return {"refused": True}, usage
    return json.loads(next(b.text for b in resp.content if b.type == "text")), usage


def run(split, out_name, prefix=None, workers=8):
    out = OUT / f"{out_name}.jsonl"
    done = set()
    if out.exists():
        done = {r["case"] for r in map(json.loads, filter(str.strip, out.read_text().splitlines())) if not r.get("error")}
    todo = [p for p in sorted((D2 / split / "model_input").glob("*.jpg")) if p.stem not in done]
    print(f"{out_name}: {len(done)} done, {len(todo)} to run", file=sys.stderr)

    def job(p):
        s = time.time()
        try:
            ans, usage, err = *ask(p, prefix), None
        except Exception as e:  # noqa: BLE001
            ans, usage, err = {}, {}, f"{type(e).__name__}: {str(e)[:200]}"
        return {"case": p.stem, "model": MODEL, **ans, "usage": usage, "error": err, "seconds": round(time.time() - s, 1)}

    with open(out, "a") as f:
        if prefix and todo:  # warm the cache with one request so the parallel ones read it
            f.write(json.dumps(job(todo.pop(0))) + "\n")
            f.flush()
        with ThreadPoolExecutor(workers) as ex:
            for n, fut in enumerate(as_completed([ex.submit(job, p) for p in todo]), 1):
                row = fut.result()
                f.write(json.dumps(row) + "\n")
                f.flush()
                if n % 25 == 0 or row["error"]:
                    print(f"  {n}/{len(todo)} {row['case']} {row.get('diagnosis', row['error'])}", file=sys.stderr)


def load(name):
    latest = {}
    for r in map(json.loads, filter(str.strip, (OUT / f"{name}.jsonl").read_text().splitlines())):
        if r["case"] not in latest or not r.get("error"):
            latest[r["case"]] = r
    return latest


def pick():
    """20 examples, 10 melanoma + 10 benign, mistakes first, then the closest calls."""
    lab, res = labels("calib"), load("calib")
    rng = random.Random(7)
    ex = []
    for label in ("melanoma", "benign"):
        rows = [r for r in res.values() if lab[r["case"]]["label"] == label and "diagnosis" in r]
        wrong = [r for r in rows if r["diagnosis"] != label]
        right = [r for r in rows if r["diagnosis"] == label]
        # closest calls first: correct answers it was least sure about
        right.sort(key=lambda r: r["melanoma_probability"] if label == "melanoma" else -r["melanoma_probability"])
        if label == "melanoma":  # keep invasive and in situ both represented
            wrong.sort(key=lambda r: lab[r["case"]]["diagnosis"])
        chosen = (wrong + right)[:10]
        print(f"{label}: {len(wrong)} mistakes in calib, using {min(len(wrong), 10)} of them", file=sys.stderr)
        ex += [{"case": r["case"], "label": label, "diagnosis": lab[r["case"]]["diagnosis"],
                "opus_cold_prob": r["melanoma_probability"], "was_mistake": r["diagnosis"] != label} for r in chosen]
    rng.shuffle(ex)
    (D2 / "examples.json").write_text(json.dumps(ex, indent=2))
    print(json.dumps(ex, indent=1))


def metrics(res, lab):
    ok = [r for r in res.values() if "diagnosis" in r]
    def n(lbl, dx): return sum(1 for r in ok if lab[r["case"]]["label"] == lbl and r["diagnosis"] == dx)
    tp, fn, tn, fp = n("melanoma", "melanoma"), n("melanoma", "benign"), n("benign", "benign"), n("benign", "melanoma")
    pos = sorted([r["melanoma_probability"] for r in ok if lab[r["case"]]["label"] == "melanoma"], reverse=True)
    neg = [r["melanoma_probability"] for r in ok if lab[r["case"]]["label"] == "benign"]
    at = {}
    for b in BENCHMARKS:
        k = -(-int(b["sensitivity"] * 1000) * len(pos) // 1000)  # ceil
        thr = pos[k - 1]
        at[b["name"][:20]] = {"spec": round(sum(x < thr for x in neg) / len(neg), 3), "derm_spec": b["specificity"]}
    u = [r["usage"] for r in res.values() if r.get("usage")]
    cost = sum(x["in"] * 4 + x["out"] * 20 + x.get("cache_write", 0) * 5 + x.get("cache_read", 0) * 0.2 for x in u) / 1e6
    return {"answered": len(ok), "refused": sum(1 for r in res.values() if r.get("refused")),
            "errors": sum(1 for r in res.values() if r.get("error")),
            "caught": tp, "missed": fn, "cleared": tn, "false_alarms": fp,
            "accuracy": round((tp + tn) / len(ok), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(auc(pos, neg), 3),
            "spec_at_derm_sensitivity": at, "cost_usd": round(cost, 2),
            "cache_read_tokens": sum(x.get("cache_read", 0) for x in u)}


def score():
    lab = labels("test")
    a, b = load("zeroshot"), load("fewshot")
    out = {"round_a_cold": metrics(a, lab), "round_b_with_examples": metrics(b, lab)}
    common = [c for c in lab if "diagnosis" in a.get(c, {}) and "diagnosis" in b.get(c, {})]
    flips = {"fixed": 0, "broke": 0}
    for c in common:
        ra, rb = a[c]["diagnosis"] == lab[c]["label"], b[c]["diagnosis"] == lab[c]["label"]
        flips["fixed"] += (not ra) and rb
        flips["broke"] += ra and not rb
    out["changed_answers"] = flips  # McNemar-style: only the cases where A and B disagree matter
    (OUT / "scorecard.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calib":
        run("calib", "calib")
    elif cmd == "pick":
        pick()
    elif cmd == "run":
        arm = sys.argv[2]
        run("test", arm, prefix=teaching_blocks() if arm == "fewshot" else None)
    elif cmd == "score":
        score()
