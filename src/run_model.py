"""Run one model blind over every image and log its answers.

Usage: run_model.py claude|fable|gpt|gemini [--limit N]

Every model gets the identical prompt and the identical resized image.
Results append to results/<model>.jsonl, so a crashed run resumes where it
stopped. Keys are read from env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY) and never printed.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "model_input"
OUT_DIR = ROOT / "results"

MODELS = {
    "claude": os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"),
    "fable": os.environ.get("FABLE_MODEL", "claude-fable-5-2"),
    "gpt": os.environ.get("GPT_MODEL", "gpt-6.2-astra"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-pro-latest"),
}

PROMPT = (
    "This is a research benchmark. The image is a dermoscopic photo of a single "
    "skin lesion that was later biopsied. Based only on the image, classify the "
    "lesion as melanoma or benign. Give your estimated probability (0 to 100) that "
    "it is melanoma, and name the visual features that drove your call in one "
    "short sentence. Answer as JSON with keys diagnosis, melanoma_probability, "
    "key_features."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "enum": ["melanoma", "benign"]},
        "melanoma_probability": {"type": "integer"},
        "key_features": {"type": "string"},
    },
    "required": ["diagnosis", "melanoma_probability", "key_features"],
    "additionalProperties": False,
}


def b64(path):
    return base64.standard_b64encode(path.read_bytes()).decode()


# ---------- Claude (official Anthropic SDK) ----------
_claude = None


def ask_claude(path, model_key="claude"):
    import anthropic

    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(max_retries=5)
    resp = _claude.messages.create(
        model=MODELS[model_key],
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        cat = resp.stop_details.category if resp.stop_details else None
        return {"refused": True, "refusal_category": cat}, usage
    text = next(b.text for b in resp.content if b.type == "text")
    return {**json.loads(text), "served_model": resp.model}, usage


# ---------- GPT (OpenAI Responses API, raw HTTP) ----------
def ask_gpt(path):
    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": MODELS["gpt"],
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(path)}"},
                    {"type": "input_text", "text": PROMPT},
                ],
            }],
            "text": {"format": {"type": "json_schema", "name": "lesion", "schema": SCHEMA, "strict": True}},
        },
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    usage = {"in": d["usage"]["input_tokens"], "out": d["usage"]["output_tokens"]}
    for item in d["output"]:
        if item["type"] == "message":
            for c in item["content"]:
                if c["type"] == "refusal":
                    return {"refused": True, "refusal_category": c.get("refusal")}, usage
                if c["type"] == "output_text":
                    return {**json.loads(c["text"]), "served_model": d.get("model")}, usage
    raise ValueError("no output_text in GPT response")


# ---------- Gemini (Generative Language API, raw HTTP) ----------
def ask_gemini(path):
    schema = {k: v for k, v in SCHEMA.items() if k != "additionalProperties"}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['gemini']}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64(path)}},
                {"text": PROMPT},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
        },
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    um = d.get("usageMetadata", {})
    usage = {"in": um.get("promptTokenCount", 0),
             "out": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}
    cand = (d.get("candidates") or [{}])[0]
    if cand.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST") or "content" not in cand:
        return {"refused": True, "refusal_category": cand.get("finishReason") or str(d.get("promptFeedback"))}, usage
    text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
    return {**json.loads(text), "served_model": d.get("modelVersion")}, usage


ASK = {"claude": ask_claude, "fable": lambda p: ask_claude(p, "fable"),
       "gpt": ask_gpt, "gemini": ask_gemini}


def with_retry(fn, path, tries=int(os.environ.get("RETRIES", 4))):
    for attempt in range(tries):
        try:
            return fn(path)
        except Exception as e:  # noqa: BLE001 - log and retry transient failures
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))  # 5, 10, 20, 40, 60s: rides out rate limits
            print(f"  retry {path.stem}: {type(e).__name__}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=ASK)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = OUT_DIR / f"{args.model}.jsonl"
    done = set()
    if out.exists():
        done = {r["case"] for r in map(json.loads, filter(str.strip, out.read_text().splitlines()))
                if not r.get("error")}  # errored rows get retried
    todo = [p for p in sorted(IMG_DIR.glob("case_*.jpg")) if p.stem not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.model} ({MODELS[args.model]}): {len(done)} done, {len(todo)} to run", file=sys.stderr)

    lock = threading.Lock()
    t0 = time.time()

    def job(p):
        s = time.time()
        try:
            ans, usage = with_retry(ASK[args.model], p)
            err = None
        except Exception as e:  # noqa: BLE001
            ans, usage, err = {}, {}, f"{type(e).__name__}: {str(e)[:200]}"
        return {"case": p.stem, "model": MODELS[args.model], **ans, "usage": usage,
                "error": err, "seconds": round(time.time() - s, 1)}

    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        for n, fut in enumerate(as_completed([ex.submit(job, p) for p in todo]), 1):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
            if n % 20 == 0 or row["error"]:
                print(f"  {n}/{len(todo)} {row['case']} {row.get('diagnosis', row['error'])}", file=sys.stderr)
    print(f"finished in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
